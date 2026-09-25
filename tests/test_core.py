import copy
import json
import tempfile
import unittest
from pathlib import Path

from memsearch.agent import parse_action
from memsearch.budget import Counter, enforce_budget, serialize_memory
from memsearch.data import convert_hotpot, load_dataset
from memsearch.memory import Memory
from memsearch.metrics import answer_scores, group_advantages, reward, score
from memsearch.runner import evaluate, load_config, make_agent
from memsearch.types import Call, Evidence, Labels, MemoryItem, MemoryState

ROOT = Path(__file__).resolve().parents[1]


class Stub:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, messages, role):
        self.calls.append(Call(role, messages, self.response, 1, 1, "lexical_proxy"))
        return self.response


class ResearchIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/demo.toml")
        self.docs, self.tasks, self.labels = load_dataset(ROOT / "data/demo")

    def test_two_hop_episode_and_no_answer_label_in_initial_prompt(self):
        episode = make_agent(self.cfg, self.docs).run(self.tasks[0])
        self.assertEqual(episode.answer, "Norvia")
        self.assertEqual(episode.searches, 2)
        self.assertEqual(set(episode.evidence_ids), {"p1:0", "c1:0"})
        first = next(c for c in episode.calls if c.role == "controller")
        self.assertNotIn("Norvia", json.dumps(first.messages))
        self.assertEqual(score(episode, self.labels["toy1"])["memory_support_recall"], 1)

    def test_gold_label_changes_cannot_change_rollout(self):
        a = make_agent(self.cfg, self.docs).run(self.tasks[0])
        self.labels["toy1"] = Labels("SECRET_TEST_LABEL", ("d1:0",))
        b = make_agent(self.cfg, self.docs).run(self.tasks[0])
        self.assertEqual(a.payload(), b.payload())
        self.assertNotIn("SECRET_TEST_LABEL", json.dumps(b.payload()))

    def test_fake_source_rejected(self):
        backend = Stub('{"items":[{"text":"fake","source_ids":["unseen:0"]}]}')
        memory = Memory("structured", 100, Counter(), backend)
        with self.assertRaisesRegex(ValueError, "unseen"):
            memory.update("q", MemoryState(), [Evidence("known:0", "Known", 0, "fact")])

    def test_memory_budget_includes_schema_and_source_ids(self):
        counter = Counter()
        state = MemoryState([MemoryItem("long sentence " * 100, ("a:0",)), MemoryItem("fact", ("b:0",))])
        bounded = enforce_budget(state, 40, counter)
        self.assertLessEqual(counter.count(serialize_memory(bounded)), 40)
        self.assertEqual(len(state.items), 2)
        with self.assertRaises(ValueError):
            enforce_budget(MemoryState(), 1, counter)

    def test_search_budget_invalid_action_and_cost_accounting(self):
        backend = Stub('{"action":"SEARCH","query":"Ada Vale"}')
        agent = make_agent(self.cfg, self.docs, overrides={"controller": backend})
        episode = agent.run(self.tasks[0])
        self.assertEqual(episode.searches, 3)
        self.assertIn("search_budget_exceeded", episode.errors)
        self.assertLess(reward(episode, self.labels["toy1"], search_cost=0.1), 0)
        with self.assertRaises(ValueError):
            parse_action('{"action":"EXECUTE","query":"bad"}')

    def test_memory_failure_is_charged_for_attempted_search(self):
        bad = Stub('{"items":[{"text":"x","source_ids":["fake:0"]}]}')
        episode = make_agent(self.cfg, self.docs, overrides={"memory": bad}).run(self.tasks[0])
        self.assertEqual(episode.searches, 1)
        self.assertTrue(episode.errors)

    def test_unseen_final_citation_rejected(self):
        bad = Stub('{"action":"FINISH","answer":"Norvia","evidence_ids":["c1:0"]}')
        episode = make_agent(self.cfg, self.docs, overrides={"controller": bad}).run(self.tasks[0])
        self.assertEqual(episode.stop_reason, "invalid_output")
        self.assertEqual(episode.answer, "unknown")

    def test_fixed_retrieval_is_independent_of_memory(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["agent"]["policy"] = "fixed"
        a = make_agent(cfg, self.docs).run(self.tasks[0])
        cfg["agent"]["memory"] = "recent"
        b = make_agent(cfg, self.docs).run(self.tasks[0])
        self.assertEqual([s["evidence_ids"] for s in a.steps], [s["evidence_ids"] for s in b.steps])

    def test_grpo_zero_variance_and_normalization(self):
        self.assertEqual(group_advantages([1, 1, 1]), [0, 0, 0])
        adv = group_advantages([0, 1, 0, 1])
        self.assertAlmostEqual(sum(adv), 0)
        self.assertGreater(adv[1], 0)
        self.assertEqual(answer_scores("The Norvia!", "Norvia"), (1, 1))

    def test_hotpot_conversion_is_lossless_and_label_separated(self):
        source = [{"_id": "x", "question": "where?", "answer": "hidden-answer",
                   "context": [["City", ["A sentence."]]], "supporting_facts": [["City", 0]]}]
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "raw.json"
            file.write_text(json.dumps(source))
            out = Path(temp) / "processed"
            convert_hotpot(file, out)
            docs, tasks, labels = load_dataset(out)
            self.assertEqual(labels["x"].answer, "hidden-answer")
            self.assertNotIn("hidden-answer", (out / "tasks.jsonl").read_text())
            self.assertNotIn("hidden-answer", (out / "corpus.jsonl").read_text())
            self.assertEqual(len(docs), 1)

    def test_evaluation_manifest_fixture_label_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            report = evaluate(self.cfg, ROOT / "data/demo", out)
            self.assertTrue(report["synthetic_fixture"])
            self.assertFalse(report["entailment_evaluated"])
            self.assertIn("lexical_proxy", report["total_usage"])
            self.assertEqual(report["em"], 1)
            self.assertTrue((out / "manifest.json").exists())
            with self.assertRaises(FileExistsError):
                evaluate(self.cfg, ROOT / "data/demo", out)


if __name__ == "__main__":
    unittest.main()
