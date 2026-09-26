import json
import unittest
from memsearch.budget import Counter
from memsearch.memory import Memory
from memsearch.types import MemoryState, Evidence
from memsearch.research import paired_bootstrap


class ResearchTests(unittest.TestCase):
    def test_summary_overflow_is_rejected_without_mutating_old_memory(self):
        class Backend:
            def complete(self, messages, role):
                return json.dumps({"items": [{"text": "fact " * 100, "source_ids": ["doc:0"]}], "unresolved": []})
        old = MemoryState()
        memory = Memory("summary", 70, Counter(), Backend(), overflow="reject")
        evidence = [Evidence("doc:0", "doc", 0, "fact")]
        with self.assertRaisesRegex(ValueError, "budget"):
            memory.update("question", old, evidence)
        self.assertEqual(old.items, [])
        trimmed = Memory("summary", 70, Counter(), Backend(), overflow="trim").update("question", old, evidence)
        self.assertEqual(trimmed.items, [])

    def test_paired_comparison_aligns_task_ids_and_rejects_different_sets(self):
        before = {"a": {"f1": 0}, "b": {"f1": 1}}
        after = {"b": {"f1": 1}, "a": {"f1": 1}}
        result = paired_bootstrap(before, after, repeats=100)
        self.assertEqual(result["delta"], .5)
        with self.assertRaises(ValueError):
            paired_bootstrap(before, {"a": {"f1": 1}})

    def test_controller_uses_memory_query_and_frozen_reader_at_budget(self):
        from memsearch.agent import Agent
        from memsearch.types import Document, Task
        class Controller:
            calls = []
            observations = []
            def complete(self, messages, role):
                obs = json.loads(messages[-1]["content"])
                self.observations.append(obs)
                self.assert_no_labels(obs)
                return json.dumps({"action": "SEARCH", "query": "follow entity" if obs["memory"]["items"] else "initial"})
            def assert_no_labels(self, obs):
                assert "answer" not in obs and "supporting_ids" not in obs
        class Reader:
            calls = []
            count = 0
            def complete(self, messages, role):
                self.count += 1
                return '{"answer":"done","evidence_ids":["doc:0"]}'
        class Retriever:
            queries = []
            def search(self, query, k, scope):
                self.queries.append(query)
                return [Document("doc", "title", ("entity fact",))]
        controller, reader, retriever = Controller(), Reader(), Retriever()
        agent = Agent(retriever, Memory("recent", 200, Counter()),
                      {"policy": "model", "max_searches": 2, "top_k": 1, "controller_reader": True}, controller, reader)
        result = agent.run(Task("t", "question", ("doc",)))
        self.assertEqual(retriever.queries, ["initial", "follow entity"])
        self.assertEqual(reader.count, 1)
        self.assertEqual(result.answer, "done")
        self.assertEqual(result.errors, [])

    def test_memory_loss_diagnostic_uses_labels_only_after_episode(self):
        from memsearch.research import mechanism
        from memsearch.types import Labels
        from types import SimpleNamespace
        episode = SimpleNamespace(steps=[{"search": 2, "memory_size": 30,
                "memory_before": {"items": [{"source_ids": ["gold", "noise"]}]},
                "memory": {"items": [{"source_ids": ["noise"]}]}}])
        result = mechanism(episode, Labels("answer", ("gold",)))
        self.assertEqual(result[0]["lost_support_ids"], ["gold"])
        self.assertEqual(result[0]["support_recall"], 0)

    def test_frozen_reader_controller_does_not_generate_an_answer(self):
        from memsearch.structured import controller_schema
        from memsearch.prompts import messages
        obs = {"QUESTION": "q", "memory": {"items": []}, "FROZEN_READER": True}
        prompt = messages("controller", obs)
        schema = controller_schema(prompt)
        self.assertEqual(schema["anyOf"][1]["required"], ["action"])
        self.assertNotIn("answer", schema["anyOf"][1]["properties"])

    def test_parallel_evaluation_keeps_task_order_and_isolated_calls(self):
        import tempfile
        from pathlib import Path
        from memsearch.runner import load_config, evaluate
        from memsearch.data import read_jsonl
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs/demo.toml")
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "one", Path(temp) / "two"
            first = evaluate(cfg, root / "data/demo", a, workers=1)
            second = evaluate(cfg, root / "data/demo", b, workers=2)
            self.assertEqual(first, second)
            self.assertEqual(list(read_jsonl(a / "metrics.jsonl")), list(read_jsonl(b / "metrics.jsonl")))

    def test_controller_schema_has_search_and_finish(self):
        from memsearch.structured import CONTROLLER_SCHEMA
        variants = CONTROLLER_SCHEMA["anyOf"]
        self.assertEqual([v["properties"]["action"]["const"] for v in variants], ["SEARCH", "FINISH"])


if __name__ == "__main__":
    unittest.main()
