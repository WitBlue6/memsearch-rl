import copy
import json
import unittest
from pathlib import Path

from memsearch.budget import Counter, serialize_memory
from memsearch.data import load_dataset
from memsearch.memory import Memory
from memsearch.memory_actions import apply_operations
from memsearch.metrics import group_advantages, memory_operation_counts, reward
from memsearch.runner import load_config, make_agent
from memsearch.training import Sample, collect_group
from memsearch.types import Call, Evidence, MemoryItem, MemoryState

ROOT = Path(__file__).resolve().parents[1]


class ScriptedPolicy:
    """Chosen actions for causal wiring tests, not an on-policy training result."""
    def __init__(self, proposals):
        self.proposals = iter(proposals)
        self.calls, self.samples = [], []

    def complete(self, messages, role):
        proposal = next(self.proposals)
        text = json.dumps(proposal)
        # Byte tokens let the test confirm operations AND arguments survive collection.
        completion_ids = list(text.encode())
        self.samples.append(Sample([1, 2], completion_ids, [-1.0] * len(completion_ids)))
        self.calls.append(Call(role, messages, text, 1, len(completion_ids), "byte_fixture"))
        return text


class MemoryDecisionTests(unittest.TestCase):
    def setUp(self):
        self.counter = Counter()
        self.evidence = [Evidence("d:0", "Doc", 0, "New fact")]
        self.old = MemoryState([MemoryItem("Old fact", ("old:0",), "m0")], [], 1)

    def apply(self, ops, old=None, budget=1000):
        return apply_operations(self.old if old is None else old, self.evidence,
                                {"operations": ops}, budget, self.counter)

    def test_add_assigns_id_update_preserves_id_delete_does_not_reuse_id(self):
        added = self.apply([{"op": "ADD", "text": "New fact", "source_ids": ["d:0"]}])
        self.assertEqual([i.memory_id for i in added.items], ["m0", "m1"])
        changed = self.apply([{"op": "UPDATE", "target_id": "m1", "text": "Updated fact",
                               "source_ids": ["d:0"]}], added)
        self.assertEqual(changed.items[1].memory_id, "m1")
        self.assertEqual(changed.items[1].text, "Updated fact")
        deleted = self.apply([{"op": "DELETE", "target_id": "m1"}], changed)
        again = self.apply([{"op": "ADD", "text": "New fact", "source_ids": ["d:0"]}], deleted)
        self.assertEqual(again.items[1].memory_id, "m2")
        self.assertEqual(self.old.items[0].text, "Old fact")

    def test_noop_preserves_state_and_ignores_new_evidence(self):
        result = self.apply([{"op": "NOOP"}])
        self.assertEqual(result.payload(), self.old.payload())
        self.assertNotIn("d:0", {s for item in result.items for s in item.source_ids})
        self.assertIsNot(result.items, self.old.items)

    def test_explicit_merge_is_update_plus_delete(self):
        old = MemoryState([MemoryItem("Fact A", ("a:0",), "m0"),
                           MemoryItem("Fact B", ("b:0",), "m1")], [], 2)
        merged = self.apply([{"op": "UPDATE", "target_id": "m0", "text": "Facts A and B",
                              "source_ids": ["a:0", "b:0"]},
                             {"op": "DELETE", "target_id": "m1"}], old)
        self.assertEqual(len(merged.items), 1)
        self.assertEqual(merged.items[0].source_ids, ("a:0", "b:0"))

    def test_overflow_rejected_without_automatic_eviction(self):
        before = copy.deepcopy(self.old.payload())
        budget = self.counter.count(serialize_memory(self.old))
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            self.apply([{"op": "ADD", "text": "a new long fact", "source_ids": ["d:0"]}], budget=budget)
        self.assertEqual(self.old.payload(), before)

    def test_invalid_second_operation_rolls_back_delete(self):
        before = copy.deepcopy(self.old.payload())
        with self.assertRaisesRegex(ValueError, "unseen"):
            self.apply([{"op": "DELETE", "target_id": "m0"},
                        {"op": "ADD", "text": "bad", "source_ids": ["unseen:9"]}])
        self.assertEqual(self.old.payload(), before)

    def test_deleted_target_and_unknown_fields_rejected(self):
        proposals = [
            [{"op": "UPDATE", "target_id": "missing", "text": "x", "source_ids": ["d:0"]}],
            [{"op": "DELETE", "target_id": "m0"}, {"op": "DELETE", "target_id": "m0"}],
            [{"op": "ADD", "text": "x", "source_ids": ["d:0"], "memory_id": "m999"}],
            [{"op": "NOOP"}, {"op": "DELETE", "target_id": "m0"}],
        ]
        for ops in proposals:
            with self.subTest(ops=ops), self.assertRaises(ValueError):
                self.apply(ops)

    def test_old_replacement_schema_is_not_an_action(self):
        policy = ScriptedPolicy([{"items": [], "unresolved": []}])
        memory = Memory("decision", 200, self.counter, policy)
        with self.assertRaisesRegex(ValueError, "explicit memory operations"):
            memory.update("question", self.old, self.evidence)
        self.assertFalse(memory.last_decision["applied"])

    def test_memory_loss_signal_depends_on_policy_decision_with_same_evidence(self):
        cfg = load_config(ROOT / "configs/demo.toml")
        cfg["agent"].update(policy="fixed", max_searches=1, top_k=5, memory_budget=500)
        docs, tasks, labels = load_dataset(ROOT / "data/demo")
        add = {"operations": [
            {"op": "ADD", "text": "Ada Vale was born in Luma.", "source_ids": ["p1:0"]},
            {"op": "ADD", "text": "Luma is in Norvia.", "source_ids": ["c1:0"]}]}
        policy = ScriptedPolicy([add, {"operations": [{"op": "NOOP"}]}])
        agent = make_agent(cfg, docs, overrides={"memory": policy})
        import io
        trace = io.StringIO()
        groups, rewards, episodes = collect_group(agent, tasks[0], labels["toy1"], policy, 2, trace_file=trace)
        traced = [json.loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual(len(traced), 2)
        self.assertEqual(traced[0]["samples"][0]["completion_ids"], groups[0][0].completion_ids)
        self.assertEqual(traced[0]["episode"]["calls"][0]["response"], policy.calls[0].response)
        self.assertEqual(rewards, [1.0, 0.0])
        self.assertEqual(episodes[0]["memory_operations"]["ADD"], 2)
        self.assertEqual(episodes[1]["memory_operations"]["NOOP"], 1)
        self.assertEqual(len(groups[0]), 1)
        decoded = bytes(groups[0][0].completion_ids).decode()
        self.assertIn('"op": "ADD"', decoded)
        self.assertIn('"source_ids"', decoded)
        self.assertIn('"NOOP"', bytes(groups[1][0].completion_ids).decode())
        advantages = group_advantages(rewards)
        self.assertGreater(advantages[0], 0)
        self.assertLess(advantages[1], 0)
        self.assertNotIn('"answer":', json.dumps(policy.calls[0].messages))

    def test_rejected_write_is_logged_and_penalized(self):
        cfg = load_config(ROOT / "configs/demo.toml")
        docs, tasks, labels = load_dataset(ROOT / "data/demo")
        policy = ScriptedPolicy([{"operations": [{"op": "ADD", "text": "fake", "source_ids": ["fake:0"]}]}])
        episode = make_agent(cfg, docs, overrides={"memory": policy}).run(tasks[0])
        self.assertEqual(episode.steps[0]["memory_before"], episode.steps[0]["memory"])
        self.assertFalse(episode.steps[0]["memory_decision"]["applied"])
        self.assertEqual(memory_operation_counts(episode)["ADD"], 0)
        self.assertLess(reward(episode, labels["toy1"]), 0)

    def test_discarded_evidence_not_in_future_memory_observation(self):
        policy = ScriptedPolicy([{"operations": [{"op": "NOOP"}]}, {"operations": [{"op": "NOOP"}]}])
        memory = Memory("decision", 200, self.counter, policy)
        state = memory.update("q", MemoryState(), [Evidence("secret:0", "Secret", 0, "discarded text")])
        memory.update("q", state, [Evidence("fresh:0", "New", 0, "fresh text")])
        self.assertNotIn("discarded text", json.dumps(policy.calls[1].messages))
        self.assertNotIn("secret:0", json.dumps(policy.calls[1].messages))


if __name__ == "__main__":
    unittest.main()
