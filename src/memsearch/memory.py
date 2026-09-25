from dataclasses import asdict

from .backends import parse_json
from .budget import enforce_budget, serialize_memory
from .memory_actions import apply_operations
from .prompts import messages
from .retrieval import words
from .types import MemoryItem, MemoryState


class Memory:
    def __init__(self, mode, budget, counter, backend=None, max_ops=16):
        if mode not in {"recent", "extractive", "summary", "structured", "decision"}:
            raise ValueError(f"Unknown memory mode: {mode}")
        self.mode, self.budget, self.counter, self.backend = mode, budget, counter, backend
        if not isinstance(max_ops, int) or max_ops < 1:
            raise ValueError("max_memory_ops must be a positive integer")
        self.max_ops, self.last_decision = max_ops, None
        if mode in {"summary", "structured", "decision"} and backend is None:
            raise ValueError("LLM memory requires an explicit backend")

    def update(self, question, old, evidence):
        self.last_decision = None
        if self.mode == "decision":
            observation = {"QUESTION": question, "OLD_MEMORY": old.payload(),
                           "NEW_EVIDENCE": [asdict(e) for e in evidence],
                           "memory_budget": self.budget, "budget_unit": self.counter.unit,
                           "memory_size": self.counter.count(serialize_memory(old)),
                           "max_memory_ops": self.max_ops}
            result = parse_json(self.backend.complete(messages("memory_decision", observation), "memory"))
            self.last_decision = {"proposal": result, "applied": False}
            state = apply_operations(old, evidence, result, self.budget, self.counter, self.max_ops)
            self.last_decision["applied"] = True
            return state
        known = {s for item in old.items for s in item.source_ids} | {e.id for e in evidence}
        if self.mode in {"recent", "extractive"}:
            # Exact sentence storage provides a factual, non-neural baseline.
            new = [MemoryItem(e.text, (e.id,)) for e in evidence]
            items = new + old.items
            unique, seen = [], set()
            for item in items:
                if item.source_ids not in seen:
                    seen.add(item.source_ids)
                    unique.append(item)
            if self.mode == "extractive":
                question_words = set(words(question))
                unique.sort(key=lambda x: -len(question_words & set(words(x.text))))
            state = MemoryState(unique)
        else:
            prompt = messages("memory" if self.mode == "structured" else "summary", {
                "QUESTION": question, "OLD_MEMORY": old.payload(),
                "NEW_EVIDENCE": [asdict(e) for e in evidence],
                "memory_budget": self.budget, "budget_unit": self.counter.unit})
            result = parse_json(self.backend.complete(prompt, "memory"))
            if not isinstance(result.get("items"), list) or len(result["items"]) > 100:
                raise ValueError("Invalid memory items")
            items = []
            for item in result["items"]:
                if not isinstance(item, dict):
                    raise ValueError("Invalid memory item")
                text, ids = item.get("text"), item.get("source_ids")
                if not isinstance(text, str) or not isinstance(ids, list) or not ids:
                    raise ValueError("Memory facts require text and source IDs")
                if any(not isinstance(s, str) for s in ids) or not set(ids) <= known:
                    raise ValueError("Memory cited unseen evidence")
                items.append(MemoryItem(text, tuple(dict.fromkeys(ids))))
            unresolved = result.get("unresolved", [])
            if not isinstance(unresolved, list) or any(not isinstance(s, str) for s in unresolved):
                raise ValueError("Invalid unresolved questions")
            state = MemoryState(items, unresolved)
        # Valid ID != entailed fact. Entailment auditing is explicitly separate.
        return enforce_budget(state, self.budget, self.counter)
