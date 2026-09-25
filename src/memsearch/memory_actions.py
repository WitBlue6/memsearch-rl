"""Deterministic memory environment. The policy chooses every mutation.

No LLM calls, automatic merging or eviction. A proposal either commits in full
or raises, leaving the previous state untouched. IDs are allocated by the env.
"""
from .budget import serialize_memory
from .types import MemoryItem, MemoryState


def apply_operations(old, evidence, proposal, budget, counter, max_ops=16):
    operations = proposal.get("operations")
    if not isinstance(operations, list) or not 1 <= len(operations) <= max_ops:
        raise ValueError("Provide 1..max_memory_ops explicit memory operations")
    if set(proposal) - {"operations", "unresolved"}:
        raise ValueError("Unexpected memory proposal fields")
    if any(not isinstance(op, dict) for op in operations):
        raise ValueError("Each memory operation must be an object")
    if any(op.get("op") == "NOOP" for op in operations):
        if operations != [{"op": "NOOP"}] or (
            "unresolved" in proposal and proposal["unresolved"] != old.unresolved
        ):
            raise ValueError("NOOP must stand alone and cannot change memory")
        if counter.count(serialize_memory(old)) > budget:
            raise ValueError("Memory budget exceeded: NOOP cannot repair an oversized state")
        return MemoryState(list(old.items), list(old.unresolved), old.next_id)

    items, next_id = list(old.items), old.next_id
    ids = [item.memory_id for item in items]
    if any(not key for key in ids) or len(ids) != len(set(ids)):
        raise ValueError("Decision memory requires unique stable memory IDs")
    known_sources = {e.id for e in evidence} | {s for item in old.items for s in item.source_ids}
    for op in operations:
        kind = op.get("op")
        if kind not in {"ADD", "UPDATE", "DELETE"}:
            raise ValueError("Memory op must be ADD, UPDATE, DELETE or NOOP")
        expected = {"op", "text", "source_ids"} if kind == "ADD" else (
            {"op", "target_id"} if kind == "DELETE" else {"op", "target_id", "text", "source_ids"})
        if set(op) != expected:
            raise ValueError(f"Invalid fields for {kind}")
        index = None
        if kind in {"UPDATE", "DELETE"}:
            target = op["target_id"]
            if not isinstance(target, str):
                raise ValueError("target_id must be a memory ID")
            index = next((i for i, item in enumerate(items) if item.memory_id == target), None)
            if index is None:
                raise ValueError("Memory target does not exist")
        if kind == "DELETE":
            items.pop(index)
            continue
        text, sources = op["text"], op["source_ids"]
        if not isinstance(text, str) or not text.strip() or not isinstance(sources, list) or not sources:
            raise ValueError("Memory write requires nonempty text and source_ids")
        if any(not isinstance(s, str) for s in sources) or not set(sources) <= known_sources:
            raise ValueError("Memory write cited unseen evidence")
        if kind == "ADD":
            key = f"m{next_id}"
            if any(item.memory_id == key for item in items):
                raise ValueError("Memory ID counter collision")
            items.append(MemoryItem(text, tuple(dict.fromkeys(sources)), key))
            next_id += 1
        else:
            items[index] = MemoryItem(text, tuple(dict.fromkeys(sources)), items[index].memory_id)
    unresolved = proposal.get("unresolved", old.unresolved)
    if not isinstance(unresolved, list) or any(not isinstance(q, str) for q in unresolved):
        raise ValueError("unresolved must be a list of strings")
    new = MemoryState(items, list(unresolved), next_id)
    if counter.count(serialize_memory(new)) > budget:
        raise ValueError("Memory budget exceeded: policy must DELETE or compress with UPDATE")
    return new
