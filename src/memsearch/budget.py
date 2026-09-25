import json
import re

from .types import MemoryState


class Counter:
    """Offline units are explicitly proxies, never reported as model tokens."""
    unit = "lexical_proxy"

    def count(self, text):
        return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", text))


class ModelCounter(Counter):
    unit = "model_token"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def serialize_memory(memory):
    return json.dumps(memory.payload(), ensure_ascii=False, separators=(",", ":"))


def enforce_budget(memory, budget, counter):
    """Drop whole entries, preserving valid IDs and JSON; no silent sentence truncation."""
    result = MemoryState(list(memory.items), list(memory.unresolved), memory.next_id)
    if counter.count(serialize_memory(MemoryState())) > budget:
        raise ValueError("Memory budget is smaller than the empty memory schema")
    while counter.count(serialize_memory(result)) > budget:
        if result.unresolved:
            result.unresolved.pop()
        elif result.items:
            result.items.pop()
        else:
            break
    return result
