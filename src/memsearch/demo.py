"""Deterministic synthetic protocol fixture. NOT an LLM, NOT a learned policy."""
import json
import re

from .budget import Counter
from .types import Call


class DemoBackend:
    def __init__(self):
        self.calls = []
        self.counter = Counter()

    def complete(self, messages, role):
        obs = json.loads(messages[-1]["content"])
        if role == "memory":
            if "max_memory_ops" in obs:
                existing = {s for item in obs["OLD_MEMORY"]["items"] for s in item["source_ids"]}
                operations = [{"op": "ADD", "text": e["text"], "source_ids": [e["id"]]}
                              for e in obs["NEW_EVIDENCE"] if e["id"] not in existing]
                result = {"operations": operations or [{"op": "NOOP"}]}
                text = json.dumps(result)
                self.calls.append(Call(role, messages, text, self.counter.count(json.dumps(messages)),
                                       self.counter.count(text), "lexical_proxy"))
                return text
            items = obs["OLD_MEMORY"]["items"] + [
                {"text": e["text"], "source_ids": [e["id"]]} for e in obs["NEW_EVIDENCE"]]
            seen, unique = set(), []
            for item in items:
                key = tuple(item["source_ids"])
                if key not in seen:
                    unique.append(item)
                    seen.add(key)
            result = {"items": unique, "unresolved": []}
        else:
            question = obs["QUESTION"]
            subject = re.search(r"What country is (.+)'s birthplace in\?", question)
            subject = subject.group(1) if subject else ""
            items = obs["memory"]["items"]
            birthplace, country, citations = None, None, []
            for item in items:
                match = re.fullmatch(re.escape(subject) + r" was born in (.+)\.", item["text"])
                if match:
                    birthplace = match.group(1)
                    citations = list(item["source_ids"])
                    break
            if birthplace:
                for item in items:
                    match = re.fullmatch(re.escape(birthplace) + r" is in (.+)\.", item["text"])
                    if match:
                        country = match.group(1)
                        citations += item["source_ids"]
                        break
            if role == "controller" and not country and obs["remaining_searches"] > 0:
                result = {"action": "SEARCH", "query": birthplace or subject}
            else:
                result = {"action": "FINISH", "answer": country or "unknown",
                          "evidence_ids": citations if country else []}
        text = json.dumps(result)
        self.calls.append(Call(role, messages, text, self.counter.count(json.dumps(messages)),
                               self.counter.count(text), "lexical_proxy"))
        return text
