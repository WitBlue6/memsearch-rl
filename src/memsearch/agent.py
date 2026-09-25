from .backends import parse_json
from .budget import serialize_memory
from .prompts import messages
from .types import Action, Episode, MemoryState


def parse_action(text):
    obj = parse_json(text)
    kind = obj.get("action")
    if kind == "SEARCH":
        query = obj.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("SEARCH requires a nonempty query")
        return Action(kind, query=query.strip())
    if kind == "FINISH":
        return parse_answer(obj)
    raise ValueError("Action must be SEARCH or FINISH")


def parse_answer(obj):
    answer, ids = obj.get("answer"), obj.get("evidence_ids", [])
    if not isinstance(answer, str) or not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
        raise ValueError("Invalid answer or evidence IDs")
    return Action("FINISH", answer=answer, evidence_ids=tuple(dict.fromkeys(ids)))


def validate_citations(action, memory):
    available = {s for item in memory.items for s in item.source_ids}
    if not set(action.evidence_ids) <= available:
        raise ValueError("Answer cited evidence absent from current memory")


class Agent:
    def __init__(self, retriever, memory, config, controller=None, reader=None):
        self.retriever, self.memory, self.config = retriever, memory, config
        self.controller, self.reader = controller, reader
        if config["policy"] not in {"fixed", "model"}:
            raise ValueError("policy must be fixed or model")
        if config["policy"] == "model" and controller is None:
            raise ValueError("Model policy requires a backend")
        if reader is None and config["policy"] == "fixed":
            raise ValueError("Fixed policy requires a reader")

    def run(self, task):
        # Deliberately receives Task, not Labels or an answer-bearing dataset row.
        cfg = self.config
        state, steps, retrieved, errors = MemoryState(), [], [], []
        backends = {id(b): b for b in (self.controller, self.reader, self.memory.backend) if b is not None}
        offsets = {k: len(b.calls) for k, b in backends.items()}
        calls = []

        def collect_calls():
            for key, backend in backends.items():
                calls.extend(backend.calls[offsets[key]:])
                offsets[key] = len(backend.calls)

        searches, action, stop = 0, Action("FINISH", answer="unknown"), "budget"
        queries = []
        for _ in range(cfg["max_searches"] + 1):
            observation = {"QUESTION": task.question, "memory": state.payload(),
                           "remaining_searches": cfg["max_searches"] - searches,
                           "previous_queries": list(queries)}
            try:
                if cfg["policy"] == "model":
                    action = parse_action(self.controller.complete(messages("controller", observation), "controller"))
                elif searches < cfg["max_searches"]:
                    # Predetermined retrieval schedule, independent of memory and labels.
                    action = Action("SEARCH", query=task.question)
                else:
                    action = parse_answer(parse_json(self.reader.complete(messages("reader", observation), "reader")))
                collect_calls()
                if action.kind == "FINISH":
                    validate_citations(action, state)
                    stop = "finish"
                    break
                if searches >= cfg["max_searches"]:
                    errors.append("search_budget_exceeded")
                    action = Action("FINISH", answer="unknown")
                    break
                searches += 1
                queries.append(action.query)
                scope = task.candidate_ids if cfg.get("retrieval_scope", "candidate") == "candidate" else None
                if cfg["policy"] == "fixed":
                    ranking = self.retriever.search(action.query, cfg["top_k"] * cfg["max_searches"], scope)
                    docs = ranking[(searches - 1) * cfg["top_k"]:searches * cfg["top_k"]]
                else:
                    docs = self.retriever.search(action.query, cfg["top_k"], scope)
                evidence = [e for d in docs for e in d.evidence()]
                # Consistent observation cap across all conditions, never use gold support to select.
                selected, used = [], 0
                for item in evidence:
                    size = self.memory.counter.count(item.text + item.id + item.title)
                    if used + size <= cfg.get("evidence_budget", 1800):
                        selected.append(item)
                        used += size
                retrieved.extend(e.id for e in selected)
                step = {"search": searches, "query": action.query,
                              "document_ids": [d.id for d in docs], "evidence_ids": [e.id for e in selected],
                              "memory_before": state.payload(), "budget_unit": self.memory.counter.unit}
                steps.append(step)
                try:
                    state = self.memory.update(task.question, state, selected)
                finally:
                    collect_calls()
                    step.update(memory=state.payload(), memory_decision=self.memory.last_decision,
                                memory_size=self.memory.counter.count(serialize_memory(state)))
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"invalid_output: {exc}")
                stop = "invalid_output"
                action = Action("FINISH", answer="unknown")
                break
        collect_calls()
        return Episode(task.id, task.question, action.answer, list(action.evidence_ids), state,
                       list(dict.fromkeys(retrieved)), steps, calls, errors, stop, searches)
