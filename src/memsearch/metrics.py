import re
import string
from collections import Counter, defaultdict


def normalize(answer):
    text = answer.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction, target):
    p, t = normalize(prediction), normalize(target)
    em = float(p == t)
    if p in {"yes", "no", "noanswer"} or t in {"yes", "no", "noanswer"}:
        return em, em
    overlap = sum((Counter(p.split()) & Counter(t.split())).values())
    f1 = 2 * overlap / (len(p.split()) + len(t.split())) if overlap else em
    return em, f1


def recall(pred, gold):
    return len(set(pred) & set(gold)) / len(set(gold)) if gold else None


def score(episode, labels):
    em, f1 = answer_scores(episode.answer, labels.answer)
    memory_ids = {s for item in episode.memory.items for s in item.source_ids}
    usage = defaultdict(lambda: {"input": 0, "output": 0})
    for call in episode.calls:
        usage[call.token_unit]["input"] += call.input_tokens
        usage[call.token_unit]["output"] += call.output_tokens
    citations = set(episode.evidence_ids)
    return {"id": episode.task_id, "em": em, "f1": f1,
            "retrieved_support_recall": recall(episode.retrieved_ids, labels.supporting_ids),
            "memory_support_recall": recall(memory_ids, labels.supporting_ids),
            "citation_support_recall": recall(citations, labels.supporting_ids),
            "citation_support_precision": len(citations & set(labels.supporting_ids)) / len(citations) if citations else 0.0,
            "searches": episode.searches, "invalid": int(bool(episode.errors)),
            "stop_reason": episode.stop_reason, "usage": dict(usage),
            "memory_operations": memory_operation_counts(episode)}


def memory_operation_counts(episode):
    counts = {op: 0 for op in ("ADD", "UPDATE", "DELETE", "NOOP")}
    for step in episode.steps:
        decision = step.get("memory_decision")
        if decision and decision["applied"]:
            for operation in decision["proposal"]["operations"]:
                counts[operation["op"]] += 1
    return counts


def aggregate(rows):
    if not rows:
        raise ValueError("No evaluation examples")
    metrics = ("em", "f1", "retrieved_support_recall", "memory_support_recall",
               "citation_support_recall", "citation_support_precision", "searches", "invalid")
    result = {"examples": len(rows)}
    for key in metrics:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = sum(values) / len(values) if values else None
    units = {unit for row in rows for unit in row["usage"]}
    result["total_usage"] = {unit: {kind: sum(r["usage"].get(unit, {}).get(kind, 0) for r in rows)
                                   for kind in ("input", "output")} for unit in sorted(units)}
    result["entailment_evaluated"] = False
    result["total_memory_operations"] = {
        op: sum(row.get("memory_operations", {}).get(op, 0) for row in rows)
        for op in ("ADD", "UPDATE", "DELETE", "NOOP")}
    return result


def reward(episode, labels, search_cost=0.0, invalid_cost=0.1):
    if search_cost < 0 or invalid_cost < 0:
        raise ValueError("Costs must be nonnegative")
    _, f1 = answer_scores(episode.answer, labels.answer)
    correctness = f1 if not episode.errors else 0.0
    return correctness - search_cost * episode.searches - invalid_cost * bool(episode.errors)


def group_advantages(rewards, epsilon=1e-6):
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two rollouts per task")
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    return [(r - mean) / (std + epsilon) for r in rewards]
