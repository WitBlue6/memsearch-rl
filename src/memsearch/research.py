"""Research helpers: label-only offline diagnostics and paired, task-level statistics."""
import random
from .metrics import recall


def mechanism(episode, labels):
    gold = set(labels.supporting_ids)
    result = []
    for step in episode.steps:
        before = {s for item in step["memory_before"]["items"] for s in item["source_ids"]}
        after = {s for item in step["memory"]["items"] for s in item["source_ids"]}
        result.append({"search": step["search"], "memory_size": step["memory_size"],
                       "support_recall": recall(after, gold),
                       "lost_support_ids": sorted((before - after) & gold),
                       "new_support_ids": sorted((after - before) & gold)})
    return result


def paired_bootstrap(before, after, key="f1", repeats=2000, seed=42):
    if not before or set(before) != set(after):
        raise ValueError("Paired comparison requires identical nonempty task IDs")
    deltas = [after[k][key] - before[k][key] for k in sorted(before)]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(repeats))
    return {"delta": sum(deltas) / len(deltas), "ci95": [samples[int(.025 * repeats)], samples[int(.975 * repeats)]],
            "examples": len(deltas), "seed": seed, "resamples": repeats,
            "note": "Task bootstrap for this pair; does not measure training-seed uncertainty."}
