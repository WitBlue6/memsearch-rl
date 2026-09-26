"""Aggregate completed research runs, training diagnostics and paired uncertainty."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from memsearch.data import read_jsonl
from memsearch.research import paired_bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = Path(a.root)
    result = {"evaluations": {}, "training": {}, "paired_vs_before": {}, "diagnostics": {}}
    for path in sorted(root.rglob("summary.json")):
        directory = path.parent
        if not (directory / "metrics.jsonl").exists():
            continue
        name = str(directory.relative_to(root))
        result["evaluations"][name] = json.loads(path.read_text())
        metrics = list(read_jsonl(directory / "metrics.jsonl"))
        valid = [r for r in metrics if not r["invalid"]]
        steps = [step for row in metrics for step in row.get("memory_trace_metrics", [])]
        errors = Counter(error for row in read_jsonl(directory / "trajectories.jsonl") for error in row["errors"])
        result["diagnostics"][name] = {
            "errors": dict(errors), "valid_only_f1": statistics.mean(r["f1"] for r in valid) if valid else None,
            "steps_dropping_gold_source": sum(bool(s["lost_support_ids"]) for s in steps),
            "memory_steps": len(steps),
            "mean_memory_size": statistics.mean(s["memory_size"] for s in steps) if steps else None,
            "note": "Source IDs are coverage proxies, not semantic entailment. Valid-only F1 is diagnostic, not the headline metric."}
    before = root / ("before-matched" if (root / "before-matched" / "metrics.jsonl").exists() else "before") / "metrics.jsonl"
    result["paired_baseline"] = str(before.parent.relative_to(root))
    if before.exists():
        baseline = {r["id"]: r for r in read_jsonl(before)}
        fingerprint = json.loads((before.parent / "manifest.json").read_text())["dataset_sha256"]
        for name in result["evaluations"]:
            directory = root / name
            if json.loads((directory / "manifest.json").read_text())["dataset_sha256"] == fingerprint:
                result["paired_vs_before"][name] = paired_bootstrap(baseline, {r["id"]: r for r in read_jsonl(directory / "metrics.jsonl")})
    for path in root.rglob("rank-*.jsonl"):
        rows = list(read_jsonl(path))
        result["training"][str(path.relative_to(root))] = {
            "steps": len(rows), "updated": sum(r.get("updated", False) for r in rows),
            "zero_variance": sum(r.get("zero_variance", False) for r in rows),
            "learning_curve": [{k: row.get(k) for k in ("step", "rewards", "diagnostics")} for row in rows]}
    seed_groups = {}
    for name, metrics in result["evaluations"].items():
        parts = name.split("-")
        if len(parts) == 3 and parts[0] == "curve":
            seed_groups.setdefault(parts[2], []).append(metrics)
    result["training_seed_variation"] = {step: {
        "runs": len(rows), "mean_f1": statistics.mean(r["f1"] for r in rows),
        "sample_std_f1": statistics.stdev(r["f1"] for r in rows) if len(rows) > 1 else None}
        for step, rows in seed_groups.items()}
    with Path(a.out).open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: {m: v[m] for m in ("em", "f1", "invalid")} for k,v in result["evaluations"].items()}, indent=2))


if __name__ == "__main__":
    main()
