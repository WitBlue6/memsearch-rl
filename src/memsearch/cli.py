import argparse
import copy
import json
from pathlib import Path

from .data import convert_hotpot, load_dataset, read_jsonl, write_jsonl
from .runner import evaluate, load_config


def main():
    p = argparse.ArgumentParser(description="Evidence memory and search RL experiments")
    subs = p.add_subparsers(dest="command", required=True)
    for name in ("eval", "matrix"):
        sub = subs.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--data", required=True)
        sub.add_argument("--out", required=True)
    sub = subs.add_parser("prepare-hotpot")
    sub.add_argument("--input", required=True)
    sub.add_argument("--out", required=True)
    sub.add_argument("--limit", type=int)
    sub = subs.add_parser("validate-data")
    sub.add_argument("--data", required=True)
    sub = subs.add_parser("export-sft")
    sub.add_argument("--trajectories", required=True)
    sub.add_argument("--metrics", required=True)
    sub.add_argument("--role", choices=["memory", "controller"], required=True)
    sub.add_argument("--out", required=True)
    sub.add_argument("--min-f1", type=float, default=1.0)
    args = p.parse_args()
    if args.command == "prepare-hotpot":
        convert_hotpot(args.input, args.out, args.limit)
    elif args.command == "validate-data":
        docs, tasks, _ = load_dataset(args.data)
        print(json.dumps({"documents": len(docs), "tasks": len(tasks)}))
    elif args.command == "export-sft":
        scores = {r["id"]: r for r in read_jsonl(args.metrics)}
        rows = []
        for trace in read_jsonl(args.trajectories):
            metric = scores[trace["task_id"]]
            if metric["f1"] < args.min_f1 or metric["invalid"]:
                continue
            for call in trace["calls"]:
                if call["role"] == args.role:
                    rows.append({"task_id": trace["task_id"], "role": args.role,
                                 "messages": call["messages"], "completion": call["response"]})
        write_jsonl(args.out, rows)
        print(json.dumps({"examples": len(rows), "note": "Success filtering does not establish memory factuality; audit before training."}))
    else:
        cfg = load_config(args.config)
        if args.command == "eval":
            print(json.dumps(evaluate(cfg, args.data, args.out), indent=2))
        else:
            root = Path(args.out)
            root.mkdir(parents=True, exist_ok=False)
            summaries = {}
            baseline = "recent" if cfg["backend"]["kind"] == "demo" else "summary"
            for name, memory, policy in (("A", baseline, "fixed"), ("B", "decision", "fixed"),
                                         ("C", baseline, "model"), ("D", "decision", "model")):
                current = copy.deepcopy(cfg)
                current["agent"].update(memory=memory, policy=policy)
                if name in {"A", "C"}:
                    # Baseline summaries must not accidentally use the trained memory checkpoint.
                    current.pop("memory_backend", None)
                    if "baseline_memory_backend" in cfg:
                        current["memory_backend"] = copy.deepcopy(cfg["baseline_memory_backend"])
                summaries[name] = evaluate(current, args.data, root / name)
            (root / "comparison.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
