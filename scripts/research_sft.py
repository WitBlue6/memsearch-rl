"""Build silver SFT actions from valid, successful TRAIN rollouts only.
Labels select episodes offline; no answer labels are added to messages.
"""
import argparse
import hashlib
import json
from pathlib import Path
from memsearch.data import load_dataset, read_jsonl
from memsearch.metrics import answer_scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data", required=True)
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-f1", type=float, default=.5)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--max-prompt", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=512)
    a = p.parse_args()
    if not 0 <= a.min_f1 <= 1:
        p.error("min-f1 must be between 0 and 1")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    excluded_length = 0
    _, tasks, labels = load_dataset(a.train_data)
    allowed = {t.id for t in tasks}
    rows, seen = [], set()
    for file in a.traces:
        for trace in read_jsonl(file):
            episode = trace["episode"]
            tid = episode["task_id"]
            if tid not in allowed:
                raise ValueError("SFT extraction attempted on a non-training task")
            if episode["errors"] or answer_scores(episode["answer"], labels[tid].answer)[1] < a.min_f1:
                continue
            for call in episode["calls"]:
                if call["role"] != "memory":
                    continue
                row = {"task_id": tid, "role": "memory", "messages": call["messages"], "completion": call["response"]}
                prompt_ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True)
                completion_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
                if len(prompt_ids) > a.max_prompt or len(completion_ids) + 1 > a.max_new:
                    excluded_length += 1
                    continue
                key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    if not rows:
        raise ValueError("No successful training trajectories; do not train on an empty SFT set")
    with Path(a.out).open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(a.out + ".manifest.json").write_text(json.dumps({"rows": len(rows), "tasks": len({r["task_id"] for r in rows}),
                "min_f1": a.min_f1, "excluded_length": excluded_length, "tokenizer": a.tokenizer, "status": "self-generated silver actions; not expert or entailment-verified",
                "sources": a.traces, "train_data": a.train_data}, indent=2))
    print(f"Silver actions: {len(rows)} from {len({r['task_id'] for r in rows})} training tasks")


if __name__ == "__main__":
    main()
