"""Export randomly sampled memory claims and cited sentences for human grounding audit.
This does not assign reward or pretend that matching source IDs proves entailment.
"""
import argparse
import json
import random
from pathlib import Path
from memsearch.data import load_dataset, read_jsonl

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data", required=True)
p.add_argument("--trajectories", required=True)
p.add_argument("--out", required=True)
p.add_argument("--limit", type=int, default=30)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
docs, _, _ = load_dataset(a.data)
sources = {e.id: e.text for doc in docs.values() for e in doc.evidence()}
rows = []
for episode in read_jsonl(a.trajectories):
    for step in episode["steps"]:
        for item in step["memory"]["items"]:
            rows.append({"task_id": episode["task_id"], "search": step["search"], "claim": item["text"],
                         "sources": [{"id": s, "text": sources.get(s)} for s in item["source_ids"]],
                         "entailed": None, "review_notes": ""})
random.Random(a.seed).shuffle(rows)
with Path(a.out).open("x") as f:
    for row in rows[:a.limit]:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Exported {min(a.limit, len(rows))} claims; unreviewed, not a grounding score")
