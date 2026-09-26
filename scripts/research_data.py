"""Create immutable, disjoint HotpotQA splits from downloaded Parquet files.
Run: uv run --locked --extra train --with pyarrow==25.0.1 python scripts/research_data.py
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
import pyarrow.parquet as pq
from memsearch.data import convert_hotpot, fingerprint


def question_key(text):
    return re.sub(r"\W+", "", text.casefold())


def read(root, split):
    files = sorted(root.glob(f"{split}-*.parquet"))
    if not files:
        raise ValueError(f"No Parquet shards for {split}")
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                yield {"_id": row["id"], "question": row["question"], "answer": row["answer"],
                       "context": list(zip(row["context"]["title"], row["context"]["sentences"], strict=True)),
                       "supporting_facts": list(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"], strict=True))}


def valid(row):
    context = dict(row["context"])
    return all(t in context and 0 <= i < len(context[t]) for t, i in row["supporting_facts"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="data/raw/hotpot-hf")
    p.add_argument("--out", default="data/processed/research-v1")
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--tune", type=int, default=80)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260925)
    a = p.parse_args()
    if min(a.train, a.tune, a.test) < 1:
        p.error("split sizes must be positive")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = list(read(Path(a.raw), "validation"))
    # All old first-200 dev tasks are excluded, even when never used successfully.
    used_ids = {x["_id"] for x in dev[:200]}
    used_questions = {question_key(x["question"]) for x in dev[:200]}
    rng = random.Random(a.seed)
    candidates = dev[200:]
    rng.shuffle(candidates)
    def take(rows, size):
        selected = []
        for row in rows:
            key = question_key(row["question"])
            if row["_id"] in used_ids or key in used_questions or not valid(row):
                continue
            used_ids.add(row["_id"])
            used_questions.add(key)
            selected.append(row)
            if len(selected) == size:
                return selected
        raise ValueError("Insufficient valid disjoint tasks")
    tune = take(candidates, a.tune)
    test = take(candidates, a.test)
    train_rows = list(read(Path(a.raw), "train"))
    rng.shuffle(train_rows)
    train = take(train_rows, a.train)
    manifest = {"seed": a.seed, "excluded_old_dev": 200,
                "deduplication": "IDs and normalized questions, not semantic near-duplicate detection",
                "selection": "Random without replacement; invalid supporting offsets excluded", "splits": {}}
    for name, rows in (("train", train), ("tune", tune), ("test", test)):
        source = out / (name + ".json")
        source.write_text(json.dumps(rows, ensure_ascii=False))
        convert_hotpot(source, out / name)
        source.unlink()
        manifest["splits"][name] = {"count": len(rows), "sha256": fingerprint(out / name), "ids": [r["_id"] for r in rows]}
    manifest["source_sha256"] = {f.name: hashlib.file_digest(f.open("rb"), "sha256").hexdigest() for f in sorted(Path(a.raw).glob("*.parquet"))}
    (out / "splits.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: {x: v[x] for x in ("count", "sha256")} for k, v in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
