"""Convert the pinned HotpotQA subset; existing outputs are never overwritten."""
import json
from itertools import islice
from pathlib import Path
import pyarrow.parquet as pq

root = Path("data/raw/hotpot-hf")

def read_rows(files):
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
            yield from batch.to_pylist()

jobs = [
    ("train", 90447, 2000, "data/raw/hotpot_train_hf_2000.json"),
    ("validation", 7405, 200, "data/raw/hotpot_dev_hf_200.json"),
]

for split, expected, limit, output in jobs:
    files = sorted(root.glob(f"{split}-*.parquet"))
    count = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    assert count == expected, f"{split}: 实际 {count} 条，预期 {expected}"
    print(f"{split}: {count} 条，元数据检查通过")

    result = []
    for row in islice(read_rows(files), limit):
        context = row["context"]
        support = row["supporting_facts"]
        result.append({
            "_id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
            "level": row["level"],
            "context": list(zip(
                context["title"], context["sentences"], strict=True)),
            "supporting_facts": list(zip(
                support["title"], support["sent_id"], strict=True)),
        })

    assert len(result) == limit
    with Path(output).open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"已生成 {output}：{len(result)} 条")
