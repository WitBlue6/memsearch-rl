import hashlib
import json
from pathlib import Path

from .types import Document, Labels, Task


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset(path):
    """Return public task observations and private labels in separate objects."""
    path = Path(path)
    docs = {r["id"]: Document(r["id"], r["title"], tuple(r["sentences"]))
            for r in read_jsonl(path / "corpus.jsonl")}
    tasks = [Task(r["id"], r["question"], tuple(r["candidate_ids"]))
             for r in read_jsonl(path / "tasks.jsonl")]
    rows = list(read_jsonl(path / "labels.jsonl"))
    labels = {r["id"]: Labels(r["answer"], tuple(r["supporting_ids"])) for r in rows}
    if len({t.id for t in tasks}) != len(tasks) or len(labels) != len(rows):
        raise ValueError("Duplicate task/label IDs")
    if {t.id for t in tasks} != set(labels):
        raise ValueError("Tasks and labels must have identical ID sets")
    evidence_ids = {e.id for d in docs.values() for e in d.evidence()}
    for task in tasks:
        if not task.candidate_ids or not set(task.candidate_ids) <= docs.keys():
            raise ValueError(f"Invalid candidate documents: {task.id}")
        if not set(labels[task.id].supporting_ids) <= evidence_ids:
            raise ValueError(f"Unknown gold evidence: {task.id}")
    return docs, tasks, labels


def fingerprint(path):
    h = hashlib.sha256()
    for name in ("corpus.jsonl", "tasks.jsonl", "labels.jsonl"):
        h.update(name.encode())
        h.update((Path(path) / name).read_bytes())
    return h.hexdigest()


def convert_hotpot(source, out, limit=None):
    """Convert official HotpotQA context JSON; never inject supporting labels into corpus."""
    rows = json.loads(Path(source).read_text(encoding="utf-8"))
    if limit is not None:
        rows = rows[:limit]
    documents, tasks, labels = {}, [], []
    for row in rows:
        title_map, candidates = {}, []
        for title, sentences in row["context"]:
            content = json.dumps([title, sentences], ensure_ascii=False)
            doc_id = hashlib.sha256(content.encode()).hexdigest()[:20]
            documents[doc_id] = {"id": doc_id, "title": title, "sentences": sentences}
            title_map[title] = doc_id
            candidates.append(doc_id)
        supporting = []
        for title, sentence_id in row["supporting_facts"]:
            if title not in title_map or sentence_id >= len(documents[title_map[title]]["sentences"]):
                raise ValueError(f"Missing supporting sentence in {row['_id']}")
            supporting.append(f"{title_map[title]}:{sentence_id}")
        tasks.append({"id": row["_id"], "question": row["question"], "candidate_ids": candidates})
        labels.append({"id": row["_id"], "answer": row["answer"], "supporting_ids": supporting})
    write_jsonl(Path(out) / "corpus.jsonl", documents.values())
    write_jsonl(Path(out) / "tasks.jsonl", tasks)
    write_jsonl(Path(out) / "labels.jsonl", labels)
    metadata = {"source": str(source), "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "setting": "candidate-pool (not full-wiki retrieval)", "examples": len(tasks)}
    (Path(out) / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    load_dataset(out)
