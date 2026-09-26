import json
import platform
import subprocess
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .agent import Agent
from .backends import ChatBackend
from .budget import Counter, ModelCounter
from .data import fingerprint, load_dataset, write_jsonl
from .demo import DemoBackend
from .memory import Memory
from .metrics import aggregate, score
from .retrieval import BM25


def load_config(path):
    with Path(path).open("rb") as f:
        cfg = tomllib.load(f)
    a = cfg["agent"]
    for key in ("max_searches", "top_k", "memory_budget", "evidence_budget"):
        if not isinstance(a[key], int) or a[key] <= 0:
            raise ValueError(f"agent.{key} must be a positive integer")
    if a.get("retrieval_scope", "candidate") not in {"candidate", "corpus"}:
        raise ValueError("retrieval_scope must be candidate or corpus")
    if a.get("evidence_order", "ranked") not in {"ranked", "reversed"}:
        raise ValueError("evidence_order must be ranked or reversed")
    if a.get("overflow", "trim") not in {"trim", "reject"}:
        raise ValueError("overflow must be trim or reject")
    if cfg.get("backend", {}).get("kind") not in {"demo", "api"}:
        raise ValueError("backend.kind must be demo or api")
    return cfg


def make_counter(cfg):
    name = cfg.get("budget", {}).get("tokenizer")
    if name:
        from transformers import AutoTokenizer
        return ModelCounter(AutoTokenizer.from_pretrained(name))
    return Counter()


def make_backend(cfg, role, counter):
    base = cfg["backend"]
    if base["kind"] == "demo":
        return DemoBackend()
    return ChatBackend({**base, **cfg.get(f"{role}_backend", {})}, counter)


def make_agent(cfg, docs, counter=None, overrides=None):
    counter, overrides = counter or make_counter(cfg), overrides or {}
    backends = {role: overrides.get(role) or make_backend(cfg, role, counter)
                for role in ("memory", "controller", "reader")}
    memory = Memory(cfg["agent"]["memory"], cfg["agent"]["memory_budget"], counter, backends["memory"],
                    cfg["agent"].get("max_memory_ops", 16), cfg["agent"].get("overflow", "trim"))
    return Agent(BM25(docs), memory, cfg["agent"], backends["controller"], backends["reader"])


def evaluate(cfg, dataset, out, overrides=None, workers=1):
    if workers < 1 or (overrides and workers != 1):
        raise ValueError("workers must be positive; local model overrides require workers=1")
    docs, tasks, labels = load_dataset(dataset)
    if cfg["backend"]["kind"] == "demo" and not (Path(dataset) / "SYNTHETIC_FIXTURE").exists():
        raise ValueError("Demo backend is only allowed on the bundled synthetic fixture")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    import concurrent.futures
    import threading
    agent = make_agent(cfg, docs, overrides=overrides) if workers == 1 else None
    local = threading.local()
    def run_task(task):
        current = agent
        if current is None:
            if not hasattr(local, "agent"):
                local.agent = make_agent(cfg, docs)
            current = local.agent
        episode = current.run(task)
        for backend in (current.memory.backend, current.controller, current.reader):
            backend.calls.clear()
            if hasattr(backend, "samples"):
                backend.samples.clear()
        return episode
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool, (out / "trajectories.jsonl").open("w", encoding="utf-8") as trace:
        for task, episode in zip(tasks, pool.map(run_task, tasks), strict=True):
            trace.write(json.dumps(episode.payload(), ensure_ascii=False) + "\n")
            trace.flush()
            row = score(episode, labels[task.id])
            from .research import mechanism
            row["memory_trace_metrics"] = mechanism(episode, labels[task.id])
            rows.append(row)
    write_jsonl(out / "metrics.jsonl", rows)
    summary = aggregate(rows)
    summary["synthetic_fixture"] = cfg["backend"]["kind"] == "demo"
    summary["training_status"] = cfg.get("experiment", {}).get("training_status", "untrained")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "dataset_sha256": fingerprint(dataset), "git_commit": commit,
                "evaluation_workers": workers, "budget_unit": agent.memory.counter.unit if agent else make_counter(cfg).unit, "retrieval_scope": cfg["agent"].get("retrieval_scope", "candidate")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary
