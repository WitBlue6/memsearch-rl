"""Render standalone scientific figures: uv run --with matplotlib python scripts/plot_research.py ..."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--report", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
out = Path(a.out)
if out.exists():
    p.error("Output already exists")
data = json.loads(Path(a.report).read_text())
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for name, info in data["training"].items():
    if not name.startswith("train-"):
        continue
    rows = info["learning_curve"]
    axes[0].plot([r["step"] for r in rows], [sum(r["rewards"])/len(r["rewards"]) for r in rows], alpha=.65, label=name.split("/")[0]+"/"+name.split("/")[-1].replace(".jsonl", ""))
    axes[1].plot([r["step"] for r in rows], [r["diagnostics"]["valid_fraction"] for r in rows], alpha=.65)
groups = {}
for name, metrics in data["evaluations"].items():
    if name.startswith("curve-"):
        _, seed, step = name.split("-")
        groups.setdefault(seed, []).append((int(step), metrics["f1"]))
for seed, points in groups.items():
    points.sort()
    axes[2].plot([p[0] for p in points], [p[1] for p in points], marker="o", label="seed "+seed)
for name in ("before-matched", "summary", "extractive", "recent"):
    if name in data["evaluations"]:
        axes[2].axhline(data["evaluations"][name]["f1"], linestyle="--", alpha=.6, label=name)
for ax, title in zip(axes, ("Training group reward", "Valid rollout fraction", "Tuning F1 (not locked test)")):
    ax.set_title(title)
    ax.set_xlabel("Optimization iteration")
    ax.grid(alpha=.2)
axes[0].legend(fontsize=6)
axes[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out, dpi=180)
