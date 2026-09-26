"""Bounded research pilot; all outputs are immutable and test evaluation is explicit.
Uses an existing frozen reader service and free training GPUs, never stops services.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
from memsearch.runner import load_config


def toml(config):
    return "".join(f"[{section}]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k,v in values.items()) + "\n" for section,values in config.items())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["setup", "baselines", "train", "curves", "mechanisms", "coldstart", "controller", "test", "report"])
    p.add_argument("--config", default="configs/generated/grammar-001/before.toml")
    p.add_argument("--root", default="outputs/research-v1")
    p.add_argument("--data", default="data/processed/research-v1")
    p.add_argument("--model", default=os.environ.get("POLICY_MODEL"))
    p.add_argument("--gpus", default="0,1,2,3,4,5")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seeds", default="42,43,44")
    a = p.parse_args()
    root, data = Path(a.root), Path(a.data)
    seeds = [int(x) for x in a.seeds.split(",")]
    gpus = a.gpus.split(",")
    if not a.model or len(gpus) != 2 * len(seeds) or a.steps < 2:
        p.error("Set --model, two free GPUs per seed, and steps >= 2")
    config = root / "config.toml"
    plan = {"model": a.model, "seeds": seeds, "steps": a.steps, "gpus": gpus, "data": str(data),
            "checkpoints": sorted(set([a.steps // 2, a.steps])),
            "test_protocol": "Frozen final-step adapters for ALL seeds; no best-seed selection",
            "source_config": str(Path(a.config).resolve()),
            "split_manifest": json.loads((data / "splits.json").read_text())}
    if a.stage == "setup":
        root.mkdir(parents=True, exist_ok=False)
        cfg = load_config(a.config)
        cfg["agent"].update(overflow="reject", controller_reader=True)
        cfg["backend"].update(reader_output="json_schema", memory_output="json_schema", controller_output="json_schema")
        config.write_text(toml(cfg))
        (root / "plan.json").write_text(json.dumps(plan, indent=2))
        return
    if json.loads((root / "plan.json").read_text()) != plan:
        p.error("Plan drift; use the original arguments or a new output root")
    def run(args, log, env=None):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as f:
            subprocess.run([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT,
                           check=True, env={**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", **(env or {})})
    def evaluate(name, memory="decision", adapter=None, gpu=None, budget=512, order="ranked", policy="fixed", split="tune", role="memory", local_base=False, frozen_memory=False, memory_adapter=None):
        directory = root / name
        args = [sys.executable, "scripts/research_eval.py", "--config", config, "--data", data / split,
                "--out", directory, "--memory", memory, "--budget", budget, "--order", order, "--policy", policy]
        if adapter is not None or local_base:
            args += ["--model", a.model, "--adapter-role", role]
        if adapter is not None:
            args += ["--adapter", adapter]
        if frozen_memory:
            args += ["--frozen-memory-model", a.model]
            if memory_adapter is not None:
                args += ["--frozen-memory-adapter", memory_adapter]
        run(args, root / "logs" / (name.replace("/", "-") + ".log"), {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {})
    if a.stage == "baselines":
        for name, memory in (("before", "decision"), ("summary", "summary"), ("extractive", "extractive"), ("recent", "recent")):
            evaluate(name, memory)
    elif a.stage == "train":
        def train(index_seed):
            i, seed = index_seed
            directory = root / f"train-{seed}"
            args = [sys.executable, "-m", "accelerate.commands.launch", "--config_file", "configs/accelerate_8x3090.yaml",
                    "--num_processes", "2", "--main_process_port", str(29610 + i), "-m", "memsearch.training",
                    "grpo", "--model", a.model, "--config", config, "--data", data / "train", "--out", directory,
                    "--steps", a.steps, "--save-every", a.steps // 2, "--seed", seed, "--group-size", "4",
                    "--max-prompt", "4096", "--max-new", "512", "--structured-memory", "--trace-rollouts"]
            run(args, root / "logs" / f"train-{seed}.log", {"CUDA_VISIBLE_DEVICES": ",".join(gpus[2*i:2*i+2])})
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as pool:
            list(pool.map(train, enumerate(seeds)))
    elif a.stage == "curves":
        evaluate("before-matched", gpu=gpus[-1], local_base=True)
        def curves(index_seed):
            i, seed = index_seed
            for step in plan["checkpoints"]:
                evaluate(f"curve-{seed}-{step}", adapter=root / f"train-{seed}" / f"step-{step:06d}", gpu=gpus[2*i])
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as pool:
            list(pool.map(curves, enumerate(seeds)))
    elif a.stage == "mechanisms":
        # Preselected first seed, never selected using outcome. Transfer, not budget-specific training.
        adapter = root / f"train-{seeds[0]}" / f"step-{a.steps:06d}"
        for budget, order in ((256, "ranked"), (1024, "ranked"), (512, "reversed")):
            for name, memory, checkpoint in (("before", "decision", None), ("summary", "summary", None), ("after", "decision", adapter)):
                evaluate(f"mechanism-{name}-{budget}-{order}", memory, checkpoint, gpus[0], budget, order)
    elif a.stage == "coldstart":
        silver = root / "silver.jsonl"
        traces = sorted(root.glob("train-*/rollouts-rank-*.jsonl"))
        run([sys.executable, "scripts/research_sft.py", "--train-data", data / "train", "--traces", *traces, "--out", silver, "--tokenizer", a.model],
            root / "logs" / "silver.log")
        for mode, steps, train_data, adapter in (("sft", 12, silver, None), ("grpo", 12, data / "train", root / "sft" / "step-000012")):
            name = "sft" if mode == "sft" else "sft-grpo"
            args = [sys.executable, "-m", "memsearch.training", mode, "--model", a.model, "--config", config,
                    "--data", train_data, "--out", root / name, "--steps", steps, "--save-every", steps,
                    "--seed", "42", "--max-prompt", "4096", "--max-new", "512"]
            if adapter:
                args += ["--adapter", adapter, "--structured-memory", "--trace-rollouts"]
            run(args, root / "logs" / (name + ".log"), {"CUDA_VISIBLE_DEVICES": gpus[0], "ACCELERATE_MIXED_PRECISION": "bf16"})
            evaluate(name + "-eval", adapter=root / name / "step-000012", gpu=gpus[0])
    elif a.stage == "controller":
        frozen = root / f"train-{seeds[0]}" / f"step-{a.steps:06d}"
        devices = ",".join(gpus[:2])
        evaluate("controller-untrained", policy="model", gpu=devices, role="controller", local_base=True,
                 frozen_memory=True, memory_adapter=frozen)
        directory = root / "controller-train"
        run([sys.executable, "-m", "memsearch.training", "grpo", "--role", "controller", "--model", a.model,
             "--config", config, "--data", data / "train", "--out", directory, "--steps", "4", "--save-every", "4",
             "--seed", "42", "--group-size", "4", "--max-prompt", "4096", "--max-new", "512", "--structured-memory", "--trace-rollouts", "--frozen-memory-adapter", frozen],
            root / "logs" / "controller-train.log", {"CUDA_VISIBLE_DEVICES": devices, "ACCELERATE_MIXED_PRECISION": "bf16"})
        # Same learned controller, two frozen memory states; fixed-search cells are before-local and curve seed 0.
        for name, checkpoint in (("controller-trained-base-memory", None), ("controller-trained", frozen)):
            evaluate(name, adapter=directory / "step-000004", gpu=devices, policy="model", role="controller",
                     frozen_memory=True, memory_adapter=checkpoint)
    elif a.stage == "test":
        evaluate("test/before-matched", gpu=gpus[0], local_base=True, split="test")
        for name, memory in (("before", "decision"), ("summary", "summary"), ("extractive", "extractive"), ("recent", "recent")):
            evaluate("test/" + name, memory, split="test")
        for i, seed in enumerate(seeds):
            evaluate(f"test/after-{seed}", adapter=root / f"train-{seed}" / f"step-{a.steps:06d}", gpu=gpus[2*i], split="test")
    elif a.stage == "report":
        run([sys.executable, "scripts/research_report.py", "--root", root, "--out", root / "report.json"], root / "logs" / "report.log")


if __name__ == "__main__":
    main()
