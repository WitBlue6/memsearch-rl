"""Evaluate matched baselines or an unmerged adapter; never modifies model services."""
import argparse
import json
import os
from pathlib import Path
from memsearch.runner import load_config, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--memory", choices=["decision", "summary", "recent", "extractive", "structured"], default="decision")
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--order", choices=["ranked", "reversed"], default="ranked")
    p.add_argument("--policy", choices=["fixed", "model"], default="fixed")
    p.add_argument("--model", help="Local base model for adapter evaluation")
    p.add_argument("--policy-api", default=os.environ.get("MEMSEARCH_RESEARCH_API"))
    p.add_argument("--workers", type=int, default=int(os.environ.get("MEMSEARCH_EVAL_WORKERS", "1")))
    p.add_argument("--adapter")
    p.add_argument("--frozen-memory-model")
    p.add_argument("--frozen-memory-adapter")
    p.add_argument("--frozen-memory-device", default="cuda:1")
    p.add_argument("--adapter-role", choices=["memory", "controller"], default="memory")
    a = p.parse_args()
    if Path(a.out).exists():
        p.error("Output already exists")
    cfg = load_config(a.config)
    cfg["agent"].update(memory=a.memory, memory_budget=a.budget, overflow="reject", evidence_order=a.order,
                        policy=a.policy, controller_reader=True)
    cfg["backend"].update(memory_output="json_schema", reader_output="json_schema", controller_output="json_schema")
    overrides = None
    if a.policy_api and a.model:
        from memsearch.lora_api import register_adapter
        cfg[a.adapter_role + "_backend"] = {"base_url": a.policy_api, "model": register_adapter(a.policy_api, a.adapter)}
    elif a.model:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        from memsearch.training import TrainBackend
        tokenizer = AutoTokenizer.from_pretrained(a.model)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda")
        if a.adapter:
            model = PeftModel.from_pretrained(model, a.adapter)
        backend = TrainBackend(model, tokenizer, "cuda", max_prompt=4096, max_new=cfg["backend"]["max_tokens"],
                               structured=True, sampling=False, record_samples=False)
        overrides = {a.adapter_role: backend}
    elif a.adapter:
        p.error("--adapter requires --model")
    if a.policy_api and a.frozen_memory_model:
        from memsearch.lora_api import register_adapter
        cfg["memory_backend"] = {"base_url": a.policy_api, "model": register_adapter(a.policy_api, a.frozen_memory_adapter)}
    elif a.frozen_memory_model:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        from memsearch.training import TrainBackend
        if overrides and "memory" in overrides:
            p.error("Cannot load two memory overrides")
        frozen_tokenizer = AutoTokenizer.from_pretrained(a.frozen_memory_model)
        frozen_tokenizer.pad_token_id = frozen_tokenizer.eos_token_id
        frozen_model = AutoModelForCausalLM.from_pretrained(a.frozen_memory_model, torch_dtype=torch.bfloat16).to(a.frozen_memory_device)
        if a.frozen_memory_adapter:
            frozen_model = PeftModel.from_pretrained(frozen_model, a.frozen_memory_adapter)
        overrides = overrides or {}
        overrides["memory"] = TrainBackend(frozen_model, frozen_tokenizer, a.frozen_memory_device, max_prompt=4096,
                                             max_new=cfg["backend"]["max_tokens"], structured=True, sampling=False, record_samples=False)
    elif a.frozen_memory_adapter:
        p.error("--frozen-memory-adapter requires --frozen-memory-model")
    cfg["experiment"] = {"training_status": "rl-checkpoint-evaluation" if a.adapter else "untrained-prompt-baselines",
                         "local_model": a.model, "adapter": a.adapter, "policy_api": a.policy_api,
                         "frozen_memory_model": a.frozen_memory_model, "frozen_memory_adapter": a.frozen_memory_adapter,
                         "evaluation_protocol": "research-v1; greedy; reject LLM overflow; same frozen reader"}
    print(json.dumps(evaluate(cfg, a.data, a.out, overrides, workers=a.workers), indent=2))


if __name__ == "__main__":
    main()
