"""Inspectable small-model SFT/GRPO reference trainer (beta=0).

One policy replica per process; explicit gradient averaging across processes.
This prioritizes trajectory/loss correctness, not throughput. It is NOT a verl
rollout implementation. All environment observations are prompt-only tokens.
"""
import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .budget import ModelCounter
from .data import fingerprint, load_dataset, read_jsonl
from .metrics import group_advantages, memory_operation_counts, reward, answer_scores
from .runner import load_config, make_agent, make_counter
from .types import Call


@dataclass
class Sample:
    prompt_ids: list[int]
    completion_ids: list[int]
    old_logprobs: list[float]
    schema: dict | None = None


def token_logprobs(model, prompt_ids, completion_ids, device, grammar=None, schema=None):
    import torch
    if not prompt_ids or not completion_ids:
        raise ValueError("Nonempty prompt and completion are required")
    ids = torch.tensor([prompt_ids + completion_ids], device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    # Only logits predicting generated action tokens contribute to the objective.
    logits = output.logits[0, len(prompt_ids) - 1:-1].float()
    if schema is not None:
        from .structured import mask_logits
        if grammar is None:
            raise ValueError("Grammar required to replay constrained token probabilities")
        logits = mask_logits(logits, grammar.masks(schema, completion_ids))
    targets = ids[0, len(prompt_ids):]
    return -torch.nn.functional.cross_entropy(logits, targets, reduction="none")


def clipped_objective(new_logprobs, old_logprobs, advantage, clip=0.2):
    import torch
    ratio = torch.exp(new_logprobs - old_logprobs)
    direct = ratio * advantage
    bounded = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage
    return -torch.minimum(direct, bounded)


def encode_prompt(tokenizer, messages):
    # Retain exact token IDs; never reconstruct a sampled action via text re-tokenization.
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def collect_group(agent, task, labels, actor, group_size, search_cost=0.0, trace_file=None):
    """Collect policy-produced decisions; their exact generated tokens enter GRPO."""
    groups, rewards, episodes = [], [], []
    actor.samples.clear()
    for _ in range(group_size):
        start = len(actor.samples)
        episode = agent.run(task)
        groups.append(actor.samples[start:])
        if trace_file is not None:
            trace_file.write(json.dumps({"episode": episode.payload(),
                                        "samples": [asdict(sample) for sample in actor.samples[start:]]},
                                       ensure_ascii=False) + "\n")
            trace_file.flush()
        rewards.append(reward(episode, labels, search_cost=search_cost))
        episodes.append({"answer_f1": answer_scores(episode.answer, labels.answer)[1],
                         "invalid_penalty": 0.1 * bool(episode.errors),
                         "search_penalty": search_cost * episode.searches,
                         "answer": episode.answer, "errors": episode.errors,
                         "searches": episode.searches, "stop_reason": episode.stop_reason,
                         "memory_operations": memory_operation_counts(episode),
                         "memory_decisions": [s.get("memory_decision") for s in episode.steps]})
    return groups, rewards, episodes


class TrainBackend:
    def __init__(self, model, tokenizer, device, max_prompt=3072, max_new=512, structured=False, sampling=True, record_samples=True):
        self.sampling, self.record_samples = sampling, record_samples
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.max_prompt, self.max_new = max_prompt, max_new
        self.calls, self.samples = [], []
        self.grammar = None
        if structured:
            from .structured import Grammar
            self.grammar = Grammar(tokenizer, model.config.vocab_size)

    def complete(self, messages, role):
        import torch
        from transformers import GenerationConfig
        prompt = encode_prompt(self.tokenizer, messages)
        if len(prompt) > self.max_prompt:
            raise ValueError("Prompt exceeds max_prompt; lower evidence/memory budgets (no silent truncation)")
        self.model.eval()
        ids = torch.tensor([prompt], device=self.device)
        # Full-distribution sampling at T=1 keeps sampling and policy logprobs aligned.
        gen = GenerationConfig(do_sample=self.sampling, temperature=1.0, top_p=1.0, top_k=0,
                               max_new_tokens=self.max_new, eos_token_id=self.tokenizer.eos_token_id,
                               pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        schema, extra = None, {}
        if self.grammar is not None:
            from .structured import memory_schema, controller_schema
            if role not in {"memory", "controller"}:
                raise ValueError("Structured policy role must be memory or controller")
            schema = memory_schema(messages) if role == "memory" else controller_schema(messages)
            extra["logits_processor"] = [self.grammar.processor(schema)]
        with torch.no_grad():
            full = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=gen, **extra)
            completion = full[0, len(prompt):].tolist()
            old = token_logprobs(self.model, prompt, completion, self.device, self.grammar, schema).cpu().tolist() if self.record_samples else []
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        if self.record_samples:
            self.samples.append(Sample(prompt, completion, old, schema))
        ended = bool(completion and completion[-1] == self.tokenizer.eos_token_id)
        self.calls.append(Call(role, messages, text, len(prompt), len(completion), "model_token",
                               finish_reason="eos" if ended else "length" if len(completion) >= self.max_new else "other",
                               decoding_mode=f"local_{role}_schema_v1_" + ("t1" if self.sampling else "greedy") if schema is not None else "local_full_distribution_t1"))
        return text


def average_gradients(model, accelerator):
    import torch
    # Every rank enters the same reductions even if its episodes had no trainable actions.
    # No DDP wrapper: ranks may have different numbers of turns/backward calls.
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad = accelerator.reduce(parameter.grad, reduction="mean")


def synchronize(accelerator):
    import torch
    # Explicit-device collective also works on CPU when a host has an MPS device.
    if accelerator.num_processes > 1:
        accelerator.reduce(torch.zeros(1, device=accelerator.device), reduction="sum")


def load_policy(args, accelerator):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.chat_template:
        raise ValueError("Use an instruction model with a chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                                lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.to(accelerator.device)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model, tokenizer


def save_checkpoint(model, tokenizer, optimizer, accelerator, out, step):
    import torch
    synchronize(accelerator)
    if accelerator.is_main_process:
        directory = Path(out) / f"step-{step:06d}"
        directory.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, directory / "optimizer.pt")
    synchronize(accelerator)


def main():
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["sft", "grpo"])
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", help="Optional SFT/previous LoRA adapter; optimizer starts fresh")
    p.add_argument("--data", required=True, help="SFT JSONL or converted QA dataset directory")
    p.add_argument("--config", help="Required for GRPO; frozen roles use configured API backends")
    p.add_argument("--frozen-memory-adapter", help="Frozen memory LoRA for single-process controller training")
    p.add_argument("--frozen-memory-device", default="cuda:1")
    p.add_argument("--role", choices=["memory", "controller"], default="memory")
    p.add_argument("--memory-mode", choices=["decision", "summary", "structured"], default="decision",
                   help="decision trains explicit memory operations; other modes are ablations")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--updates-per-rollout", type=int, default=1)
    p.add_argument("--max-prompt", type=int, default=3072)
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--search-cost", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--structured-memory", action="store_true", help="Grammar-constrained memory sampling with masked log-prob replay")
    p.add_argument("--trace-rollouts", action="store_true",
                   help="Save raw rollout calls and sampled token IDs for diagnosis (may be large)")
    args = p.parse_args()
    if min(args.steps, args.max_prompt, args.max_new, args.save_every, args.updates_per_rollout) < 1:
        p.error("Step/length limits must be positive")
    if args.group_size < 2 or args.search_cost < 0:
        p.error("group-size >= 2 and search-cost >= 0 are required")
    accelerator = Accelerator()
    if accelerator.mixed_precision not in {"no", "bf16"}:
        p.error("Reference trainer supports no/bf16 precision, not fp16 loss scaling")
    out = Path(args.out)
    if accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=False)
    synchronize(accelerator)
    set_seed(args.seed)  # Same initial adapters on every rank.
    model, tokenizer = load_policy(args, accelerator)
    counter = ModelCounter(tokenizer)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=args.learning_rate)
    set_seed(args.seed + accelerator.process_index + 1)
    if args.structured_memory and args.mode != "grpo":
        p.error("--structured-memory requires GRPO policy training")
    actor = TrainBackend(model, tokenizer, accelerator.device, args.max_prompt, args.max_new, args.structured_memory)
    config = None
    if args.mode == "grpo":
        if not args.config:
            p.error("GRPO requires --config")
        config = load_config(args.config)
        if config["backend"]["kind"] != "api":
            p.error("GRPO requires real frozen API roles; the demo fixture is not training data")
        config["agent"]["memory"] = args.memory_mode
        config["agent"]["policy"] = "fixed" if args.role == "memory" else "model"
        counter = make_counter(config)  # Honor a fixed budget tokenizer across model sizes.
        docs, tasks, labels = load_dataset(args.data)
        overrides = {args.role: actor}
        if args.frozen_memory_adapter:
            if args.role != "controller" or accelerator.num_processes != 1:
                p.error("Frozen local memory requires single-process controller training")
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            frozen = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.frozen_memory_device)
            frozen = PeftModel.from_pretrained(frozen, args.frozen_memory_adapter, is_trainable=False)
            overrides["memory"] = TrainBackend(frozen, tokenizer, args.frozen_memory_device, args.max_prompt, args.max_new,
                                               structured=True, sampling=False, record_samples=False)
        agent = make_agent(config, docs, counter, overrides)
        rows = tasks
    else:
        rows = list(read_jsonl(args.data))
        if any(row.get("role") != args.role for row in rows):
            p.error("SFT rows must match --role")
    if not rows:
        p.error("Empty training dataset")
    random.Random(args.seed).shuffle(rows)
    if accelerator.is_main_process:
        manifest = {"arguments": vars(args), "world_size": accelerator.num_processes,
                    "algorithm": "SFT" if args.mode == "sft" else "trajectory-normalized clipped GRPO, beta=0",
                    "config": config, "torch": torch.__version__,
                    "policy_decoding": "memory_schema_v1_t1" if args.structured_memory else "full_distribution_t1",
                    "dataset_sha256": fingerprint(args.data) if args.mode == "grpo" else None,
                    "status": "training-started; not an evaluation result"}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    trace_file = (out / f"rollouts-rank-{accelerator.process_index}.jsonl").open("x", encoding="utf-8") if args.trace_rollouts else None
    with (out / f"rank-{accelerator.process_index}.jsonl").open("w", encoding="utf-8") as log:
        for step in range(args.steps):
            row = rows[(step * accelerator.num_processes + accelerator.process_index) % len(rows)]
            if args.mode == "sft":
                prompt = encode_prompt(tokenizer, row["messages"])
                completion = tokenizer.encode(row["completion"], add_special_tokens=False)
                if tokenizer.eos_token_id is not None:
                    completion.append(tokenizer.eos_token_id)
                if len(prompt) > args.max_prompt or len(completion) > args.max_new:
                    raise ValueError("SFT sequence exceeds configured limit; prepare/filter data explicitly")
                optimizer.zero_grad(set_to_none=True)
                model.train()
                loss = -token_logprobs(model, prompt, completion, accelerator.device).mean()
                accelerator.backward(loss)
                average_gradients(model, accelerator)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                info = {"step": step + 1, "task_id": row.get("task_id"), "loss": loss.item()}
            else:
                groups, rewards, episodes = collect_group(
                    agent, row, labels[row.id], actor, args.group_size, args.search_cost, trace_file)
                advantages = group_advantages(rewards)
                local_trainable = float(any(g and abs(a) > 1e-8 for g, a in zip(groups, advantages)))
                active = accelerator.reduce(torch.tensor(local_trainable, device=accelerator.device), reduction="sum").item()
                total_loss = 0.0
                diagnostics = {"tokens": 0, "ratio_sum": 0.0, "clip_count": 0, "approx_kl_sum": 0.0}
                grad_norms = []
                if active:
                    for _ in range(args.updates_per_rollout):
                        optimizer.zero_grad(set_to_none=True)
                        model.train()
                        for samples, advantage in zip(groups, advantages):
                            n_tokens = sum(len(sample.completion_ids) for sample in samples)
                            if not n_tokens or abs(advantage) < 1e-8:
                                continue
                            for sample in samples:
                                new = token_logprobs(model, sample.prompt_ids, sample.completion_ids, accelerator.device, actor.grammar, sample.schema)
                                old = torch.tensor(sample.old_logprobs, device=accelerator.device)
                                with torch.no_grad():
                                    delta = new.detach() - old
                                    ratio = delta.exp()
                                    diagnostics["tokens"] += ratio.numel()
                                    diagnostics["ratio_sum"] += ratio.sum().item()
                                    diagnostics["clip_count"] += ((ratio < .8) | (ratio > 1.2)).sum().item()
                                    diagnostics["approx_kl_sum"] += (ratio - 1 - delta).sum().item()
                                loss = clipped_objective(new, old, advantage).sum() / n_tokens / args.group_size
                                accelerator.backward(loss)
                                total_loss += loss.item()
                        average_gradients(model, accelerator)
                        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
                        optimizer.step()
                count = diagnostics["tokens"]
                diagnostics.update(ratio_mean=diagnostics["ratio_sum"] / count if count else None,
                                   clip_fraction=diagnostics["clip_count"] / count if count else None,
                                   approx_kl=diagnostics["approx_kl_sum"] / count if count else None,
                                   gradient_norms=grad_norms,
                                   completion_tokens=[sum(len(s.completion_ids) for s in g) for g in groups],
                                   valid_fraction=sum(not e["errors"] for e in episodes) / len(episodes))
                info = {"diagnostics": diagnostics, "step": step + 1, "task_id": row.id, "rewards": rewards,
                        "advantages": advantages, "zero_variance": max(rewards) == min(rewards),
                        "updated": bool(active), "loss": total_loss, "episodes": episodes}
                # Avoid retaining all prompts and trajectories across optimization steps.
                for backend in {id(b): b for b in (agent.memory.backend, agent.controller, agent.reader)}.values():
                    backend.calls.clear()
                actor.samples.clear()
            log.write(json.dumps(info, ensure_ascii=False) + "\n")
            log.flush()
            if accelerator.is_main_process:
                print(json.dumps(info, ensure_ascii=False), flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                save_checkpoint(model, tokenizer, optimizer, accelerator, out, step + 1)

    if trace_file is not None:
        trace_file.close()


if __name__ == "__main__":
    main()
