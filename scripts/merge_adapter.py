"""Merge a completed LoRA adapter for a separately managed inference server."""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--adapter", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
if Path(args.out).exists():
    p.error("Output already exists")
base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
policy = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
policy.save_pretrained(args.out, safe_serialization=True)
AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
