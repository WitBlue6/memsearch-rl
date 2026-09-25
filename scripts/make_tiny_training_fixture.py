"""Create a random tiny local model + SFT sample for CPU integration checks only."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
args = p.parse_args()
root = Path(args.out)
root.mkdir(parents=True, exist_ok=False)
vocab = {token: i for i, token in enumerate([
    "[PAD]", "[UNK]", "[EOS]", "system", "user", "assistant", "Remember", "a", "fact", "operations", "op", "ADD",
    "text", "source_ids", "unresolved", "Ada", "Luma", "born", "in", "was", "p1", "0", ":", ",", "{", "}", "[", "]", '"'])}
base = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
base.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=base, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
tokenizer.chat_template = "{% for message in messages %}{{message['role']}} {{message['content']}} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
tokenizer.save_pretrained(root / "model")
model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=512, n_embd=16, n_layer=1,
                                  n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                                  bos_token_id=None, eos_token_id=2, pad_token_id=0))
model.save_pretrained(root / "model")
row = {"task_id": "tiny-fixture", "role": "memory", "messages": [{"role": "user", "content": "Remember a fact"}],
       "completion": '{"operations":[{"op":"ADD","text":"Ada was born in Luma","source_ids":["p1:0"]}],"unresolved":[]}'}
(root / "sft.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
