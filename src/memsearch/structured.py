"""Shared API/local schemas and replayable grammar masks; semantic checks stay in the environment."""
import json


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}


def memory_schema(messages):
    observation = json.loads(messages[-1]["content"])
    text = {"type": "string", "minLength": 1}
    sources = {"type": "array", "items": text, "minItems": 1}
    unresolved = {"type": "array", "items": {"type": "string"}}
    if "VALID_MEMORY_IDS" not in observation:
        return obj({"items": {"type": "array", "items": obj({"text": text, "source_ids": sources})},
                    "unresolved": unresolved})
    variants = []
    for name, fields in (("ADD", {"text": text, "source_ids": sources}),
                         ("UPDATE", {"target_id": text, "text": text, "source_ids": sources}),
                         ("DELETE", {"target_id": text}), ("NOOP", {})):
        variants.append(obj({"op": {"const": name}, **fields}))
    return obj({"operations": {"type": "array", "minItems": 1,
                               "maxItems": observation.get("max_memory_ops", 16),
                               "items": {"anyOf": variants}}, "unresolved": unresolved}, ["operations"])


class Grammar:
    def __init__(self, tokenizer, vocab_size):
        import xgrammar as xgr
        self.xgr, self.vocab_size = xgr, vocab_size
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size,
                                                stop_token_ids=[tokenizer.eos_token_id])
        self.compiler = xgr.GrammarCompiler(info)
        self.cache = {}

    def compiled(self, schema):
        key = json.dumps(schema, sort_keys=True)
        if key not in self.cache:
            self.cache[key] = self.compiler.compile_json_schema(schema)
        return self.cache[key]

    def masks(self, schema, completion):
        matcher = self.xgr.GrammarMatcher(self.compiled(schema))
        masks = self.xgr.allocate_token_bitmask(len(completion), self.vocab_size)
        for i, token in enumerate(completion):
            matcher.fill_next_token_bitmask(masks, i)
            if not matcher.accept_token(token):
                raise ValueError("Recorded completion violates its sampling grammar")
        return masks

    def processor(self, schema):
        grammar = self
        matcher = self.xgr.GrammarMatcher(self.compiled(schema))
        mask = self.xgr.allocate_token_bitmask(1, self.vocab_size)
        class Processor:
            started = False
            def __call__(self, input_ids, scores):
                if input_ids.shape[0] != 1:
                    raise ValueError("Reference grammar processor requires batch size one")
                if self.started and not matcher.accept_token(input_ids[0, -1].item()):
                    raise ValueError("Sampled token violates grammar")
                self.started = True
                matcher.fill_next_token_bitmask(mask)
                return mask_logits(scores, mask)
        return Processor()


def mask_logits(logits, packed):
    # Out-of-place PyTorch masking keeps gradients correct; never mutate logits via a CUDA kernel in backward.
    import torch
    positions = torch.arange(logits.shape[-1], device=logits.device)
    packed = packed.to(logits.device)
    allowed = ((packed[:, positions // 32] >> (positions % 32)) & 1).bool()
    return logits.masked_fill(~allowed, float("-inf"))
