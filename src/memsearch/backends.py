import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .budget import Counter
from .types import Call


READER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "evidence_ids"],
    "additionalProperties": False,
}


class ChatBackend:
    """Minimal OpenAI-compatible protocol; no SDK or credentials in artifacts."""
    def __init__(self, config, counter=None):
        self.config = config
        self.counter = counter or Counter()
        self.calls = []

    def complete(self, messages, role):
        cfg = self.config
        url = cfg["base_url"].rstrip("/") + "/chat/completions"
        parsed = urllib.parse.urlsplit(url)
        if parsed.username or parsed.password or parsed.query:
            raise ValueError("Do not put credentials in base_url; use an environment variable")
        key = os.environ.get(cfg.get("api_key_env", "MEMSEARCH_API_KEY"), "")
        body = {"model": cfg["model"], "messages": messages,
                "temperature": cfg.get("temperature", 0), "max_tokens": cfg.get("max_tokens", 512)}
        mode = cfg.get("reader_output", "text")
        if mode not in {"text", "json_schema"}:
            raise ValueError("reader_output must be text or json_schema")
        constrained = role == "reader" and mode == "json_schema"
        if constrained:
            body["structured_outputs"] = {"json": READER_SCHEMA}
        memory_mode = cfg.get("memory_output", "text")
        if memory_mode not in {"text", "json_schema"}:
            raise ValueError("memory_output must be text or json_schema")
        memory_constrained = role == "memory" and memory_mode == "json_schema"
        if memory_constrained:
            from .structured import memory_schema
            body["structured_outputs"] = {"json": memory_schema(messages)}
        controller_mode = cfg.get("controller_output", "text")
        if controller_mode not in {"text", "json_schema"}:
            raise ValueError("controller_output must be text or json_schema")
        controller_constrained = role == "controller" and controller_mode == "json_schema"
        if controller_constrained:
            from .structured import controller_schema
            body["structured_outputs"] = {"json": controller_schema(messages)}
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=cfg.get("timeout", 120)) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Model endpoint HTTP {exc.code}; check server logs/config") from None
        except urllib.error.URLError:
            raise RuntimeError("Model endpoint unavailable; check base_url and server") from None
        text = result["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("Model returned non-text content")
        usage = result.get("usage") or {}
        exact = "prompt_tokens" in usage and "completion_tokens" in usage
        self.calls.append(Call(role, messages, text,
                               usage["prompt_tokens"] if exact else self.counter.count(json.dumps(messages)),
                               usage["completion_tokens"] if exact else self.counter.count(text),
                               "model_token" if exact else "estimated_" + self.counter.unit,
                               finish_reason=result["choices"][0].get("finish_reason"),
                               decoding_mode="reader_json_schema_v1" if constrained else "memory_json_schema_v1" if memory_constrained else "controller_json_schema_v1" if controller_constrained else "text"))
        return text


def parse_json(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value
