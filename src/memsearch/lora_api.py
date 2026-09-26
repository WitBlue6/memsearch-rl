"""Register immutable local research adapters on a loopback-only vLLM service."""
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def register_adapter(base_url, adapter=None):
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password or parsed.query:
        raise ValueError("Research adapter registration requires a loopback endpoint")
    if adapter is None:
        return "memsearch-research"
    path = Path(adapter).resolve(strict=True)
    weights = path / "adapter_model.safetensors"
    config = path / "adapter_config.json"
    h = hashlib.sha256()
    h.update(weights.read_bytes())
    h.update(config.read_bytes())
    name = "research-" + h.hexdigest()[:24]
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as response:
        models = json.load(response)
    if name not in {m["id"] for m in models["data"]}:
        request = urllib.request.Request(base_url.rstrip("/") + "/load_lora_adapter",
                  data=json.dumps({"lora_name": name, "lora_path": str(path)}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            response.read()
    return name
