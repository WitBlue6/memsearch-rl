#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
source "${MEMSEARCH_SETTINGS:-configs/experiment.local.env}"
set +a
export PATH="$SERVING_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${RESEARCH_GPU:-7}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec "$SERVING_ENV/bin/vllm" serve "$POLICY_MODEL" \
  --served-model-name memsearch-research --host 127.0.0.1 --port "${RESEARCH_PORT:-8002}" \
  --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-6144}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" --max-num-seqs 4 \
  --enable-lora --max-lora-rank 16 --max-loras 4 --max-cpu-loras 16
