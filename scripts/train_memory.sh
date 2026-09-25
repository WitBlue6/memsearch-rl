#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
: "${TRAIN_DATA:?Set TRAIN_DATA to the converted training dataset directory}"
: "${RUN_OUT:?Set RUN_OUT to a new output directory}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
uv run --locked --extra train accelerate launch --config_file configs/accelerate_8x3090.yaml \
  --num_processes "${TRAIN_PROCESSES:-6}" \
  -m memsearch.training grpo --role memory \
  --model "${POLICY_MODEL:?Set POLICY_MODEL to the training model path or repository ID}" \
  --config "${EXPERIMENT_CONFIG:-configs/api.toml}" \
  --data "$TRAIN_DATA" --out "$RUN_OUT" "$@"
