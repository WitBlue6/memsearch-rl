#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
settings="${MEMSEARCH_SETTINGS:-configs/experiment.local.env}"
[[ -f "$settings" ]] || { echo "Copy configs/experiment.env.example to $settings first" >&2; exit 1; }
set -a
source "$settings"
set +a
: "${EXPERIMENT_ID:?}" "${READER_MODEL:?}" "${POLICY_MODEL:?}" "${BUDGET_TOKENIZER:?}"
export PATH="$SERVING_ENV/bin:$HOME/.local/bin:$PATH"
extras=(--extra train)
if [[ "${MEMORY_OUTPUT:-text}" == json_schema ]]; then extras+=(--extra structured); fi
uv run --locked "${extras[@]}" python scripts/experiment.py "$@"
