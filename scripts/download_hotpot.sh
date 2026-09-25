#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base=https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve
base="$base/1908d6afbbead072334abe2965f91bd2709910ab/distractor"
mkdir -p data/raw/hotpot-hf
for name in train-00000-of-00002.parquet train-00001-of-00002.parquet validation-00000-of-00001.parquet; do
  target="data/raw/hotpot-hf/$name"
  if [[ -f "$target" ]]; then echo "Keeping $target; conversion will check metadata"; continue; fi
  curl -fL --connect-timeout 15 --max-time 1800 --retry 3 -o "$target.part" "$base/$name"
  mv "$target.part" "$target"
done
