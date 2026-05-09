#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
HEAD_DIM="${HEAD_DIM:-64}"
QUERY_COUNT="${QUERY_COUNT:-2048}"
MEMORY_UNITS="${MEMORY_UNITS:-32,64,128,256,512,1024}"
STEPS="${STEPS:-500}"
LR="${LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-512}"
COSINE_WEIGHT="${COSINE_WEIGHT:-0.1}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float32}"
SEED="${SEED:-13}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/temp_ffn/random}"
RUN_NAME="${RUN_NAME:-random_qkv}"
INIT="${INIT:-sample}"
SAVE_STUDENTS="${SAVE_STUDENTS:-0}"

cmd=(
  python examples/run_temp_ffn_compression.py
  random
  --context-length "$CONTEXT_LENGTH"
  --head-dim "$HEAD_DIM"
  --query-count "$QUERY_COUNT"
  --memory-units "$MEMORY_UNITS"
  --steps "$STEPS"
  --lr "$LR"
  --batch-size "$BATCH_SIZE"
  --cosine-weight "$COSINE_WEIGHT"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --init "$INIT"
)

if [[ "$SAVE_STUDENTS" == "1" ]]; then
  cmd+=(--save-students)
fi

"${cmd[@]}"
