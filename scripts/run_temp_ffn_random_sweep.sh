#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

HEAD_DIM="${HEAD_DIM:-64}"
QUERY_COUNT="${QUERY_COUNT:-2048}"
MEMORY_UNITS="${MEMORY_UNITS:-32,64,128,256,512}"
STEPS="${STEPS:-400}"
LR="${LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-512}"
COSINE_WEIGHT="${COSINE_WEIGHT:-0.1}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float32}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/temp_ffn/random_sweep}"
INIT="${INIT:-sample}"

for CONTEXT_LENGTH in ${CONTEXT_LENGTHS:-512 1024 2048 4096 8192}; do
  RUN_NAME="random_N${CONTEXT_LENGTH}_M${MEMORY_UNITS//,/p}"
  python examples/run_temp_ffn_compression.py \
    random \
    --context-length "$CONTEXT_LENGTH" \
    --head-dim "$HEAD_DIM" \
    --query-count "$QUERY_COUNT" \
    --memory-units "$MEMORY_UNITS" \
    --steps "$STEPS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --cosine-weight "$COSINE_WEIGHT" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --seed "$CONTEXT_LENGTH" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    --init "$INIT"
done

python examples/run_temp_ffn_compression.py \
  memory \
  --context-lengths "${CONTEXT_LENGTHS_CSV:-512,1024,2048,4096,8192}" \
  --memory-units "$MEMORY_UNITS" \
  --layers "${LAYERS:-1}" \
  --heads "${HEADS:-1}" \
  --head-dim "$HEAD_DIM" \
  --output-dir "$OUTPUT_DIR" \
  --run-name memory_table
