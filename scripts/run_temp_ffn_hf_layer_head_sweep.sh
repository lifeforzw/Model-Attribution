#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

MODEL="${MODEL:-gpt2}"
LAYERS="${LAYERS:-0 3 6 9 11}"
HEADS="${HEADS:-0 1 2 3}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
QUERY_COUNT="${QUERY_COUNT:-128}"
MEMORY_UNITS="${MEMORY_UNITS:-32,64,128,256}"
STEPS="${STEPS:-350}"
LR="${LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-128}"
COSINE_WEIGHT="${COSINE_WEIGHT:-0.1}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/temp_ffn/hf_layer_head_sweep}"
TEXT="${TEXT:-}"
TEXT_FILE="${TEXT_FILE:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
INIT="${INIT:-sample}"
QKV_LAYOUT="${QKV_LAYOUT:-gpt2}"

for LAYER in $LAYERS; do
  for HEAD in $HEADS; do
    RUN_NAME="${MODEL##*/}_layer${LAYER}_head${HEAD}"
    cmd=(
      python examples/run_temp_ffn_compression.py
      hf-activations
      --model "$MODEL"
      --layer "$LAYER"
      --head "$HEAD"
      --context-length "$CONTEXT_LENGTH"
      --query-count "$QUERY_COUNT"
      --memory-units "$MEMORY_UNITS"
      --steps "$STEPS"
      --lr "$LR"
      --batch-size "$BATCH_SIZE"
      --cosine-weight "$COSINE_WEIGHT"
      --device "$DEVICE"
      --dtype "$DTYPE"
      --output-dir "$OUTPUT_DIR"
      --run-name "$RUN_NAME"
      --init "$INIT"
      --qkv-layout "$QKV_LAYOUT"
    )
    if [[ -n "$TEXT" ]]; then
      cmd+=(--text "$TEXT")
    fi
    if [[ -n "$TEXT_FILE" ]]; then
      cmd+=(--text-file "$TEXT_FILE")
    fi
    if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
      cmd+=(--trust-remote-code)
    fi
    "${cmd[@]}"
  done
done
