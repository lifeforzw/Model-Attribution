#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

MODEL="${MODEL:-gpt2}"
LAYER="${LAYER:-6}"
HEAD="${HEAD:-0}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
QUERY_COUNT="${QUERY_COUNT:-128}"
MEMORY_UNITS="${MEMORY_UNITS:-32,64,128,256}"
STEPS="${STEPS:-500}"
LR="${LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-128}"
COSINE_WEIGHT="${COSINE_WEIGHT:-0.1}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/temp_ffn/hf_activations}"
RUN_NAME="${RUN_NAME:-}"
TEXT="${TEXT:-}"
TEXT_FILE="${TEXT_FILE:-}"
QKV_MODULE_PATTERN="${QKV_MODULE_PATTERN:-}"
QKV_LAYOUT="${QKV_LAYOUT:-gpt2}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
SAVE_STUDENTS="${SAVE_STUDENTS:-0}"
INIT="${INIT:-sample}"

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
  --init "$INIT"
  --qkv-layout "$QKV_LAYOUT"
)

if [[ -n "$RUN_NAME" ]]; then
  cmd+=(--run-name "$RUN_NAME")
fi
if [[ -n "$TEXT" ]]; then
  cmd+=(--text "$TEXT")
fi
if [[ -n "$TEXT_FILE" ]]; then
  cmd+=(--text-file "$TEXT_FILE")
fi
if [[ -n "$QKV_MODULE_PATTERN" ]]; then
  cmd+=(--qkv-module-pattern "$QKV_MODULE_PATTERN")
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi
if [[ "$SAVE_STUDENTS" == "1" ]]; then
  cmd+=(--save-students)
fi

"${cmd[@]}"
