#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

MODEL="${MODEL:-gpt2}"
PROMPT="${PROMPT:-The capital of France is}"
TARGET_TEXT="${TARGET_TEXT:- Paris}"
TARGET_TOKEN_ID="${TARGET_TOKEN_ID:-}"
HOOK_PATTERN="${HOOK_PATTERN:-.*mlp.*}"
HOOK_TYPE="${HOOK_TYPE:-forward}"
TENSOR_PATH="${TENSOR_PATH:-}"
TARGET_POSITION="${TARGET_POSITION:-last}"
STEPS="${STEPS:-32}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/attribution}"
RUN_NAME="${RUN_NAME:-ig}"

cmd=(
  python examples/run_attribution_cli.py
  --method ig
  --model "$MODEL"
  --prompt "$PROMPT"
  --hook-pattern "$HOOK_PATTERN"
  --hook-type "$HOOK_TYPE"
  --target-position "$TARGET_POSITION"
  --steps "$STEPS"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
)

if [[ -n "$TARGET_TOKEN_ID" ]]; then
  cmd+=(--target-token-id "$TARGET_TOKEN_ID")
else
  cmd+=(--target-text "$TARGET_TEXT")
fi

if [[ -n "$TENSOR_PATH" ]]; then
  cmd+=(--tensor-path "$TENSOR_PATH")
fi

"${cmd[@]}"
