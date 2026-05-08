#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

MODEL="${MODEL:-gpt2}"
CLEAN_PROMPT="${CLEAN_PROMPT:-The capital of France is}"
CORRUPT_PROMPT="${CORRUPT_PROMPT:-The capital of Italy is}"
TARGET_TEXT="${TARGET_TEXT:- Paris}"
TARGET_TOKEN_ID="${TARGET_TOKEN_ID:-}"
HOOK_PATTERN="${HOOK_PATTERN:-.*mlp.*}"
HOOK_TYPE="${HOOK_TYPE:-forward}"
TENSOR_PATH="${TENSOR_PATH:-}"
TARGET_POSITION="${TARGET_POSITION:-last}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/attribution}"
RUN_NAME="${RUN_NAME:-atp}"

cmd=(
  python examples/run_attribution_cli.py
  --method atp
  --model "$MODEL"
  --clean-prompt "$CLEAN_PROMPT"
  --corrupt-prompt "$CORRUPT_PROMPT"
  --hook-pattern "$HOOK_PATTERN"
  --hook-type "$HOOK_TYPE"
  --target-position "$TARGET_POSITION"
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
