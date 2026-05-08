#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

MODEL="${MODEL:-gpt2}"
PROMPT="${PROMPT:-The capital of France is}"
SUBJECT="${SUBJECT:-France}"
TARGET_TEXT="${TARGET_TEXT:- Paris}"
TARGET_TOKEN_ID="${TARGET_TOKEN_ID:-}"
HOOK_PATTERN="${HOOK_PATTERN:-transformer\\.h\\.[0-9]+$}"
HOOK_TYPE="${HOOK_TYPE:-forward}"
TENSOR_PATH="${TENSOR_PATH:-0}"
TARGET_POSITION="${TARGET_POSITION:-last}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1}"
BETAS="${BETAS:-$ALPHAS}"
PCA_TOKEN_POSITIONS="${PCA_TOKEN_POSITIONS:-last}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/sr_demo}"
RUN_NAME="${RUN_NAME:-sr_demo}"

cmd=(
  python examples/run_sr_propagation_cli.py
  --model "$MODEL"
  --prompt "$PROMPT"
  --subject "$SUBJECT"
  --hook-pattern "$HOOK_PATTERN"
  --hook-type "$HOOK_TYPE"
  --tensor-path "$TENSOR_PATH"
  --target-position "$TARGET_POSITION"
  --alphas "$ALPHAS"
  --betas "$BETAS"
  --pca-token-positions "$PCA_TOKEN_POSITIONS"
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

"${cmd[@]}"
