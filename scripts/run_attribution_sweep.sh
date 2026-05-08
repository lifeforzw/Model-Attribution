#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-gpt2}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/attribution_sweep}"
TARGET_TEXT="${TARGET_TEXT:- Paris}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
STEPS="${STEPS:-16}"

HOOK_PATTERNS=(
  "${HOOK_PATTERNS_0:-.*attn.*}"
  "${HOOK_PATTERNS_1:-.*mlp.*}"
)

for pattern in "${HOOK_PATTERNS[@]}"; do
  safe_name="$(printf '%s' "$pattern" | tr -cs '[:alnum:]' '_')"
  MODEL="$MODEL" \
  PROMPT="${PROMPT:-The capital of France is}" \
  TARGET_TEXT="$TARGET_TEXT" \
  HOOK_PATTERN="$pattern" \
  DEVICE="$DEVICE" \
  DTYPE="$DTYPE" \
  STEPS="$STEPS" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  RUN_NAME="ig_${safe_name}" \
    bash scripts/run_integrated_gradients.sh

  MODEL="$MODEL" \
  CLEAN_PROMPT="${CLEAN_PROMPT:-The capital of France is}" \
  CORRUPT_PROMPT="${CORRUPT_PROMPT:-The capital of Italy is}" \
  TARGET_TEXT="$TARGET_TEXT" \
  HOOK_PATTERN="$pattern" \
  DEVICE="$DEVICE" \
  DTYPE="$DTYPE" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  RUN_NAME="atp_${safe_name}" \
    bash scripts/run_attribution_patching.sh
done
