#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
TORCH_PROFILE="${TORCH_PROFILE:-cpu}"
INSTALL_PROJECT="${INSTALL_PROJECT:-1}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -U pip setuptools wheel

case "$TORCH_PROFILE" in
  skip)
    ;;
  cpu)
    "$VENV_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
    ;;
  cuda121)
    "$VENV_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
    ;;
  cuda124)
    "$VENV_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
    ;;
  existing)
    ;;
  *)
    echo "Unsupported TORCH_PROFILE=$TORCH_PROFILE" >&2
    echo "Use one of: skip, cpu, cuda121, cuda124, existing" >&2
    exit 2
    ;;
esac

if [[ "$INSTALL_PROJECT" == "1" ]]; then
  "$VENV_DIR/bin/python" -m pip install -e ".[dev]"
  "$VENV_DIR/bin/python" -m pip install transformers datasets accelerate
fi

cat <<EOF
Environment ready.

Activate:
  source $VENV_DIR/bin/activate

Use source tree:
  export PYTHONPATH="$ROOT_DIR/src:\${PYTHONPATH:-}"

Torch profile:
  $TORCH_PROFILE
EOF
