#!/usr/bin/env bash
set -euo pipefail

# Install the five-model runtime into the environment that is already active,
# then download every registry-pinned checkpoint.  This script never creates
# or activates a virtual environment.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
HF_HOME="${HF_HOME:-$ROOT_DIR/checkpoints_huggingface}"
export HF_HOME

if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ "$PYTHON" != */* ]]; then
    echo "python executable is not available: $PYTHON" >&2
    exit 2
fi
if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "the complete five-model runtime requires Python 3.10 or newer" >&2
    exit 2
fi

echo "Using active interpreter: $($PYTHON -c 'import sys; print(sys.executable)')"
echo "Installing complete TSFM dependencies into the active environment..."
"$PYTHON" -m pip install -r "$ROOT_DIR/requirements-modern.txt"

mkdir -p -- "$HF_HOME"
echo "Downloading pinned checkpoints into HF_HOME=$HF_HOME"
"$PYTHON" "$ROOT_DIR/scripts/download_foundation_weights.py" "$@"

echo "Foundation runtime setup completed."
echo "The experiment scripts will reuse HF_HOME=$HF_HOME by default."
