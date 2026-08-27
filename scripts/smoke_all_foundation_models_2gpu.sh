#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SMOKE=1
exec "$SCRIPT_DIR/run_all_foundation_models_2gpu.sh" "$@"
