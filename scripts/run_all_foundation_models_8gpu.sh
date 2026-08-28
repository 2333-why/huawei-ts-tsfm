#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/_run_all_foundation_models_fixed_gpu.sh" 8 "0 1 2 3 4 5 6 7" "$@"
