#!/usr/bin/env bash
set -uo pipefail

# Smoke test 与正式训练使用完全相同的任务矩阵，只把每个阶段限制为 1 个 batch。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SMOKE=1

exec "$SCRIPT_DIR/run_all_pure_time_series.sh" "$@"
