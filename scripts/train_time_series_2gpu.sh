#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ==================== 必须修改：数据集地址 ====================
# 可直接修改下面两个默认值，也可在命令行用同名环境变量覆盖。
SKIPPD_PARQUET="${SKIPPD_PARQUET:-/REPLACE_WITH_ABSOLUTE_PATH/skippd_luoyang.parquet}"
PVOD_PARQUET="${PVOD_PARQUET:-/REPLACE_WITH_ABSOLUTE_PATH/station00_ylj.parquet}"
# =============================================================

if [[ ! -f "$SKIPPD_PARQUET" ]]; then
    echo "SKIPPD_PARQUET 数据集地址不存在: $SKIPPD_PARQUET" >&2
    exit 2
fi
if [[ ! -f "$PVOD_PARQUET" ]]; then
    echo "PVOD_PARQUET 数据集地址不存在: $PVOD_PARQUET" >&2
    exit 2
fi

export SKIPPD_PARQUET PVOD_PARQUET
export GPUS="${GPUS:-0 1}"
export EXPECTED_GPU_COUNT=2
export INCLUDE_BASELINES=0
exec bash "$ROOT_DIR/scripts/run_all_pure_time_series.sh"
