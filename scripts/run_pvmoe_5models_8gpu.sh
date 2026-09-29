#!/usr/bin/env bash
set -euo pipefail

# Full 68-task matrix on PVMMoE: Sundial, both TimeMoE sizes, Chronos2, TiRex.
# TimesFM is intentionally excluded until its local checkpoint is tested.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/use_pvmoe_local_weights.sh"
unset TIMESFM_WEIGHT_DIR

export PYTHON="${PYTHON:-python}"
export CONFIG_PYTHON="${CONFIG_PYTHON:-$PYTHON}"
export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
export FOUNDATION_MODELS="Sundial TimeMoE TimeMoE200M Chronos2 TiRex"
export SKIPPD_PARQUET="${SKIPPD_PARQUET:-/data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/Luoyang-Unified_format-V1.parquet}"
export PVOD_PARQUET="${PVOD_PARQUET:-/data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/YLJ-Unified_format.parquet}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/results_foundation_5models_8gpu}"
RECENT_CONFIG_DIR="$OUTPUT_ROOT/recent_four_months_configs"
export SKIPPD_CONFIG="$RECENT_CONFIG_DIR/luoyang_recent_four_months.json"
export PVOD_CONFIG="${PVOD_CONFIG:-$ROOT_DIR/configs/datasets/pvod_station00_ylj.yaml}"
export SUMMARY_PATH="${SUMMARY_PATH:-$OUTPUT_ROOT/run_summary.tsv}"
export RESUME="${RESUME:-1}"
export SMOKE=0

mkdir -p "$OUTPUT_ROOT"

echo "[1/4] 检查 Luoyang 数据范围并生成前两月训练、后两月测试配置"
"$PYTHON" scripts/prepare_recent_four_months.py \
  --luoyang-config "$ROOT_DIR/configs/datasets/skippd_luoyang.json" \
  --luoyang-parquet "$SKIPPD_PARQUET" \
  --output-dir "$RECENT_CONFIG_DIR"

echo "YLJ 保持原配置不变: $PVOD_CONFIG"

echo "[2/4] 校验五个本地权重"
"$PYTHON" scripts/check_local_weights.py \
  --models Sundial TimeMoE TimeMoE200M Chronos2 TiRex \
  --verify-sha256 --json | tee "$OUTPUT_ROOT/local_weights.json"

echo "[3/4] 校验环境、数据、GPU 与五模型依赖"
"$PYTHON" scripts/check_foundation_environment.py \
  --offline --models Sundial TimeMoE TimeMoE200M Chronos2 TiRex --json \
  | tee "$OUTPUT_ROOT/preflight.json"

echo "[4/4] 运行 8 GPU 完整 68 任务实验矩阵"
bash scripts/run_all_foundation_models_8gpu.sh

"$PYTHON" scripts/foundation_results_to_md.py \
  --summary "$SUMMARY_PATH" \
  --output "$OUTPUT_ROOT/results_summary.md"

echo "全部 68 个实验完成。"
echo "TSV: $SUMMARY_PATH"
echo "Markdown: $OUTPUT_ROOT/results_summary.md"
