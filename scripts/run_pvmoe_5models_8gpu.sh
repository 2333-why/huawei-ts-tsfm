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
export SKIPPD_PARQUET="${SKIPPD_PARQUET:-/data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/Luoyang-Unified_format-V1-with_DNI_DHI.parquet}"
export PVOD_PARQUET="${PVOD_PARQUET:-/data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/YLJ-Unified_format-with_DNI_DHI.parquet}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/results_foundation_5models_8gpu}"
DATASET_PREFLIGHT_DIR="$OUTPUT_ROOT/dataset_preflight"
export SKIPPD_CONFIG="$DATASET_PREFLIGHT_DIR/luoyang_recent_four_months.json"
export PVOD_CONFIG="$DATASET_PREFLIGHT_DIR/ylj_original_split.json"
export SUMMARY_PATH="${SUMMARY_PATH:-$OUTPUT_ROOT/run_summary.tsv}"
export RESUME="${RESUME:-1}"
export SMOKE=0

mkdir -p "$OUTPUT_ROOT"

echo "[1/4] 读取并检查两个数据集；自动生成 Luoyang 新切分和 YLJ 原切分配置"
"$PYTHON" scripts/check_datasets_and_prepare.py \
  --luoyang-config "$ROOT_DIR/configs/datasets/skippd_luoyang.json" \
  --luoyang-parquet "$SKIPPD_PARQUET" \
  --ylj-config "$ROOT_DIR/configs/datasets/pvod_station00_ylj.yaml" \
  --ylj-parquet "$PVOD_PARQUET" \
  --output-dir "$DATASET_PREFLIGHT_DIR"

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
echo "数据检查报告: $DATASET_PREFLIGHT_DIR/dataset_preflight.json"
