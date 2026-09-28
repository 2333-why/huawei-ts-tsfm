#!/usr/bin/env bash
set -euo pipefail

# One-command validation for the updated TSFM branch.
# TEST_SCOPE=smoke (default): unit tests + offline preflight + real-weight GPU smoke.
# TEST_SCOPE=unit: code-only validation without datasets, weights, or GPUs.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
TEST_SCOPE="${TEST_SCOPE:-smoke}"
GPUS="${GPUS:-0 1}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/results_updated_tsfm_test}"
HF_HOME="${HF_HOME:-$ROOT_DIR/checkpoints_huggingface}"

case "$TEST_SCOPE" in
    unit|smoke) ;;
    *)
        echo "TEST_SCOPE 仅支持 unit 或 smoke: $TEST_SCOPE" >&2
        exit 2
        ;;
esac

mkdir -p "$RESULTS_ROOT"

echo "[1/4] Python 与代码静态检查"
"$PYTHON" --version
"$PYTHON" -m compileall -q run.py models data_provider utils scripts
"$PYTHON" run.py --list-models | tee "$RESULTS_ROOT/models.txt"
"$PYTHON" run.py --list-modes | tee "$RESULTS_ROOT/modes.txt"

echo "[2/4] pytest 全量测试"
"$PYTHON" -m pytest -q | tee "$RESULTS_ROOT/pytest.log"

if [[ "$TEST_SCOPE" == "unit" ]]; then
    echo "代码测试通过。结果目录: $RESULTS_ROOT"
    exit 0
fi

echo "[3/4] 离线环境、数据、GPU与固定权重预检"
export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$PYTHON" scripts/check_foundation_environment.py --offline --json \
    | tee "$RESULTS_ROOT/preflight.json"

read -r -a GPU_IDS <<<"$GPUS"
if (( ${#GPU_IDS[@]} == 2 )); then
    RUNNER="$ROOT_DIR/scripts/smoke_all_foundation_models_2gpu.sh"
elif (( ${#GPU_IDS[@]} == 8 )); then
    RUNNER="$ROOT_DIR/scripts/run_all_foundation_models_8gpu.sh"
else
    echo "真实权重 smoke 仅支持 2 或 8 张 GPU，当前 GPUS='$GPUS'" >&2
    exit 2
fi

echo "[4/4] ${#GPU_IDS[@]} GPU 真实权重 smoke 矩阵"
SMOKE=1 \
RESUME="${RESUME:-0}" \
GPUS="$GPUS" \
OUTPUT_ROOT="$RESULTS_ROOT/smoke" \
SUMMARY_PATH="$RESULTS_ROOT/smoke/smoke_summary.tsv" \
PYTHON="$PYTHON" \
CONFIG_PYTHON="${CONFIG_PYTHON:-$PYTHON}" \
  bash "$RUNNER"

echo "全部测试通过。"
echo "pytest 日志: $RESULTS_ROOT/pytest.log"
echo "环境预检: $RESULTS_ROOT/preflight.json"
echo "smoke 汇总: $RESULTS_ROOT/smoke/smoke_summary.tsv"
