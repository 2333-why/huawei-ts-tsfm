#!/usr/bin/env bash
set -uo pipefail

# Run the selected pure time-series models in one sequential queue per GPU.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/data/private/penv/time/bin/python}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-40}"
RESUME="${RESUME:-0}"
GPUS="${GPUS:-0 1}"

if [[ "$SMOKE" == "1" ]]; then
    DEFAULT_OUTPUT_ROOT="$ROOT_DIR/results_pure_time_series_smoke"
    DEFAULT_SUMMARY_NAME="smoke_summary.tsv"
else
    DEFAULT_OUTPUT_ROOT="$ROOT_DIR/results_pure_time_series"
    DEFAULT_SUMMARY_NAME="run_summary.tsv"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
SUMMARY_PATH="${SUMMARY_PATH:-$OUTPUT_ROOT/$DEFAULT_SUMMARY_NAME}"
SUMMARY_HEADER=$'seq_len\tpred_len\tdataset\tmodel\tstatus\toutput_dir\texit_code\tlaunch_gpu'

DATASETS=("skippd_luoyang" "pvod_station00_ylj")
EXPECTED_MODELS=(
    "TSMixer" "Pyraformer" "SegRNN" "Transformer"
    "LightTS" "Crossformer" "FreTS" "MICN"
)

# The horizon is either one output point or four physical hours. The latter
# depends on the source sampling interval, not on a fixed number of points.
prediction_points() {
    local dataset="$1"
    local horizon="$2"

    if [[ "$horizon" == "1" ]]; then
        printf '1\n'
        return 0
    fi
    if [[ "$horizon" == "4" ]]; then
        case "$dataset" in
            skippd_luoyang) printf '48\n' ;;
            pvod_station00_ylj) printf '16\n' ;;
            *)
                echo "未知数据集: $dataset" >&2
                return 2
                ;;
        esac
        return 0
    fi

    echo "未知预测时长: $horizon" >&2
    return 2
}

# 本脚本固定使用两张不同的显卡，默认是 GPU 0 和 GPU 1。
read -r GPU_0 GPU_1 EXTRA_GPU <<<"$GPUS"
if [[ -z "${GPU_0:-}" || -z "${GPU_1:-}" || -n "${EXTRA_GPU:-}" || "$GPU_0" == "$GPU_1" ]]; then
    echo "GPUS 必须包含两张不同的显卡，例如: GPUS='0 1'" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$(dirname "$SUMMARY_PATH")"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

# Keep these rows literal: they are the complete experiment contract.
# Columns: output label, history points, physical horizon.
SETTING_ROWS=(
    "seq24_pred1 24 1"
    "seq48_pred1 48 1"
    "seq48_h4 48 4"
    "seq96_h4 96 4"
)

MODEL_FILE="$TEMP_DIR/models.txt"
if ! CUDA_VISIBLE_DEVICES="$GPU_0" \
    "$PYTHON" "$ROOT_DIR/run_time_series.py" --list-models >"$MODEL_FILE"; then
    echo "无法读取纯时序模型列表" >&2
    exit 2
fi
mapfile -t MODELS <"$MODEL_FILE"

if (( ${#MODELS[@]} != ${#EXPECTED_MODELS[@]} )); then
    echo "应有 ${#EXPECTED_MODELS[@]} 个模型，实际读取到 ${#MODELS[@]} 个" >&2
    exit 2
fi
for index in "${!EXPECTED_MODELS[@]}"; do
    if [[ "${MODELS[$index]}" != "${EXPECTED_MODELS[$index]}" ]]; then
        echo "模型列表顺序或名称不符合纯时序 top-8: ${MODELS[*]}" >&2
        exit 2
    fi
done

# ==================== 把 64 个任务平均分给两张卡 ====================
QUEUE_0="$TEMP_DIR/gpu-0-tasks.tsv"
QUEUE_1="$TEMP_DIR/gpu-1-tasks.tsv"
: >"$QUEUE_0"
: >"$QUEUE_1"

task_index=0
for row in "${SETTING_ROWS[@]}"; do
    read -r setting_label seq_len horizon <<<"$row"
    for dataset in "${DATASETS[@]}"; do
        if ! pred_len="$(prediction_points "$dataset" "$horizon")"; then
            exit 2
        fi
        for model in "${MODELS[@]}"; do
            queue="$QUEUE_0"
            if (( task_index % 2 == 0 )); then
                queue="$QUEUE_0"
            else
                queue="$QUEUE_1"
            fi
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$setting_label" "$seq_len" "$pred_len" "$dataset" "$model" >>"$queue"
            task_index=$((task_index + 1))
        done
    done
done
EXPECTED_TASKS=$task_index

# Load reusable rows before writing the new summary. A task is resumable only
# when its previous row was PASS for the same setting/dataset/model and all
# three expected artifacts are non-empty.
declare -A RESUME_PASS_DIRS=()
declare -A RESUME_PASS_LAUNCH_GPUS=()
resume_key() {
    local setting_label="$1"
    local dataset="$2"
    local model="$3"
    printf '%s|%s|%s' "$setting_label" "$dataset" "$model"
}

resume_key_from_output_dir() {
    local output_dir="$1"
    local model dataset setting_label
    model="$(basename "$output_dir")"
    dataset="$(basename "$(dirname "$output_dir")")"
    setting_label="$(basename "$(dirname "$(dirname "$output_dir")")")"
    resume_key "$setting_label" "$dataset" "$model"
}

load_resume_summary() {
    [[ "$RESUME" == "1" && -s "$SUMMARY_PATH" ]] || return 0

    local actual_header
    IFS= read -r actual_header <"$SUMMARY_PATH" || return 0
    [[ "$actual_header" == "$SUMMARY_HEADER" ]] || return 0

    local seq_len pred_len dataset model status output_dir exit_code launch_gpu key
    while IFS=$'\t' read -r seq_len pred_len dataset model status output_dir exit_code launch_gpu; do
        # The exact header gate above rejects legacy schemas before any row is
        # considered; malformed current-schema rows are skipped as well.
        [[ "$status" == "PASS" && -n "$output_dir" && -n "$launch_gpu" ]] || continue
        key="$(resume_key_from_output_dir "$output_dir")"
        RESUME_PASS_DIRS["$key"]="$output_dir"
        RESUME_PASS_LAUNCH_GPUS["$key"]="$launch_gpu"
    done < <(tail -n +2 "$SUMMARY_PATH")
}

load_resume_summary

RUN_MODE_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
    RUN_MODE_ARGS=(--smoke)
fi

# RESUME=1 时，只复用前次 summary 中 PASS 且三个结果文件非空的任务。
results_exist() {
    local output_dir="$1"
    [[ -s "$output_dir/best.pt" \
        && -s "$output_dir/metrics.json" \
        && -s "$output_dir/predictions.csv" ]]
}

can_resume() {
    local setting_label="$1"
    local dataset="$2"
    local model="$3"
    local output_dir="$4"
    local key
    key="$(resume_key "$setting_label" "$dataset" "$model")"
    [[ "${RESUME_PASS_DIRS[$key]:-}" == "$output_dir" ]] \
        && results_exist "$output_dir"
}

# 一张卡一次只训练一个模型；两张卡的队列会同时运行。
run_gpu_queue() {
    local gpu="$1"
    local queue_file="$2"
    local result_file="$3"
    local setting_label seq_len pred_len dataset model output_dir log_file exit_code status
    local key launch_gpu

    : >"$result_file"
    while IFS=$'\t' read -r setting_label seq_len pred_len dataset model; do
        output_dir="$OUTPUT_ROOT/$setting_label/$dataset/$model"
        log_file="$output_dir/run.log"
        mkdir -p "$output_dir"
        launch_gpu="$gpu"

        if [[ "$RESUME" == "1" ]] \
            && can_resume "$setting_label" "$dataset" "$model" "$output_dir"; then
            key="$(resume_key "$setting_label" "$dataset" "$model")"
            launch_gpu="${RESUME_PASS_LAUNCH_GPUS[$key]}"
            echo "[跳过 GPU $gpu] $setting_label / $dataset / $model"
            exit_code=0
            status="PASS"
        else
            echo "[运行 GPU $gpu] $setting_label / $dataset / $model"
            CUDA_VISIBLE_DEVICES="$gpu" \
                "$PYTHON" "$ROOT_DIR/run_time_series.py" \
                --dataset "$dataset" \
                --model "$model" \
                --seq_len "$seq_len" \
                --pred_len "$pred_len" \
                --epochs "$EPOCHS" \
                "${RUN_MODE_ARGS[@]}" \
                --device cuda:0 \
                --output_dir "$output_dir" \
                >"$log_file" 2>&1
            exit_code=$?

            if (( exit_code == 0 )); then
                status="PASS"
            else
                status="FAIL"
                echo "[失败 GPU $gpu] $setting_label / $dataset / $model，日志: $log_file" >&2
            fi
        fi

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$seq_len" "$pred_len" "$dataset" "$model" \
            "$status" "$output_dir" "$exit_code" "$launch_gpu" \
            >>"$result_file"
    done <"$queue_file"
}

# ==================== 同时启动 GPU 0 和 GPU 1 ====================
RESULT_0="$TEMP_DIR/gpu-0-results.tsv"
RESULT_1="$TEMP_DIR/gpu-1-results.tsv"

run_gpu_queue "$GPU_0" "$QUEUE_0" "$RESULT_0" &
PID_0=$!
run_gpu_queue "$GPU_1" "$QUEUE_1" "$RESULT_1" &
PID_1=$!

worker_failed=0
wait "$PID_0" || worker_failed=1
wait "$PID_1" || worker_failed=1

# ==================== 汇总结果 ====================
{
    printf '%s\n' "$SUMMARY_HEADER"
    cat "$RESULT_0" "$RESULT_1"
} >"$SUMMARY_PATH"

result_count=$(($(wc -l <"$SUMMARY_PATH") - 1))
if (( worker_failed != 0 || result_count != EXPECTED_TASKS )) \
    || grep -q $'\tFAIL\t' "$SUMMARY_PATH"; then
    echo "有实验失败，请查看: $SUMMARY_PATH" >&2
    exit 1
fi

echo "全部 $EXPECTED_TASKS 个实验完成，汇总文件: $SUMMARY_PATH"
