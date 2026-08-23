#!/usr/bin/env bash
set -uo pipefail

# ==================== 配置 ====================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/data/private/penv/time/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/results_pure_time_series}"
SUMMARY_PATH="${SUMMARY_PATH:-$OUTPUT_ROOT/run_summary.tsv}"
EPOCHS="${EPOCHS:-40}"
RESUME="${RESUME:-0}"
SMOKE="${SMOKE:-0}"
GPUS="${GPUS:-0 1}"
DATASETS=("skippd_luoyang" "pvod_station00_ylj")
SETTINGS=("24 1" "48 12")

# 本脚本固定使用两张不同的显卡，默认是 GPU 0 和 GPU 1。
read -r GPU_0 GPU_1 EXTRA_GPU <<<"$GPUS"
if [[ -z "${GPU_0:-}" || -z "${GPU_1:-}" || -n "${EXTRA_GPU:-}" || "$GPU_0" == "$GPU_1" ]]; then
    echo "GPUS 必须包含两张不同的显卡，例如: GPUS='0 1'" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$(dirname "$SUMMARY_PATH")"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

# ==================== 获取模型列表 ====================
MODEL_FILE="$TEMP_DIR/models.txt"
if ! CUDA_VISIBLE_DEVICES="$GPU_0" \
    "$PYTHON" "$ROOT_DIR/run_time_series.py" --list-models >"$MODEL_FILE"; then
    echo "无法读取纯时序模型列表" >&2
    exit 2
fi
mapfile -t MODELS <"$MODEL_FILE"

if (( ${#MODELS[@]} != 32 )); then
    echo "应有 32 个模型，实际读取到 ${#MODELS[@]} 个" >&2
    exit 2
fi

declare -A MODEL_SEEN=()
for model in "${MODELS[@]}"; do
    if [[ -z "$model" || -n "${MODEL_SEEN[$model]:-}" ]]; then
        echo "模型列表包含空名称或重复名称: $model" >&2
        exit 2
    fi
    MODEL_SEEN["$model"]=1
done

# ==================== 把 128 个任务平均分给两张卡 ====================
QUEUE_0="$TEMP_DIR/gpu-0-tasks.tsv"
QUEUE_1="$TEMP_DIR/gpu-1-tasks.tsv"
: >"$QUEUE_0"
: >"$QUEUE_1"

task_index=0
for setting in "${SETTINGS[@]}"; do
    read -r seq_len pred_len <<<"$setting"
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            if (( task_index % 2 == 0 )); then
                printf '%s\t%s\t%s\t%s\n' \
                    "$seq_len" "$pred_len" "$dataset" "$model" >>"$QUEUE_0"
            else
                printf '%s\t%s\t%s\t%s\n' \
                    "$seq_len" "$pred_len" "$dataset" "$model" >>"$QUEUE_1"
            fi
            task_index=$((task_index + 1))
        done
    done
done
EXPECTED_TASKS=$task_index

RUN_MODE_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
    RUN_MODE_ARGS=(--smoke)
fi

# RESUME=1 时，三个结果文件都存在就跳过该任务。
results_exist() {
    local output_dir="$1"
    [[ -s "$output_dir/best.pt" \
        && -s "$output_dir/metrics.json" \
        && -s "$output_dir/predictions.csv" ]]
}

# 一张卡一次只训练一个模型；两张卡的队列会同时运行。
run_gpu_queue() {
    local gpu="$1"
    local queue_file="$2"
    local result_file="$3"
    local seq_len pred_len dataset model output_dir log_file exit_code status

    : >"$result_file"
    while IFS=$'\t' read -r seq_len pred_len dataset model; do
        output_dir="$OUTPUT_ROOT/seq${seq_len}_pred${pred_len}/$dataset/$model"
        log_file="$output_dir/run.log"
        mkdir -p "$output_dir"

        if [[ "$RESUME" == "1" ]] && results_exist "$output_dir"; then
            echo "[跳过 GPU $gpu] seq=$seq_len pred=$pred_len / $dataset / $model"
            exit_code=0
            status="PASS"
        else
            echo "[运行 GPU $gpu] seq=$seq_len pred=$pred_len / $dataset / $model"
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
                echo "[失败 GPU $gpu] seq=$seq_len pred=$pred_len / $dataset / $model，日志: $log_file" >&2
            fi
        fi

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$seq_len" "$pred_len" "$dataset" "$model" \
            "$status" "$output_dir" "$exit_code" \
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
    printf 'seq_len\tpred_len\tdataset\tmodel\tstatus\toutput_dir\texit_code\n'
    cat "$RESULT_0" "$RESULT_1"
} >"$SUMMARY_PATH"

result_count=$(($(wc -l <"$SUMMARY_PATH") - 1))
if (( worker_failed != 0 || result_count != EXPECTED_TASKS )) \
    || grep -q $'\tFAIL\t' "$SUMMARY_PATH"; then
    echo "有实验失败，请查看: $SUMMARY_PATH" >&2
    exit 1
fi

echo "全部 $EXPECTED_TASKS 个实验完成，汇总文件: $SUMMARY_PATH"
