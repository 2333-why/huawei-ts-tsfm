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
COMPLETION_HEADER=$'schema_version\tdataset\tmodel\tseq_len\tpred_len\toutput_dir\trun_mode\tepochs\tmax_train_steps\tmax_eval_steps\tmax_test_steps\ttrain_steps\tval_steps\ttest_steps\tbest_sha256\tpredictions_sha256\tmetrics_sha256'

DATASETS=("skippd_luoyang" "pvod_station00_ylj")
EXPECTED_MODELS=(
    "TSMixer" "Pyraformer" "SegRNN" "Transformer"
    "LightTS" "Crossformer" "FreTS" "MICN"
)
EXPECTED_BASELINES=(
    "Persistence" "SmartPersistence" "SeasonalPersistence" "Climatology"
    "MovingMedian" "DriftPersistence" "ClearSkyEWMA" "ClearSkyAR"
    "SimilarDay" "PersistenceClimatologyBlend"
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
OUTPUT_ROOT_CANONICAL="$(realpath -m "$OUTPUT_ROOT")"
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

setting_label_for_task() {
    local seq_len="$1"
    local pred_len="$2"
    local dataset="$3"
    case "$seq_len:$pred_len:$dataset" in
        24:1:skippd_luoyang|24:1:pvod_station00_ylj) printf 'seq24_pred1\n' ;;
        48:1:skippd_luoyang|48:1:pvod_station00_ylj) printf 'seq48_pred1\n' ;;
        48:48:skippd_luoyang|48:16:pvod_station00_ylj) printf 'seq48_h4\n' ;;
        96:48:skippd_luoyang|96:16:pvod_station00_ylj) printf 'seq96_h4\n' ;;
        *) return 1 ;;
    esac
}

model_is_expected() {
    local candidate="$1"
    local model
    for model in "${EXPECTED_MODELS[@]}"; do
        [[ "$candidate" == "$model" ]] && return 0
    done
    return 1
}

baseline_is_expected() {
    local candidate="$1"
    local baseline
    for baseline in "${EXPECTED_BASELINES[@]}"; do
        [[ "$candidate" == "$baseline" ]] && return 0
    done
    return 1
}

method_is_expected() {
    model_is_expected "$1" || baseline_is_expected "$1"
}

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

# Baseline discovery is independent of GPU discovery. Keep this invocation
# device-free because baseline tasks always run on the CPU queue.
BASELINE_FILE="$TEMP_DIR/baselines.txt"
if ! "$PYTHON" "$ROOT_DIR/run_time_series.py" --list-baselines >"$BASELINE_FILE"; then
    echo "无法读取经典基线列表" >&2
    exit 2
fi
mapfile -t BASELINES <"$BASELINE_FILE"

if (( ${#BASELINES[@]} != ${#EXPECTED_BASELINES[@]} )); then
    echo "应有 ${#EXPECTED_BASELINES[@]} 个基线，实际读取到 ${#BASELINES[@]} 个" >&2
    exit 2
fi
for index in "${!EXPECTED_BASELINES[@]}"; do
    if [[ "${BASELINES[$index]}" != "${EXPECTED_BASELINES[$index]}" ]]; then
        echo "基线列表顺序或名称不符合要求: ${BASELINES[*]}" >&2
        exit 2
    fi
done

# ==================== 把 64 个神经任务平均分给两张卡 ====================
QUEUE_0="$TEMP_DIR/gpu-0-tasks.tsv"
QUEUE_1="$TEMP_DIR/gpu-1-tasks.tsv"
CPU_QUEUE="$TEMP_DIR/cpu-tasks.tsv"
: >"$QUEUE_0"
: >"$QUEUE_1"
: >"$CPU_QUEUE"

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
EXPECTED_NEURAL_TASKS=$task_index

baseline_task_index=0
for row in "${SETTING_ROWS[@]}"; do
    read -r setting_label seq_len horizon <<<"$row"
    for dataset in "${DATASETS[@]}"; do
        if ! pred_len="$(prediction_points "$dataset" "$horizon")"; then
            exit 2
        fi
        for baseline in "${BASELINES[@]}"; do
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$setting_label" "$seq_len" "$pred_len" "$dataset" "$baseline" >>"$CPU_QUEUE"
            baseline_task_index=$((baseline_task_index + 1))
        done
    done
done
EXPECTED_BASELINE_TASKS=$baseline_task_index
EXPECTED_TASKS=$((EXPECTED_NEURAL_TASKS + EXPECTED_BASELINE_TASKS))

# Load reusable rows before writing the new summary. A task is resumable only
# when its previous row and completion manifest describe the exact current
# task and all three artifact hashes still match.
declare -A RESUME_PASS_DIRS=()
declare -A RESUME_PASS_LAUNCH_GPUS=()
declare -A RESUME_SEEN_KEYS=()
declare -A RESUME_INVALID_KEYS=()
resume_key() {
    local setting_label="$1"
    local dataset="$2"
    local model="$3"
    printf '%s|%s|%s' "$setting_label" "$dataset" "$model"
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

completion_is_valid() {
    local output_dir="$1"
    local expected_seq_len="$2"
    local expected_pred_len="$3"
    local expected_dataset="$4"
    local expected_model="$5"
    local manifest="$output_dir/completion.tsv"
    local actual_header manifest_line
    local schema dataset model seq_len pred_len manifest_output_dir run_mode epochs
    local max_train_steps max_eval_steps max_test_steps train_steps val_steps test_steps
    local best_sha256 predictions_sha256 metrics_sha256 extra
    local expected_mode expected_epochs expected_train_limit expected_eval_limit expected_test_limit

    [[ -s "$manifest" ]] || return 1
    IFS= read -r actual_header <"$manifest" || return 1
    [[ "$actual_header" == "$COMPLETION_HEADER" ]] || return 1
    mapfile -t manifest_rows < <(tail -n +2 "$manifest")
    (( ${#manifest_rows[@]} == 1 )) || return 1
    manifest_line="${manifest_rows[0]}"
    IFS=$'\t' read -r schema dataset model seq_len pred_len manifest_output_dir \
        run_mode epochs max_train_steps max_eval_steps max_test_steps \
        train_steps val_steps test_steps best_sha256 predictions_sha256 metrics_sha256 extra \
        <<<"$manifest_line"
    [[ -z "$extra" ]] || return 1
    [[ "$schema" == "1" \
        && "$dataset" == "$expected_dataset" \
        && "$model" == "$expected_model" \
        && "$seq_len" == "$expected_seq_len" \
        && "$pred_len" == "$expected_pred_len" \
        && "$manifest_output_dir" == "$output_dir" ]] || return 1

    if [[ "$SMOKE" == "1" ]]; then
        expected_mode="smoke"
        expected_epochs="1"
        expected_train_limit="1"
        expected_eval_limit="1"
        expected_test_limit="1"
    else
        expected_mode="full"
        expected_train_limit="0"
        expected_eval_limit="0"
        expected_test_limit="0"
    fi
    if baseline_is_expected "$expected_model"; then
        expected_epochs="0"
    elif [[ "$SMOKE" != "1" ]]; then
        expected_epochs="$EPOCHS"
    fi
    [[ "$run_mode" == "$expected_mode" \
        && "$epochs" == "$expected_epochs" \
        && "$max_train_steps" == "$expected_train_limit" \
        && "$max_eval_steps" == "$expected_eval_limit" \
        && "$max_test_steps" == "$expected_test_limit" ]] || return 1

    if baseline_is_expected "$expected_model"; then
        [[ "$train_steps" == "0" && "$val_steps" == "0" \
            && "$test_steps" =~ ^[1-9][0-9]*$ ]] || return 1
    elif [[ "$SMOKE" == "1" ]]; then
        [[ "$train_steps" == "1" && "$val_steps" == "1" && "$test_steps" == "1" ]] || return 1
    else
        [[ "$train_steps" =~ ^[1-9][0-9]*$ \
            && "$val_steps" =~ ^[1-9][0-9]*$ \
            && "$test_steps" =~ ^[1-9][0-9]*$ ]] || return 1
    fi
    [[ "$best_sha256" =~ ^[0-9a-fA-F]{64}$ \
        && "$predictions_sha256" =~ ^[0-9a-fA-F]{64}$ \
        && "$metrics_sha256" =~ ^[0-9a-fA-F]{64}$ ]] || return 1
    [[ -s "$output_dir/best.pt" \
        && -s "$output_dir/predictions.csv" \
        && -s "$output_dir/metrics.json" ]] || return 1
    [[ "$(sha256_file "$output_dir/best.pt")" == "$best_sha256" \
        && "$(sha256_file "$output_dir/predictions.csv")" == "$predictions_sha256" \
        && "$(sha256_file "$output_dir/metrics.json")" == "$metrics_sha256" ]]
}

load_resume_summary() {
    [[ "$RESUME" == "1" && -s "$SUMMARY_PATH" ]] || return 0

    local actual_header
    IFS= read -r actual_header <"$SUMMARY_PATH" || return 0
    [[ "$actual_header" == "$SUMMARY_HEADER" ]] || return 0

    local seq_len pred_len dataset model status output_dir exit_code launch_gpu key
    local setting_label expected_output_dir
    while IFS=$'\t' read -r seq_len pred_len dataset model status output_dir exit_code launch_gpu; do
        # The exact header gate above rejects legacy schemas before any row is
        # considered. Task identity comes from the row itself, never from the
        # output path.
        [[ "$seq_len" =~ ^[1-9][0-9]*$ \
            && "$pred_len" =~ ^[1-9][0-9]*$ \
            && -n "$dataset" && -n "$model" ]] || continue
        setting_label="$(setting_label_for_task "$seq_len" "$pred_len" "$dataset")" || continue
        key="$(resume_key "$setting_label" "$dataset" "$model")"
        if [[ -n "${RESUME_SEEN_KEYS[$key]:-}" ]]; then
            RESUME_INVALID_KEYS["$key"]=1
            unset 'RESUME_PASS_DIRS[$key]'
            unset 'RESUME_PASS_LAUNCH_GPUS[$key]'
            continue
        fi
        RESUME_SEEN_KEYS["$key"]=1
        [[ -z "${RESUME_INVALID_KEYS[$key]:-}" ]] || continue
        [[ "$status" == "PASS" \
            && -n "$output_dir" \
            && "$exit_code" == "0" \
            && "$launch_gpu" =~ ^[^[:space:]]+$ ]] || {
            RESUME_INVALID_KEYS["$key"]=1
            continue
        }
        expected_output_dir="$OUTPUT_ROOT_CANONICAL/$setting_label/$dataset/$model"
        if ! method_is_expected "$model" || [[ "$output_dir" != "$expected_output_dir" ]]; then
            RESUME_INVALID_KEYS["$key"]=1
            continue
        fi
        if baseline_is_expected "$model" && [[ "$launch_gpu" != "cpu" ]]; then
            RESUME_INVALID_KEYS["$key"]=1
            continue
        fi
        completion_is_valid "$output_dir" "$seq_len" "$pred_len" "$dataset" "$model" || continue
        [[ -z "${RESUME_INVALID_KEYS[$key]:-}" ]] || continue
        RESUME_PASS_DIRS["$key"]="$output_dir"
        RESUME_PASS_LAUNCH_GPUS["$key"]="$launch_gpu"
    done < <(tail -n +2 "$SUMMARY_PATH")
}

load_resume_summary

RUN_MODE_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
    RUN_MODE_ARGS=(--smoke)
fi

can_resume() {
    local setting_label="$1"
    local seq_len="$2"
    local pred_len="$3"
    local dataset="$4"
    local model="$5"
    local output_dir="$6"
    local key
    key="$(resume_key "$setting_label" "$dataset" "$model")"
    [[ "${RESUME_PASS_DIRS[$key]:-}" == "$output_dir" ]] \
        && completion_is_valid "$output_dir" "$seq_len" "$pred_len" "$dataset" "$model"
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
        output_dir="$OUTPUT_ROOT_CANONICAL/$setting_label/$dataset/$model"
        log_file="$output_dir/run.log"
        mkdir -p "$output_dir"
        launch_gpu="$gpu"

        if [[ "$RESUME" == "1" ]] \
            && can_resume "$setting_label" "$seq_len" "$pred_len" "$dataset" "$model" "$output_dir"; then
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

# Baselines are evaluated serially in one CPU queue. Keep CUDA visibility
# explicitly empty so a CPU task cannot accidentally claim or initialize a GPU.
run_cpu_queue() {
    local queue_file="$1"
    local result_file="$2"
    local setting_label seq_len pred_len dataset baseline output_dir log_file exit_code status epoch_value
    local key launch_gpu

    : >"$result_file"
    while IFS=$'\t' read -r setting_label seq_len pred_len dataset baseline; do
        output_dir="$OUTPUT_ROOT_CANONICAL/$setting_label/$dataset/$baseline"
        log_file="$output_dir/run.log"
        mkdir -p "$output_dir"
        launch_gpu="cpu"

        if [[ "$RESUME" == "1" ]] \
            && can_resume "$setting_label" "$seq_len" "$pred_len" "$dataset" "$baseline" "$output_dir"; then
            key="$(resume_key "$setting_label" "$dataset" "$baseline")"
            launch_gpu="${RESUME_PASS_LAUNCH_GPUS[$key]}"
            echo "[跳过 CPU] $setting_label / $dataset / $baseline"
            exit_code=0
            status="PASS"
        else
            echo "[运行 CPU] $setting_label / $dataset / $baseline"
            epoch_value=0
            CUDA_VISIBLE_DEVICES="" \
                "$PYTHON" "$ROOT_DIR/run_time_series.py" \
                --dataset "$dataset" \
                --model "$baseline" \
                --seq_len "$seq_len" \
                --pred_len "$pred_len" \
                --epochs "$epoch_value" \
                "${RUN_MODE_ARGS[@]}" \
                --device cpu \
                --output_dir "$output_dir" \
                >"$log_file" 2>&1
            exit_code=$?

            if (( exit_code == 0 )); then
                status="PASS"
            else
                status="FAIL"
                echo "[失败 CPU] $setting_label / $dataset / $baseline，日志: $log_file" >&2
            fi
        fi

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$seq_len" "$pred_len" "$dataset" "$baseline" \
            "$status" "$output_dir" "$exit_code" "$launch_gpu" \
            >>"$result_file"
    done <"$queue_file"
}

# ==================== 同时启动两张 GPU 和 CPU 队列 ====================
RESULT_0="$TEMP_DIR/gpu-0-results.tsv"
RESULT_1="$TEMP_DIR/gpu-1-results.tsv"
RESULT_CPU="$TEMP_DIR/cpu-results.tsv"

run_gpu_queue "$GPU_0" "$QUEUE_0" "$RESULT_0" &
PID_0=$!
run_gpu_queue "$GPU_1" "$QUEUE_1" "$RESULT_1" &
PID_1=$!
run_cpu_queue "$CPU_QUEUE" "$RESULT_CPU" &
PID_CPU=$!

worker_failed=0
wait "$PID_0" || worker_failed=1
wait "$PID_1" || worker_failed=1
wait "$PID_CPU" || worker_failed=1

# ==================== 汇总结果 ====================
{
    printf '%s\n' "$SUMMARY_HEADER"
    cat "$RESULT_0" "$RESULT_1" "$RESULT_CPU"
} >"$SUMMARY_PATH"

result_count=$(($(wc -l <"$SUMMARY_PATH") - 1))
if (( worker_failed != 0 || result_count != EXPECTED_TASKS )) \
    || grep -q $'\tFAIL\t' "$SUMMARY_PATH"; then
    echo "有实验失败，请查看: $SUMMARY_PATH" >&2
    exit 1
fi

echo "全部 $EXPECTED_TASKS 个实验完成，汇总文件: $SUMMARY_PATH"
