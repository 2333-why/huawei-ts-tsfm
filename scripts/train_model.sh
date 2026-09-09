#!/usr/bin/env bash
set -euo pipefail

# A small Time-Series-Library-style entry point for the default TimesFM adapter.
# The runner owns data loading, lifecycle accounting, and artifact validation;
# this script only enumerates the example dataset/shape matrix.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Use the interpreter from the environment that is already active.  PYTHON can
# still be overridden explicitly when a different executable is required.
PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

model_name="${MODEL_NAME:-TimesFM}"
seq_len="${SEQ_LEN:-48}"
debug_mode="${DEBUG_MODE:-0}"
results_root="${RESULTS_ROOT:-$ROOT_DIR/results_foundation_models_train}"

if [[ "$debug_mode" != "0" && "$debug_mode" != "1" ]]; then
    echo "DEBUG_MODE must be 0 or 1" >&2
    exit 2
fi
if [[ "$model_name" != "TimesFM" ]]; then
    echo "MODEL_NAME must be TimesFM for this adapter example" >&2
    exit 2
fi
if ! [[ "$seq_len" =~ ^[1-9][0-9]*$ ]]; then
    echo "SEQ_LEN must be a positive integer" >&2
    exit 2
fi
if [[ "$debug_mode" == "1" ]]; then
    debug_value="True"
else
    debug_value="False"
fi

cd "$ROOT_DIR"

run_adapter() {
    local dataset="$1"
    local shape_seq_len="$2"
    local pred_len="$3"
    local setting="$4"
    local output_dir="$5"
    local experiment_model_id="${model_name}_${dataset}_${setting}"
    "$PYTHON" -u run.py \
        --data "$dataset" \
        --model "$model_name" \
        --model_id "$experiment_model_id" \
        --seq_len "$shape_seq_len" \
        --pred_len "$pred_len" \
        --train_epochs 1 \
        --debug "$debug_value" \
        --mode adapter \
        --task_name long_term_forecast \
        --is_training 1 \
        --output_dir "$output_dir"
}

if [[ "$debug_mode" == "1" ]]; then
    # One valid, bounded smoke run is useful for checking the installation.
    run_adapter "skippd_luoyang" "$seq_len" 1 "debug" \
        "$results_root/debug/${model_name}/adapter"
    exit 0
fi

datasets=("skippd_luoyang" "pvod_station00_ylj")
shape_names=("seq48_pred1" "seq96_h4")
shape_seq_lens=(48 96)
for dataset in "${datasets[@]}"; do
    for shape_index in "${!shape_names[@]}"; do
        shape_name="${shape_names[$shape_index]}"
        current_seq_len="${shape_seq_lens[$shape_index]}"
        if [[ "$shape_name" == "seq48_pred1" ]]; then
            pred_lens=(1)
        elif [[ "$dataset" == "skippd_luoyang" ]]; then
            pred_lens=(48)
        else
            pred_lens=(16)
        fi
        for pred_len in "${pred_lens[@]}"; do
            output_dir="$results_root/$shape_name/$dataset/$model_name/adapter"
            run_adapter "$dataset" "$current_seq_len" "$pred_len" "$shape_name" "$output_dir"
        done
    done
done
