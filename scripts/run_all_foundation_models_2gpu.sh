#!/usr/bin/env bash
set -euo pipefail

# Run the fixed foundation-model matrix with one serial queue per GPU.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/data/private/penv/time/bin/python}"
CONFIG_PYTHON="${CONFIG_PYTHON:-$PYTHON}"
SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-0}"
GPUS="${GPUS:-0 1}"
SKIPPD_PARQUET="${SKIPPD_PARQUET:-}"
PVOD_PARQUET="${PVOD_PARQUET:-}"

if [[ "$SMOKE" != "0" && "$SMOKE" != "1" ]]; then
    echo "SMOKE must be 0 or 1" >&2
    exit 2
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
    echo "RESUME must be 0 or 1" >&2
    exit 2
fi

python_is_usable() {
    local interpreter="$1"
    if [[ "$interpreter" == */* ]]; then
        [[ -x "$interpreter" ]]
    else
        command -v "$interpreter" >/dev/null 2>&1
    fi
}

if ! python_is_usable "$PYTHON"; then
    echo "PYTHON is not executable: $PYTHON" >&2
    exit 2
fi
if ! python_is_usable "$CONFIG_PYTHON"; then
    echo "CONFIG_PYTHON is not executable: $CONFIG_PYTHON" >&2
    exit 2
fi
if [[ ! -f "$ROOT_DIR/run_foundation_model.py" ]]; then
    echo "run_foundation_model.py is missing" >&2
    exit 2
fi

if [[ "$GPUS" == *$'\n'* || "$GPUS" == *$'\t'* ]]; then
    echo "GPUS must contain exactly two canonical distinct physical GPU ordinals (for example: GPUS='0 1')" >&2
    exit 2
fi
read -r -a GPU_IDS <<<"$GPUS"
if (( ${#GPU_IDS[@]} != 2 )); then
    echo "GPUS must contain exactly two canonical distinct physical GPU ordinals (for example: GPUS='0 1')" >&2
    exit 2
fi
for gpu_id in "${GPU_IDS[@]}"; do
    if ! [[ "$gpu_id" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "GPUS must contain exactly two canonical distinct physical GPU ordinals (for example: GPUS='0 1')" >&2
        exit 2
    fi
done
if [[ "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
    echo "GPUS must contain exactly two canonical distinct physical GPU ordinals (for example: GPUS='0 1')" >&2
    exit 2
fi

if [[ -n "${OUTPUT_ROOT:-}" && ( "$OUTPUT_ROOT" == *$'\n'* || "$OUTPUT_ROOT" == *$'\t'* ) ]]; then
    echo "OUTPUT_ROOT must not contain tabs or newlines" >&2
    exit 2
fi
if [[ "$SMOKE" == "1" ]]; then
    DEFAULT_OUTPUT_ROOT="$ROOT_DIR/results_foundation_models_smoke"
    DEFAULT_SUMMARY_NAME="smoke_summary.tsv"
else
    DEFAULT_OUTPUT_ROOT="$ROOT_DIR/results_foundation_models"
    DEFAULT_SUMMARY_NAME="run_summary.tsv"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
SUMMARY_PATH="${SUMMARY_PATH:-$OUTPUT_ROOT/$DEFAULT_SUMMARY_NAME}"
SUMMARY_HEADER=$'seq_len\tpred_len\tdataset\tmodel\tmode\tstatus\toutput_dir\texit_code\tlaunch_gpu\tdata_fingerprint'
COMPLETION_HEADER=$'schema_version\tdataset\tmodel\tseq_len\tpred_len\toutput_dir\trun_mode\tepochs\tmax_train_steps\tmax_eval_steps\tmax_test_steps\ttrain_steps\tval_steps\ttest_steps\tbest_sha256\tpredictions_sha256\tmetrics_sha256\tmode\tmodel_id\tmodel_revision'

if ! mkdir -p -- "$OUTPUT_ROOT"; then
    echo "could not create OUTPUT_ROOT: $OUTPUT_ROOT" >&2
    exit 2
fi
OUTPUT_ROOT_CANONICAL="$(realpath -m -- "$OUTPUT_ROOT")"
SUMMARY_PATH="$(realpath -m -- "$SUMMARY_PATH")"
SUMMARY_DIR="$(dirname -- "$SUMMARY_PATH")"
if ! mkdir -p -- "$SUMMARY_DIR"; then
    echo "could not create summary directory: $SUMMARY_DIR" >&2
    exit 2
fi

TEMP_DIR=""
SUMMARY_TEMP=""
if ! TEMP_DIR="$(mktemp -d)"; then
    echo "could not create a private temporary directory" >&2
    exit 2
fi
WORKER_PIDS=()

cleanup_temp_files() {
    if [[ -n "${SUMMARY_TEMP:-}" && -e "$SUMMARY_TEMP" ]]; then
        rm -f -- "$SUMMARY_TEMP" || true
    fi
    if [[ -n "${TEMP_DIR:-}" && -d "$TEMP_DIR" ]]; then
        rm -rf -- "$TEMP_DIR" || true
    fi
}

collect_process_tree() {
    local pid="$1"
    local child
    printf '%s\n' "$pid"
    while IFS= read -r child; do
        [[ -n "$child" ]] || continue
        collect_process_tree "$child"
    done < <(ps -o pid= --ppid "$pid" 2>/dev/null | awk '{print $1}')
}

worker_is_owned() {
    local pid="$1"
    local parent_pid
    parent_pid="$(ps -o ppid= -p "$pid" 2>/dev/null | awk '{print $1}')"
    [[ "$parent_pid" == "$$" ]]
}

handle_signal() {
    local signal_code="$1"
    local worker
    local -a process_tree=()
    local -a worker_tree=()
    trap - INT TERM

    for worker in "${WORKER_PIDS[@]}"; do
        if kill -0 "$worker" 2>/dev/null && worker_is_owned "$worker"; then
            worker_tree=()
            mapfile -t worker_tree < <(collect_process_tree "$worker")
            process_tree+=("${worker_tree[@]}")
        fi
    done
    if (( ${#process_tree[@]} > 0 )); then
        kill -TERM "${process_tree[@]}" 2>/dev/null || true
        sleep 0.2 || true
        # Reuse the snapshot as well as a fresh tree. A stubborn descendant
        # may be reparented when its worker exits, so it must still be killed.
        kill -KILL "${process_tree[@]}" 2>/dev/null || true
        for worker in "${WORKER_PIDS[@]}"; do
            if kill -0 "$worker" 2>/dev/null && worker_is_owned "$worker"; then
                worker_tree=()
                mapfile -t worker_tree < <(collect_process_tree "$worker")
                if (( ${#worker_tree[@]} > 0 )); then
                    kill -KILL "${worker_tree[@]}" 2>/dev/null || true
                fi
            fi
        done
    fi
    for worker in "${WORKER_PIDS[@]}"; do
        wait "$worker" 2>/dev/null || true
    done
    exit "$signal_code"
}

trap cleanup_temp_files EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

declare -A RUNTIME_CONFIGS=()
declare -A DATA_FINGERPRINTS=()

prepare_runtime_config() {
    local dataset="$1"
    local source_config="$2"
    local override_path="$3"
    local runtime_config="$TEMP_DIR/config-$dataset.json"
    local fingerprint_path="$TEMP_DIR/fingerprint-$dataset.txt"

    if ! "$CONFIG_PYTHON" - "$source_config" "$override_path" "$runtime_config" "$fingerprint_path" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1]).expanduser().resolve()
override_value = sys.argv[2]
runtime_path = Path(sys.argv[3])
fingerprint_path = Path(sys.argv[4])

if not source_path.is_file():
    raise SystemExit("dataset config is not a regular file")
if source_path.suffix.lower() == ".json":
    config = json.loads(source_path.read_text(encoding="utf-8"))
else:
    import yaml
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
    raise SystemExit("dataset config must contain paths")

if override_value:
    parquet_path = Path(override_value).expanduser().resolve()
else:
    configured = config["paths"].get("parquet_file")
    if not isinstance(configured, str) or not configured:
        raise SystemExit("dataset config must name paths.parquet_file")
    parquet_path = Path(configured).expanduser()
    if not parquet_path.is_absolute():
        parquet_path = source_path.parent / parquet_path
    parquet_path = parquet_path.resolve()
if not parquet_path.is_file():
    raise SystemExit(f"parquet file is not a regular file: {parquet_path}")

config["paths"]["parquet_file"] = str(parquet_path)
effective_bytes = (
    json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    + "\n"
).encode("utf-8")
runtime_path.write_bytes(effective_bytes)
parquet_digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
fingerprint = hashlib.sha256(
    b"foundation-data-v1\0"
    + effective_bytes
    + b"\0"
    + str(parquet_path).encode("utf-8")
    + b"\0"
    + parquet_digest.encode("ascii")
).hexdigest()
fingerprint_path.write_text("foundation-data-v1:" + fingerprint + "\n", encoding="utf-8")
PY
    then
        echo "could not normalize $dataset dataset config or parquet path" >&2
        return 1
    fi
    RUNTIME_CONFIGS["$dataset"]="$runtime_config"
    DATA_FINGERPRINTS["$dataset"]="$(<"$fingerprint_path")"
    [[ "${DATA_FINGERPRINTS[$dataset]}" =~ ^foundation-data-v1:[0-9a-f]{64}$ ]]
}

if ! prepare_runtime_config \
    "skippd_luoyang" "$ROOT_DIR/configs/datasets/skippd_luoyang.json" "$SKIPPD_PARQUET"; then
    exit 2
fi
if ! prepare_runtime_config \
    "pvod_station00_ylj" "$ROOT_DIR/configs/datasets/pvod_station00_ylj.yaml" "$PVOD_PARQUET"; then
    exit 2
fi

REGISTRY_MODELS="$TEMP_DIR/registry-models.txt"
REGISTRY_MODES="$TEMP_DIR/registry-modes.txt"
if ! "$CONFIG_PYTHON" - "$ROOT_DIR" "$REGISTRY_MODELS" "$REGISTRY_MODES" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
models_path = Path(sys.argv[2])
modes_path = Path(sys.argv[3])
sys.path.insert(0, str(root))
from models.registry import MODEL_NAMES, RUN_MODES

models_path.write_text("".join(name + "\n" for name in MODEL_NAMES), encoding="utf-8")
modes_path.write_text("".join(mode + "\n" for mode in RUN_MODES), encoding="utf-8")
PY
then
    echo "could not query the lazy foundation registry" >&2
    exit 2
fi

mapfile -t EXPECTED_MODELS <"$REGISTRY_MODELS"
mapfile -t EXPECTED_MODES <"$REGISTRY_MODES"
if (( ${#EXPECTED_MODELS[@]} != 2 || ${#EXPECTED_MODES[@]} != 4 )); then
    echo "foundation registry must expose exactly two models and four modes" >&2
    exit 2
fi

MODEL_FILE="$TEMP_DIR/models.txt"
MODE_FILE="$TEMP_DIR/modes.txt"
if ! CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
    "$PYTHON" "$ROOT_DIR/run_foundation_model.py" --list-models >"$MODEL_FILE"; then
    echo "could not read the foundation model list" >&2
    exit 2
fi
if ! CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
    "$PYTHON" "$ROOT_DIR/run_foundation_model.py" --list-modes >"$MODE_FILE"; then
    echo "could not read the foundation mode list" >&2
    exit 2
fi
mapfile -t MODELS <"$MODEL_FILE"
mapfile -t MODES <"$MODE_FILE"
if (( ${#MODELS[@]} != ${#EXPECTED_MODELS[@]} || ${#MODES[@]} != ${#EXPECTED_MODES[@]} )); then
    echo "runner catalog does not match the lazy registry" >&2
    exit 2
fi
for index in "${!EXPECTED_MODELS[@]}"; do
    if [[ "${MODELS[$index]}" != "${EXPECTED_MODELS[$index]}" ]]; then
        echo "runner model catalog does not match the lazy registry" >&2
        exit 2
    fi
done
for index in "${!EXPECTED_MODES[@]}"; do
    if [[ "${MODES[$index]}" != "${EXPECTED_MODES[$index]}" ]]; then
        echo "runner mode catalog does not match the lazy registry" >&2
        exit 2
    fi
done

TASK_FILE="$TEMP_DIR/tasks.tsv"
if ! "$CONFIG_PYTHON" - "$ROOT_DIR" >"$TASK_FILE" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from models.registry import get_model_spec
from models.tasks import iter_experiment_tasks

for task in iter_experiment_tasks():
    spec = get_model_spec(task.model)
    values = (
        task.setting,
        task.dataset,
        task.model,
        task.mode,
        str(task.seq_len),
        str(task.pred_len),
        spec.model_id,
        spec.revision,
    )
    print("\t".join(values))
PY
then
    echo "could not query the foundation task matrix" >&2
    exit 2
fi

if ! "$CONFIG_PYTHON" - "$ROOT_DIR" "$TASK_FILE" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
task_path = Path(sys.argv[2])
sys.path.insert(0, str(root))
from models.registry import MODEL_NAMES, RUN_MODES, get_model_spec
from models.tasks import iter_experiment_tasks

if tuple(MODEL_NAMES) != ("Sundial", "TimeMoE"):
    raise SystemExit("unexpected foundation model catalog")
if tuple(RUN_MODES) != ("zero_shot", "adapter", "full", "last_layer"):
    raise SystemExit("unexpected foundation mode catalog")
actual = []
for raw in task_path.read_text(encoding="utf-8").splitlines():
    fields = raw.split("\t")
    if len(fields) != 8 or any(not field for field in fields):
        raise SystemExit("malformed foundation task row")
    actual.append(tuple(fields))
expected = []
for task in iter_experiment_tasks():
    spec = get_model_spec(task.model)
    expected.append((
        task.setting,
        task.dataset,
        task.model,
        task.mode,
        str(task.seq_len),
        str(task.pred_len),
        spec.model_id,
        spec.revision,
    ))
if len(expected) != 32 or actual != expected or len(set(actual)) != 32:
    raise SystemExit("foundation task matrix is not the exact unique 32-row catalog")
PY
then
    echo "foundation task matrix is not the exact supported catalog" >&2
    exit 2
fi

component_is_safe() {
    local component="$1"
    [[ -n "$component" && "$component" != "." && "$component" != ".." \
        && "$component" != */* && "$component" != *$'\t'* && "$component" != *$'\n'* ]]
}

path_is_strictly_below_root() {
    local candidate="$1"
    if [[ "$OUTPUT_ROOT_CANONICAL" == "/" ]]; then
        [[ "$candidate" == /* && "$candidate" != "/" ]]
    else
        [[ "$candidate" == "$OUTPUT_ROOT_CANONICAL/"* \
            && "$candidate" != "$OUTPUT_ROOT_CANONICAL" ]]
    fi
}

GPU_QUEUES=()
GPU_RESULTS=()
declare -A TASK_OUTPUTS=()
declare -A TASK_CANONICAL_OUTPUTS=()
declare -A TASK_MODEL_IDS=()
declare -A TASK_REVISIONS=()
declare -A TASK_ORDINALS=()

for index in 0 1; do
    GPU_QUEUES[$index]="$TEMP_DIR/gpu-$index-tasks.tsv"
    GPU_RESULTS[$index]="$TEMP_DIR/gpu-$index-results.tsv"
    : >"${GPU_QUEUES[$index]}"
done

mapfile -t TASK_ROWS <"$TASK_FILE"
for ordinal in "${!TASK_ROWS[@]}"; do
    row="${TASK_ROWS[$ordinal]}"
    IFS=$'\t' read -r setting dataset model mode seq_len pred_len model_id revision extra <<<"$row"
    if [[ -n "${extra:-}" ]] || ! component_is_safe "$setting" \
        || ! component_is_safe "$dataset" || ! component_is_safe "$model" \
        || ! component_is_safe "$mode" \
        || ! [[ "$seq_len" =~ ^[1-9][0-9]*$ ]] \
        || ! [[ "$pred_len" =~ ^[1-9][0-9]*$ ]] \
        || [[ "$model_id" =~ [[:space:]] || "$revision" =~ [[:space:]] ]]; then
        echo "unsafe or malformed foundation task component" >&2
        exit 2
    fi
    output_dir="$(realpath -m -- "$OUTPUT_ROOT_CANONICAL/$setting/$dataset/$model/$mode")"
    if ! path_is_strictly_below_root "$output_dir"; then
        echo "foundation task output path escapes OUTPUT_ROOT: $output_dir" >&2
        exit 2
    fi
    key="$seq_len|$pred_len|$dataset|$model|$mode"
    if [[ -n "${TASK_CANONICAL_OUTPUTS[$output_dir]+present}" ]]; then
        echo "duplicate canonical foundation task output path: $output_dir" >&2
        exit 2
    fi
    if [[ -n "${TASK_OUTPUTS[$key]+present}" ]]; then
        echo "duplicate foundation task identity: $key" >&2
        exit 2
    fi
    TASK_OUTPUTS["$key"]="$output_dir"
    TASK_CANONICAL_OUTPUTS["$output_dir"]="$key"
    TASK_MODEL_IDS["$key"]="$model_id"
    TASK_REVISIONS["$key"]="$revision"
    TASK_ORDINALS["$key"]="$ordinal"
    queue_index=$((ordinal % 2))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ordinal" "$setting" "$dataset" "$model" "$mode" "$seq_len" \
        "$pred_len" "$model_id" "$revision" >>"${GPU_QUEUES[$queue_index]}"
done

if (( ${#TASK_ROWS[@]} != 32 || ${#TASK_OUTPUTS[@]} != 32 )); then
    echo "exactly 32 unique foundation tasks are required" >&2
    exit 2
fi

declare -A RESUME_DIRS=()
declare -A RESUME_GPUS=()
declare -A RESUME_SEEN=()
declare -A RESUME_INVALID=()

task_key() {
    printf '%s|%s|%s|%s|%s' "$1" "$2" "$3" "$4" "$5"
}

validate_artifacts() {
    local output_dir="$1"
    local dataset="$2"
    local model="$3"
    local mode="$4"
    local seq_len="$5"
    local pred_len="$6"
    local model_id="$7"
    local revision="$8"
    local fingerprint="$9"
    "$CONFIG_PYTHON" - "$output_dir" "$dataset" "$model" "$mode" "$seq_len" "$pred_len" \
        "$model_id" "$revision" "$fingerprint" "$SMOKE" "$COMPLETION_HEADER" <<'PY'
import hashlib
import json
import re
import stat
import sys
from pathlib import Path

(output_value, expected_dataset, expected_model, expected_mode, expected_seq,
 expected_pred, expected_model_id, expected_revision, expected_fingerprint,
 smoke_value, completion_header) = sys.argv[1:]
output = Path(output_value)
expected_seq = int(expected_seq)
expected_pred = int(expected_pred)
expected_smoke = smoke_value == "1"
expected_run_mode = "smoke" if expected_smoke else "full"
expected_limits = {"train": 1, "val": 1, "test": 1} if expected_smoke else {"train": 0, "val": 0, "test": 0}

def fail(message):
    raise SystemExit(message)

def regular_nonempty(path):
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return not path.is_symlink() and stat.S_ISREG(mode) and path.stat().st_size > 0

if not output.is_dir() or output.is_symlink():
    fail("output directory is not a real directory")
for name in ("best.pt", "predictions.csv", "metrics.json"):
    if not regular_nonempty(output / name):
        fail("missing or non-regular artifact: " + name)

completion = output / "completion.tsv"
try:
    completion_bytes = completion.read_bytes()
except OSError:
    fail("missing completion manifest")
if not completion_bytes.endswith(b"\n"):
    fail("completion manifest is not LF terminated")
parts = completion_bytes.split(b"\n")
if len(parts) != 3 or parts[-1] != b"":
    fail("completion manifest must contain exactly two rows")
try:
    header = parts[0].decode("utf-8")
    row_text = parts[1].decode("utf-8")
except UnicodeDecodeError:
    fail("completion manifest is not UTF-8")
if "\r" in header or "\r" in row_text:
    fail("completion manifest contains CR characters")
if header != completion_header:
    fail("completion header mismatch")
header_fields = header.split("\t")
values = row_text.split("\t")
if len(values) != len(header_fields) or len(values) != 20:
    fail("completion row width mismatch")
manifest = dict(zip(header_fields, values))
expected_epochs = "0" if expected_mode == "zero_shot" else "1"
expected_values = {
    "schema_version": "2",
    "dataset": expected_dataset,
    "model": expected_model,
    "seq_len": str(expected_seq),
    "pred_len": str(expected_pred),
    "output_dir": str(output.resolve()),
    "run_mode": expected_run_mode,
    "epochs": expected_epochs,
    "max_train_steps": str(expected_limits["train"]),
    "max_eval_steps": str(expected_limits["val"]),
    "max_test_steps": str(expected_limits["test"]),
    "mode": expected_mode,
    "model_id": expected_model_id,
    "model_revision": expected_revision,
}
for key, value in expected_values.items():
    if manifest.get(key) != value:
        fail("completion identity mismatch for " + key)

def non_negative(value, name):
    if not re.fullmatch(r"[0-9]+", value):
        fail(name + " is not a non-negative integer")
    return int(value)

train_steps = non_negative(manifest["train_steps"], "train_steps")
val_steps = non_negative(manifest["val_steps"], "val_steps")
test_steps = non_negative(manifest["test_steps"], "test_steps")
if expected_mode == "zero_shot":
    if train_steps != 0 or val_steps != 0:
        fail("zero-shot training lifecycle mismatch")
    if (expected_smoke and test_steps != 1) or (not expected_smoke and test_steps <= 0):
        fail("zero-shot test lifecycle mismatch")
elif expected_smoke:
    if (train_steps, val_steps, test_steps) != (1, 1, 1):
        fail("smoke training lifecycle mismatch")
elif min(train_steps, val_steps, test_steps) <= 0:
    fail("full training lifecycle mismatch")

for name, artifact in (("best_sha256", "best.pt"), ("predictions_sha256", "predictions.csv"), ("metrics_sha256", "metrics.json")):
    digest = manifest.get(name, "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        fail(name + " is not a lowercase SHA-256")
    observed = hashlib.sha256((output / artifact).read_bytes()).hexdigest()
    if observed != digest:
        fail(name + " does not match artifact")

predictions = output / "predictions.csv"
prediction_lines = predictions.read_bytes().split(b"\n")
if not prediction_lines or prediction_lines[0] != b"issue_time,target_time,horizon_minutes,y_true,y_pred":
    fail("predictions header mismatch")
if len(prediction_lines) < 3 or prediction_lines[-1] != b"":
    fail("predictions must contain a terminated data row")

try:
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
except (OSError, ValueError, UnicodeError):
    fail("metrics are not parseable JSON")
if not isinstance(metrics, dict):
    fail("metrics must be an object")
metric_identity = {
    "dataset": expected_dataset,
    "model": expected_model,
    "model_id": expected_model_id,
    "model_revision": expected_revision,
    "effective_revision": expected_revision,
    "revision": expected_revision,
    "mode": expected_mode,
    "seq_len": expected_seq,
    "pred_len": expected_pred,
    "epochs": int(expected_epochs),
    "run_mode": expected_run_mode,
    "limits": expected_limits,
}
for key, value in metric_identity.items():
    if metrics.get(key) != value:
        fail("metrics identity mismatch for " + key)
phase = {"train": train_steps, "val": val_steps, "test": test_steps}
for metric_key, phase_key in (("train_steps", "train"), ("val_steps", "val"), ("test_steps", "test")):
    if metrics.get(metric_key) != phase[phase_key]:
        fail("metrics top-level phase step mismatch")
if metrics.get("phase_steps") != phase:
    fail("metrics nested phase steps mismatch")
for key in ("trainability", "trainability_report"):
    if not isinstance(metrics.get(key), dict) or metrics[key].get("mode") != expected_mode:
        fail("metrics trainability mode mismatch")

fingerprint_path = output / "data_fingerprint.txt"
try:
    fingerprint_bytes = fingerprint_path.read_bytes()
except OSError:
    fail("missing data fingerprint")
if fingerprint_bytes != (expected_fingerprint + "\n").encode("utf-8"):
    fail("data fingerprint mismatch")
PY
}

load_resume_summary() {
    [[ "$RESUME" == "1" && -s "$SUMMARY_PATH" ]] || return 0
    local header
    if ! IFS= read -r header <"$SUMMARY_PATH"; then
        return 0
    fi
    [[ "$header" == "$SUMMARY_HEADER" ]] || return 0

    local seq_len pred_len dataset model mode status output_dir exit_code launch_gpu fingerprint key
    local resume_line field_count
    local -a resume_fields
    while IFS= read -r resume_line; do
        [[ -n "$resume_line" ]] || continue
        field_count="$(awk -F $'\t' '{print NF; exit}' <<<"$resume_line")"
        [[ "$field_count" == "10" ]] || continue
        resume_line="${resume_line//$'\t'/$'\x1f'}"
        IFS=$'\x1f' read -r -a resume_fields <<<"$resume_line"
        (( ${#resume_fields[@]} == 10 )) || continue
        seq_len="${resume_fields[0]}"
        pred_len="${resume_fields[1]}"
        dataset="${resume_fields[2]}"
        model="${resume_fields[3]}"
        mode="${resume_fields[4]}"
        status="${resume_fields[5]}"
        output_dir="${resume_fields[6]}"
        exit_code="${resume_fields[7]}"
        launch_gpu="${resume_fields[8]}"
        fingerprint="${resume_fields[9]}"
        key="$(task_key "$seq_len" "$pred_len" "$dataset" "$model" "$mode")"
        [[ -n "${TASK_OUTPUTS[$key]+present}" ]] || continue
        if [[ -n "${RESUME_SEEN[$key]:-}" ]]; then
            RESUME_INVALID["$key"]=1
            unset 'RESUME_DIRS[$key]'
            unset 'RESUME_GPUS[$key]'
            continue
        fi
        RESUME_SEEN["$key"]=1
        if [[ -n "${extra:-}" ]] \
            || [[ "$status" != "PASS" || "$exit_code" != "0" ]] \
            || [[ -z "$launch_gpu" || "$launch_gpu" =~ [[:space:]] ]] \
            || [[ "$fingerprint" != "${DATA_FINGERPRINTS[$dataset]:-}" ]] \
            || [[ "$output_dir" != "${TASK_OUTPUTS[$key]}" ]]; then
            RESUME_INVALID["$key"]=1
            continue
        fi
        RESUME_DIRS["$key"]="$output_dir"
        RESUME_GPUS["$key"]="$launch_gpu"
    done < <(tail -n +2 "$SUMMARY_PATH")
}

load_resume_summary

write_data_fingerprint() {
    local output_dir="$1"
    local dataset="$2"
    local fingerprint_file="$output_dir/data_fingerprint.txt"
    local temporary_file
    if ! temporary_file="$(mktemp "$output_dir/.data_fingerprint.txt.tmp.XXXXXX")"; then
        return 1
    fi
    if printf '%s\n' "${DATA_FINGERPRINTS[$dataset]}" >"$temporary_file" \
        && mv -f -- "$temporary_file" "$fingerprint_file"; then
        return 0
    fi
    rm -f -- "$temporary_file" "$fingerprint_file" || true
    return 1
}

record_result() {
    local result_file="$1"
    local ordinal="$2"
    local seq_len="$3"
    local pred_len="$4"
    local dataset="$5"
    local model="$6"
    local mode="$7"
    local status="$8"
    local output_dir="$9"
    local exit_code="${10}"
    local launch_gpu="${11}"
    local fingerprint="${12}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ordinal" "$seq_len" "$pred_len" "$dataset" "$model" "$mode" \
        "$status" "$output_dir" "$exit_code" "$launch_gpu" "$fingerprint" >>"$result_file"
}

can_resume() {
    local output_dir="$1"
    local dataset="$2"
    local model="$3"
    local mode="$4"
    local seq_len="$5"
    local pred_len="$6"
    local model_id="$7"
    local revision="$8"
    local fingerprint="$9"
    local key
    key="$(task_key "$seq_len" "$pred_len" "$dataset" "$model" "$mode")"
    [[ -z "${RESUME_INVALID[$key]:-}" ]] \
        && [[ "${RESUME_DIRS[$key]:-}" == "$output_dir" ]] \
        && validate_artifacts "$output_dir" "$dataset" "$model" "$mode" "$seq_len" \
            "$pred_len" "$model_id" "$revision" "$fingerprint"
}

run_gpu_queue() {
    local gpu="$1"
    local queue_file="$2"
    local result_file="$3"
    local ordinal setting dataset model mode seq_len pred_len model_id revision extra
    local output_dir log_file status exit_code launch_gpu key
    : >"$result_file"
    while IFS=$'\t' read -r ordinal setting dataset model mode seq_len pred_len model_id revision extra; do
        [[ -n "$ordinal" ]] || continue
        output_dir="${TASK_OUTPUTS[$(task_key "$seq_len" "$pred_len" "$dataset" "$model" "$mode")]}"
        launch_gpu="$gpu"
        exit_code=0
        status="FAIL"

        if ! mkdir -p -- "$output_dir"; then
            record_result "$result_file" "$ordinal" "$seq_len" "$pred_len" "$dataset" \
                "$model" "$mode" "$status" "$output_dir" 1 "$launch_gpu" \
                "${DATA_FINGERPRINTS[$dataset]}"
            continue
        fi

        if [[ "$RESUME" == "1" ]] && can_resume "$output_dir" "$dataset" "$model" "$mode" \
            "$seq_len" "$pred_len" "$model_id" "$revision" "${DATA_FINGERPRINTS[$dataset]}"; then
            key="$(task_key "$seq_len" "$pred_len" "$dataset" "$model" "$mode")"
            launch_gpu="${RESUME_GPUS[$key]}"
            status="PASS"
            exit_code=0
        else
            rm -f -- "$output_dir/completion.tsv" "$output_dir/data_fingerprint.txt" || true
            run_args=(
                --dataset "$dataset"
                --config "${RUNTIME_CONFIGS[$dataset]}"
                --model "$model"
                --mode "$mode"
                --seq_len "$seq_len"
                --pred_len "$pred_len"
                --device cuda:0
                --output_dir "$output_dir"
            )
            if [[ "$SMOKE" == "1" ]]; then
                run_args+=(--smoke)
            fi
            log_file="$output_dir/run.log"
            if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT_DIR/run_foundation_model.py" \
                "${run_args[@]}" >"$log_file" 2>&1; then
                exit_code=0
            else
                exit_code=$?
            fi
            if (( exit_code == 0 )) \
                && write_data_fingerprint "$output_dir" "$dataset" \
                && validate_artifacts "$output_dir" "$dataset" "$model" "$mode" \
                    "$seq_len" "$pred_len" "$model_id" "$revision" "${DATA_FINGERPRINTS[$dataset]}"; then
                status="PASS"
            else
                if (( exit_code == 0 )); then
                    exit_code=1
                fi
                rm -f -- "$output_dir/completion.tsv" "$output_dir/data_fingerprint.txt" || true
                status="FAIL"
            fi
        fi
        record_result "$result_file" "$ordinal" "$seq_len" "$pred_len" "$dataset" \
            "$model" "$mode" "$status" "$output_dir" "$exit_code" "$launch_gpu" \
            "${DATA_FINGERPRINTS[$dataset]}"
    done <"$queue_file"
}

for index in 0 1; do
    run_gpu_queue "${GPU_IDS[$index]}" "${GPU_QUEUES[$index]}" "${GPU_RESULTS[$index]}" &
    WORKER_PIDS+=("$!")
done

worker_failed=0
for worker in "${WORKER_PIDS[@]}"; do
    if wait "$worker"; then
        :
    else
        worker_failed=1
    fi
done

ALL_RESULTS="$TEMP_DIR/all-results.tsv"
SORTED_RESULTS="$TEMP_DIR/sorted-results.tsv"
: >"$ALL_RESULTS"
for result_file in "${GPU_RESULTS[@]}"; do
    if [[ -f "$result_file" ]]; then
        cat "$result_file" >>"$ALL_RESULTS"
    else
        worker_failed=1
    fi
done
if ! sort -t $'\t' -n -k1,1 "$ALL_RESULTS" >"$SORTED_RESULTS"; then
    echo "could not collect foundation task results" >&2
    exit 1
fi

if ! SUMMARY_TEMP="$(mktemp "$SUMMARY_DIR/.$(basename -- "$SUMMARY_PATH").tmp.XXXXXX")"; then
    echo "could not create adjacent summary temporary file" >&2
    exit 1
fi
{
    printf '%s\n' "$SUMMARY_HEADER"
    while IFS=$'\t' read -r ordinal seq_len pred_len dataset model mode status output_dir exit_code launch_gpu fingerprint extra; do
        [[ -n "$ordinal" ]] || continue
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$seq_len" "$pred_len" "$dataset" "$model" "$mode" "$status" \
            "$output_dir" "$exit_code" "$launch_gpu" "$fingerprint"
    done <"$SORTED_RESULTS"
} >"$SUMMARY_TEMP"

if ! "$CONFIG_PYTHON" - "$SUMMARY_TEMP" "$TASK_FILE" "$OUTPUT_ROOT_CANONICAL" "$SUMMARY_HEADER" <<'PY'
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
task_path = Path(sys.argv[2])
root = Path(sys.argv[3]).resolve()
header_expected = sys.argv[4]

tasks = []
for raw in task_path.read_text(encoding="utf-8").splitlines():
    setting, dataset, model, mode, seq, pred, _model_id, _revision = raw.split("\t")
    tasks.append((seq, pred, dataset, model, mode, str((root / setting / dataset / model / mode).resolve())))
if len(tasks) != 32:
    raise SystemExit("task count changed before summary publication")
expected = {row[:5]: row[5] for row in tasks}
data = summary_path.read_bytes()
if not data.endswith(b"\n"):
    raise SystemExit("summary is not LF terminated")
try:
    lines = data.decode("utf-8").splitlines()
except UnicodeDecodeError:
    raise SystemExit("summary is not UTF-8")
if not lines or lines[0] != header_expected:
    raise SystemExit("summary header mismatch")
if len(lines) != 33:
    raise SystemExit("summary must contain exactly 32 result rows")
seen = set()
for line in lines[1:]:
    fields = line.split("\t")
    if len(fields) != 10:
        raise SystemExit("summary row width mismatch")
    key = tuple(fields[:5])
    if key in seen or key not in expected:
        raise SystemExit("summary task identity is not unique/current")
    seen.add(key)
    if fields[6] != expected[key]:
        raise SystemExit("summary output path mismatch")
    if fields[5] not in {"PASS", "FAIL"}:
        raise SystemExit("summary status mismatch")
    if not re.fullmatch(r"-?[0-9]+", fields[7]):
        raise SystemExit("summary exit code mismatch")
    code = int(fields[7])
    if (fields[5] == "PASS" and code != 0) or (fields[5] == "FAIL" and code == 0):
        raise SystemExit("summary status/exit code mismatch")
    if not fields[8] or re.search(r"\s", fields[8]):
        raise SystemExit("summary launch GPU mismatch")
    if not re.fullmatch(r"foundation-data-v1:[0-9a-f]{64}", fields[9]):
        raise SystemExit("summary fingerprint mismatch")
if seen != set(expected):
    raise SystemExit("summary identity set mismatch")
PY
then
    rm -f -- "$SUMMARY_TEMP" || true
    SUMMARY_TEMP=""
    echo "foundation result collection was incomplete or invalid" >&2
    exit 1
fi

if ! mv -f -- "$SUMMARY_TEMP" "$SUMMARY_PATH"; then
    rm -f -- "$SUMMARY_TEMP" || true
    SUMMARY_TEMP=""
    echo "could not atomically publish foundation summary" >&2
    exit 1
fi
SUMMARY_TEMP=""

if (( worker_failed != 0 )) || grep -q $'\tFAIL\t' "$SUMMARY_PATH"; then
    echo "one or more foundation-model tasks failed; see $SUMMARY_PATH" >&2
    exit 1
fi
echo "all 32 foundation-model tasks completed; summary: $SUMMARY_PATH"
