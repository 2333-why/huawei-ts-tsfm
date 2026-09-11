# TSFM-only power forecasting

[English](README.md) | [简体中文](README.zh-CN.md)

This repository provides power-only forecasting with five time-series foundation
models: Sundial, TimeMoE, Chronos2, TiRex, and TimesFM.  Each run consumes a
single normalized power channel with shape `[B, L, 1]` and returns
`[B, H, 1]`.  The implementations live under `models/`; optional model
packages are imported lazily so catalog and task-list commands work offline.
There is no `foundation_models/` implementation package.

## Models and capabilities

The registry is the source of truth for checkpoint identity, revisions, modes,
and last-layer selectors.

| model | checkpoint ID | pinned revision | modes | last-layer selector |
| --- | --- | --- | --- | --- |
| `Sundial` | `thuml/sundial-base-128m` | `3212e42564493f520593e5414af4367fc4b49226` | `zero_shot`, `adapter`, `full`, `last_layer` | `flow_loss` |
| `TimeMoE` | `Maple728/TimeMoE-50M` | `446753ee48ff3726d0606a81d0092d54acee995e` | `zero_shot`, `adapter`, `full`, `last_layer` | `lm_heads` |
| `Chronos2` | `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` | `zero_shot`, `adapter`, `full`, `last_layer` | `output_patch_embedding` |
| `TiRex` | `NX-AI/TiRex` | `63c740922493f5fbe60b277609ec62babfba2762` | `zero_shot` | — |
| `TimesFM` | `google/timesfm-2.5-200m-transformers` | `5a9806b9b291fad9233b5249d88263f1846304d3` | `zero_shot`, `adapter`, `full`, `last_layer` | `output_projection_point` |

Modes are explicit trainability policies:

- `zero_shot` freezes the model and runs inference without an optimizer.
- `adapter` injects LoRA modules and trains only adapter parameters.
- `full` trains all model parameters.
- `last_layer` trains only the named selector subtree.

TiRex is intentionally zero-shot-only; unsupported modes are rejected before
its optional package or checkpoint loader is constructed.

## Data and task matrix

The runner accepts only the two repository power configurations:

- `configs/datasets/skippd_luoyang.json` (`SKIPPD_PARQUET` may override its
  configured Parquet path; 5-minute sampling)
- `configs/datasets/pvod_station00_ylj.yaml` (`PVOD_PARQUET` may override its
  configured Parquet path; 15-minute sampling)

The supported dataset/window rows are fixed:

| setting | dataset | `seq_len` | `pred_len` |
| --- | --- | ---: | ---: |
| `seq48_pred1` | `skippd_luoyang` | 48 | 1 |
| `seq48_pred1` | `pvod_station00_ylj` | 48 | 1 |
| `seq96_h4` | `skippd_luoyang` | 96 | 48 |
| `seq96_h4` | `pvod_station00_ylj` | 96 | 16 |

There are four dataset/window settings.  Each has four trainable models with
four modes plus one TiRex zero-shot mode:

`4 settings × (4 trainable models × 4 modes + 1 TiRex mode) = 68 tasks`.

## Runtime assumption

Activate your existing Python environment before running any command in this
repository.  The repository does not create, activate, or modify an
environment.  All experiment scripts use `python` from the active shell by
default; set `PYTHON=/path/to/python` only when an explicit override is needed.

## Install dependencies and download checkpoints

The following command installs into the environment that is already active; it
does not create or activate a virtual environment.  It then downloads the
pinned Sundial, TimeMoE, Chronos2, TiRex, and TimesFM revisions declared in
`models/registry.py`:

```bash
bash scripts/setup_foundation_runtime.sh
```

The default cache is `checkpoints_huggingface/` under the repository, and the
experiment scripts automatically reuse it.  If dependencies are already
installed, download checkpoints only:

```bash
python scripts/download_foundation_weights.py
```

Set `HF_HOME=/path/to/huggingface-cache` consistently on setup, download, and
experiment commands to use a different shared cache.  After downloading, run
the offline readiness check with:

```bash
HF_HOME="$PWD/checkpoints_huggingface" \
python scripts/check_foundation_environment.py --offline --json
```

## Runner

Run from the repository root.  The catalog commands do not load optional model
backends:

```bash
python run.py --list-models
python run.py --list-modes
```

Single-run example (Sundial):

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --dataset skippd_luoyang \
  --model Sundial \
  --mode zero_shot \
  --seq_len 48 \
  --pred_len 1 \
  --device cuda:0 \
  --output_dir results_foundation_models/example
```

The Time-Series-Library-style aliases are also supported:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --data skippd_luoyang \
  --model TimesFM \
  --model_id timesfm_skippd_adapter_example \
  --mode adapter \
  --seq_len 48 \
  --pred_len 1 \
  --train_epochs 1 \
  --debug True \
  --task_name long_term_forecast \
  --is_training 1 \
  --device cuda:0 \
  --output_dir results_foundation_models/example-adapter
```

`--data` aliases `--dataset`; `--train_epochs` aliases `--epochs`; and
`--debug` aliases `--smoke`.  `--debug` requires an explicit `True` or
`False` value.  Supplying both spellings is allowed only when their values
agree; conflicts fail early.  `--model_id` is metadata for the experiment
label only and never replaces the registry's pinned checkpoint ID or revision.
`--task_name` and `--is_training` are optional consistency assertions:
`zero_shot` corresponds to `zero_shot_forecast` / `0`, while training modes
correspond to `long_term_forecast` / `1`.  Unknown and abbreviated options are
rejected.

## Simple training script

`scripts/train_model.sh` follows the requested shell-loop style.  Its default
model is `TimesFM` in `adapter` mode.  With `DEBUG_MODE=1` it launches exactly
one bounded adapter smoke run; otherwise it launches the four real
dataset/window rows.  Useful variables are `PYTHON`, `CUDA_VISIBLE_DEVICES`,
`MODEL_NAME` (must remain `TimesFM` for this example), `SEQ_LEN`,
`DEBUG_MODE`, and `RESULTS_ROOT`.

```bash
DEBUG_MODE=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_model.sh
```

The normal branch runs the four real dataset/window rows:

```bash
DEBUG_MODE=0 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_model.sh
```

## Two-GPU batch scripts

`scripts/smoke_all_foundation_models_2gpu.sh` is the bounded wrapper for
`scripts/run_all_foundation_models_2gpu.sh`; the latter executes every one of
the 68 supported tasks.  Both use `GPUS="0 1"` by default and create two
process-level queues.  Each queue is serial, while the two queues may overlap.
The queue launches a physical ordinal as
`CUDA_VISIBLE_DEVICES=<ordinal>` and the runner uses the process-local
`--device cuda:0`; no DataParallel is used.

Relevant variables are:

- `PYTHON` and `CONFIG_PYTHON`: interpreters for runs and catalog/config checks.
- `GPUS`: exactly two distinct physical GPU ordinals.
- `SMOKE`: `0` for full runs or `1` for one-step smoke runs.
- `RESUME`: `1` enables validated resume, otherwise `0` reruns tasks.
- `OUTPUT_ROOT`: result root; `SUMMARY_PATH`: summary TSV destination.
- `SKIPPD_PARQUET` and `PVOD_PARQUET`: optional Parquet path overrides.

Example smoke invocation:

```bash
GPUS="0 1" \
RESUME=0 \
OUTPUT_ROOT=results_foundation_models_smoke \
  bash scripts/smoke_all_foundation_models_2gpu.sh
```

For a full batch run, or to resume validated tasks, use the non-smoke script:

```bash
GPUS="0 1" \
RESUME=0 \
OUTPUT_ROOT=results_foundation_models \
  bash scripts/run_all_foundation_models_2gpu.sh

GPUS="0 1" \
RESUME=1 \
OUTPUT_ROOT=results_foundation_models \
SUMMARY_PATH=results_foundation_models/run_summary.tsv \
  bash scripts/run_all_foundation_models_2gpu.sh
```

## Eight-GPU batch script

`scripts/run_all_foundation_models_8gpu.sh` is the fixed eight-GPU batch entry
for all 68 supported tasks.  It defaults to `GPUS='0 1 2 3 4 5 6 7'` and
requires exactly eight distinct canonical physical GPU ordinals.  Tasks are
assigned by `ordinal % 8`; each GPU queue is serial, while the eight queues may
run in parallel.  Each process launches one physical ordinal as
`CUDA_VISIBLE_DEVICES=<ordinal>` and the runner uses the process-local
`--device cuda:0`; no DataParallel is used.

The entry supports `SMOKE=0` or `SMOKE=1` and `RESUME=0` or `RESUME=1`.  It
shares the same validation and artifact contract as
`scripts/run_all_foundation_models_2gpu.sh`: identity, schema, data fingerprint,
artifact hashes, checkpoint/metrics fields, model ID, and pinned revision must
validate before a task is resumed.  Relevant variables are `PYTHON`,
`CONFIG_PYTHON`, `GPUS`, `SMOKE`, `RESUME`, `OUTPUT_ROOT`, `SUMMARY_PATH`,
`SKIPPD_PARQUET`, and `PVOD_PARQUET`.

Example eight-GPU smoke invocation:

```bash
GPUS='0 1 2 3 4 5 6 7' \
SMOKE=1 \
RESUME=0 \
OUTPUT_ROOT=results_foundation_models_smoke \
  bash scripts/run_all_foundation_models_8gpu.sh
```

For a full eight-GPU batch run, or to resume validated tasks:

```bash
GPUS='0 1 2 3 4 5 6 7' \
SMOKE=0 \
RESUME=0 \
OUTPUT_ROOT=results_foundation_models \
  bash scripts/run_all_foundation_models_8gpu.sh

GPUS='0 1 2 3 4 5 6 7' \
SMOKE=0 \
RESUME=1 \
OUTPUT_ROOT=results_foundation_models \
SUMMARY_PATH=results_foundation_models/run_summary.tsv \
  bash scripts/run_all_foundation_models_8gpu.sh
```

## Artifacts and resume

Each task directory contains four final artifacts:

- `best.pt`: the best/restored model state and lifecycle metadata;
- `predictions.csv`: valid power predictions in the original power units;
- `metrics.json`: metrics, identities, trainability, and phase counts;
- `completion.tsv`: the completion manifest written last.

Batch runs additionally write `data_fingerprint.txt` per task, `run.log` per
task, and a TSV summary at `SUMMARY_PATH`.  With `RESUME=1`, a task is skipped
only after its identity, schema, data fingerprint, artifact hashes,
checkpoint/metrics fields, model ID, and pinned revision all validate.  Any
missing, changed, or tampered artifact is rerun.

The historical result-directory component `foundation_models` is retained for
resume compatibility.  It is a result path only, not a Python package or
implementation directory.
