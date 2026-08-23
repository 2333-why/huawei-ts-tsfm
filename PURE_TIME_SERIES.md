# Pure time-series forecasting

This entry point trains forecasting-capable, non-foundation
Time-Series-Library models using historical power only.  The runner supports
these dataset identifiers and physical source columns:

| Dataset identifier | Target/power column |
| --- | --- |
| `skippd_luoyang` | `final_power` |
| `pvod_station00_ylj` | `observe_power` |

The data adapters normalize the target into channel zero.  The model boundary
passes only `batch_x[..., :1]`; images, weather/NWP, ramp features, time marks,
privileged Teacher data, and future targets are not model inputs.  Encoder-
decoder models receive historical power followed by zero-valued forecast
positions.

## Model catalog

The authoritative catalog is printed by `run_time_series.py --list-models`.
It contains exactly these 32 models:

`Autoformer`, `Crossformer`, `DLinear`, `ETSformer`, `FEDformer`, `FiLM`,
`FreTS`, `Informer`, `Koopa`, `LightTS`, `MICN`, `MSGNet`, `Mamba`,
`MambaSimple`, `MultiPatchFormer`, `Nonstationary_Transformer`, `PAttn`,
`PatchTST`, `Pyraformer`, `Reformer`, `SCINet`, `SegRNN`, `TSMixer`,
`TemporalFusionTransformer`, `TiDE`, `TimeFilter`, `TimeMixer`, `TimeXer`,
`TimesNet`, `Transformer`, `WPMixer`, and `iTransformer`.

The registry also records these seven explicit exclusions:

| Excluded model | Reason |
| --- | --- |
| `Chronos` | Foundation time-series model; zero-shot foundation path. |
| `Chronos2` | Foundation time-series model; zero-shot foundation path. |
| `Sundial` | Foundation time-series model; zero-shot foundation path. |
| `TimeMoE` | Foundation time-series model; zero-shot foundation path. |
| `TiRex` | Foundation time-series model; zero-shot foundation path. |
| `TimesFM` | Foundation time-series model; zero-shot foundation path. |
| `KANAD` | Source implements anomaly detection, not supervised forecasting. |

## One model

Use the required environment and process-local CUDA device explicitly.  The
following example writes artifacts to a model-specific directory:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/data/private/penv/time/bin/python run_time_series.py \
  --dataset skippd_luoyang --model PatchTST --epochs 40 \
  --device cuda:0 --output_dir results_pure_time_series/skippd_luoyang/PatchTST
```

For a one-model run, `CUDA_VISIBLE_DEVICES` may be either physical GPU `0` or
`1`; the process-local address remains `cuda:0`.

The CLI also accepts `--config`, model tuning flags, and explicit phase limits
(`--max_train_steps`, `--max_eval_steps`, `--max_test_steps`); zero means no
limit.  `--list-models` prints the 32 names and exits without requiring a
dataset or model:

```bash
/opt/data/private/penv/time/bin/python run_time_series.py --list-models
```

## Full 64-run matrix

The full batch script discovers the model list from that CLI and runs all
32 models on both datasets.  It uses `EPOCHS` (default `40`) and does not add
smoke flags or phase step limits:

```bash
PYTHON=/opt/data/private/penv/time/bin/python \
OUTPUT_ROOT="$PWD/results_pure_time_series" EPOCHS=40 GPUS="0 1" RESUME=0 \
  scripts/run_all_pure_time_series.sh
```

By default, both scripts use physical GPUs `0 1` (override with the
space-separated `GPUS` environment variable).  They create one sequential
worker queue per selected GPU and assign the deterministic 64-row matrix
round-robin, so both cards are active concurrently while no card runs two
model processes at once.  Each model subprocess receives
`CUDA_VISIBLE_DEVICES=<physical-gpu>` and uses `--device cuda:0`.

A failed subprocess is recorded and the remaining combinations continue; the
script waits for both queues, aggregates their results, and exits non-zero
after writing the summary if any combination failed.  To resume, use
`RESUME=1`: completed rows are reused only when their summary status is `PASS`
and all three expected artifacts exist, while failed or incomplete rows are
retried.

The historical full-matrix path remains a compatibility alias and forwards
all arguments and environment variables unchanged:

```bash
scripts/run_time_series_models.sh
```

## Bounded smoke matrix

The smoke script expands exactly `32 x 2 = 64` dataset/model combinations and
passes `--smoke --device cuda:0` to every model subprocess:

```bash
PYTHON=/opt/data/private/penv/time/bin/python \
OUTPUT_ROOT="$PWD/results_pure_time_series_smoke" GPUS="0 1" RESUME=0 \
  scripts/smoke_all_pure_time_series.sh
```

`--smoke` forces the runner to one epoch container, batch size 2, and exactly
one training, validation, and test iteration.  It is intentionally bounded;
the smoke command never runs a complete unrestricted dataset epoch.  The
script uses the two GPU queues concurrently, continues after individual failures, writes
`results_pure_time_series_smoke/smoke_summary.tsv`, and exits non-zero if any
row is `FAIL`.  Use `RESUME=1` with the same `OUTPUT_ROOT` to retry only failed
or incomplete rows.

The historical smoke path is also a transparent compatibility alias:

```bash
scripts/smoke_time_series_models.sh
```

## Artifacts and optional dependency

Each dataset/model combination has its own directory, so concurrent matrix
rows cannot overwrite one another:

```text
<OUTPUT_ROOT>/<dataset>/<model>/best.pt
<OUTPUT_ROOT>/<dataset>/<model>/metrics.json
<OUTPUT_ROOT>/<dataset>/<model>/predictions.csv
<OUTPUT_ROOT>/<dataset>/<model>/run.log
<OUTPUT_ROOT>/smoke_summary.tsv       # smoke matrix
<OUTPUT_ROOT>/run_summary.tsv         # full matrix
```

`metrics.json` contains phase step counts; the expected bounded smoke values
are `train_steps=1`, `val_steps=1`, and `test_steps=1`.  `predictions.csv`
contains valid test targets in original power units.  Do not claim a complete
64-row smoke success until the summary and all per-combination artifacts have
been independently checked.

`Mamba` requires the optional `mamba_ssm` package compatible with the pinned
Python 3.8 / PyTorch 2.3.1 / CUDA 11.8 environment.  The registry keeps
`Mamba` lazy so other models can be listed without that package; a missing
dependency is reported with the model and package name.  `Mamba` must not be
silently replaced by `MambaSimple`.
