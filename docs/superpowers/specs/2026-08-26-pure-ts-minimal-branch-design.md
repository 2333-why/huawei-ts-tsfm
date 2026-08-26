# Pure Time-Series Minimal Branch Design

## Goal

Create a standalone `pure-ts` branch at `/opt/data/private/code/pure-ts` for
training and evaluating only the eight models with the best aggregate ranking
across the completed pure time-series experiments. Remove other models and the
multimodal, Teacher/KD, visualization, legacy-dataset, and unrelated experiment
code while preserving a small, testable power-only workflow.

## Branch and workspace

- Branch: `pure-ts`
- Worktree: `/opt/data/private/code/pure-ts`
- Base: `main` at `9e2cef3`
- The untracked reports, PDF, figures, and visualization files in the original
  `main` worktree are not copied, deleted, or committed.

## Retained models

The model registry contains exactly these eight names:

1. `TSMixer`
2. `Pyraformer`
3. `SegRNN`
4. `Transformer`
5. `LightTS`
6. `Crossformer`
7. `FreTS`
8. `MICN`

All other model implementation files are removed. Shared neural-network layers
are retained only when one of these eight implementations imports them.
`Crossformer` must no longer import `PatchTST`; its small `FlattenHead` helper is
made local or moved to a neutral shared-layer module.

## Data contract

The workflow supports exactly two dataset identifiers:

| Identifier | Source target | Sampling interval | Power unit | Rated power |
| --- | --- | ---: | --- | ---: |
| `skippd_luoyang` | `final_power` | 5 minutes | kW | 30.1 |
| `pvod_station00_ylj` | `observe_power` | 15 minutes | MW | 6.6 |

The model receives normalized historical power only. It never receives images,
weather/NWP variables, ramp variables, calendar marks, privileged Teacher data,
or future targets as inputs.

A dedicated power-only Parquet dataset replaces the multimodal adapters. It
reads only the configured timestamp and target columns, retains chronological
train/validation/test boundaries, requires exact timestamp continuity for each
history/forecast window, emits an explicit future-target validity mask, and
normalizes power by the configured `power_scale`. Missing historical power is
filled causally and then by the training mean; missing future targets remain
masked and do not contribute to loss or metrics.

The batch boundary is a small mapping containing historical power, future
target, target mask, and issue timestamp. No dummy image/weather tensors are
created.

## Forecast tasks

The batch scripts execute four user-selected tasks:

| Task label | History points | SKIPP'D prediction points | PVOD prediction points |
| --- | ---: | ---: | ---: |
| `24-1` | 24 | 1 | 1 |
| `48-1` | 48 | 1 | 1 |
| `48-4h` | 48 | 48 | 16 |
| `96-4h` | 96 | 48 | 16 |

The four-hour tasks represent four hours of physical time. Their point counts
therefore differ by dataset sampling interval. History length remains a point
count, as requested.

With eight models and two datasets, the full and smoke matrices each contain
`4 × 2 × 8 = 64` tasks.

## Training and evaluation

`run_time_series.py` remains the single-model entry point. It:

- constructs one selected model through the registry and factory;
- trains with masked mean squared error;
- selects the best validation checkpoint;
- evaluates predictions in normalized model space;
- restores predictions and targets to original power units;
- writes `best.pt`, `predictions.csv`, and `metrics.json`;
- supports explicit iteration caps and `--smoke`.

`--smoke` forces one epoch container, batch size two, and exactly one training,
one validation, and one test iteration. It must never silently run a complete
epoch.

The orchestration script uses physical GPUs 0 and 1 by default, with one
sequential queue per GPU and both queues running concurrently. The 64 tasks are
distributed evenly: 32 tasks per GPU. Failed tasks are recorded without
preventing the other queue from completing, and resume mode skips only complete
artifacts.

## Retained repository surface

The final branch keeps only:

- the single-model runner;
- the eight model files and their registry/factory/adapter;
- the dedicated power-only data loader;
- the two dataset configurations;
- the exact shared layers required by the eight models;
- full and smoke dual-GPU scripts;
- focused unit/integration tests;
- minimal dependency metadata, `.gitignore`, `pytest.ini`, README, and pure-TS
  usage documentation;
- this design and its implementation plan.

The branch removes:

- all other forecasting and foundation-model implementations;
- `MTS_31`, `MTS_31F`, image encoders, Teacher/KD experiments and entry points;
- multimodal Luoyang/YLJ adapters after their power-only semantics are covered;
- visualization, monitoring, legacy dataset conversion/pipeline code, and old
  experiment documentation;
- old tests whose subjects no longer exist;
- historical experiment outputs and generated figures;
- dependencies used only by removed code.

## Tests and acceptance criteria

Implementation follows test-first changes. Acceptance requires:

1. `run_time_series.py --list-models` prints exactly the eight retained names.
2. Importing every retained model succeeds without importing any removed model.
3. Each retained model produces finite `[batch, pred_len, 1]` output for the
   supported point shapes exercised by focused CPU tests.
4. The data loader reads only timestamp/power fields, preserves split and mask
   behavior, and yields normalized power-only batches.
5. Script contract tests prove exactly 64 unique tasks, the required four task
   settings, two datasets, two GPUs, and 32 tasks per GPU.
6. Script contract tests prove smoke passes `--smoke` to every task and full
   training does not.
7. Removed model names are rejected by the CLI and their source files are absent.
8. Focused unit tests pass in `/opt/data/private/penv/time/bin/python`.
9. The real dual-GPU smoke matrix completes using GPUs 0 and 1, with every phase
   limited to one batch. No full epoch or formal training is run during
   verification.
