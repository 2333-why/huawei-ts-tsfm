# Pure Time-Series Minimal Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the `pure-ts` branch to a standalone power-only forecasting repository containing exactly the aggregate top-eight models and four requested task settings.

**Architecture:** Replace the multimodal dataset adapters with one generic Parquet power loader that emits a small mapping, keep the existing runner/model boundary, and narrow the lazy registry to eight implementations. Retain only model dependencies, focused tests, two-GPU orchestration, and concise documentation.

**Tech Stack:** Python 3.8, PyTorch 2.3, pandas, PyArrow, NumPy, PyYAML, einops, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-08-26-pure-ts-minimal-branch-design.md`

## Global Constraints

- Work only in `/opt/data/private/code/pure-ts` on branch `pure-ts`.
- Do not modify or delete anything in `/opt/data/private/code/FACTS-Luoyang-why`.
- Retain exactly `TSMixer`, `Pyraformer`, `SegRNN`, `Transformer`, `LightTS`, `Crossformer`, `FreTS`, and `MICN`.
- Model inputs contain historical normalized power only.
- Supported tasks are `24-1`, `48-1`, `48-4h`, and `96-4h`.
- Four hours means 48 outputs for 5-minute SKIPP'D data and 16 outputs for 15-minute PVOD data.
- Smoke verification is one train, one validation, and one test batch per task; never run a complete epoch.
- Use `/opt/data/private/penv/time/bin/python` and physical GPUs 0 and 1.
- Preserve the original `main` worktree's untracked reports, PDF, figures, and visualization files.

---

### Task 1: Dedicated power-only dataset and batch contract

**Files:**
- Create: `data_provider/power_only.py`
- Create: `tests/test_power_only_dataset.py`
- Modify: `models/tslib_adapter.py`
- Modify: `run_time_series.py`
- Modify: `configs/datasets/skippd_luoyang.json`
- Modify: `configs/datasets/pvod_station00_ylj.yaml`

**Interfaces:**
- Produces: `PowerOnlyParquetDataset(config_path, flag, history_points, forecast_steps)` with `seq_len`, `pred_len`, `power_scale`, `forecast_step`, and mapping samples.
- Produces sample keys: `history`, `target`, `target_mask`, `issue_time_ns`.
- `power_only_batch(batch, device) -> PowerBatch` consumes the collated mapping.
- `run_time_series.DATASET_LOADERS` maps both identifiers to `PowerOnlyParquetDataset`.

- [ ] **Step 1: Write failing dataset tests**

Create literal synthetic 5-minute and 15-minute Parquet fixtures. Tests must demonstrate these observable contracts:

```python
def test_loader_emits_only_power_batch_fields(synthetic_power_config):
    dataset = PowerOnlyParquetDataset(
        synthetic_power_config, "train", history_points=24, forecast_steps=1
    )
    sample = dataset[0]
    assert set(sample) == {"history", "target", "target_mask", "issue_time_ns"}
    assert sample["history"].shape == (24, 1)
    assert sample["target"].shape == (1, 1)


def test_loader_normalizes_by_configured_power_scale(synthetic_power_config):
    dataset = PowerOnlyParquetDataset(
        synthetic_power_config, "train", history_points=2, forecast_steps=1
    )
    sample = dataset[0]
    assert sample["history"][-1, 0] == pytest.approx(0.5)
    assert sample["target"][0, 0] == pytest.approx(0.6)


def test_loader_masks_missing_future_target_but_keeps_valid_window(
    synthetic_power_config_with_missing_future,
):
    dataset = PowerOnlyParquetDataset(
        synthetic_power_config_with_missing_future,
        "test",
        history_points=2,
        forecast_steps=2,
    )
    sample = dataset[0]
    assert sample["target_mask"].tolist() == [False, True]
    assert np.isfinite(sample["target"]).all()


def test_loader_rejects_timestamp_gaps_inside_requested_window(
    synthetic_power_config_with_gap,
):
    dataset = PowerOnlyParquetDataset(
        synthetic_power_config_with_gap, "test", history_points=2, forecast_steps=2
    )
    assert len(dataset) == 0
```

Add a real `DataLoader` collation test that passes one batch through
`power_only_batch` and asserts `[B, history, 1]`, `[B, forecast, 1]`, and
`[B, forecast]` shapes. Each expected value must be a hand-written literal.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_power_only_dataset.py
```

Expected: collection fails because `data_provider.power_only` does not exist.

- [ ] **Step 3: Implement the minimal dataset**

Implement config loading for JSON/YAML, read only `[timestamp, target]` via
`pd.read_parquet(..., columns=...)`, validate finite unique timestamps, and
construct exact timestamp windows using `np.searchsorted`. Compute validation
start only from pre-test candidates, preserve horizon-within-split guards, and
compute the training mean strictly before validation start.

Return float32 normalized history/target arrays, a boolean target mask, and an
int64 issue timestamp. Historical missing values use within-window forward fill
then the training mean. Missing targets use a finite zero placeholder after
mask creation. Apply `target_floor` only when present in configuration.

Strip both dataset configs to these keys only:

```text
dataset, paths.parquet_file, fields.timestamp, fields.target,
time.sampling_interval_minutes, time.forecast_step_minutes,
time.train_start_timestamp, time.test_start_timestamp,
time.test_end_exclusive_timestamp, time.validation_fraction,
power.power_scale, power.rated_power, power.units, optional power.target_floor,
paths.results_root
```

Update `run_time_series.py` to use the generic loader and pass no image,
weather, privileged, or dataset-specific kwargs. Update issue-time extraction to
read the batch mapping. Remove the unreachable return following the empty-loader
exception.

- [ ] **Step 4: Run Task 1 tests and focused runner tests**

Run:

```bash
/opt/data/private/penv/time/bin/python -m pytest -q \
  tests/test_power_only_dataset.py tests/test_run_time_series_all_models.py
```

Expected: PASS after adapting runner tests away from multimodal tuples as needed.

- [ ] **Step 5: Commit**

```bash
git add data_provider/power_only.py models/tslib_adapter.py run_time_series.py \
  configs/datasets/skippd_luoyang.json \
  configs/datasets/pvod_station00_ylj.yaml \
  tests/test_power_only_dataset.py tests/test_run_time_series_all_models.py
git commit -m "refactor: add dedicated power-only data path"
```

### Task 2: Restrict model surface to the aggregate top eight

**Files:**
- Modify: `models/tslib_registry.py`
- Modify: `models/tslib_factory.py`
- Modify: `models/tslib_adapter.py`
- Modify: `models/__init__.py`
- Modify: `models/Crossformer.py`
- Modify: `layers/tslib/SelfAttention_Family.py`
- Create: `tests/test_top_eight_models.py`
- Delete: every `models/*.py` implementation except the eight selected models, `__init__.py`, `tslib_registry.py`, `tslib_factory.py`, and `tslib_adapter.py`.

**Interfaces:**
- `SELECTED_MODEL_NAMES` is the ranking-order tuple of eight required names.
- `load_model_class(name)` lazy-loads only those eight.
- `build_power_model(name, args, dataset)` supports every retained model for all required shapes.

- [ ] **Step 1: Write failing registry and forward tests**

```python
EXPECTED_MODELS = (
    "TSMixer", "Pyraformer", "SegRNN", "Transformer",
    "LightTS", "Crossformer", "FreTS", "MICN",
)


def test_registry_exposes_only_ranked_top_eight():
    assert SELECTED_MODEL_NAMES == EXPECTED_MODELS


@pytest.mark.parametrize("name", EXPECTED_MODELS)
@pytest.mark.parametrize(
    "seq_len,pred_len",
    [(24, 1), (48, 1), (48, 16), (48, 48), (96, 16), (96, 48)],
)
def test_retained_model_forecasts_supported_shapes(name, seq_len, pred_len):
    args = tiny_args()
    dataset = SimpleNamespace(seq_len=seq_len, pred_len=pred_len)
    model, config = build_power_model(name, args, dataset)
    prediction = forward_power_model(
        name, model.eval(), torch.randn(2, seq_len, 1), config.label_len, pred_len
    )
    assert prediction.shape == (2, pred_len, 1)
    assert torch.isfinite(prediction).all()
```

Also assert CLI behavior rather than source text:

```python
def test_removed_model_is_rejected_by_cli():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "skippd_luoyang", "--model", "DLinear"])
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_top_eight_models.py
```

Expected: registry assertion fails because 32 models are still exposed.

- [ ] **Step 3: Narrow registry/factory and detach Crossformer from PatchTST**

Set the registry tuple exactly as shown above. Remove `EXCLUDED_MODELS`, Koopa
spectrum preparation, MultiPatchFormer overrides, and PatchTST/PAttn constructor
branches. Remove factory config fields that no retained model reads, while
retaining all fields consumed by the eight constructors.

Move the 13-line `FlattenHead` helper used by `Crossformer` into
`models/Crossformer.py`, then remove `from models.PatchTST import FlattenHead`.
Do not alter Crossformer forecasting behavior.

Simplify `models/__init__.py` to expose only registry APIs; it must not advertise
legacy models.

Remove the unused `ReformerLayer` and its eager `reformer_pytorch` import from
`layers/tslib/SelfAttention_Family.py`. The retained models use
`FullAttention`, `AttentionLayer`, and `TwoStageAttentionLayer`; this removal
must not change those classes.

- [ ] **Step 4: Delete nonselected model files using an explicit allowlist**

Resolve `git ls-files 'models/*.py'`, compare it to this allowlist, and delete
only paths outside it:

```text
models/__init__.py
models/Crossformer.py
models/FreTS.py
models/LightTS.py
models/MICN.py
models/Pyraformer.py
models/SegRNN.py
models/TSMixer.py
models/Transformer.py
models/tslib_adapter.py
models/tslib_factory.py
models/tslib_registry.py
```

- [ ] **Step 5: Run model tests and import audit**

Run:

```bash
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_top_eight_models.py
/opt/data/private/penv/time/bin/python run_time_series.py --list-models
```

Expected: all tests pass; CLI prints the eight names in ranking order.

- [ ] **Step 6: Commit**

```bash
git add -A models layers/tslib/SelfAttention_Family.py \
  tests/test_top_eight_models.py
git commit -m "refactor: retain aggregate top-eight models"
```

### Task 3: Four-setting dual-GPU scripts and focused documentation

**Files:**
- Modify: `scripts/run_all_pure_time_series.sh`
- Modify: `scripts/smoke_all_pure_time_series.sh`
- Modify: `tests/test_pure_time_series_scripts.py`
- Modify: `PURE_TIME_SERIES.md`
- Modify: `README.md`
- Modify: `requirements.txt`
- Delete: `scripts/run_time_series_models.sh`
- Delete: `scripts/smoke_time_series_models.sh`

**Interfaces:**
- Full script expands 64 unique `(setting, dataset, model)` tasks.
- Smoke wrapper sets `SMOKE=1` and delegates to the same matrix.
- Output setting directories are `seq24_pred1`, `seq48_pred1`, `seq48_h4`, and `seq96_h4`.

- [ ] **Step 1: Change script contract tests first**

Set test literals to:

```python
EXPECTED_MODELS = (
    "TSMixer", "Pyraformer", "SegRNN", "Transformer",
    "LightTS", "Crossformer", "FreTS", "MICN",
)
EXPECTED_TASKS = {
    (24, 1, "skippd_luoyang"),
    (24, 1, "pvod_station00_ylj"),
    (48, 1, "skippd_luoyang"),
    (48, 1, "pvod_station00_ylj"),
    (48, 48, "skippd_luoyang"),
    (48, 16, "pvod_station00_ylj"),
    (96, 48, "skippd_luoyang"),
    (96, 16, "pvod_station00_ylj"),
}
```

Run the real shell scripts against the existing fake Python boundary. Assert 64
runs, eight task triples times eight models, 32 runs on each GPU, no same-GPU
overlap, unique output directories, `--smoke` only in smoke mode, complete
summary rows, and resume retry behavior.

- [ ] **Step 2: Run script tests and verify RED**

Run:

```bash
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_pure_time_series_scripts.py
```

Expected: failures report the old 32-model/128-task contract.

- [ ] **Step 3: Implement the readable 64-task matrix**

Use literal setting rows and a small `prediction_points(dataset, horizon)`
function. A one-point horizon always returns one. A four-hour horizon returns 48
for `skippd_luoyang` and 16 for `pvod_station00_ylj`; unknown inputs exit 2.
Require exactly eight unique registry names. Keep one sequential queue per GPU,
round-robin assignment, resume checks, per-task logs, and final summary behavior.

Use readable task labels in output directories so both 48-history tasks remain
distinct. Preserve `GPUS="0 1"`, the configured Python default, and `--device
cuda:0` inside each process.

- [ ] **Step 4: Simplify docs and dependencies**

Document only the two datasets, eight models, four settings, single-model usage,
64-task full run, and bounded smoke run. Replace stale 32/128 references.

Reduce requirements to packages imported by retained runtime code. At minimum:

```text
einops>=0.6,<1
numpy>=1.24,<2
pandas>=1.5,<3
pyarrow>=10
PyYAML>=6,<7
torch>=2.0
```

If an import audit proves another package is required by a retained module, add
that exact package with the existing compatible version bounds and record why in
the worker report.

- [ ] **Step 5: Run script tests**

```bash
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_pure_time_series_scripts.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A scripts tests/test_pure_time_series_scripts.py \
  PURE_TIME_SERIES.md README.md requirements.txt
git commit -m "feat: add four-setting top-eight experiment matrix"
```

### Task 4: Remove unrelated repository surface and verify the branch

**Files:**
- Retain only the final manifest below plus any imported shared-layer file proved necessary by `modulefinder`/import execution.
- Delete all other tracked source, config, test, script, result, and legacy documentation files.
- Create: `tests/test_runner.py` if runner tests need a focused replacement name.

**Interfaces:**
- The branch remains runnable from the root with `run_time_series.py` and the two orchestration scripts.
- No retained Python module imports a deleted local module.

- [ ] **Step 1: Establish the final tracked-file manifest**

```text
.gitignore
README.md
PURE_TIME_SERIES.md
pytest.ini
requirements.txt
run_time_series.py
configs/datasets/skippd_luoyang.json
configs/datasets/pvod_station00_ylj.yaml
data_provider/__init__.py
data_provider/power_only.py
models/__init__.py
models/Crossformer.py
models/FreTS.py
models/LightTS.py
models/MICN.py
models/Pyraformer.py
models/SegRNN.py
models/TSMixer.py
models/Transformer.py
models/tslib_adapter.py
models/tslib_factory.py
models/tslib_registry.py
layers/__init__.py
layers/tslib/__init__.py
layers/tslib/Autoformer_EncDec.py
layers/tslib/Crossformer_EncDec.py
layers/tslib/Embed.py
layers/tslib/Pyraformer_EncDec.py
layers/tslib/SelfAttention_Family.py
layers/tslib/Transformer_EncDec.py
utils/__init__.py
utils/masking.py
scripts/run_all_pure_time_series.sh
scripts/smoke_all_pure_time_series.sh
tests/test_power_only_dataset.py
tests/test_top_eight_models.py
tests/test_run_time_series_all_models.py
tests/test_pure_time_series_scripts.py
docs/superpowers/specs/2026-08-26-pure-ts-minimal-branch-design.md
docs/superpowers/plans/2026-08-26-pure-ts-minimal-branch.md
```

Before deletion, run an import closure over `run_time_series.py` and the eight
model files. If a local file outside the manifest is imported, retain only that
proved dependency and record it in the worker report. Do not retain files merely
because old tests or deleted entry points reference them.

- [ ] **Step 2: Delete files outside the resolved manifest**

Use `git ls-files` to enumerate exact paths. Never target the repository root,
an unresolved variable, or a wildcard with `rm -rf`. Delete only reviewed
tracked paths outside the manifest.

- [ ] **Step 3: Run static and focused verification**

```bash
/opt/data/private/penv/time/bin/python -m compileall -q .
/opt/data/private/penv/time/bin/python -m pytest -q
/opt/data/private/penv/time/bin/python run_time_series.py --list-models
bash -n scripts/run_all_pure_time_series.sh
bash -n scripts/smoke_all_pure_time_series.sh
```

Expected: compile succeeds, all retained tests pass, exactly eight model names
print, and both scripts parse.

- [ ] **Step 4: Run real two-GPU bounded smoke verification**

```bash
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" SMOKE=1 \
OUTPUT_ROOT=/opt/data/private/code/pure-ts/results_pure_time_series_smoke \
bash scripts/smoke_all_pure_time_series.sh
```

Expected: exactly 64 summary rows; every row is `PASS`; every per-task metrics
file reports one train, one validation, and one test step. This command must not
run any complete epoch.

- [ ] **Step 5: Commit final pruning**

```bash
git add -A
git commit -m "chore: remove non-pure-ts repository code"
```

- [ ] **Step 6: Record final evidence**

The worker report must include branch, worktree, commit hashes, retained tracked
file count, deleted tracked file count, pytest output, eight-name CLI output,
smoke summary counts by status/GPU/setting/dataset/model, and any remaining risk.
