# Classical Forecast Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Persistence, SmartPersistence, SeasonalPersistence, and Climatology as CPU baselines with the same observable outputs as the eight retained neural forecasting models.

**Architecture:** Keep a dedicated baseline registry and implementation package beside the unchanged neural registry. Branch inside the existing runner between the current neural training path and a no-optimizer baseline path, then include both catalogs in one 96-task batch summary.

**Tech Stack:** Python 3.8, PyTorch 2.3, NumPy, pandas, pvlib 0.10.5-0.11.x, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-08-26-classical-baselines-design.md`

## Global Constraints

- Preserve the exact eight-name neural catalog and its order.
- Add the exact baseline order: `Persistence`, `SmartPersistence`, `SeasonalPersistence`, `Climatology`.
- `--list-models` lists only neural models; `--list-baselines` lists only baselines; `--model` accepts their union.
- Neural model inputs and behavior remain historical normalized power only.
- Baselines may additionally use issue timestamps, static site metadata, and past raw power, but never future observations or weather/NWP data.
- Smart Persistence uses pvlib Ineichen clear sky and array-plane irradiance with a 1 W/m2 denominator threshold.
- All baseline predictions are finite normalized `[B, pred_len, 1]` arrays clipped to `[0, 1]`.
- Baselines write `best.pt`, `predictions.csv`, `metrics.json`, and `completion.tsv` under the existing layout.
- The batch matrix contains exactly 96 tasks: 64 neural tasks on two GPU queues and 32 baseline tasks on CPU.
- Follow RED-GREEN-REFACTOR and retain command/output evidence in the task report.

---

### Task 1: Baseline algorithms and site-aware clear-sky curve

**Files:**
- Create: `baselines/__init__.py`
- Create: `baselines/registry.py`
- Create: `baselines/methods.py`
- Modify: `data_provider/power_only.py`
- Modify: `configs/datasets/skippd_luoyang.json`
- Modify: `configs/datasets/pvod_station00_ylj.yaml`
- Modify: `requirements.txt`
- Create: `tests/test_classical_baselines.py`
- Modify: `tests/test_power_only_dataset.py`

**Interfaces:**
- Produces: `BASELINE_NAMES: tuple[str, ...]` in the exact required order.
- Produces: `load_baseline_class(name: str) -> type[ForecastBaseline]` and `build_baseline(name: str, train_dataset) -> ForecastBaseline`.
- Produces: baseline `predict(history: torch.Tensor, issue_time_ns, dataset) -> torch.Tensor` and `checkpoint_payload() -> dict`.
- Extends: `PowerOnlyParquetDataset` with validated `site_config`, `normalized_power_at(timestamps, fallback)` and finite training-observation access needed by baselines.

- [ ] **Step 1: Write failing registry and formula tests**

Add literal tests asserting the exact catalog; Persistence repeats the last
value; SmartPersistence produces `[0.2, 0.4]` from current power `0.1` and
injected clear POA `[100, 200, 400]`; SeasonalPersistence reads exact previous-day
values and falls back per missing step; Climatology excludes validation/test
values and uses the circular +/-15-day same-clock bucket. Each test names the
wrong behavior it catches and derives expected values by hand.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_classical_baselines.py tests/test_power_only_dataset.py
```

Expected: collection fails because `baselines` and `site_config` do not exist.

- [ ] **Step 3: Implement the minimal baseline package and dataset accessors**

Implement the four semantics exactly as the spec states. Keep pvlib calculation
behind one `clear_sky_poa(site_config, naive_local_times)` function so formula
tests inject literal POA values without mocking pvlib internals. Localize source
wall-clock timestamps with each site's IANA timezone. Validate latitude in
`[-90, 90]`, longitude in `[-180, 180]`, tilt in `[0, 90]`, azimuth in
`[0, 360)`, and a non-empty timezone string.

Add these exact config values:

```text
SKIPP'D: latitude=37.427, longitude=-122.174,
timezone=America/Los_Angeles, surface_tilt=37, surface_azimuth=195
PVOD: latitude=38.04778, longitude=114.95139,
timezone=Asia/Shanghai, surface_tilt=33, surface_azimuth=180
```

Add `pvlib>=0.10.5,<0.12` to requirements.

- [ ] **Step 4: Add real pvlib and validation tests, then verify GREEN**

Use fixed local summer timestamps and assert both sites have finite positive POA
at noon and zero POA at midnight. Test malformed site fields and baseline output
shape/finite/clipping behavior. Run the focused command from Step 2 and require
all tests to pass without warnings.

- [ ] **Step 5: Run the full suite and commit**

```bash
/opt/data/private/penv/time/bin/python -m pytest -q
git add baselines data_provider/power_only.py configs/datasets requirements.txt tests/test_classical_baselines.py tests/test_power_only_dataset.py
git commit -m "feat: add classical photovoltaic baselines"
```

### Task 2: Runner dispatch and uniform artifacts

**Files:**
- Modify: `run_time_series.py`
- Modify: `tests/test_run_time_series_all_models.py`
- Modify: `tests/test_top_eight_models.py`
- Create: `tests/test_baseline_runner.py`

**Interfaces:**
- Consumes: `BASELINE_NAMES` and `build_baseline` from Task 1.
- Produces: `_collect_baseline_predictions(baseline, loader, dataset, device, max_steps, phase="test") -> dict` using the same result keys as `_collect_predictions`.
- Produces: a baseline branch in `run(args)` with truthful zero training/validation steps and a serializable `best.pt`.
- Extends: CLI catalog behavior without changing the exact neural `SELECTED_MODEL_NAMES` tuple.

- [ ] **Step 1: Write failing CLI and runner behavior tests**

Test `--list-baselines` without dataset/model, union acceptance through
`--model`, and rejection of unknown methods. With a real synthetic Parquet
dataset and `Persistence`, assert predictions are repeated last history values,
all four artifacts exist, checkpoint identifies `method_type=baseline`, and
metrics contain `training_skipped=true`, `best_epoch=null`,
`phase_steps={train: 0, val: 0, test: 1}` in smoke mode. Patch neural model
construction to raise and prove the baseline branch never calls it; do not
assert on a mock call count.

- [ ] **Step 2: Run the new runner tests and verify RED**

```bash
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_baseline_runner.py tests/test_run_time_series_all_models.py tests/test_top_eight_models.py
```

Expected: failures show missing CLI catalog and baseline dispatch.

- [ ] **Step 3: Implement baseline collection, dispatch, and artifacts**

Keep the existing neural branch structurally intact. Baseline collection uses
the same mask weighting, empty-loader errors, timestamp export, metrics writer,
hashing, and completion writer as neural evaluation. Save `best.pt` with method
name/type, dataset identity, effective config, and `checkpoint_payload()`.
Record effective epochs=0 for baseline manifests and retain the requested epoch
count only for neural methods. Add `method_type` and `training_skipped` to both
metrics payload variants.

- [ ] **Step 4: Verify GREEN and neural non-regression**

Run the command from Step 2, then:

```bash
/opt/data/private/penv/time/bin/python run_time_series.py --list-models
/opt/data/private/penv/time/bin/python run_time_series.py --list-baselines
```

Require the exact two catalogs and all focused tests to pass.

- [ ] **Step 5: Run the full suite and commit**

```bash
/opt/data/private/penv/time/bin/python -m pytest -q
git add run_time_series.py tests/test_run_time_series_all_models.py tests/test_top_eight_models.py tests/test_baseline_runner.py
git commit -m "feat: run baselines with uniform artifacts"
```

### Task 3: Unified 96-task orchestration and documentation

**Files:**
- Modify: `scripts/run_all_pure_time_series.sh`
- Modify: `tests/test_pure_time_series_scripts.py`
- Modify: `README.md`
- Modify: `PURE_TIME_SERIES.md`

**Interfaces:**
- Consumes: `--list-models`, `--list-baselines`, and baseline completion semantics from Task 2.
- Produces: exactly 96 unique `(setting, dataset, method)` summary rows.
- Preserves: existing eight-column summary format and artifact-hash resume checks.

- [ ] **Step 1: Change executable script tests first**

Extend the fake runner boundary to return both catalogs. Assert 64 neural calls
use `--device cuda:0`, distribute 32/32 across the configured GPUs, and retain
existing smoke limits. Assert 32 baseline calls use `--device cpu`,
`launch_gpu=cpu`, never set a CUDA device requirement, and appear exactly once
for every setting/dataset/baseline combination. Assert all 96 output directories
and summary identities are unique.

Add resume fixtures with baseline completion rows (epochs/train/val steps all
zero, test steps positive) and prove valid artifacts skip while corrupted hashes,
nonzero baseline training steps, or wrong method identity rerun.

- [ ] **Step 2: Run script tests and verify RED**

```bash
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_pure_time_series_scripts.py
```

Expected: task count/catalog/CPU assertions fail against the 64-task script.

- [ ] **Step 3: Implement neural GPU queues plus a baseline CPU queue**

Load and validate the exact two ordered catalogs separately. Keep neural
round-robin queue behavior. Generate a third task file for baselines and run it
with `launch_gpu=cpu`, `--device cpu`, and no `CUDA_VISIBLE_DEVICES` assignment.
Completion validation accepts train_steps=val_steps=epochs=0 only for registered
baseline names; neural full/smoke validation remains unchanged. Merge all three
result files under the existing summary header.

- [ ] **Step 4: Update human documentation**

Document the two catalogs, four formulas, site metadata and pvlib dependency,
single baseline invocation, 96-task total, CPU/GPU scheduling, output identity,
and resume rules. Preserve the statement that neural models receive only
historical power and explicitly state the extra static/time inputs used only by
baselines.

- [ ] **Step 5: Verify scripts and commit**

```bash
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_pure_time_series_scripts.py
bash -n scripts/run_all_pure_time_series.sh
bash -n scripts/smoke_all_pure_time_series.sh
/opt/data/private/penv/time/bin/python -m pytest -q
git add scripts tests/test_pure_time_series_scripts.py README.md PURE_TIME_SERIES.md
git commit -m "feat: add baselines to experiment matrix"
```

