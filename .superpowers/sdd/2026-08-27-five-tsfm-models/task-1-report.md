# Task 1 report — migrate existing TSFM core

## Implementation

- Moved the shared foundation backend contract and forecast validation to `models/base.py`.
- Split shared validation, normalization, lazy Transformers loading, generated-backend setup, and native-loss validation into `models/common.py`.
- Moved Sundial and TimeMoE backends into `models/Sundial.py` and `models/TimeMoE.py`, preserving their generation, denormalization, native training-loss, mask, and device behavior.
- Moved the registry, lazy factory, capability-filtered task matrix, and trainability/audit policies to `models/registry.py`, `models/factory.py`, `models/tasks.py`, and `models/trainability.py`.
- Made `models/__init__.py` keep registry/task imports lightweight and resolve runtime helpers lazily, without importing Transformers or PEFT.
- Moved `PowerBatch`/`power_only_batch` into `data_provider/power_only.py`.
- Moved `json_safe`, `sha256`, and `write_predictions` (with the private legacy aliases used by the runner) into `utils/artifacts.py`.
- Updated `run_foundation_model.py`, the retained environment preflight, retained two-GPU shell snippets, and foundation tests to use the new boundaries.
- Deleted the `foundation_models/` compatibility package, pure/classical models and layers, `models/tslib_*`, `run_time_series.py`, pure-time-series scripts/tests/docs, and `utils/masking.py`.
- Kept `FOUNDATION_MODELS.md` and the old 2026-08-27 two-model plan/spec as directed by the scope correction.

## RED

First changed the focused tests to import the desired `models.*` and `data_provider.power_only` boundaries, then ran:

```text
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_foundation_registry.py tests/test_foundation_tasks.py tests/test_foundation_backends.py tests/test_foundation_trainability.py tests/test_power_only_dataset.py
```

Expected collection failure was observed:

```text
ImportError: cannot import name 'MODEL_NAMES' from 'models'
ModuleNotFoundError: No module named 'models.registry'
ImportError: cannot import name 'power_only_batch' from 'data_provider.power_only'
3 errors during collection
```

## GREEN

After the minimal migration and test-boundary updates, the same command produced:

```text
........................................................................ [ 93%]
.....                                                                    [100%]
77 passed in 5.98s
```

The retained runner/preflight/script tests also passed in the full run. A separate retained-test check before the final full run reported 170 passed; the isolated script harness was changed to copy only required files, avoiding the known `codex` special-file copy failure.

## Full suite

Command:

```text
/opt/data/private/penv/time/bin/python -m pytest -q
```

Output:

```text
........................................................................ [ 36%]
........................................................................ [ 72%]
......................................................                   [100%]
198 passed in 100.65s (0:01:40)
```

Additional verification:

- Imported `models`, registry/tasks, both backend modules, `PowerBatch`, `power_only_batch`, and all artifact helpers in a subprocess-style smoke check; registry import left `transformers` and `peft` unloaded.
- `git diff --check` passed.
- Explicit filesystem checks confirmed `foundation_models/`, `baselines/`, `layers/`, `run_time_series.py`, `models/tslib_*`, and `utils/masking.py` are absent.
- `git status --short` confirmed `task.md` and the modified `时序基础模型.md` remain unstaged.

## Self-review

- Existing Sundial/TimeMoE fake-loader, shape, normalization, native-loss, target-mask, and trainability tests remain behavior assertions and pass unchanged apart from import boundaries.
- The catalog package no longer eagerly imports torch/optional model modules; backend construction remains the lazy boundary.
- Shared generated-backend validation is centralized in `models/common.py`; model-specific prediction/loss logic is kept in separate model files.
- Artifact output formats and private runner aliases are preserved.

## Concerns

- No real checkpoint/GPU smoke was run; this environment does not provide the required downloaded model weights, so verification uses the existing complete fake-model contracts.
- The retained foundation shell batch script still intentionally validates the existing two-model/32-task matrix because Task 1 was instructed not to change runner CLI or batch counts; later model-matrix work must update those counts from registry capabilities.

## Fix round 1 — isolated checkout fixture

Reviewer finding: the isolated default-path fixture had been narrowed to an
allowlist, which removed representative repository surface unrelated to the
import migration. The fixture now restores the original complete-tree
`shutil.copytree` and retains the existing `.git`, `__pycache__`, and `*.pyc`
filters, with only the root-level workspace `codex` special entry additionally
ignored. No production or shell-script behavior was changed.

Covering command:

```text
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_foundation_scripts.py::test_unset_output_and_summary_paths_use_smoke_defaults_in_isolated_repo
```

Exact output:

```text
.                                                                        [100%]
1 passed in 2.93s
```

Self-review: the fixture again copies the full checkout, so root-level files,
retained model files, configs, scripts, and any future integration dependency
are represented; only the known non-regular `codex` entry is excluded at the
workspace root. User files remain unstaged. Remaining concern is unchanged:
real model weights/GPU smoke is unavailable in this environment.
