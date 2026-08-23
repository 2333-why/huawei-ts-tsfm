# All Pure Time-Series Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate and minimally GPU-smoke all 32 forecasting-capable, non-foundation Time-Series-Library models on both power datasets.

**Architecture:** Keep the existing causal dataset adapters and enforce one power-only model boundary. Copy the 28 missing models into `models/`, isolate copied TSLib layer dependencies under `layers/tslib/`, and use a lazy registry plus small model-specific adapters to normalize construction, inputs, and outputs.

**Tech Stack:** Python 3.8, PyTorch 2.3.1 CUDA 11.8, NumPy, pandas, PyYAML, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-08-21-all-pure-time-series-models-design.md`

## Global Constraints

- Use `/opt/data/private/penv/time/bin/python` for every Python command.
- GPU work must select a physical GPU with `CUDA_VISIBLE_DEVICES` and use
  process-local `cuda:0`; Task6 defaults to physical GPUs `0 1` and assigns
  one sequential worker queue per card.
- Model input is only normalized historical power from channel zero.
- Never pass images, weather, NWP, ramp variables, privileged Teacher data, or future targets to a model.
- Select exactly the 32 models listed in the spec; foundation models and KANAD stay excluded.
- Smoke verification is exactly one train, one validation, and one test iteration per model/dataset combination.
- Do not run a complete unrestricted dataset epoch.
- The current workspace has no `.git` directory. Do not attempt commits; record files, commands, and outputs in each worker report instead.
- Multiple agents share the workspace. Do not revert or overwrite unrelated user or agent changes.
- Production behavior changes follow red-green TDD: add a focused failing test, run it and record the expected failure, implement minimally, then rerun it.

---

### Task 1: Authoritative lazy model registry

**Files:**
- Create: `models/tslib_registry.py`
- Modify: `models/__init__.py`
- Modify: `tests/test_pure_time_series.py`

**Interfaces:**
- Produces: `SELECTED_MODEL_NAMES: tuple[str, ...]`
- Produces: `EXCLUDED_MODELS: dict[str, str]`
- Produces: `ModelSpec(module: str, constructor: str = "Model")`
- Produces: `MODEL_SPECS: dict[str, ModelSpec]`
- Produces: `load_model_class(name: str) -> type[torch.nn.Module]`
- Preserves: `PURE_TIME_SERIES_MODELS` as a mapping-like lazy compatibility view for existing callers.

- [ ] **Step 1: Replace the four-name registry assertion with exact selection and exclusion behavior tests**

Write literal expected names in `tests/test_pure_time_series.py`:

```python
EXPECTED_MODELS = {
    "Autoformer", "Crossformer", "DLinear", "ETSformer", "FEDformer",
    "FiLM", "FreTS", "Informer", "Koopa", "LightTS", "MICN", "MSGNet",
    "Mamba", "MambaSimple", "MultiPatchFormer", "Nonstationary_Transformer",
    "PAttn", "PatchTST", "Pyraformer", "Reformer", "SCINet", "SegRNN",
    "TSMixer", "TemporalFusionTransformer", "TiDE", "TimeFilter",
    "TimeMixer", "TimeXer", "TimesNet", "Transformer", "WPMixer",
    "iTransformer",
}

def test_registry_has_every_forecasting_non_foundation_model():
    assert set(SELECTED_MODEL_NAMES) == EXPECTED_MODELS
    assert set(MODEL_SPECS) == EXPECTED_MODELS

def test_registry_exclusions_are_explicit():
    assert set(EXCLUDED_MODELS) == {
        "Chronos", "Chronos2", "KANAD", "Sundial", "TimeMoE", "TiRex", "TimesFM",
    }
    assert "forecast" in EXCLUDED_MODELS["KANAD"].lower()
    for name in {"Chronos", "Chronos2", "Sundial", "TimeMoE", "TiRex", "TimesFM"}:
        assert "foundation" in EXCLUDED_MODELS[name].lower()

def test_registry_is_lazy_for_optional_mamba_dependency(monkeypatch):
    real_import = importlib.import_module
    calls = []
    def recording_import(name, package=None):
        calls.append(name)
        return real_import(name, package)
    monkeypatch.setattr(importlib, "import_module", recording_import)
    assert "Mamba" in SELECTED_MODEL_NAMES
    assert "models.Mamba" not in calls
```

- [ ] **Step 2: Run the focused tests and verify the expected RED state**

Run:

```bash
/opt/data/private/penv/time/bin/python -m pytest -p no:cacheprovider tests/test_pure_time_series.py -q
```

Expected: registry-name imports or exact-set assertions fail because only four models are registered.

- [ ] **Step 3: Implement the registry and lazy compatibility view**

Use an immutable sorted tuple for `SELECTED_MODEL_NAMES`, literal exclusion reasons, and `importlib.import_module(spec.module)`. Unknown names must raise `KeyError` listing valid names. Wrap missing dependencies as an `ImportError` that names both the model and original package.

The compatibility view must implement `collections.abc.Mapping`; `__getitem__` calls `load_model_class`, while `__iter__` and `__len__` never import model modules.

- [ ] **Step 4: Run the focused tests GREEN**

Run the same pytest command. Expected: all registry and existing four-model tests pass.

---

### Task 2: Reproducibly copy the 28 missing models and isolated TSLib layers

**Files:**
- Create: `scripts/sync_tslib_models.py`
- Create: `layers/tslib/__init__.py`
- Create: `layers/tslib/*.py` copied from the source `layers/`
- Create: the 28 missing selected `models/*.py` modules
- Modify: `tests/test_pure_time_series.py`

**Interfaces:**
- Consumes: `SELECTED_MODEL_NAMES` from Task 1.
- Produces: `sync_models(source_root: Path, destination_root: Path) -> list[Path]`.
- Invariant: the script never changes `/opt/data/private/code/Time-Series-Library`.

- [ ] **Step 1: Add a failing migration completeness test**

```python
def test_every_selected_model_file_exists():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    missing = sorted(name for name in EXPECTED_MODELS if not (model_dir / f"{name}.py").is_file())
    assert missing == []

def test_foundation_and_nonforecast_files_are_not_copied():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    for name in EXCLUDED_MODELS:
        assert not (model_dir / f"{name}.py").exists()
```

- [ ] **Step 2: Verify RED**

Run the focused test file. Expected: the first test reports the 28 missing modules.

- [ ] **Step 3: Implement and run the deterministic sync script**

The script must use these constants:

```python
SOURCE_ROOT = Path("/opt/data/private/code/Time-Series-Library")
PRESERVE_EXISTING = {"DLinear", "PatchTST", "TimesNet", "iTransformer"}
EXCLUDED = {"Chronos", "Chronos2", "KANAD", "Sundial", "TimeMoE", "TiRex", "TimesFM"}
```

For every selected model not in `PRESERVE_EXISTING`, copy its source text to
`models/<name>.py`, replacing `from layers.` with `from layers.tslib.` and
`import layers.` with `import layers.tslib.`. Copy every source `layers/*.py`
to `layers/tslib/` with the same internal import rewrite. Preserve UTF-8 text
and final newlines. Reject a source inventory that differs from the registry
rather than silently omitting a model.

Run:

```bash
/opt/data/private/penv/time/bin/python scripts/sync_tslib_models.py
```

- [ ] **Step 4: Verify file completeness and import all modules except Mamba**

Add and run:

```python
@pytest.mark.parametrize("name", sorted(EXPECTED_MODELS - {"Mamba"}))
def test_selected_model_module_imports(name):
    module = importlib.import_module(f"models.{name}")
    assert issubclass(module.Model, torch.nn.Module)
```

Expected: module imports succeed. If copied source has an invalid current-project import, fix only that copied import and record it.

---

### Task 3: Power-only input and output adapter

**Files:**
- Create: `models/tslib_adapter.py`
- Create: `tests/test_tslib_adapter.py`

**Interfaces:**
- Produces: `PowerBatch(x: Tensor, y: Tensor, mask: Tensor)`.
- Produces: `power_only_batch(batch, device) -> PowerBatch`.
- Produces: `decoder_from_history(x, label_len, pred_len) -> Tensor`.
- Produces: `prepare_model_inputs(model_name, x, label_len, pred_len) -> tuple`.
- Produces: `normalize_forecast(model_name, output, batch_size, pred_len) -> Tensor`.
- Produces: `forward_power_model(model_name, model, x, label_len, pred_len) -> Tensor`.

- [ ] **Step 1: Write failing behavior tests for channel isolation and no future leakage**

Use literal tensors:

```python
def test_power_batch_keeps_only_channel_zero():
    x = torch.tensor([[[1., 101.], [2., 102.], [3., 103.]]])
    y = torch.tensor([[[4., 104.], [5., 105.]]])
    batch = (x, y, None, None, None, None, None, None, None,
             {"target_mask": torch.tensor([[True, False]])})
    result = power_only_batch(batch, torch.device("cpu"))
    assert result.x.tolist() == [[[1.], [2.], [3.]]]
    assert result.y.tolist() == [[[4.], [5.]]]
    assert result.mask.tolist() == [[True, False]]

def test_decoder_uses_history_then_future_zeros():
    x = torch.tensor([[[1.], [2.], [3.], [4.]]])
    decoder = decoder_from_history(x, label_len=2, pred_len=3)
    assert decoder.tolist() == [[[3.], [4.], [0.], [0.], [0.]]]

def test_normalize_forecast_slices_history_prefix():
    output = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    result = normalize_forecast("SCINet", output, batch_size=1, pred_len=3)
    assert result.tolist() == [[[4.], [5.], [6.]]]
```

- [ ] **Step 2: Verify RED**

Run `tests/test_tslib_adapter.py`; expected failure is a missing module/function.

- [ ] **Step 3: Implement the minimal adapter**

`prepare_model_inputs` returns `(x_model, None, x_dec, None)` by default.
For `TemporalFusionTransformer`, return zero marks of shape `[B, seq_len, 4]`
and `[B, label_len + pred_len, 4]`. For `MultiPatchFormer`, left-pad power
history to 32 by repeating its first value. Do not accept or reference `y`.

`normalize_forecast` accepts a tensor or a tuple whose first element is a
tensor, validates batch size and final channel count 1, takes the final
`pred_len` time steps, and rejects too-short or non-3D outputs.

- [ ] **Step 4: Run adapter tests GREEN**

Expected: all adapter tests pass with no warning.

---

### Task 4: Unified model configuration, construction, and source compatibility fixes

**Files:**
- Create: `models/tslib_factory.py`
- Modify: `models/Koopa.py`
- Modify: `models/TemporalFusionTransformer.py`
- Modify: `models/MICN.py`
- Modify: `models/FiLM.py`
- Modify: other copied model files only when a focused failing test proves the need
- Create: `tests/test_tslib_factory.py`

**Interfaces:**
- Produces: `build_model_config(args, dataset) -> SimpleNamespace`.
- Produces: `compute_koopa_mask(dataset, alpha=0.2) -> Tensor`.
- Produces: `build_power_model(name, args, dataset) -> tuple[nn.Module, SimpleNamespace]`.

- [ ] **Step 1: Write failing factory tests**

Tests must assert these observable behaviors:

```python
def test_factory_sets_univariate_dimensions(base_args, fake_dataset):
    config = build_model_config(base_args, fake_dataset)
    assert (config.enc_in, config.dec_in, config.c_out) == (1, 1, 1)
    assert config.features == "M"
    assert config.data == "power_only"

def test_multipatchformer_factory_uses_32_point_internal_history(base_args, fake_dataset):
    _, config = build_power_model("MultiPatchFormer", base_args, fake_dataset)
    assert config.seq_len == 32

def test_koopa_mask_uses_only_power_channel(fake_multichannel_dataset):
    mask = compute_koopa_mask(fake_multichannel_dataset)
    fake_multichannel_dataset.fail_if_non_power_channel_was_read()
    assert mask.ndim == 1
```

- [ ] **Step 2: Verify RED**

Run `tests/test_tslib_factory.py`; expected failure is missing factory functions.

- [ ] **Step 3: Implement the exhaustive common configuration**

Set all source-required fields explicitly, including:

```python
task_name="long_term_forecast", enc_in=1, dec_in=1, c_out=1,
seq_len=dataset.seq_len, label_len=dataset.seq_len, pred_len=dataset.pred_len,
d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers,
d_layers=args.d_layers, d_ff=args.d_ff, factor=args.factor,
dropout=args.dropout, activation="gelu", embed="fixed", freq="h",
moving_avg=odd_positive(args.moving_avg), top_k=args.top_k,
num_kernels=args.num_kernels, num_class=1, features="M", data="power_only",
distil=True, channel_independence=1, decomp_method="moving_avg", use_norm=1,
down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None,
seg_len=min(8, dataset.seq_len), expand=2, d_conv=4,
p_hidden_dims=[16, 16], p_hidden_layers=2, patch_len=min(args.patch_len, dataset.seq_len),
alpha=0.1, top_p=0.5, pos=1, node_dim=8, gcn_depth=2,
propalpha=0.3, conv_channel=8, skip_channel=8, individual=False,
batch_size=args.batch_size, device=args.device, use_amp=False,
```

Model-specific constructor arguments are centralized in `build_power_model`:
PatchTST and PAttn receive valid patch/stride values; MultiPatchFormer gets
`seq_len=32`; Koopa gets `config.mask_spectrum`; all others use `Model(config)`.

- [ ] **Step 4: Apply narrow source compatibility fixes**

- Koopa: remove `data_provider` import and `_get_mask_spectrum` data loading;
  consume `configs.mask_spectrum` and reject an absent/empty mask.
- TemporalFusionTransformer: add `"power_only": TypePos([], [0])`; ensure its
  attention mask broadcasts as `[1, 1, T, T]` and slice output through the
  common adapter.
- MICN: remove constructor-time hardcoding to `cuda:0`; allocate new tensors on
  the current input/module device.
- FiLM: do not choose a global CUDA device at import time; register CPU-created
  buffers and allocate forward temporaries on the input device.

- [ ] **Step 5: Run factory and earlier tests GREEN**

Run:

```bash
/opt/data/private/penv/time/bin/python -m pytest -p no:cacheprovider \
  tests/test_pure_time_series.py tests/test_tslib_adapter.py tests/test_tslib_factory.py -q
```

---

### Task 5: Integrate all models into the real training runner

**Files:**
- Modify: `run_time_series.py`
- Create: `tests/test_run_time_series_all_models.py`

**Interfaces:**
- Consumes: registry, factory, and adapter from Tasks 1-4.
- CLI behavior: `--list-models` prints the 32 names and exits zero.
- CLI behavior: `--smoke` forces one epoch container with all three step limits set to 1 and batch size 2.
- Preserves: checkpoint, prediction CSV, and metrics JSON output contracts.

- [ ] **Step 1: Write failing runner-contract tests**

Test parser and a tiny fake model/loader without mocking model outputs:

```python
def test_smoke_arguments_limit_every_phase():
    args = parse_args(["--dataset", "skippd_luoyang", "--model", "DLinear", "--smoke"])
    assert (args.epochs, args.max_train_steps, args.max_eval_steps,
            args.max_test_steps, args.batch_size) == (1, 1, 1, 1, 2)

def test_one_step_limit_is_exact():
    loss, steps = _run_epoch(real_tiny_model, three_batch_loader, optimizer,
                             torch.device("cpu"), max_steps=1, model_name="DLinear",
                             label_len=2, pred_len=1)
    assert steps == 1
```

- [ ] **Step 2: Verify RED**

Run the new test file; expected failures are missing `parse_args`, `--smoke`, or new runner signature.

- [ ] **Step 3: Refactor the runner to registry/factory/adapter interfaces**

Replace the fixed four-model construction path. `_run_epoch` and
`_collect_predictions` must call `forward_power_model`, and the loss still
uses `target_mask`. Construct the model after the train dataset so Koopa can
derive its mask. Add model name and exact phase step counts to `metrics.json`.

Expose `parse_args(argv)` for tests. `main` must handle `--list-models` before
requiring dataset/model. `--smoke` overrides any supplied unrestricted limits
to the exact `(epochs=1, train=1, val=1, test=1, batch=2)` contract.

- [ ] **Step 4: Run runner and all focused tests GREEN**

Run all four focused files. Expected: zero failures.

---

### Task 6: Full-run and smoke orchestration plus documentation

**Files:**
- Create: `scripts/run_all_pure_time_series.sh`
- Create: `scripts/smoke_all_pure_time_series.sh`
- Create: compatibility wrappers `scripts/run_time_series_models.sh` and
  `scripts/smoke_time_series_models.sh`
- Modify: `PURE_TIME_SERIES.md`
- Create: `tests/test_pure_time_series_scripts.py`

**Interfaces:**
- Full runner discovers model names from `run_time_series.py --list-models`.
- Smoke runner executes 64 subprocesses and writes `smoke_summary.tsv`.

- [ ] **Step 1: Write a failing executable orchestration test**

Run the scripts (including both compatibility wrappers) against a temporary
fake Python executable that records arguments and timing. Assert 64 unique
invocations, every smoke invocation contains `--smoke`, the model subprocess
environment selects physical GPUs `0` and `1` with 32 calls per card, and no
same-card overlap occurs. This tests script behavior rather than grepping its
text.

- [ ] **Step 2: Verify RED**

Expected: current script makes only eight invocations and lacks the new smoke summary behavior.

- [ ] **Step 3: Implement orchestration**

Both primary scripts use:

```bash
#!/usr/bin/env bash
set -uo pipefail
PYTHON="${PYTHON:-/opt/data/private/penv/time/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="${GPUS:-0 1}"
```

Discover models with `mapfile -t MODELS < <("$PYTHON" "$ROOT_DIR/run_time_series.py" --list-models)`.
Assign the 64 combinations round-robin to one sequential worker queue per
selected physical GPU. Each model child gets
`CUDA_VISIBLE_DEVICES=<physical>` and `--device cuda:0`; the parent waits for
all queues, aggregates failures, and exits nonzero when any combination fails.
The full script uses user-configurable `EPOCHS` and no step limits. The smoke
script passes `--smoke --device cuda:0`; `RESUME=1` reuses only completed rows
with their expected artifacts. The two historical `*_time_series_models.sh`
paths are executable wrappers that `exec` the corresponding primary script.

- [ ] **Step 4: Update documentation**

Document the exact 32 selected and seven excluded models, power column mapping,
one-model CLI, full matrix command, smoke command, GPU mapping, interpreter,
artifacts, and Mamba dependency. Do not claim smoke success until Task 8.

- [ ] **Step 5: Run script behavior and focused test suite GREEN**

Expected: fake-runner test observes exactly 64 smoke commands.

---

### Task 7: Synthetic GPU forward/backward compatibility matrix, including Mamba

**Files:**
- Create: `tests/test_tslib_model_matrix.py`
- Modify: copied model/layer files only to fix reproduced compatibility failures
- Modify: `requirements.txt` if an additional runtime package is required

**Interfaces:**
- Test iterates `SELECTED_MODEL_NAMES` and reports the model name on failure.
- Each case constructs, forwards, computes MSE, backpropagates, and takes one optimizer step.

- [ ] **Step 1: Add a parameterized synthetic matrix test**

Use batch size 2, source history length 16, prediction length 16, `d_model=8`,
`n_heads=2`, one encoder/decoder layer, `d_ff=16`, and CUDA. Build through the
real factory and call the real adapter. Assert output shape `(2, 16, 1)`, finite
loss, and at least one finite parameter gradient before the optimizer step.
Do not skip Mamba or any other selected model.

- [ ] **Step 2: Run the matrix and record the RED model list**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/data/private/penv/time/bin/python -m pytest \
  -p no:cacheprovider tests/test_tslib_model_matrix.py -q
```

Expected: initial failures identify concrete source/environment incompatibilities.

- [ ] **Step 3: Make Mamba available without substituting MambaSimple**

First try importing `mamba_ssm` in the specified interpreter. If absent,
install a release compatible with Python 3.8, PyTorch 2.3.1, and CUDA 11.8 into
that interpreter, using `--no-build-isolation` when source build metadata would
otherwise select a different PyTorch. Record the exact command/version in the
report and `requirements.txt`. Verify `models.Mamba.Model` is the class used.

- [ ] **Step 4: Fix each reproduced model failure minimally**

For every failure, add or tighten a focused regression test, observe it fail,
then patch only the copied model, isolated layer, factory, or adapter needed.
Do not remove a model, mark it skipped, use future targets, or introduce
non-power inputs to obtain green tests.

- [ ] **Step 5: Run the entire synthetic GPU matrix GREEN**

Expected: 32 selected models pass forward, backward, and optimizer step.

---

### Task 8: Real two-dataset 64-combination GPU smoke matrix

**Files:**
- Modify: compatibility code only for failures reproduced on real dataset batches
- Produce: `results_pure_time_series_smoke/smoke_summary.tsv`
- Produce: per-combination `best.pt`, `metrics.json`, and `predictions.csv`

**Interfaces:**
- Consumes the Task 6 smoke script and real configured datasets.
- Evidence requires `train_steps=1`, `val_steps=1`, and `test_steps=1` for every row.

- [ ] **Step 1: Run the smoke matrix concurrently on physical GPUs 0 and 1**

```bash
OUTPUT_ROOT=/opt/data/private/code/FACTS-Luoyang-why/results_pure_time_series_smoke \
GPUS="0 1" scripts/smoke_all_pure_time_series.sh
```

This is 64 bounded runs, not a full epoch.

- [ ] **Step 2: Fix real-batch failures through red-green regression tests**

When a combination fails, preserve its log, create the narrowest automated
test that reproduces the same shape/device/data-contract failure, verify RED,
fix it, verify GREEN, then rerun that combination before resuming the matrix.

- [ ] **Step 3: Validate every result artifact and exact iteration count**

Use a verifier that reads the summary and each `metrics.json`; require 64
unique dataset/model rows, all status `PASS`, all three artifacts present,
`history` length 1, `history[0].train_steps == 1`,
`history[0].val_steps == 1`, and `test_steps == 1`.

- [ ] **Step 4: Run the complete focused test suite once more**

```bash
/opt/data/private/penv/time/bin/python -m pytest -p no:cacheprovider \
  tests/test_pure_time_series.py tests/test_tslib_adapter.py \
  tests/test_tslib_factory.py tests/test_run_time_series_all_models.py \
  tests/test_tslib_model_matrix.py -q
```

Expected: zero failures.

---

### Task 9: Completion audit and user-facing handoff

**Files:**
- Modify: `PURE_TIME_SERIES.md` only if verification exposed inaccurate instructions
- Produce: worker report with requirement-by-requirement evidence

**Interfaces:**
- Consumes all source files, tests, smoke summary, metrics, and artifacts.
- Produces no new behavior.

- [ ] **Step 1: Audit the source inventory**

Compare the 39 source model filenames with selected/excluded sets and prove the
partition is exact: 32 selected + 6 foundation + KANAD.

- [ ] **Step 2: Audit power-only behavior**

Trace both dataset batches through `power_only_batch`, decoder construction,
and model forward; cite tests proving no non-power or future-target input.

- [ ] **Step 3: Audit commands, environment, and smoke scope**

Confirm the scripts use the specified interpreter, physical GPUs 0 and 1,
both datasets, all 32 models, and exact 1/1/1 iteration limits.

- [ ] **Step 4: Audit artifacts and tests**

Confirm 64 passing summary rows and all 192 expected artifacts, then run the
fresh verification commands required by the verification-before-completion
skill. Report any gap as incomplete rather than weakening the acceptance
criteria.
