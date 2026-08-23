# P0 Correctness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the four confirmed P0 correctness failures in the registered multimodal encoders and Stanford data/Teacher pipelines.

**Architecture:** Make each image/weather Crossformer encoder own one fixed, registered Encoder whose sequence length is set by its Student or Teacher owner. Keep Stanford samples date-atomic during cross-validation, align precomputed power arrays to selected rows, and construct Teacher image sequences chronologically before encoding.

**Tech Stack:** Python 3, PyTorch, NumPy, h5py, pandas, pytest.

## Global Constraints

- Do not change P1/P2 behavior in this implementation.
- Old checkpoints without registered Encoder parameters are intentionally incomplete and require retraining.
- Correct Teacher image order is `[oldest, ..., current, future_1, ..., future_P]`.
- Use deterministic Stanford fold seeds already present in the loader.
- The checkout has no `.git`; replace commit steps with explicit verification checkpoints.

> **Execution note:** The listed pytest commands were executed. In this
> workspace, Torch-dependent modules are blocked during collection/import by a
> missing or broken `torch` runtime; checked steps record execution, not a
> Torch runtime pass. See `.superpowers/sdd/2026-08-14-p0-correctness-fixes/task-4-report.md`.

---

### Task 1: Date-Atomic Stanford Splits and Precomputed Alignment

**Files:**
- Modify: `data_provider/data_loader_stanford.py:356-423`
- Create: `tests/test_stanford_data_loader_p0.py`

**Interfaces:**
- Consumes: `StanfordSolarForecastDataset._day_block_cv_indices(sample_times)` and selected one-dimensional `sample_indices` in precomputed mode.
- Produces: disjoint train/validation index arrays and subset-aligned cloud-event labels.

- [x] **Step 1: Write failing date-atomic and subset-alignment tests**

```python
def test_day_block_fold_keeps_each_date_atomic():
    dataset = object.__new__(StanfordSolarForecastDataset)
    dataset.num_fold = 3
    dataset.fold_index = 1
    counts = [5, 7, 4, 8]
    times = np.concatenate([
        np.arange(count, dtype="timedelta64[m]")
        + np.datetime64(f"2019-01-{day:02d}")
        for day, count in enumerate(counts, start=1)
    ])
    train, validation = dataset._day_block_cv_indices(times)
    assert set(times[train].astype("datetime64[D]")).isdisjoint(
        set(times[validation].astype("datetime64[D]")))
    assert set(train).isdisjoint(set(validation))
    assert sorted(np.concatenate([train, validation]).tolist()) == list(range(len(times)))


def test_precomputed_cloud_events_use_selected_rows(tmp_path):
    path = tmp_path / "forecast.h5"
    selected = np.asarray([0, 2, 4, 6, 8, 9])
    with h5py.File(path, "w") as handle:
        group = handle.create_group("trainval")
        history = np.ones((10, 2), dtype=np.float32)
        target = np.ones(10, dtype=np.float32)
        target[selected] = 7.0
        group.create_dataset("pv_log", data=history)
        group.create_dataset("pv_pred", data=target)
    dataset = object.__new__(StanfordSolarForecastDataset)
    dataset.precomputed_forecast = True
    dataset.data_path = str(path)
    dataset.group_name = "trainval"
    dataset.history_order = "current_first"
    dataset.sample_indices = selected
    dataset.sample_times = np.asarray(
        [np.datetime64("2019-01-01") + np.timedelta64(i, "m") for i in range(6)])
    mask, dates = dataset._build_cloud_event_day_mask()
    assert mask.tolist() == [True] * 6
    assert dates.tolist() == [np.datetime64("2019-01-01")]
```

- [x] **Step 2: Run the focused tests and confirm the original failures**

Run: `python -m pytest tests/test_stanford_data_loader_p0.py -q`

Expected before implementation: the date sets overlap and the precomputed test raises a boolean-index length `IndexError`.

- [x] **Step 3: Split shuffled date blocks before concatenating samples**

```python
dates = sample_times.astype("datetime64[D]")
blocks = [np.flatnonzero(dates == value) for value in np.unique(dates)]
if self.num_fold > len(blocks):
    raise ValueError(
        f"num_fold={self.num_fold} exceeds distinct date count={len(blocks)}")
block_rng = np.random.RandomState(1)
block_rng.shuffle(blocks)
folds = np.array_split(np.arange(len(blocks)), self.num_fold)
validation_block_ids = set(folds[fold_index].tolist())
train_positions = np.concatenate([
    block for index, block in enumerate(blocks)
    if index not in validation_block_ids
])
val_positions = np.concatenate([
    blocks[index] for index in folds[fold_index]
])
```

Shuffle both returned arrays with `np.random.RandomState(fold_index)` after the complete-date selection.

- [x] **Step 4: Align precomputed HDF5 rows to the selected indices**

```python
selected = np.asarray(self.sample_indices, dtype=np.int64)
history = np.asarray(group["pv_log"])[selected]
target = np.asarray(group["pv_pred"]).reshape(-1)[selected]
if len(history) != len(self.sample_times) or len(target) != len(self.sample_times):
    raise ValueError("precomputed power rows do not align with selected sample times")
```

- [x] **Step 5: Re-run the focused tests**

Run: `python -m pytest tests/test_stanford_data_loader_p0.py -q`

Expected: both tests pass.

- [x] **Step 6: Verification checkpoint**

Record the focused test output. No commit is possible because this checkout has no `.git` directory.

---

### Task 2: Register the Image/Weather Encoder

**Files:**
- Modify: `models/encoder_img.py:39-116`
- Modify: `models/MTS_31.py:1-269`
- Modify: `models/MTS_31F.py:1-75`
- Create: `tests/test_encoder_img_registration.py`

**Interfaces:**
- Consumes: a copied branch config whose `seq_len` equals the branch's actual runtime input length.
- Produces: `encoder_img.Model.encoder: Encoder`, `registered_seg_num: int`, and deterministic eval output.

- [x] **Step 1: Write failing registration, determinism, and mismatch tests**

```python
def _config(seq_len=16):
    return SimpleNamespace(
        enc_in=1, pred_len=1, seq_len=seq_len, task_name="long_term_forecast",
        d_model=4, e_layers=1, n_heads=1, d_ff=8, dropout=0.0, factor=1)


def test_encoder_is_registered_and_eval_is_deterministic():
    torch.manual_seed(7)
    model = encoder_img.Model(_config()).eval()
    assert any(name.startswith("encoder.") for name, _ in model.named_parameters())
    assert any(name.startswith("encoder.") for name in model.state_dict())
    inputs = torch.randn(2, 16, 1)
    first = model(inputs)
    second = model(inputs)
    torch.testing.assert_close(first, second)


def test_encoder_rejects_an_unregistered_runtime_length():
    model = encoder_img.Model(_config(seq_len=16))
    with pytest.raises(ValueError, match="configured segment count"):
        model(torch.randn(2, 25, 1))
```

- [x] **Step 2: Run the focused tests and confirm failure**

Run: `python -m pytest tests/test_encoder_img_registration.py -q`

Expected before implementation: no `encoder.*` parameter exists, repeated outputs differ, and a mismatched length does not raise the specified error.

- [x] **Step 3: Register one fixed Encoder in `__init__`**

```python
self.configured_seq_len = int(configs.seq_len)
self.registered_seg_num = ceil(self.configured_seq_len / self.seg_len)
self.encoder = self._create_encoder(self.registered_seg_num)
```

In `forecast`, replace `_create_encoder()` and `.cuda()` with:

```python
if seg_num != self.registered_seg_num:
    raise ValueError(
        "encoder_img received segment count "
        f"{seg_num}, but configured segment count is {self.registered_seg_num} "
        f"for seq_len={self.configured_seq_len}")
enc_out, attns = self.encoder(x_enc)
```

- [x] **Step 4: Give Student branches their actual history length**

In `MTS_31.Model.__init__`, shallow-copy `args_img` and `args_weather`, set each copied `seq_len = args.seq_len`, and construct BOP/weather encoders from those copies. CNN image encoders may consume the copied image config too.

```python
student_img_args = copy.copy(args_img)
student_img_args.seq_len = int(args.seq_len)
student_weather_args = copy.copy(args_weather)
student_weather_args.seq_len = int(args.seq_len)
```

- [x] **Step 5: Give Teacher branches their combined history/future length**

In `MTS_31F.Model.__init__`, shallow-copy both configs and set:

```python
teacher_seq_len = int(args.seq_len) + int(args.pred_len)
teacher_img_args.seq_len = teacher_seq_len
teacher_weather_args.seq_len = teacher_seq_len
```

Construct `encoder_img` branches from these copies.

- [x] **Step 6: Re-run focused tests**

Run: `python -m pytest tests/test_encoder_img_registration.py -q`

Expected: all tests pass and state dicts contain `encoder.encode_blocks.*` keys.

- [x] **Step 7: Verification checkpoint**

Record focused results and confirm `rg -n "_create_encoder\(seg_num\)|encoder = encoder\.cuda" models/encoder_img.py` returns no forward-time construction.

---

### Task 3: Make Teacher Image Ordering Chronological

**Files:**
- Modify: `models/MTS_31F.py:50-126`
- Modify: `scripts/pipeline.py:189-242`
- Create: `tests/test_teacher_image_ordering.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `MTS_31F.Model._ordered_teacher_images(x_img_h, x_img_f)` returning a chronological tensor.
- Consumes: `history_order` captured from the original image config before configuring the Teacher CNN encoder as `past_first`.

- [x] **Step 1: Write failing ordering tests**

```python
@pytest.mark.parametrize(
    ("history_order", "history"),
    [
        ("current_first", [3.0, 2.0, 1.0]),
        ("past_first", [1.0, 2.0, 3.0]),
    ],
)
def test_teacher_orders_history_before_appending_future(history_order, history):
    model = object.__new__(MTS_31F.Model)
    model.image_history_order = history_order
    x_h = torch.tensor(history).view(1, 3, 1, 1)
    x_f = torch.tensor([4.0, 5.0]).view(1, 2, 1, 1)
    ordered = model._ordered_teacher_images(x_h, x_f)
    assert ordered.flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
```

Extend `tests/test_pipeline.py` so every standard Student command asserts:

```python
assert "--teacher_legacy_image_order" not in command
```

- [x] **Step 2: Run the ordering and pipeline tests and confirm failure**

Run: `python -m pytest tests/test_teacher_image_ordering.py tests/test_pipeline.py -q`

Expected before implementation: the helper is absent and the pipeline command still contains the legacy flag.

- [x] **Step 3: Implement history-only reordering**

```python
def _ordered_teacher_images(self, x_img_h, x_img_f):
    if self.image_history_order == "current_first":
        x_img_h = torch.flip(x_img_h, dims=[1])
    elif self.image_history_order != "past_first":
        raise ValueError(
            f"unsupported image history_order: {self.image_history_order}")
    return torch.cat([x_img_h, x_img_f], dim=1)
```

Store `self.image_history_order` from `args_img`. For Teacher CNN encoders, set the copied `teacher_img_args.history_order = "past_first"` because the combined tensor is already chronological. Replace the direct `torch.cat` in `forward` with this helper.

- [x] **Step 4: Remove the legacy flag from standard pipeline commands**

Delete `"--teacher_legacy_image_order"` from `scripts.pipeline.student_command`. Keep the CLI option and explicit KD compatibility branch unchanged.

- [x] **Step 5: Verify terminal future-frame semantics**

Add a lightweight fake spatial encoder test in which `encode_spatial_sequence` returns the scalar input as its spatial map. Run `MTS_31F.forward` and assert `last_image_spatial_target` equals the final future frame value `5.0`.

- [x] **Step 6: Re-run focused tests**

Run: `python -m pytest tests/test_teacher_image_ordering.py tests/test_pipeline.py -q`

Expected: ordering, future-target, and command-contract tests pass.

- [x] **Step 7: Verification checkpoint**

Record test output. Confirm the explicit legacy CLI option still exists while standard generated commands no longer use it.

---

### Task 4: Integrated Verification and Documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-08-14-p0-correctness-fixes-design.md` only if implementation details required a documented correction.
- Modify: `docs/superpowers/plans/2026-08-14-p0-correctness-fixes.md` by checking completed steps.

**Interfaces:**
- Consumes: all code and focused tests from Tasks 1-3.
- Produces: final verification evidence and a concise migration summary.

- [x] **Step 1: Run syntax compilation**

Run: `python -m compileall -q data_provider models scripts tests`

Expected: exit code 0.

- [x] **Step 2: Run all focused P0 tests**

Run: `python -m pytest tests/test_stanford_data_loader_p0.py tests/test_encoder_img_registration.py tests/test_teacher_image_ordering.py tests/test_pipeline.py -q`

Expected: all dependency-available tests pass.

- [x] **Step 3: Run the full test suite**

Run: `python -m pytest -q`

Expected in a complete environment: all tests pass. If the current interpreter still lacks PyTorch or `reformer_pytorch`, retain the exact collection error as an environment limitation and separately report the dependency-free results.

- [x] **Step 4: Inspect source invariants**

Run:

```bash
rg -n "_create_encoder\(seg_num\)|encoder = encoder\.cuda" models/encoder_img.py
rg -n "teacher_legacy_image_order" scripts/pipeline.py
```

Expected: neither prohibited production pattern is present.

- [x] **Step 5: Review the complete patch against scope**

Confirm only P0-related source, focused tests, and design/plan documents changed. Confirm no permissive old-checkpoint loading was introduced.

- [x] **Step 6: Final verification checkpoint**

Summarize each P0 root cause, implementation, tests, checkpoint migration impact, and any unexecuted verification caused by missing local dependencies.
