# P0 Correctness Fixes Design

## Scope

Fix the four confirmed P0 correctness defects without changing unrelated model
or dataset behavior:

1. `models/encoder_img.py` creates an unregistered random Encoder in every
   forward pass.
2. Stanford precomputed train/validation datasets mix subset timestamps with
   full HDF5 arrays while building cloud-event labels.
3. Stanford day-block cross-validation can split one date across train and
   validation.
4. The privileged Teacher reverses a combined history/future image sequence,
   so its terminal representation and spatial KD target refer to the current
   image instead of the future image.

## Registered Image/Weather Encoder

`encoder_img.Model` will construct one Encoder in `__init__` from the configured
input sequence length. The runtime segment count must match the registered
segment count; a mismatch raises a descriptive `ValueError` instead of creating
new parameters.

`MTS_31` and `MTS_31F` will pass branch-specific copies of their configuration:

- Student image and weather branches use `args.seq_len`.
- Teacher image and weather branches use `args.seq_len + args.pred_len`.

This makes the fixed runtime length explicit at the ownership boundary. The
Encoder is included in `named_parameters()`, optimizer parameter groups,
`state_dict()`, device/dtype moves, and train/eval mode changes.

## Stanford Precomputed Subset Alignment

In precomputed mode, cloud-event computation will index `pv_log` and `pv_pred`
with the already selected one-dimensional `sample_indices` before calculating
ramps. The selected power arrays and `sample_times` must have equal lengths; a
defensive error will report any remaining mismatch.

Dynamic-nowcast behavior remains unchanged.

## Date-Atomic Stanford Folds

Day-block cross-validation will:

1. Group sample positions by date.
2. Deterministically shuffle complete date blocks with the existing seed.
3. Divide the shuffled block list among folds.
4. Select whole validation blocks and concatenate all remaining blocks for
   training.
5. Shuffle sample order inside each result with the existing fold seed.

No date can appear in both train and validation. The two index sets must be
disjoint and their union must cover every input sample. Configuration with more
folds than distinct dates raises a clear error instead of producing empty
validation folds.

## Teacher Image Ordering

`MTS_31F` will order the history portion before combining modalities:

- `current_first`: reverse only `x_img_h`.
- `past_first`: retain `x_img_h` as supplied.
- append `x_img_f` after the ordered history.

The resulting encoder input is always
`[oldest, ..., current, future_1, ..., future_P]`. CNN image encoders owned by
the Teacher will be configured as `past_first` so they do not reverse this
already chronological sequence. BOP image inputs receive the same ordering.

The standard Stanford pipeline will stop passing
`--teacher_legacy_image_order`. The option remains available only for explicit
legacy experiments.

## Checkpoint Policy

Old checkpoints are not considered complete after this fix because they do not
contain the newly registered Encoder parameters. Existing Stanford Teacher
checkpoints also learned under the incorrect image ordering. The implementation
will not add permissive loading that silently invents missing Encoder weights.
Teacher and Student models must be retrained for comparable results.

## Tests

Add focused regression tests for:

- Encoder parameters and state-dict keys being registered at construction.
- Two eval forwards producing identical output.
- Runtime length mismatch failing explicitly.
- A precomputed Stanford train/validation subset constructing without a boolean
  indexing mismatch and computing labels from selected rows.
- Date-block train/validation sets having no shared date, being disjoint, and
  covering all samples.
- Teacher ordering producing chronological history followed by future frames,
  with the terminal spatial target derived from the final future frame.
- The standard Stanford pipeline no longer emitting the legacy-order flag.

Run syntax compilation and all runnable unit tests. PyTorch-dependent tests may
remain blocked in the current local interpreter; if so, report that limitation
separately and do not treat collection failures as passing verification.

## Non-Goals

This change does not address the P1/P2 findings, including mask propagation,
DataParallel side-channel state, CPU fallback, KD option validation, resume
state, or repeated test-set evaluation.

## Success Criteria

- No trainable module is created during `encoder_img.Model.forward`.
- Stanford precomputed train and validation datasets can build cloud-event
  labels from their selected rows.
- Stanford day-block folds are date-disjoint.
- Teacher image features and spatial KD targets use the future image as the
  terminal frame.
- Standard pipeline commands use the corrected ordering.
- Focused regression tests pass in an environment with project dependencies.
