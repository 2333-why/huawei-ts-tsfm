# All Pure Time-Series Models Design

## Goal

Integrate every forecasting-capable, non-foundation time-series model from
`/opt/data/private/code/Time-Series-Library` into this project, and train each
model using only historical power to predict future power on
`skippd_luoyang` and `pvod_station00_ylj`.

The implementation must use `/opt/data/private/penv/time/bin/python`. GPU runs
must reserve physical GPU 1 with `CUDA_VISIBLE_DEVICES=1`; inside each process
that device is addressed as `cuda:0`.

## Model Boundary

The source directory contains 39 model modules. The integration selects the
following 32 modules because they implement a forecasting path and are not
time-series foundation models:

- Autoformer
- Crossformer
- DLinear
- ETSformer
- FEDformer
- FiLM
- FreTS
- Informer
- Koopa
- LightTS
- MICN
- MSGNet
- Mamba
- MambaSimple
- MultiPatchFormer
- Nonstationary_Transformer
- PAttn
- PatchTST
- Pyraformer
- Reformer
- SCINet
- SegRNN
- TSMixer
- TemporalFusionTransformer
- TiDE
- TimeFilter
- TimeMixer
- TimeXer
- TimesNet
- Transformer
- WPMixer
- iTransformer

Six source modules are excluded because the source README classifies them as
large/foundation time-series models used for zero-shot evaluation:
Chronos, Chronos2, Sundial, TimeMoE, TiRex, and TimesFM.

KANAD is also excluded. It is a pure time-series model, but its source
explicitly raises `NotImplementedError` for forecasting and only implements
anomaly detection. It therefore cannot participate in supervised future-power
forecasting without inventing a new model that is not present in the source.

The four models already present in this project (DLinear, PatchTST,
iTransformer, and TimesNet) remain part of the 32-model registry. The other 28
model modules are copied into `models/` from the current local source tree.
The source tree is treated as read-only.

## Integration Architecture

### Isolated TSLib dependencies

The current project already has modified modules under `layers/` that are used
by the multimodal models. Replacing those files wholesale risks regressions.
The Time-Series-Library layer modules required by the new models will instead
be copied into an isolated `layers/tslib/` package. Imports in copied model and
layer modules will be changed from `layers.<name>` to
`layers.tslib.<name>`. Existing multimodal code and the four already-working
models keep their present dependencies.

The copied implementation may receive only narrow compatibility fixes needed
for the specified Python, PyTorch, CUDA, sequence lengths, or the unified
power-only interface. Algorithmic rewrites are out of scope.

### Lazy model registry

`models/tslib_registry.py` will be the single authoritative model catalog. A
model specification records its module, constructor behavior, input adapter,
and any model-specific configuration overrides. Imports are lazy so one
optional dependency cannot prevent all other models from being listed or run.

`models/__init__.py` will expose the catalog and a model-class loader without
eagerly importing all 32 implementations. This is required for Mamba, whose
source depends on the optional `mamba_ssm` package.

### Unified power-only contract

Both existing data adapters place normalized target power in channel zero:

- `skippd_luoyang`: source column `final_power`
- `pvod_station00_ylj`: source column `observe_power`

These are the dataset-specific physical columns implementing the logical
target named `power` in this task. The training runner will validate this
mapping and then select only `batch_x[..., :1]` and `batch_y[..., :1]`.
Images, weather, NWP, ramp features, other time-series channels, privileged
Teacher inputs, and real future target values are never forwarded to a model.

For encoder-decoder architectures, the decoder input is constructed only from
the trailing historical power values followed by zeros for the forecast
horizon. It never contains future ground truth. Most models receive `None` for
time marks. Models whose code structurally requires time-feature tensors
receive zero-valued marks; these constants contain no timestamp or exogenous
information.

Every model adapter returns exactly `[batch, pred_len, 1]`. Models that return
history plus forecast are sliced to their final `pred_len` steps. Any other
shape is a hard error rather than being silently broadcast.

### Model-specific compatibility

Compatibility decisions stay in the registry/adapter layer unless the source
model itself has a device or shape bug:

- Autoformer, FEDformer, Informer, MICN, Nonstationary Transformer, and
  Transformer receive a historical-power-plus-zero decoder tensor.
- TemporalFusionTransformer receives a power-only dataset definition and zero
  known-feature marks. Its sole observed variable is power.
- TiDE receives zero temporal features when none are supplied.
- TimeXer runs in univariate `M` mode so no external variable is introduced.
- Koopa receives a dominant-frequency mask computed from training histories'
  power channel. Its copied model must not call the source repository's data
  factory.
- MultiPatchFormer requires a look-back of at least 32 for its largest patch.
  Its adapter left-pads the 16-point power history with the earliest available
  power value and constructs it with `seq_len=32`. No external or future data
  is introduced.
- Patch-based models cap patch length and stride to valid values for the
  configured look-back.
- WPMixer receives the actual device, batch size, and AMP setting through its
  configuration.
- Mamba remains in the 32-model catalog. The implementation will attempt to
  install a version of `mamba_ssm` compatible with the specified environment
  and will verify a real forward/backward step. It must not be silently
  replaced by MambaSimple.

## Training and Evaluation

`run_time_series.py` remains the one-model/one-dataset entry point. It will:

1. seed Python, NumPy, and PyTorch;
2. load the existing causal train, validation, and test splits;
3. construct a model through the registry;
4. train with masked MSE on normalized power;
5. select the best validation checkpoint;
6. evaluate valid test targets in original power units; and
7. write `best.pt`, `metrics.json`, and `predictions.csv`.

The CLI will retain model-specific tuning flags needed for full experiments
while central defaults permit every registered model to start. The batch
runner will enumerate the registry's complete model list rather than maintain
a second hand-written list that can drift.

## Smoke Verification

Smoke verification is deliberately iteration-bounded, not a full epoch over a
dataset. For each of 32 models on each of 2 datasets (64 combinations), it
runs:

- exactly 1 training iteration;
- exactly 1 validation iteration; and
- exactly 1 test iteration.

Each subprocess uses batch size 2, small hidden dimensions where the model
permits them, `/opt/data/private/penv/time/bin/python`, and physical GPU 1.
The orchestration writes one result directory per model/dataset and a summary
that records pass/fail status. It never launches an unrestricted training
epoch during verification.

Before the GPU matrix, focused tests must prove:

- the registry contains exactly the 32 selected models and records all seven
  exclusions with reasons;
- only channel zero is passed through the power-only batch adapter;
- decoder construction contains historical power and future zeros, never
  future targets;
- output normalization produces `[B, pred_len, 1]` or raises a useful error;
- each available model can perform forward, loss, backward, and optimizer step
  on a minimal synthetic batch; and
- the real dataset runner honors one-step train/validation/test limits.

## Error Handling

Missing optional packages produce an error naming the model, missing package,
and installation action. Shape and sequence-length incompatibilities name the
model and received dimensions. A failed smoke combination is recorded in the
summary and blocks completion; it is not reported as a skip or success.

## Acceptance Criteria

The work is complete only when:

1. all 28 new selected model modules exist under `models/`, alongside the four
   existing models;
2. all required isolated layer dependencies exist and import successfully;
3. the registry exposes exactly the 32-model selected set;
4. the runner trains and evaluates every model using only power;
5. the full-training batch command targets physical GPU 1 and the specified
   Python interpreter;
6. focused tests pass; and
7. all 64 minimal GPU smoke combinations complete one train, one validation,
   and one test iteration and produce their expected artifacts.

No multi-epoch or complete-dataset training run is part of this implementation
task.
