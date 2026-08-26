# Classical Forecast Baselines Design

## Goal

Add four reproducible photovoltaic forecasting baselines to the power-only
benchmark while preserving the existing eight neural models and their training
behavior. Baselines and neural models share the same CLI selection point,
prediction/metric formats, completion manifest, and batch summary.

## Public catalog

The neural catalog remains exactly:

`TSMixer`, `Pyraformer`, `SegRNN`, `Transformer`, `LightTS`, `Crossformer`,
`FreTS`, `MICN`.

A separate baseline catalog contains, in this order:

`Persistence`, `SmartPersistence`, `SeasonalPersistence`, `Climatology`.

`--list-models` continues to print only neural models. `--list-baselines`
prints only baselines. The existing `--model` option accepts either catalog so
single-method invocation and output paths stay compatible.

## Baseline semantics

All baseline predictions use normalized power and are clipped to `[0, 1]`,
which maps to `[0, rated_power]` for both configured datasets.

- `Persistence`: repeat the most recent finite historical power for every
  forecast step.
- `SmartPersistence`: persist the current clear-sky power index. Compute
  clear-sky GHI/DNI/DHI with pvlib's Ineichen model, transpose irradiance onto
  the configured array plane, and forecast
  `P(t+h) = P(t) * POA_clear(t+h) / POA_clear(t)`. If current clear-sky POA is
  at most 1 W/m2, return zero for the horizon to avoid an unstable ratio.
- `SeasonalPersistence`: for each target timestamp, use the observed power at
  exactly 24 hours earlier. A missing timestamp or non-finite value falls back
  to ordinary persistence for that forecast step. This remains causal for the
  supported four-hour maximum horizon.
- `Climatology`: fit only on raw observations strictly inside the training
  interval. For each target timestamp, average finite normalized power at the
  same minute of day within a circular +/-15-day day-of-year window. If that
  bucket is empty, use the finite training mean.

## Site configuration

Each dataset config gains a required `site` mapping:

| Dataset | Latitude | Longitude | Timezone | Tilt | Azimuth |
| --- | ---: | ---: | --- | ---: | ---: |
| SKIPP'D Stanford | 37.427 | -122.174 | America/Los_Angeles | 37 | 195 |
| PVOD station00 | 38.04778 | 114.95139 | Asia/Shanghai | 33 | 180 |

Timestamps in the source files are naive local wall-clock values. Clear-sky
calculation localizes them with the configured IANA timezone. Array azimuth uses
pvlib's convention: clockwise degrees from north.

The implementation depends on `pvlib>=0.10.5,<0.12`, compatible with the
repository's Python 3.8 environment.

## Architecture

Keep neural and baseline construction separate. `models/tslib_registry.py` and
`models/tslib_factory.py` continue to own only trainable models. A focused
`baselines/` package owns the baseline registry, clear-sky calculation, fitting,
prediction, and serializable state.

`run_time_series.py` dispatches after constructing the three datasets:

- neural name: retain the current Adam/epoch/best-validation-checkpoint path;
- baseline name: construct and fit the baseline, skip optimizer and epochs,
  collect test predictions through a baseline-specific collector, and save a
  metadata/state checkpoint to `best.pt`.

Baseline metrics explicitly record `method_type="baseline"`,
`training_skipped=true`, `best_epoch=null`, `best_val_loss=null`, and phase
steps of train=0, val=0, test=the number of collected test batches. Neural
payload values remain unchanged apart from an added `method_type="neural"` and
`training_skipped=false` discriminator.

The completion manifest keeps its existing columns. Baseline manifests record
epochs=0, train_steps=0, and val_steps=0; neural manifests retain existing
positive-step behavior. Resume validation distinguishes the two registered
catalogs and validates those semantics without weakening artifact hash checks.

## Batch execution

The full matrix becomes four settings x two datasets x twelve methods = 96
tasks. The 64 neural tasks remain divided evenly between two sequential GPU
queues. The 32 baseline tasks run on CPU and write `launch_gpu=cpu`; they never
reserve or initialize CUDA. All tasks share the existing summary schema and
result directory layout.

Smoke mode limits each neural phase to one batch. For baselines it limits test
collection to one batch; fitting `Climatology` still reads the complete training
interval because that fit defines the method rather than model optimization.

## Error handling

- Config loading rejects a missing or malformed `site` mapping and non-finite
  coordinates/angles.
- Unknown baseline names fail with the complete baseline-name list.
- Baseline history and issue-time batch dimensions must agree.
- Baseline output must satisfy the existing finite `[B, pred_len, 1]` contract.
- Empty test loaders and zero valid test targets retain the runner's current
  failure behavior and do not produce a successful completion manifest.

## Testing and acceptance

1. Unit tests prove all four formulas with hand-derived synthetic values.
2. A real pvlib test proves POA is zero at night and positive near local noon
   for each configured site.
3. Seasonal fallback and climatology's training-only fit are explicitly tested.
4. CLI tests prove catalog separation and union selection behavior.
5. Runner tests prove baselines skip model construction, Adam, and training;
   save all four artifacts; and report truthful zero train/validation steps.
6. Script behavior tests prove 96 unique tasks, 64 neural GPU tasks, 32 baseline
   CPU tasks, and correct smoke/resume behavior.
7. Existing neural model shape tests and the full retained suite remain green.

