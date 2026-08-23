# Folsom compatibility conversion

`scripts/build_folsom_parquet.py` creates two Parquet datasets from
`Folsom_combined_no_zero.csv`. They deliberately keep the existing adapter
names (`LuoyangParquet` and `YLJParquet`) so the model and batch contracts stay
unchanged.

## Luoyang-compatible output

`luoyang/folsom_luoyang.parquet` has one row per 5-minute aggregate. The
required columns are:

- `timestamp`
- `final_power` (`ghi`)
- `dni`, `dhi`, `air_temp`, `relhum`, `press`, `windsp`, `winddir`, `max_windsp`
- `asi_path` and `asi_path_timestamps`

The last two columns contain JSON-serialized lists. Each path is relative to
the configured `image_root`, and each timestamp is the original image capture
time. Missing image intervals are represented by `[]`. The conversion exports
one 32x32 JPG from the image cache per output interval, so
`history_aligned_ffill` can use only an image captured no later than the
history slot.

The generated config is 16 history points, 48 forecast points, 5 minutes per
step, and a 240-minute horizon. The eight exogenous columns are privileged
future time-series channels for the Teacher and are hidden from the Student by
the existing causal pipeline settings.

## YLJ-compatible output

`ylj/folsom_ylj.parquet` has one row per 15-minute aggregate. It contains:

- `timestamp`
- `observe_power`
- seven `*_observe` columns mapped from Folsom GHI, temperature, wind speed,
  wind direction, relative humidity, pressure, and maximum wind speed
- fourteen forecast columns: `*_forecast_1day` and `*_forecast_4hour`

The 1-day forecast at row `u` is the observed value at `u - 24h`; the 4-hour
forecast is the observed value at `u - 4h`. These are causal stand-ins for
forecast products because the available Folsom source has observations but no
forecast files. The loader still applies its configured alternative,
persistence, and training-mean fallbacks.

## Validation

Run the format inspector against either generated config:

```bash
python scripts/inspect_dataset_format.py /path/to/folsom_luoyang.json --pretty
python scripts/inspect_dataset_format.py /path/to/folsom_ylj.yaml --pretty
```

The inspector reports required/missing columns, timestamp continuity, numeric
missingness, and image-list shape. The Dataset adapters additionally reject
samples whose history or forecast window crosses an input-data gap.
