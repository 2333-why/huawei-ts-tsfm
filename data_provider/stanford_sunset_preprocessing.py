"""SUNSET-compatible PV preprocessing and processed-data validation.

This module intentionally follows ``data_preprocess_PV.ipynb`` at repository
commit c4c3d0acf953d32f06c9748ab9fdee083c65593c.  In particular, it keeps the
notebook's naive local datetimes, one-second ``interp1d`` interpolation,
10-second ``first`` resampling, 22 manually selected invalid intervals,
greater-than-one-hour missing-record mask, and strict ``PV > 0`` filter.

The released Stanford nowcast HDF5 consumed by FACTS is downstream of that
notebook.  ``validate_sunset_processed_data`` makes this otherwise implicit
data contract explicit at load time.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


SUNSET_START_DATE = dt.datetime(2017, 1, 1)
# Kept exactly as in the notebook.  This is midnight at the start of Dec 31.
SUNSET_END_DATE = dt.datetime(2019, 12, 31)


SUNSET_INVALID_INTERVALS: Tuple[Tuple[dt.datetime, dt.datetime], ...] = (
    (dt.datetime(2017, 3, 10), dt.datetime(2017, 3, 13)),
    (dt.datetime(2017, 3, 13), dt.datetime(2017, 3, 16)),
    (dt.datetime(2017, 3, 18), dt.datetime(2017, 3, 20)),
    (dt.datetime(2017, 3, 31), dt.datetime(2017, 4, 1)),
    (dt.datetime(2017, 4, 5, 8, 34), dt.datetime(2017, 4, 5, 9, 51)),
    (dt.datetime(2017, 5, 6, 18, 50), dt.datetime(2017, 5, 6, 21, 8)),
    (dt.datetime(2017, 7, 17), dt.datetime(2017, 7, 18)),
    (dt.datetime(2017, 9, 8, 8, 59), dt.datetime(2017, 9, 8, 11, 26)),
    (dt.datetime(2017, 10, 9, 10, 3), dt.datetime(2017, 10, 9, 12, 12)),
    (dt.datetime(2017, 11, 5), dt.datetime(2017, 11, 6)),
    (dt.datetime(2017, 12, 30), dt.datetime(2018, 1, 1)),
    (dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 5)),
    (dt.datetime(2018, 3, 11), dt.datetime(2018, 3, 13)),
    (dt.datetime(2018, 4, 2), dt.datetime(2018, 4, 14)),
    (dt.datetime(2018, 6, 13), dt.datetime(2018, 6, 14)),
    (dt.datetime(2018, 8, 22), dt.datetime(2018, 9, 18)),
    (dt.datetime(2018, 9, 25), dt.datetime(2018, 10, 1, 6)),
    (dt.datetime(2018, 10, 16), dt.datetime(2018, 10, 17)),
    (dt.datetime(2018, 11, 4), dt.datetime(2018, 11, 26)),
    (dt.datetime(2019, 1, 28, 10, 29), dt.datetime(2019, 1, 28, 11, 47)),
    (dt.datetime(2019, 5, 22, 14, 18), dt.datetime(2019, 5, 22, 17, 3)),
    (dt.datetime(2019, 10, 27), dt.datetime(2019, 10, 28)),
)


def require_naive_datetimes(raw_times: Iterable[object], context: str = "timestamps") -> None:
    """Reject timezone-aware timestamps; SUNSET stores naive local time."""

    for value in np.asarray(raw_times, dtype=object).reshape(-1):
        timezone = getattr(value, "tzinfo", None)
        if timezone is not None and value.utcoffset() is not None:
            raise ValueError(f"{context} must use SUNSET naive datetimes; found timezone-aware value {value!r}")


def _as_naive_datetime64_seconds(raw_times: Sequence[object], context: str) -> np.ndarray:
    require_naive_datetimes(raw_times, context=context)
    try:
        return np.asarray(raw_times, dtype="datetime64[s]")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} cannot be converted to naive datetime64 seconds") from exc


def interpolate_to_sunset_10_seconds(
    record_times: Sequence[object], record_outputs: Sequence[float]
) -> pd.Series:
    """Reproduce the notebook's 1-second interpolation then 10-second sampling."""

    record_times_object = np.asarray(record_times, dtype=object)
    record_outputs = np.asarray(record_outputs)
    if record_times_object.ndim != 1 or record_outputs.ndim != 1:
        raise ValueError("record_times and record_outputs must be one-dimensional")
    if len(record_times_object) != len(record_outputs):
        raise ValueError("record_times and record_outputs must have the same length")
    if len(record_times_object) < 2:
        raise ValueError("SUNSET interpolation requires at least two raw records")

    record_times_seconds = _as_naive_datetime64_seconds(record_times_object, "raw PV timestamps")
    if np.any(record_times_seconds[1:] <= record_times_seconds[:-1]):
        raise ValueError("raw PV timestamps must be strictly increasing")

    start_time = pd.Timestamp(record_times_object[0]).to_pydatetime()
    elapsed_seconds = (record_times_seconds - record_times_seconds[0]).astype(np.int64)
    interpolation = interp1d(elapsed_seconds, record_outputs)

    # np.arange excludes the final raw timestamp, exactly as the notebook does.
    interpolated_elapsed = np.arange(0, elapsed_seconds[-1], 1, dtype=np.int64)
    interpolated_outputs = interpolation(interpolated_elapsed)
    interpolated_times = pd.DatetimeIndex(
        start_time + interpolated_elapsed * dt.timedelta(seconds=1)
    )
    interpolated_series = pd.Series(interpolated_outputs, index=interpolated_times)

    # Despite the function name in the notebook, this is not an average.
    pv_10_seconds = interpolated_series.resample("10s").first()
    if not np.all(pv_10_seconds.index.second.to_numpy() % 10 == 0):
        raise AssertionError("SUNSET 10-second timestamps are not aligned to 10-second boundaries")
    return pv_10_seconds


def plot_sunset_daily_profiles(pv_10_seconds: pd.Series, output_dir: str) -> None:
    """Save the daily plots used before manually specifying the 22 intervals."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    first_day = pv_10_seconds.index[0].normalize()
    last_day = pv_10_seconds.index[-1].normalize()
    for day in pd.date_range(first_day, last_day, freq="1D"):
        day_end = day + dt.timedelta(days=1)
        values = pv_10_seconds.loc[day:day_end]
        figure, axis = plt.subplots()
        axis.plot(values.index, values.values, "-o", markerfacecolor="None")
        axis.grid(True)
        axis.set_xlabel("Time of the day")
        axis.set_ylabel("panel output (kW)")
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        figure.savefig(destination / f"{day:%Y%m%d}.png", bbox_inches="tight")
        plt.close(figure)


def apply_sunset_filters(
    pv_10_seconds: pd.Series,
    raw_record_times: Sequence[object],
) -> Tuple[pd.Series, dict]:
    """Apply SUNSET's manual, missing-record and strict-positive filters."""

    raw_times = _as_naive_datetime64_seconds(raw_record_times, "raw PV timestamps")
    invalid_dates_mask = pd.Series(False, index=pv_10_seconds.index)
    for start, end in SUNSET_INVALID_INTERVALS:
        # Pandas .loc slicing is inclusive at both ends, matching the notebook.
        invalid_dates_mask.loc[start:end] = True

    record_intervals = raw_times[1:] - raw_times[:-1]
    large_gap_mask = record_intervals > np.timedelta64(1, "h")
    invalid_gap_starts = raw_times[:-1][large_gap_mask]
    invalid_gap_ends = raw_times[1:][large_gap_mask]
    missing_mask = pd.Series(False, index=pv_10_seconds.index)
    for start, end in zip(invalid_gap_starts, invalid_gap_ends):
        missing_mask.loc[pd.Timestamp(start):pd.Timestamp(end)] = True

    positive_mask = pv_10_seconds > 0
    valid = pv_10_seconds[(~invalid_dates_mask) & (~missing_mask) & positive_mask]
    audit = {
        "resampled_10s_count": int(len(pv_10_seconds)),
        "manual_invalid_interval_count": len(SUNSET_INVALID_INTERVALS),
        "manual_invalid_10s_count": int(invalid_dates_mask.sum()),
        "raw_gap_gt_1h_count": int(large_gap_mask.sum()),
        "missing_gap_10s_count": int(missing_mask.sum()),
        "nonpositive_10s_count": int((~positive_mask).sum()),
        "valid_10s_count": int(len(valid)),
    }
    return valid, audit


def clean_sunset_pv_csv(
    csv_path: str,
    plot_daily_dir: Optional[str] = None,
    start_date: dt.datetime = SUNSET_START_DATE,
    end_date: dt.datetime = SUNSET_END_DATE,
) -> Tuple[pd.Series, dict]:
    """Run the original SUNSET PV-cleaning chain on ``pv_output.csv``."""

    raw = pd.read_csv(csv_path)
    required_columns = {"Date", "Huang_E4102_kW"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"raw PV CSV is missing columns: {sorted(missing_columns)}")

    timestamps = raw["Date"].apply(lambda value: dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S"))
    selected = pd.DataFrame(
        {"PV_output_kw": raw["Huang_E4102_kW"].to_numpy()},
        index=pd.DatetimeIndex(timestamps, name="Time"),
    )
    selected = selected[(selected.index >= start_date) & (selected.index <= end_date)]
    record_times = np.asarray([value.to_pydatetime() for value in selected.index], dtype=object)
    record_outputs = selected["PV_output_kw"].to_numpy()

    pv_10_seconds = interpolate_to_sunset_10_seconds(record_times, record_outputs)
    if plot_daily_dir is not None:
        plot_sunset_daily_profiles(pv_10_seconds, plot_daily_dir)
    cleaned, audit = apply_sunset_filters(pv_10_seconds, record_times)
    audit.update(
        {
            "raw_csv_count": int(len(raw)),
            "selected_raw_count": int(len(selected)),
            "interpolated_1s_count": int(
                (np.datetime64(record_times[-1], "s") - np.datetime64(record_times[0], "s"))
                / np.timedelta64(1, "s")
            ),
            "naive_datetime": True,
            "strict_positive_filter": True,
            "large_gap_rule": ">1 hour removed; <=1 hour linearly interpolated",
        }
    )
    return cleaned, audit


def validate_sunset_processed_data(
    raw_times: Sequence[object],
    pv_values: np.ndarray,
    context: str,
) -> dict:
    """Validate invariants visible in a released post-cleaning SUNSET artifact."""

    times = _as_naive_datetime64_seconds(raw_times, f"{context} timestamps")
    values = np.asarray(pv_values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{context} contains non-finite PV values")
    if np.any(values <= 0):
        minimum = float(np.min(values))
        raise ValueError(f"{context} violates SUNSET PV > 0 filtering; minimum={minimum}")

    epoch_seconds = times.astype(np.int64)
    if np.any(epoch_seconds % 10 != 0):
        raise ValueError(f"{context} timestamps are not aligned to SUNSET 10-second samples")

    invalid_count = 0
    for start, end in SUNSET_INVALID_INTERVALS:
        start64 = np.datetime64(start, "s")
        end64 = np.datetime64(end, "s")
        invalid_count += int(np.count_nonzero((times >= start64) & (times <= end64)))
    if invalid_count:
        raise ValueError(f"{context} contains {invalid_count} timestamps in SUNSET manual invalid intervals")

    return {
        "timestamp_count": int(len(times)),
        "pv_value_count": int(values.size),
        "pv_min": float(np.min(values)),
        "pv_max": float(np.max(values)),
        "naive_datetime": True,
        "ten_second_aligned": True,
        "strictly_positive": True,
        "manual_invalid_overlap": 0,
    }
