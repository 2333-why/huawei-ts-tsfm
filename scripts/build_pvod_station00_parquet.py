#!/usr/bin/env python3
"""Convert the PVOD station00 CSV to the configuration-driven YLJ format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = Path("/opt/data/private/data/PVOD/station00.csv")
DEFAULT_OUTPUT = Path("/opt/data/private/data/PVOD/station00_ylj.parquet")
SAMPLING_INTERVAL = pd.Timedelta(minutes=15)

HISTORY_COLUMNS = [
    "observe_power",
    "lmd_totalirrad",
    "lmd_diffuseirrad",
    "lmd_temperature",
    "lmd_pressure",
    "lmd_winddirection",
    "lmd_windspeed",
]
FORECAST_COLUMNS = [
    "nwp_globalirrad",
    "nwp_directirrad",
    "nwp_temperature",
    "nwp_humidity",
    "nwp_windspeed",
    "nwp_winddirection",
    "nwp_pressure",
]
SOURCE_COLUMNS = [
    "date_time",
    "nwp_globalirrad",
    "nwp_directirrad",
    "nwp_temperature",
    "nwp_humidity",
    "nwp_windspeed",
    "nwp_winddirection",
    "nwp_pressure",
    "lmd_totalirrad",
    "lmd_diffuseirrad",
    "lmd_temperature",
    "lmd_pressure",
    "lmd_winddirection",
    "lmd_windspeed",
    "power",
]
OUTPUT_COLUMNS = ["timestamp", *HISTORY_COLUMNS, *FORECAST_COLUMNS]


def _read_source(source: str | Path) -> pd.DataFrame:
    """Read and strictly validate the station CSV before aggregation."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"PVOD station00 CSV not found: {source}")

    frame = pd.read_csv(source)
    missing = [name for name in SOURCE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"PVOD source is missing required columns: {missing}")

    timestamp = pd.to_datetime(frame["date_time"], errors="coerce")
    if timestamp.isna().any():
        count = int(timestamp.isna().sum())
        raise ValueError(f"PVOD source contains {count} invalid timestamps")
    if isinstance(timestamp.dtype, pd.DatetimeTZDtype):
        raise ValueError("PVOD source timestamps must be naive UTC")
    try:
        off_grid = (
            (timestamp.dt.minute % 15 != 0)
            | (timestamp.dt.second != 0)
            | (timestamp.dt.microsecond != 0)
            | (timestamp.dt.nanosecond != 0)
        )
    except AttributeError as exc:
        raise ValueError("PVOD source timestamps must be naive UTC") from exc
    if off_grid.any():
        raise ValueError("PVOD source timestamps must lie on a 15-minute grid")

    numeric_columns = [name for name in SOURCE_COLUMNS if name != "date_time"]
    for name in numeric_columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        nonfinite = values.isna() | values.isin([float("inf"), float("-inf")])
        if nonfinite.any():
            count = int(nonfinite.sum())
            raise ValueError(f"PVOD source column {name!r} has {count} non-finite values")
        frame[name] = values

    frame["timestamp"] = timestamp
    return frame[["timestamp", *numeric_columns]]


def _aggregate(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse duplicate timestamps by numeric mean and retain sorted rows."""
    numeric_columns = [name for name in SOURCE_COLUMNS if name != "date_time"]
    grouped = (
        frame.groupby("timestamp", as_index=False, sort=True)[numeric_columns]
        .mean()
    )
    duplicate_rows_collapsed = int(len(frame) - len(grouped))
    output = grouped.rename(columns={"power": "observe_power"})
    output = output[OUTPUT_COLUMNS]
    if output.empty:
        raise ValueError("PVOD source contains no rows")
    return output, duplicate_rows_collapsed


def _gap_count(frame: pd.DataFrame) -> int:
    span_steps = (frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]) / SAMPLING_INTERVAL
    if span_steps != int(span_steps):
        raise ValueError("PVOD output range is not aligned to a 15-minute grid")
    expected_rows = int(span_steps) + 1
    return expected_rows - len(frame)


def build_parquet(
    source: str | Path = DEFAULT_SOURCE,
    output: str | Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Build the station00 YLJ Parquet file and return its conversion summary."""
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    raw = _read_source(source_path)
    converted, duplicate_rows_collapsed = _aggregate(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(output_path, index=False, engine="pyarrow")
    timestamps = converted["timestamp"]
    summary: dict[str, object] = {
        "source": str(source_path),
        "source_rows": int(len(raw)),
        "output": str(output_path),
        "output_rows": int(len(converted)),
        "duplicate_rows_collapsed": duplicate_rows_collapsed,
        "gap_count": _gap_count(converted),
        "range": {
            "start": timestamps.iloc[0].isoformat(sep=" "),
            "end": timestamps.iloc[-1].isoformat(sep=" "),
        },
    }
    return summary


# Keep the conventional builder entry point available to callers.
build = build_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_parquet(args.source, args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
