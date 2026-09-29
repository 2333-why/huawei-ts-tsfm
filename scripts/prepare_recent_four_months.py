#!/usr/bin/env python3
"""Inspect updated Luoyang data and build its last-four-calendar-month config."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.power_only import load_power_only_config  # noqa: E402


DEFAULT_DATASETS = {
    "luoyang": {
        "config": REPO_ROOT / "configs/datasets/skippd_luoyang.json",
        "parquet": Path(
            "/data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/"
            "Luoyang-Unified_format-V1-with_DNI_DHI.parquet"
        ),
    },
}


def _timestamp_text(value: pd.Timestamp) -> str:
    return value.isoformat(sep=" ")


def prepare_dataset(
    name: str,
    base_config_path: Path,
    parquet_path: Path,
    output_path: Path,
    *,
    strict_grid: bool = False,
) -> dict[str, Any]:
    config = copy.deepcopy(load_power_only_config(base_config_path))
    parquet_path = parquet_path.expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(f"{name}: Parquet file not found: {parquet_path}")

    timestamp_column = config["fields"]["timestamp"]
    target_column = config["fields"]["target"]
    frame = pd.read_parquet(parquet_path, columns=[timestamp_column, target_column])
    if frame.empty:
        raise ValueError(f"{name}: Parquet file is empty")

    parsed = pd.to_datetime(frame[timestamp_column], errors="coerce")
    invalid_timestamps = int(parsed.isna().sum())
    if invalid_timestamps:
        raise ValueError(f"{name}: timestamp contains {invalid_timestamps} invalid values")
    timestamps = pd.DatetimeIndex(parsed)
    if timestamps.tz is not None:
        timestamps = timestamps.tz_localize(None)
    duplicate_timestamps = int(timestamps.duplicated().sum())
    if duplicate_timestamps:
        raise ValueError(
            f"{name}: timestamp contains {duplicate_timestamps} duplicate values"
        )
    timestamps = timestamps.sort_values()

    step_minutes = int(config["time"]["sampling_interval_minutes"])
    if step_minutes <= 0:
        raise ValueError(f"{name}: sampling_interval_minutes must be positive")
    step = pd.Timedelta(minutes=step_minutes)
    data_start = timestamps.min()
    data_end = timestamps.max()
    end_exclusive = data_end + step
    train_start = end_exclusive - pd.DateOffset(months=4)
    test_start = end_exclusive - pd.DateOffset(months=2)
    if data_start > train_start:
        raise ValueError(
            f"{name}: only covers {data_start} through {data_end}; "
            "four calendar months are required"
        )

    selected = timestamps[(timestamps >= train_start) & (timestamps < end_exclusive)]
    expected = pd.date_range(train_start, end_exclusive, freq=step, inclusive="left")
    missing = expected.difference(selected)
    off_grid = selected.difference(expected)
    if strict_grid and (len(missing) or len(off_grid)):
        raise ValueError(
            f"{name}: selected range has {len(missing)} missing and "
            f"{len(off_grid)} off-grid timestamps"
        )

    numeric_target = pd.to_numeric(frame[target_column], errors="coerce")
    timestamp_series = pd.Series(pd.DatetimeIndex(parsed).tz_localize(None))
    finite = np.isfinite(numeric_target.to_numpy(dtype=np.float64))
    train_mask = (timestamp_series >= train_start) & (timestamp_series < test_start)
    test_mask = (timestamp_series >= test_start) & (timestamp_series < end_exclusive)
    train_rows = int(train_mask.sum())
    test_rows = int(test_mask.sum())
    finite_train = int((finite & train_mask.to_numpy()).sum())
    finite_test = int((finite & test_mask.to_numpy()).sum())
    if not train_rows or not test_rows or not finite_train or not finite_test:
        raise ValueError(f"{name}: selected train or test interval has no usable targets")

    config["paths"]["parquet_file"] = str(parquet_path)
    config["time"]["train_start_timestamp"] = _timestamp_text(train_start)
    config["time"]["test_start_timestamp"] = _timestamp_text(test_start)
    config["time"]["test_end_exclusive_timestamp"] = _timestamp_text(end_exclusive)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)

    return {
        "dataset": name,
        "parquet_file": str(parquet_path),
        "config_file": str(output_path),
        "rows": int(len(frame)),
        "full_data_range": [_timestamp_text(data_start), _timestamp_text(data_end)],
        "selected_range": [_timestamp_text(train_start), _timestamp_text(end_exclusive)],
        "train_range": [_timestamp_text(train_start), _timestamp_text(test_start)],
        "test_range": [_timestamp_text(test_start), _timestamp_text(end_exclusive)],
        "train_rows": train_rows,
        "test_rows": test_rows,
        "finite_train_targets": finite_train,
        "finite_test_targets": finite_test,
        "missing_selected_timestamps": int(len(missing)),
        "off_grid_selected_timestamps": int(len(off_grid)),
        "sampling_interval_minutes": step_minutes,
        "status": "PASS",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--luoyang-parquet", type=Path, default=DEFAULT_DATASETS["luoyang"]["parquet"]
    )
    parser.add_argument(
        "--luoyang-config", type=Path, default=DEFAULT_DATASETS["luoyang"]["config"]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "generated_configs/recent_four_months",
    )
    parser.add_argument(
        "--strict-grid",
        action="store_true",
        help="fail if any timestamp is missing or does not match the configured grid",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    try:
        results = [
            prepare_dataset(
                "luoyang",
                args.luoyang_config,
                args.luoyang_parquet,
                output_dir / "luoyang_recent_four_months.json",
                strict_grid=args.strict_grid,
            )
        ]
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"RANGE CHECK FAIL: {exc}")
        return 2

    report = {
        "status": "PASS",
        "policy": "Luoyang latest 4 calendar months: 2 train + 2 test; YLJ unchanged",
        "datasets": results,
    }
    report_path = output_dir / "data_range_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(
                f"PASS {result['dataset']}: full={result['full_data_range']} "
                f"train={result['train_range']} test={result['test_range']} "
                f"missing={result['missing_selected_timestamps']} "
                f"config={result['config_file']}"
            )
        print(f"RANGE CHECK PASS: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
