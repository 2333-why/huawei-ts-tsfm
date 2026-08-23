#!/usr/bin/env python3
"""Inspect the Parquet schema expected by the Luoyang/YLJ adapters.

The script intentionally inspects the configured columns without constructing
the Dataset.  This makes it useful when the source data is not available on
the current machine, and it reports the exact missing columns separately from
loader-level validation errors.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _as_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    return [value]


def _path_from_config(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.parent / path


def _column_report(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in columns:
        values = frame[name]
        numeric = pd.to_numeric(values, errors="coerce")
        item: dict[str, Any] = {
            "dtype": str(values.dtype),
            "missing": int(values.isna().sum()),
            "non_numeric": int((values.notna() & numeric.isna()).sum()),
        }
        finite = numeric[np.isfinite(numeric.to_numpy(dtype=np.float64, na_value=np.nan))]
        if len(finite):
            item["min"] = float(finite.min())
            item["max"] = float(finite.max())
        report[name] = item
    return report


def _list_report(frame: pd.DataFrame, path_column: str, timestamp_column: str) -> dict[str, Any]:
    lengths: list[int] = []
    malformed = 0
    mismatched = 0
    for paths, stamps in zip(frame[path_column], frame[timestamp_column]):
        path_values = _as_list(paths)
        stamp_values = _as_list(stamps)
        lengths.append(len(path_values))
        if isinstance(paths, str) and paths.strip() and path_values == [paths.strip()]:
            malformed += 1
        if len(path_values) != len(stamp_values):
            mismatched += 1
    return {
        "path_column": path_column,
        "timestamp_column": timestamp_column,
        "rows_with_entries": int(sum(length > 0 for length in lengths)),
        "max_entries_per_row": max(lengths, default=0),
        "average_entries_per_row": float(np.mean(lengths)) if lengths else 0.0,
        "malformed_rows": malformed,
        "length_mismatch_rows": mismatched,
    }


def inspect_config(config_path: str | Path, kind: str = "auto") -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    if kind == "auto":
        kind = "ylj" if config_path.suffix.lower() in {".yaml", ".yml"} else "luoyang"
    if kind not in {"luoyang", "ylj"}:
        raise ValueError("kind must be auto, luoyang, or ylj")

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle) if kind == "luoyang" else yaml.safe_load(handle)

    if kind == "luoyang":
        fields = config["fields"]
        parquet_path = _path_from_config(config_path, config["paths"]["parquet_file"])
        required = list(dict.fromkeys([
            fields["timestamp"],
            *fields["timeseries_columns"],
            fields["image_paths"],
            fields["image_timestamps"],
        ]))
        list_fields = (fields["image_paths"], fields["image_timestamps"])
        description = {
            "adapter": "LuoyangParquetDataset",
            "row_layout": "one timestamp per row; image fields are serialized lists",
            "timeseries_columns": fields["timeseries_columns"],
            "privileged_future_columns": config["privileged_teacher"]["future_timeseries_columns"],
        }
    else:
        fields = config["fields"]
        parquet_path = _path_from_config(config_path, config["paths"]["parquet_file"])
        required = list(dict.fromkeys([
            fields["timestamp"], *fields["time_series_columns"]
        ]))
        list_fields = None
        description = {
            "adapter": "YLJParquetDataset",
            "row_layout": "one timestamp per row; observed history and forecast columns share the row index",
            "history_columns": fields["history_columns"],
            "forecast_window_columns": fields["forecast_window_columns"],
            "privileged_future_columns": config["privileged_teacher"]["future_timeseries_columns"],
        }

    result: dict[str, Any] = {
        "config": str(config_path),
        "parquet": str(parquet_path),
        "required_columns": required,
        "description": description,
    }
    if not parquet_path.is_file():
        result.update({"exists": False, "missing_columns": required})
        return result

    available = pd.read_parquet(parquet_path, engine="pyarrow").columns.tolist()
    missing = [name for name in required if name not in available]
    result["exists"] = True
    result["available_columns"] = available
    result["missing_columns"] = missing
    if missing:
        return result

    frame = pd.read_parquet(parquet_path, columns=required, engine="pyarrow")
    timestamp = fields["timestamp"]
    parsed = pd.to_datetime(frame[timestamp], errors="coerce")
    result["rows"] = int(len(frame))
    result["timestamp"] = {
        "dtype": str(frame[timestamp].dtype),
        "invalid": int(parsed.isna().sum()),
        "duplicates": int(frame[timestamp].duplicated().sum()),
        "min": str(parsed.min()) if len(parsed) else None,
        "max": str(parsed.max()) if len(parsed) else None,
        "interval_minutes": sorted(
            set(parsed.sort_values().diff().dt.total_seconds().div(60).dropna().round(6).tolist())
        )[:50],
    }
    numeric_columns = [
        name for name in required
        if name != fields["timestamp"] and (list_fields is None or name not in list_fields)
    ]
    result["columns"] = _column_report(frame, numeric_columns)
    if list_fields is not None:
        result["lists"] = _list_report(frame, *list_fields)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--kind", choices=["auto", "luoyang", "ylj"], default="auto")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    result = inspect_config(args.config, args.kind)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None, default=str))
    if result.get("missing_columns"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
