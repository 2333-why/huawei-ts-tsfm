#!/usr/bin/env python3
"""Validate both PV Parquet datasets and prepare runnable experiment configs."""

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

from data_provider.power_only import (  # noqa: E402
    PowerOnlyParquetDataset,
    load_power_only_config,
)
DEFAULT_LUOYANG = Path(
    "/data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/"
    "Luoyang-Unified_format-V1-with_DNI_DHI.parquet"
)
DEFAULT_YLJ = Path(
    "/data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/"
    "YLJ-Unified_format-with_DNI_DHI.parquet"
)
LUOYANG_FIXED_TIME = {
    "train_start_timestamp": "2026-04-05 00:00:00",
    "test_start_timestamp": "2026-05-11 00:00:00",
    "test_end_exclusive_timestamp": "2026-06-12 00:00:00",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _inspect_content(name: str, parquet_path: Path, config_path: Path) -> dict[str, Any]:
    config = load_power_only_config(config_path)
    parquet_path = parquet_path.expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(f"{name}: Parquet file not found: {parquet_path}")
    if parquet_path.stat().st_size <= 0:
        raise ValueError(f"{name}: Parquet file is empty: {parquet_path}")

    timestamp_column = config["fields"]["timestamp"]
    target_column = config["fields"]["target"]
    try:
        frame = pd.read_parquet(parquet_path, columns=[timestamp_column, target_column])
    except Exception as exc:
        raise ValueError(
            f"{name}: cannot read required columns {timestamp_column!r}, "
            f"{target_column!r}: {exc}"
        ) from exc
    if frame.empty:
        raise ValueError(f"{name}: Parquet contains no rows")

    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    invalid_timestamps = int(timestamps.isna().sum())
    duplicate_timestamps = int(timestamps.duplicated().sum())
    if invalid_timestamps:
        raise ValueError(f"{name}: {invalid_timestamps} invalid timestamps")
    if duplicate_timestamps:
        raise ValueError(f"{name}: {duplicate_timestamps} duplicate timestamps")
    timestamp_index = pd.DatetimeIndex(timestamps)
    if timestamp_index.tz is not None:
        timestamp_index = timestamp_index.tz_localize(None)

    targets = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite_targets = int(np.isfinite(targets).sum())
    if not finite_targets:
        raise ValueError(f"{name}: target column has no finite numeric values")
    return {
        "dataset": name,
        "parquet_file": str(parquet_path),
        "file_size_bytes": int(parquet_path.stat().st_size),
        "rows": int(len(frame)),
        "timestamp_column": timestamp_column,
        "target_column": target_column,
        "data_start": timestamp_index.min().isoformat(sep=" "),
        "data_end": timestamp_index.max().isoformat(sep=" "),
        "finite_targets": finite_targets,
        "non_finite_targets": int(len(frame) - finite_targets),
        "status": "PASS",
    }


def _copy_config_with_parquet(
    base_config_path: Path,
    parquet_path: Path,
    output_path: Path,
    *,
    time_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(load_power_only_config(base_config_path))
    config["paths"]["parquet_file"] = str(parquet_path.expanduser().resolve())
    if time_overrides:
        config["time"].update(time_overrides)
    _write_json(output_path, config)
    time_config = config["time"]
    return {
        "config_file": str(output_path.resolve()),
        "train_validation_range": [
            str(time_config["train_start_timestamp"]),
            str(time_config["test_start_timestamp"]),
        ],
        "test_range": [
            str(time_config["test_start_timestamp"]),
            str(time_config["test_end_exclusive_timestamp"]),
        ],
    }


def _validate_splits(
    name: str, config_path: Path, windows: tuple[tuple[int, int], ...]
) -> dict[str, Any]:
    experiment_windows: dict[str, Any] = {}
    for history_points, forecast_steps in windows:
        label = f"seq{history_points}_pred{forecast_steps}"
        counts: dict[str, int] = {}
        audits: dict[str, Any] = {}
        for split in ("train", "val", "test"):
            dataset = PowerOnlyParquetDataset(
                config_path,
                flag=split,
                history_points=history_points,
                forecast_steps=forecast_steps,
            )
            count = len(dataset)
            if count <= 0:
                raise ValueError(
                    f"{name}: {label} generated {split} split contains no samples"
                )
            counts[split] = count
            audits[split] = dict(dataset.candidate_audit)
        experiment_windows[label] = {
            "history_points": history_points,
            "forecast_steps": forecast_steps,
            "sample_counts": counts,
            "candidate_audits": audits,
            "status": "PASS",
        }
    return {"experiment_windows": experiment_windows}


def run_preflight(
    *,
    luoyang_parquet: Path,
    ylj_parquet: Path,
    luoyang_config: Path,
    ylj_config: Path,
    output_dir: Path,
    strict_grid: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    luoyang_output = output_dir / "luoyang_one_month_train_one_month_test.json"
    ylj_output = output_dir / "ylj_original_split.json"

    luoyang_content = _inspect_content("luoyang", luoyang_parquet, luoyang_config)
    ylj_content = _inspect_content("ylj", ylj_parquet, ylj_config)
    # Keep the previously verified fixed Luoyang split from the base config.
    # The sequence-format candidate is deliberately not used by this row-wise loader.
    luoyang_range = _copy_config_with_parquet(
        luoyang_config,
        luoyang_parquet,
        luoyang_output,
        time_overrides=LUOYANG_FIXED_TIME,
    )
    ylj_range = _copy_config_with_parquet(ylj_config, ylj_parquet, ylj_output)

    luoyang_splits = _validate_splits(
        "luoyang", luoyang_output, ((48, 1), (96, 48))
    )
    ylj_splits = _validate_splits("ylj", ylj_output, ((48, 1), (96, 16)))
    report = {
        "status": "PASS",
        "policy": (
            "Luoyang keeps the verified one-month train/validation plus one-month "
            "test split from its base config; YLJ keeps its configured split"
        ),
        "datasets": [
            {**luoyang_content, **luoyang_range, **luoyang_splits},
            {
                **ylj_content,
                **ylj_splits,
                **ylj_range,
                "split_policy": "unchanged from base config",
            },
        ],
    }
    _write_json(output_dir / "dataset_preflight.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--luoyang-parquet", type=Path, default=DEFAULT_LUOYANG)
    parser.add_argument("--ylj-parquet", type=Path, default=DEFAULT_YLJ)
    parser.add_argument(
        "--luoyang-config",
        type=Path,
        default=REPO_ROOT / "configs/datasets/skippd_luoyang.json",
    )
    parser.add_argument(
        "--ylj-config",
        type=Path,
        default=REPO_ROOT / "configs/datasets/pvod_station00_ylj.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "generated_configs/preflight"
    )
    parser.add_argument(
        "--strict-grid",
        action="store_true",
        help="retained for CLI compatibility; split validation always checks exact windows",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_preflight(
            luoyang_parquet=args.luoyang_parquet,
            ylj_parquet=args.ylj_parquet,
            luoyang_config=args.luoyang_config,
            ylj_config=args.ylj_config,
            output_dir=args.output_dir,
            strict_grid=args.strict_grid,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"DATASET PREFLIGHT FAIL: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for dataset in report["datasets"]:
            counts = ", ".join(
                f"{label}="
                f"{window['sample_counts']['train']}/"
                f"{window['sample_counts']['val']}/"
                f"{window['sample_counts']['test']}"
                for label, window in dataset["experiment_windows"].items()
            )
            print(
                f"PASS {dataset['dataset']}: rows={dataset['rows']} "
                f"range={dataset['data_start']}..{dataset['data_end']} "
                f"samples(train/val/test): {counts}"
            )
        print(f"DATASET PREFLIGHT PASS: {args.output_dir.resolve() / 'dataset_preflight.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
