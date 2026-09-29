from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.check_datasets_and_prepare import run_preflight


def _write_fixture(
    tmp_path: Path,
    name: str,
    target: str,
    start: str,
    end: str,
    test_start: str,
    train_start: str | None = None,
    test_end_exclusive: str | None = None,
) -> tuple[Path, Path]:
    timestamps = pd.date_range(start, f"{end} 23:00:00", freq="1h")
    parquet_path = tmp_path / f"{name}-with_DNI_DHI.parquet"
    pd.DataFrame(
        {
            "timestamp": timestamps,
            target: range(len(timestamps)),
            "DNI": 100.0,
            "DHI": 20.0,
        }
    ).to_parquet(parquet_path, index=False)
    config = {
        "paths": {"parquet_file": str(parquet_path), "results_root": str(tmp_path)},
        "fields": {"timestamp": "timestamp", "target": target},
        "time": {
            "sampling_interval_minutes": 60,
            "forecast_step_minutes": 60,
            "train_start_timestamp": train_start or start,
            "test_start_timestamp": test_start,
            "test_end_exclusive_timestamp": test_end_exclusive
            or (pd.Timestamp(end) + pd.Timedelta(days=1)).isoformat(sep=" "),
            "validation_fraction": 0.15,
        },
        "power": {"power_scale": 1.0, "rated_power": 1.0},
        "site": {
            "latitude": 34.0,
            "longitude": 112.0,
            "timezone": "Asia/Shanghai",
            "surface_tilt": 34.0,
            "surface_azimuth": 180.0,
        },
    }
    config_path = tmp_path / f"{name}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return parquet_path, config_path


def test_preflight_reads_extra_columns_and_validates_all_splits(tmp_path):
    luoyang_parquet, luoyang_config = _write_fixture(
        tmp_path,
        "luoyang",
        "final_power",
        "2026-01-01",
        "2026-06-30",
        "2026-05-01",
        train_start="2026-04-01",
        test_end_exclusive="2026-06-01",
    )
    ylj_parquet, ylj_config = _write_fixture(
        tmp_path, "ylj", "observe_power", "2024-01-01", "2024-06-30", "2024-05-01"
    )

    report = run_preflight(
        luoyang_parquet=luoyang_parquet,
        ylj_parquet=ylj_parquet,
        luoyang_config=luoyang_config,
        ylj_config=ylj_config,
        output_dir=tmp_path / "output",
        strict_grid=True,
    )

    assert report["status"] == "PASS"
    assert all(
        count > 0
        for dataset in report["datasets"]
        for window in dataset["experiment_windows"].values()
        for count in window["sample_counts"].values()
    )
    assert (tmp_path / "output/dataset_preflight.json").is_file()
    luoyang_generated = json.loads(
        (
            tmp_path / "output/luoyang_one_month_train_one_month_test.json"
        ).read_text(encoding="utf-8")
    )
    assert luoyang_generated["time"]["train_start_timestamp"] == "2026-04-05 00:00:00"
    assert luoyang_generated["time"]["test_start_timestamp"] == "2026-05-11 00:00:00"
    assert (
        luoyang_generated["time"]["test_end_exclusive_timestamp"]
        == "2026-06-12 00:00:00"
    )
    ylj_generated = json.loads(
        (tmp_path / "output/ylj_original_split.json").read_text(encoding="utf-8")
    )
    assert ylj_generated["time"]["test_start_timestamp"] == "2024-05-01"
    assert ylj_generated["paths"]["parquet_file"] == str(ylj_parquet.resolve())


def test_preflight_fails_before_configs_when_required_column_is_missing(tmp_path):
    luoyang_parquet, luoyang_config = _write_fixture(
        tmp_path,
        "luoyang",
        "final_power",
        "2026-01-01",
        "2026-06-30",
        "2026-05-01",
        train_start="2026-04-01",
        test_end_exclusive="2026-06-01",
    )
    ylj_parquet, ylj_config = _write_fixture(
        tmp_path, "ylj", "observe_power", "2024-01-01", "2024-06-30", "2024-05-01"
    )
    broken = pd.read_parquet(ylj_parquet).drop(columns=["observe_power"])
    broken.to_parquet(ylj_parquet, index=False)

    with pytest.raises(ValueError, match="cannot read required columns"):
        run_preflight(
            luoyang_parquet=luoyang_parquet,
            ylj_parquet=ylj_parquet,
            luoyang_config=luoyang_config,
            ylj_config=ylj_config,
            output_dir=tmp_path / "output",
        )

    assert not (
        tmp_path / "output/luoyang_one_month_train_one_month_test.json"
    ).exists()
