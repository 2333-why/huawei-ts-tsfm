from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_recent_four_months import prepare_dataset


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    timestamps = pd.date_range(
        "2026-01-01 00:00:00",
        "2026-06-30 23:45:00",
        freq="15min",
    )
    parquet_path = tmp_path / "updated.parquet"
    pd.DataFrame(
        {"timestamp": timestamps, "power": range(len(timestamps))}
    ).to_parquet(parquet_path, index=False)
    config = {
        "paths": {"parquet_file": str(parquet_path), "results_root": str(tmp_path)},
        "fields": {"timestamp": "timestamp", "target": "power"},
        "history_points": 16,
        "forecast_steps": 16,
        "time": {
            "sampling_interval_minutes": 15,
            "forecast_step_minutes": 15,
            "train_start_timestamp": "2026-01-01 00:00:00",
            "test_start_timestamp": "2026-03-01 00:00:00",
            "test_end_exclusive_timestamp": "2026-05-01 00:00:00",
            "validation_fraction": 0.15,
        },
        "power": {"power_scale": 1.0, "rated_power": 1.0},
    }
    config_path = tmp_path / "base.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, parquet_path


def test_prepare_dataset_selects_latest_four_calendar_months(tmp_path):
    config_path, parquet_path = _fixture(tmp_path)
    output_path = tmp_path / "generated.json"

    report = prepare_dataset(
        "fixture", config_path, parquet_path, output_path, strict_grid=True
    )
    generated = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["full_data_range"] == [
        "2026-01-01 00:00:00",
        "2026-06-30 23:45:00",
    ]
    assert report["train_range"] == [
        "2026-03-01 00:00:00",
        "2026-05-01 00:00:00",
    ]
    assert report["test_range"] == [
        "2026-05-01 00:00:00",
        "2026-07-01 00:00:00",
    ]
    assert report["missing_selected_timestamps"] == 0
    assert generated["time"]["train_start_timestamp"] == "2026-03-01 00:00:00"
    assert generated["time"]["test_start_timestamp"] == "2026-05-01 00:00:00"
    assert generated["time"]["test_end_exclusive_timestamp"] == "2026-07-01 00:00:00"


def test_prepare_dataset_reports_gaps_and_strict_mode_rejects_them(tmp_path):
    config_path, parquet_path = _fixture(tmp_path)
    frame = pd.read_parquet(parquet_path)
    frame = frame.drop(frame.index[-100])
    frame.to_parquet(parquet_path, index=False)

    report = prepare_dataset(
        "fixture", config_path, parquet_path, tmp_path / "non_strict.json"
    )
    assert report["missing_selected_timestamps"] == 1

    with pytest.raises(ValueError, match="1 missing"):
        prepare_dataset(
            "fixture",
            config_path,
            parquet_path,
            tmp_path / "strict.json",
            strict_grid=True,
        )
