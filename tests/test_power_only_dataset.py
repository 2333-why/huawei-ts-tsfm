from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from data_provider.power_only import PowerOnlyParquetDataset
from models.tslib_adapter import power_only_batch


def _write_power_fixture(
    root: Path,
    *,
    interval_minutes: int,
    values: list[float],
    test_start: str = "2025-01-01 04:00:00",
    test_end: str = "2025-01-01 05:00:00",
    missing_timestamps: tuple[str, ...] = (),
) -> Path:
    timestamps = pd.date_range(
        "2025-01-01 00:00:00",
        periods=len(values),
        freq=f"{interval_minutes}min",
    )
    frame = pd.DataFrame({"timestamp": timestamps, "power": values})
    if missing_timestamps:
        frame = frame.loc[~frame["timestamp"].isin(pd.to_datetime(missing_timestamps))]
    parquet_path = root / f"power_{interval_minutes}min.parquet"
    frame.to_parquet(parquet_path, index=False)
    config = {
        "dataset": "PowerOnlyParquet",
        "paths": {"parquet_file": str(parquet_path), "results_root": str(root / "results")},
        "fields": {"timestamp": "timestamp", "target": "power"},
        "time": {
            "sampling_interval_minutes": interval_minutes,
            "forecast_step_minutes": interval_minutes,
            "train_start_timestamp": "2025-01-01 00:00:00",
            "test_start_timestamp": test_start,
            "test_end_exclusive_timestamp": test_end,
            "validation_fraction": 0.5,
        },
        "power": {
            "power_scale": 10.0,
            "rated_power": 10.0,
            "units": "kW",
        },
    }
    config_path = root / f"power_{interval_minutes}min.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


@pytest.fixture
def synthetic_power_config(tmp_path: Path) -> Path:
    return _write_power_fixture(
        tmp_path,
        interval_minutes=5,
        values=[
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
            24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
            34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
            44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0,
            54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
        ],
    )


@pytest.fixture
def synthetic_power_config_with_missing_future(tmp_path: Path) -> Path:
    config_path = _write_power_fixture(
        tmp_path,
        interval_minutes=5,
        values=[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
                23.0, 24.0, 25.0, 26.0, 27.0],
        test_start="2025-01-01 01:00:00",
        test_end="2025-01-01 01:20:00",
    )
    frame = pd.read_parquet(json.loads(config_path.read_text())["paths"]["parquet_file"])
    frame.loc[frame["timestamp"] == pd.Timestamp("2025-01-01 01:05:00"), "power"] = np.nan
    frame.to_parquet(json.loads(config_path.read_text())["paths"]["parquet_file"], index=False)
    return config_path


@pytest.fixture
def synthetic_power_config_with_gap(tmp_path: Path) -> Path:
    return _write_power_fixture(
        tmp_path,
        interval_minutes=5,
        values=[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
                23.0, 24.0, 25.0, 26.0, 27.0],
        test_start="2025-01-01 00:15:00",
        test_end="2025-01-01 00:30:00",
        missing_timestamps=("2025-01-01 00:20:00",),
    )


def test_loader_emits_only_power_batch_fields(synthetic_power_config):
    dataset = PowerOnlyParquetDataset(
        synthetic_power_config, "train", history_points=24, forecast_steps=1
    )
    sample = dataset[0]
    assert set(sample) == {"history", "target", "target_mask", "issue_time_ns"}
    assert sample["history"].shape == (24, 1)
    assert sample["target"].shape == (1, 1)


def test_loader_normalizes_by_configured_power_scale(synthetic_power_config):
    dataset = PowerOnlyParquetDataset(
        synthetic_power_config, "train", history_points=2, forecast_steps=1
    )
    sample = dataset[0]
    assert sample["history"][-1, 0] == pytest.approx(0.5)
    assert sample["target"][0, 0] == pytest.approx(0.6)


def test_loader_masks_missing_future_target_but_keeps_valid_window(
    synthetic_power_config_with_missing_future,
):
    dataset = PowerOnlyParquetDataset(
        synthetic_power_config_with_missing_future,
        "test",
        history_points=2,
        forecast_steps=2,
    )
    sample = dataset[0]
    assert sample["target_mask"].tolist() == [False, True]
    assert np.isfinite(sample["target"]).all()


def test_loader_rejects_timestamp_gaps_inside_requested_window(
    synthetic_power_config_with_gap,
):
    dataset = PowerOnlyParquetDataset(
        synthetic_power_config_with_gap, "test", history_points=2, forecast_steps=2
    )
    assert len(dataset) == 0


def test_dataloader_collation_matches_power_only_batch_contract(tmp_path):
    config_path = _write_power_fixture(
        tmp_path,
        interval_minutes=15,
        values=[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        ],
        test_start="2025-01-01 03:00:00",
        test_end="2025-01-01 04:30:00",
    )
    dataset = PowerOnlyParquetDataset(
        config_path, "test", history_points=3, forecast_steps=2
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    result = power_only_batch(batch, device="cpu")
    assert result.x.shape == (2, 3, 1)
    assert result.y.shape == (2, 2, 1)
    assert result.mask.shape == (2, 2)
    assert result.mask.tolist() == [[True, True], [True, True]]
