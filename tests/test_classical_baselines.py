from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from baselines import BASELINE_NAMES, Climatology, build_baseline, load_baseline_class
from baselines.methods import clear_sky_poa
from data_provider.power_only import PowerOnlyParquetDataset


SITE = {
    "latitude": 37.427,
    "longitude": -122.174,
    "timezone": "America/Los_Angeles",
    "surface_tilt": 37,
    "surface_azimuth": 195,
}


def _write_fixture(
    root: Path,
    *,
    start: str = "2025-01-01 00:00:00",
    periods: int = 96,
    freq: str = "1h",
    values: list[float] | None = None,
    train_start: str = "2025-01-01 00:00:00",
    test_start: str = "2025-01-03 00:00:00",
    test_end: str = "2025-01-05 00:00:00",
    validation_fraction: float = 0.25,
    missing_timestamps: tuple[str, ...] = (),
) -> Path:
    timestamps = pd.date_range(start, periods=periods, freq=freq)
    if values is None:
        values = [float(index + 1) for index in range(periods)]
    frame = pd.DataFrame({"timestamp": timestamps, "power": values})
    if missing_timestamps:
        frame = frame.loc[
            ~frame["timestamp"].isin(pd.to_datetime(missing_timestamps))
        ]
    parquet_path = root / "power.parquet"
    frame.to_parquet(parquet_path, index=False)
    config = {
        "dataset": "PowerOnlyParquet",
        "paths": {"parquet_file": str(parquet_path), "results_root": str(root / "results")},
        "fields": {"timestamp": "timestamp", "target": "power"},
        "time": {
            "sampling_interval_minutes": 60,
            "forecast_step_minutes": 60,
            "train_start_timestamp": train_start,
            "test_start_timestamp": test_start,
            "test_end_exclusive_timestamp": test_end,
            "validation_fraction": validation_fraction,
        },
        "power": {"power_scale": 10.0, "rated_power": 10.0, "units": "kW"},
        "site": SITE,
    }
    config_path = root / "power.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _dataset(tmp_path: Path, **kwargs) -> PowerOnlyParquetDataset:
    return PowerOnlyParquetDataset(
        _write_fixture(tmp_path, **kwargs),
        "train",
        history_points=2,
        forecast_steps=2,
    )


def _ns(*timestamps: str) -> np.ndarray:
    return pd.to_datetime(timestamps).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def test_baseline_catalog_is_exact_and_ordered():
    assert BASELINE_NAMES == (
        "Persistence",
        "SmartPersistence",
        "SeasonalPersistence",
        "Climatology",
    )
    assert tuple(load_baseline_class(name).__name__ for name in BASELINE_NAMES) == BASELINE_NAMES


def test_persistence_repeats_the_last_history_value(tmp_path: Path):
    dataset = _dataset(tmp_path)
    baseline = build_baseline("Persistence", dataset)
    history = torch.tensor([[[0.05], [0.25]]], dtype=torch.float32)

    prediction = baseline.predict(history, _ns("2025-01-02 00:00:00"), dataset)

    assert prediction.shape == (1, 2, 1)
    assert prediction.tolist() == [[[0.25], [0.25]]]


def test_smart_persistence_uses_clear_sky_ratio_and_current_power(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset = _dataset(tmp_path)
    baseline = build_baseline("SmartPersistence", dataset)
    monkeypatch.setattr(
        "baselines.methods.clear_sky_poa",
        lambda site_config, naive_local_times: np.asarray([100.0, 200.0, 400.0]),
    )
    history = torch.tensor([[[0.05], [0.10]]], dtype=torch.float32)

    prediction = baseline.predict(history, _ns("2025-01-02 00:00:00"), dataset)

    np.testing.assert_allclose(prediction.numpy(), [[[0.2], [0.4]]])


def test_smart_persistence_treats_one_w_m2_as_nighttime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset = _dataset(tmp_path)
    baseline = build_baseline("SmartPersistence", dataset)
    monkeypatch.setattr(
        "baselines.methods.clear_sky_poa",
        lambda site_config, naive_local_times: np.asarray([1.0, 100.0, 100.0]),
    )
    history = torch.tensor([[[0.05], [0.50]]], dtype=torch.float32)

    prediction = baseline.predict(history, _ns("2025-01-02 00:00:00"), dataset)

    np.testing.assert_allclose(prediction.numpy(), [[[0.0], [0.0]]])
    assert baseline.checkpoint_payload()["clear_sky_denominator_threshold_w_m2"] == 1.0


def test_seasonal_persistence_uses_exact_previous_day_and_falls_back_per_step(
    tmp_path: Path,
):
    timestamps = pd.date_range("2025-01-01", periods=50, freq="1h")
    values = [5.0] * len(timestamps)
    values[timestamps.get_loc(pd.Timestamp("2025-01-01 01:00:00"))] = 3.0
    config_path = _write_fixture(
        tmp_path,
        periods=50,
        values=values,
        missing_timestamps=("2025-01-01 02:00:00",),
    )
    dataset = PowerOnlyParquetDataset(config_path, "train", history_points=2, forecast_steps=2)
    baseline = build_baseline("SeasonalPersistence", dataset)
    history = torch.tensor([[[0.40], [0.50]]], dtype=torch.float32)

    prediction = baseline.predict(history, _ns("2025-01-02 00:00:00"), dataset)

    np.testing.assert_allclose(prediction.numpy(), [[[0.3], [0.5]]])


def test_climatology_uses_training_only_circular_same_clock_bucket(tmp_path: Path):
    timestamps = pd.date_range("2024-12-20", periods=312, freq="1h")
    values = [float("nan")] * len(timestamps)
    values[timestamps.get_loc(pd.Timestamp("2024-12-31 12:00:00"))] = 3.0
    values[timestamps.get_loc(pd.Timestamp("2024-12-31 13:00:00"))] = 3.0
    values[timestamps.get_loc(pd.Timestamp("2025-01-01 12:00:00"))] = 5.0
    config_path = _write_fixture(
        tmp_path,
        start="2024-12-20",
        periods=312,
        values=values,
        train_start="2024-12-20",
        test_start="2025-01-01 00:00:00",
        test_end="2025-01-02 00:00:00",
        validation_fraction=0.0,
    )
    dataset = PowerOnlyParquetDataset(config_path, "train", history_points=2, forecast_steps=2)
    baseline = build_baseline("Climatology", dataset)
    history = torch.tensor([[[0.10], [0.20]]], dtype=torch.float32)

    prediction = baseline.predict(history, _ns("2025-01-01 11:00:00"), dataset)

    # Dec 31 is within the circular 15-day window; Jan 1 is test-only and
    # must not raise the same-clock forecast above the training value.
    np.testing.assert_allclose(prediction.numpy(), [[[0.3], [0.3]]])


def test_climatology_checkpoint_restores_without_training_observations(tmp_path: Path):
    dataset = _dataset(tmp_path)
    baseline = build_baseline("Climatology", dataset)
    history = torch.tensor([[[0.10], [0.20]]], dtype=torch.float32)
    issue_time_ns = _ns("2025-01-02 00:00:00")

    expected = baseline.predict(history, issue_time_ns, dataset)
    payload = json.loads(json.dumps(baseline.checkpoint_payload()))

    dataset._target_values_raw[:] = np.nan
    dataset.parquet_path.unlink()
    restored = Climatology.from_checkpoint_payload(payload)
    actual = restored.predict(history, issue_time_ns, dataset)

    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    assert payload["profile"]["day_of_year"]
    assert payload["profile"]["minute_of_day"]


@pytest.mark.parametrize(
    "site_override",
    [
        {"latitude": 91},
        {"longitude": -181},
        {"surface_tilt": 91},
        {"surface_azimuth": 360},
        {"timezone": ""},
    ],
)
def test_dataset_rejects_malformed_site_fields(tmp_path: Path, site_override: dict):
    config_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["site"].update(site_override)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="site"):
        PowerOnlyParquetDataset(config_path, "train", history_points=2, forecast_steps=2)


def test_dataset_rejects_unknown_nonempty_timezone(tmp_path: Path):
    config_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["site"]["timezone"] = "Mars/Olympus"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="site.timezone"):
        PowerOnlyParquetDataset(config_path, "train", history_points=2, forecast_steps=2)


@pytest.mark.parametrize(
    "site",
    [
        SITE,
        {
            "latitude": 38.04778,
            "longitude": 114.95139,
            "timezone": "Asia/Shanghai",
            "surface_tilt": 33,
            "surface_azimuth": 180,
        },
    ],
)
def test_clear_sky_poa_is_positive_at_local_noon_and_zero_at_midnight(site):
    local_times = pd.DatetimeIndex(
        [pd.Timestamp("2025-06-21 12:00:00"), pd.Timestamp("2025-06-21 00:00:00")]
    )

    poa = clear_sky_poa(site, local_times)

    assert np.isfinite(poa).all()
    assert poa[0] > 0
    assert poa[1] == pytest.approx(0.0, abs=1e-8)


def test_baseline_predictions_are_finite_clipped_and_serializable(tmp_path: Path):
    dataset = _dataset(tmp_path)
    history = torch.tensor([[[2.0], [3.0]]], dtype=torch.float32)
    issue_time_ns = _ns("2025-01-02 00:00:00")

    for name in BASELINE_NAMES:
        baseline = build_baseline(name, dataset)
        prediction = baseline.predict(history, issue_time_ns, dataset)
        assert prediction.shape == (1, 2, 1)
        assert torch.isfinite(prediction).all()
        assert torch.all((prediction >= 0) & (prediction <= 1))
        assert isinstance(baseline.checkpoint_payload(), dict)
