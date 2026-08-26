from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from baselines import BASELINE_NAMES, build_baseline, load_baseline_class


SITE = {
    "latitude": 37.427,
    "longitude": -122.174,
    "timezone": "America/Los_Angeles",
    "surface_tilt": 37,
    "surface_azimuth": 195,
}


class TrainingDataset:
    def __init__(
        self,
        timestamps: pd.DatetimeIndex,
        values: list[float],
        *,
        pred_len: int = 2,
        step_minutes: int = 15,
    ) -> None:
        self.pred_len = pred_len
        self.seq_len = 5
        self.step = pd.Timedelta(minutes=step_minutes)
        self.forecast_step = self.step
        self.site_config = SITE
        self.normalized_train_mean = float(np.mean(values))
        self.power_scale = 1.0
        self.train_mean = self.normalized_train_mean
        self.finite_training_observations = pd.DataFrame(
            {"timestamp": timestamps, "normalized_power": values, "power": values}
        )


def _ns(timestamp: str) -> np.ndarray:
    return np.asarray([pd.Timestamp(timestamp).value], dtype=np.int64)


def test_extended_baseline_catalog_is_exact_and_ordered():
    assert BASELINE_NAMES == (
        "Persistence",
        "SmartPersistence",
        "SeasonalPersistence",
        "Climatology",
        "MovingMedian",
        "DriftPersistence",
        "ClearSkyEWMA",
        "ClearSkyAR",
        "SimilarDay",
        "PersistenceClimatologyBlend",
    )


def test_moving_median_uses_the_last_physical_hour_and_ignores_nonfinite_values():
    dataset = TrainingDataset(pd.date_range("2025-01-01", periods=8, freq="15min"), [0.1] * 8)
    baseline = build_baseline("MovingMedian", dataset)
    history = torch.tensor(
        [[[0.99], [0.10], [float("nan")], [0.70], [0.30], [0.50]]],
        dtype=torch.float32,
    )

    prediction = baseline.predict(history, _ns("2025-01-02 12:00:00"), dataset)

    np.testing.assert_allclose(prediction.numpy(), [[[0.4], [0.4]]])
    assert baseline.checkpoint_payload()["window_minutes"] == 60


def test_drift_persistence_extrapolates_the_recent_physical_hour_slope():
    dataset = TrainingDataset(pd.date_range("2025-01-01", periods=8, freq="15min"), [0.1] * 8)
    baseline = build_baseline("DriftPersistence", dataset)
    history = torch.tensor(
        [[[0.95], [0.10], [0.20], [0.30], [0.40], [0.50]]],
        dtype=torch.float32,
    )

    prediction = baseline.predict(history, _ns("2025-01-02 12:00:00"), dataset)

    np.testing.assert_allclose(prediction.numpy(), [[[0.6], [0.7]]], atol=1e-6)


def test_clear_sky_ewma_averages_recent_clear_sky_indices(
    monkeypatch: pytest.MonkeyPatch,
):
    dataset = TrainingDataset(pd.date_range("2025-01-01", periods=8, freq="15min"), [0.1] * 8)
    baseline = build_baseline("ClearSkyEWMA", dataset)
    history = torch.tensor([[[0.10], [0.20], [0.40]]], dtype=torch.float32)
    monkeypatch.setattr(
        "baselines.extended.clear_sky_poa",
        lambda site, times: np.asarray([100.0, 100.0, 100.0, 200.0, 50.0]),
    )

    prediction = baseline.predict(history, _ns("2025-01-02 12:00:00"), dataset)

    np.testing.assert_allclose(
        prediction.numpy(),
        [[[0.5359246], [0.13398115]]],
        atol=1e-6,
    )
    assert baseline.checkpoint_payload()["half_life_minutes"] == 30


def test_clear_sky_ar_restores_coefficients_and_recurses_in_index_space(
    monkeypatch: pytest.MonkeyPatch,
):
    dataset = TrainingDataset(pd.date_range("2025-01-01", periods=8, freq="15min"), [0.1] * 8)
    baseline_class = load_baseline_class("ClearSkyAR")
    payload = {
        "name": "ClearSkyAR",
        "order": 1,
        "intercept": 0.0,
        "coefficients": [0.5],
        "training_index_mean": 0.001,
        "step_ns": pd.Timedelta(minutes=15).value,
        "forecast_step_ns": pd.Timedelta(minutes=15).value,
        "ridge": 1e-8,
        "clear_sky_model": "ineichen",
        "clear_sky_denominator_threshold_w_m2": 1.0,
    }
    baseline = baseline_class.from_checkpoint_payload(payload)
    history = torch.tensor([[[0.10], [0.20], [0.40]]], dtype=torch.float32)
    monkeypatch.setattr(
        "baselines.extended.clear_sky_poa",
        lambda site, times: np.asarray([100.0, 100.0, 100.0, 100.0, 100.0]),
    )

    prediction = baseline.predict(history, _ns("2025-01-02 12:00:00"), dataset)

    np.testing.assert_allclose(prediction.numpy(), [[[0.2], [0.1]]], atol=1e-6)


def test_clear_sky_ar_fitted_checkpoint_reproduces_predictions(
    monkeypatch: pytest.MonkeyPatch,
):
    timestamps = pd.date_range("2025-06-01 08:00", periods=16, freq="15min")
    dataset = TrainingDataset(timestamps, [0.05 + 0.02 * index for index in range(16)])
    monkeypatch.setattr(
        "baselines.extended.clear_sky_poa",
        lambda site, times: np.full(len(pd.DatetimeIndex(times)), 100.0),
    )
    baseline = build_baseline("ClearSkyAR", dataset)
    history = torch.tensor([[[0.15], [0.17], [0.19], [0.21], [0.23]]], dtype=torch.float32)
    issue = _ns("2025-06-02 09:00")

    expected = baseline.predict(history, issue, dataset)
    payload = json.loads(json.dumps(baseline.checkpoint_payload()))
    restored = load_baseline_class("ClearSkyAR").from_checkpoint_payload(payload)

    np.testing.assert_allclose(restored.predict(history, issue, dataset), expected)
    assert len(payload["coefficients"]) == payload["order"]


def test_similar_day_uses_only_training_analog_trajectory():
    timestamps = pd.date_range("2025-01-01 00:00", periods=7, freq="15min")
    dataset = TrainingDataset(timestamps, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    baseline = build_baseline("SimilarDay", dataset)
    history = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5]]], dtype=torch.float32)

    prediction = baseline.predict(history, _ns("2025-01-08 01:00"), dataset)

    np.testing.assert_allclose(prediction.numpy(), [[[0.6], [0.7]]], atol=1e-6)
    payload = baseline.checkpoint_payload()
    assert payload["candidate_issue_time_ns"] == [pd.Timestamp("2025-01-01 01:00").value]


def test_similar_day_checkpoint_restores_without_training_dataset():
    timestamps = pd.date_range("2025-01-01 00:00", periods=7, freq="15min")
    dataset = TrainingDataset(timestamps, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    baseline = build_baseline("SimilarDay", dataset)
    history = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5]]], dtype=torch.float32)
    issue = _ns("2025-01-08 01:00")
    payload = json.loads(json.dumps(baseline.checkpoint_payload()))

    restored = load_baseline_class("SimilarDay").from_checkpoint_payload(payload)

    np.testing.assert_allclose(restored.predict(history, issue, dataset), [[[0.6], [0.7]]])


def test_similar_day_empty_candidate_checkpoint_restores_persistence_fallback():
    timestamps = pd.date_range("2025-01-01 00:00", periods=4, freq="15min")
    dataset = TrainingDataset(timestamps, [0.1, 0.2, 0.3, 0.4])
    baseline = build_baseline("SimilarDay", dataset)
    payload = json.loads(json.dumps(baseline.checkpoint_payload()))

    restored = load_baseline_class("SimilarDay").from_checkpoint_payload(payload)
    history = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5]]])

    np.testing.assert_allclose(
        restored.predict(history, _ns("2025-01-08 01:00"), dataset),
        [[[0.5], [0.5]]],
    )


def test_persistence_climatology_blend_applies_per_horizon_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    dataset = TrainingDataset(pd.date_range("2025-01-01", periods=8, freq="15min"), [0.3] * 8)
    baseline_class = load_baseline_class("PersistenceClimatologyBlend")
    payload = {
        "name": "PersistenceClimatologyBlend",
        "weights": [1.0, 0.0],
        "step_ns": pd.Timedelta(minutes=15).value,
        "forecast_step_ns": pd.Timedelta(minutes=15).value,
        "smart_persistence": {
            "name": "SmartPersistence",
            "clear_sky_model": "ineichen",
            "clear_sky_denominator_threshold_w_m2": 1.0,
        },
        "climatology": {
            "name": "Climatology",
            "window_days": 15,
            "training_mean": 0.3,
            "profile": {
                "day_of_year": [2, 2],
                "minute_of_day": [735, 750],
                "sum_normalized_power": [0.3, 0.3],
                "count": [1, 1],
            },
        },
    }
    baseline = baseline_class.from_checkpoint_payload(payload)
    history = torch.tensor([[[0.10], [0.20]]], dtype=torch.float32)
    monkeypatch.setattr(
        "baselines.methods.clear_sky_poa",
        lambda site, times: np.full(len(pd.DatetimeIndex(times)), 100.0),
    )

    prediction = baseline.predict(history, _ns("2025-01-02 12:00"), dataset)

    np.testing.assert_allclose(prediction.numpy(), [[[0.2], [0.3]]], atol=1e-6)


def test_blend_fitted_weights_are_training_only_serializable_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
):
    timestamps = pd.date_range("2025-01-01 08:00", periods=32, freq="15min")
    values = [0.1 + 0.01 * (index % 8) for index in range(32)]
    dataset = TrainingDataset(timestamps, values)
    monkeypatch.setattr(
        "baselines.methods.clear_sky_poa",
        lambda site, times: np.full(len(pd.DatetimeIndex(times)), 100.0),
    )
    baseline = build_baseline("PersistenceClimatologyBlend", dataset)
    payload = json.loads(json.dumps(baseline.checkpoint_payload()))

    assert len(payload["weights"]) == dataset.pred_len
    assert all(0.0 <= weight <= 1.0 for weight in payload["weights"])
    restored = load_baseline_class("PersistenceClimatologyBlend").from_checkpoint_payload(payload)
    history = torch.tensor([[[0.11], [0.12], [0.13], [0.14], [0.15]]])
    issue = _ns("2025-01-10 12:00")
    np.testing.assert_allclose(
        restored.predict(history, issue, dataset),
        baseline.predict(history, issue, dataset),
    )


@pytest.mark.parametrize("name", ["ClearSkyAR", "SimilarDay", "PersistenceClimatologyBlend"])
def test_fitted_checkpoint_rejects_a_different_time_grid(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
):
    timestamps = pd.date_range("2025-06-01 08:00", periods=32, freq="15min")
    dataset = TrainingDataset(timestamps, [0.1 + 0.01 * (index % 8) for index in range(32)])
    monkeypatch.setattr(
        "baselines.extended.clear_sky_poa",
        lambda site, times: np.full(len(pd.DatetimeIndex(times)), 100.0),
    )
    monkeypatch.setattr(
        "baselines.methods.clear_sky_poa",
        lambda site, times: np.full(len(pd.DatetimeIndex(times)), 100.0),
    )
    payload = json.loads(json.dumps(build_baseline(name, dataset).checkpoint_payload()))
    restored = load_baseline_class(name).from_checkpoint_payload(payload)
    dataset.step = pd.Timedelta(minutes=5)
    dataset.forecast_step = pd.Timedelta(minutes=5)
    history = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5]]])

    with pytest.raises(ValueError, match="time grid"):
        restored.predict(history, _ns("2025-06-10 12:00"), dataset)


def test_clear_sky_ar_checkpoint_rejects_changed_clear_sky_semantics(
    monkeypatch: pytest.MonkeyPatch,
):
    timestamps = pd.date_range("2025-06-01 08:00", periods=16, freq="15min")
    dataset = TrainingDataset(timestamps, [0.1 + 0.01 * index for index in range(16)])
    monkeypatch.setattr(
        "baselines.extended.clear_sky_poa",
        lambda site, times: np.full(len(pd.DatetimeIndex(times)), 100.0),
    )
    payload = build_baseline("ClearSkyAR", dataset).checkpoint_payload()
    payload["clear_sky_denominator_threshold_w_m2"] = 999.0

    with pytest.raises(ValueError, match="ClearSkyAR checkpoint"):
        load_baseline_class("ClearSkyAR").from_checkpoint_payload(payload)


def test_blend_checkpoint_requires_compatible_smart_persistence_payload(
    monkeypatch: pytest.MonkeyPatch,
):
    timestamps = pd.date_range("2025-01-01 08:00", periods=32, freq="15min")
    dataset = TrainingDataset(timestamps, [0.1 + 0.01 * (index % 8) for index in range(32)])
    monkeypatch.setattr(
        "baselines.methods.clear_sky_poa",
        lambda site, times: np.full(len(pd.DatetimeIndex(times)), 100.0),
    )
    payload = build_baseline("PersistenceClimatologyBlend", dataset).checkpoint_payload()
    payload.pop("smart_persistence")

    with pytest.raises(ValueError, match="PersistenceClimatologyBlend checkpoint"):
        load_baseline_class("PersistenceClimatologyBlend").from_checkpoint_payload(payload)
