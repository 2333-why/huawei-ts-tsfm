"""Classical photovoltaic forecasting baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import torch

from data_provider.power_only import _validate_site_config


def clear_sky_poa(site_config: dict, naive_local_times: Any) -> np.ndarray:
    """Calculate Ineichen clear-sky plane-of-array irradiance in W/m2.

    The input timestamps represent source-file wall-clock values and are
    localized using the site's IANA timezone before pvlib is called.
    """

    from pvlib.irradiance import get_total_irradiance
    from pvlib.location import Location

    site = _validate_site_config(site_config)
    parsed = pd.to_datetime(naive_local_times, errors="coerce")
    scalar = isinstance(parsed, pd.Timestamp)
    if scalar:
        if parsed is pd.NaT:
            raise ValueError("naive_local_times must contain finite timestamps")
        times = pd.DatetimeIndex([parsed])
    else:
        times = pd.DatetimeIndex(parsed)
    if times.isna().any():
        raise ValueError("naive_local_times must contain finite timestamps")
    if times.tz is None:
        localized = times.tz_localize(site["timezone"])
    else:
        localized = times.tz_convert(site["timezone"])
    if not len(localized):
        return np.empty(0, dtype=np.float64)

    location = Location(
        site["latitude"],
        site["longitude"],
        tz=site["timezone"],
    )
    clearsky = location.get_clearsky(localized, model="ineichen")
    solar_position = location.get_solarposition(localized)
    poa = get_total_irradiance(
        surface_tilt=site["surface_tilt"],
        surface_azimuth=site["surface_azimuth"],
        solar_zenith=solar_position["zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )["poa_global"]
    result = np.asarray(poa, dtype=np.float64).reshape(-1)
    result[~np.isfinite(result)] = 0.0
    return np.maximum(result, 0.0)


def _validate_history(history: torch.Tensor, dataset: Any) -> int:
    if not isinstance(history, torch.Tensor):
        raise TypeError("history must be a torch.Tensor")
    if history.ndim != 3 or history.shape[-1] != 1:
        raise ValueError("history must have shape [batch, history_points, 1]")
    pred_len = getattr(dataset, "pred_len", None)
    if pred_len is None or int(pred_len) <= 0:
        raise ValueError("dataset must provide a positive pred_len")
    return int(pred_len)


def _issue_times_ns(issue_time_ns: Any, batch_size: int) -> np.ndarray:
    if isinstance(issue_time_ns, torch.Tensor):
        raw = issue_time_ns.detach().cpu().numpy()
    else:
        raw = np.asarray(issue_time_ns)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    raw = raw.reshape(-1)
    if len(raw) != batch_size:
        raise ValueError("issue_time_ns batch dimension must match history")
    try:
        values = raw.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("issue_time_ns must contain integer nanosecond timestamps") from exc
    return values


def _fallback_mean(dataset: Any) -> float:
    value = getattr(dataset, "normalized_train_mean", None)
    if value is None:
        value = float(dataset.train_mean) / float(dataset.power_scale)
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _last_finite_history(history: torch.Tensor, dataset: Any) -> np.ndarray:
    values = history.detach().to(dtype=torch.float64).cpu().numpy()[..., 0]
    fallback = _fallback_mean(dataset)
    result = np.full(values.shape[0], fallback, dtype=np.float64)
    for row_index, row in enumerate(values):
        finite = np.flatnonzero(np.isfinite(row))
        if len(finite):
            result[row_index] = row[finite[-1]]
    return result


def _future_timestamps(issue_ns: np.ndarray, pred_len: int, dataset: Any) -> np.ndarray:
    step = getattr(dataset, "forecast_step", None)
    if step is None:
        raise ValueError("dataset must provide forecast_step")
    step_ns = int(pd.Timedelta(step).value)
    offsets = np.arange(1, pred_len + 1, dtype=np.int64) * step_ns
    return issue_ns[:, None] + offsets[None, :]


def _finish_prediction(values: np.ndarray, history: torch.Tensor) -> torch.Tensor:
    values = np.asarray(values, dtype=np.float64)
    values[~np.isfinite(values)] = 0.0
    values = np.clip(values, 0.0, 1.0)
    dtype = history.dtype if history.is_floating_point() else torch.float32
    return torch.as_tensor(values, dtype=dtype, device=history.device)


def _canonical_day_of_year(index: pd.DatetimeIndex) -> np.ndarray:
    """Map leap-year dates onto a 365-day circular calendar."""

    day = index.dayofyear.to_numpy(dtype=np.int64)
    leap_after_february = index.is_leap_year & (day > 59)
    day = day.copy()
    day[leap_after_february] -= 1
    return day


class ForecastBaseline(ABC):
    """Common interface shared by all non-trainable forecasting methods."""

    name = "ForecastBaseline"

    def __init__(self, train_dataset: Any) -> None:
        self.train_dataset = train_dataset

    @abstractmethod
    def predict(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any) -> torch.Tensor:
        """Return normalized ``[batch, pred_len, 1]`` predictions."""

    def checkpoint_payload(self) -> dict:
        return {
            "name": self.name,
            "method_name": self.name,
            "method_type": "baseline",
        }

    def _inputs(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any):
        pred_len = _validate_history(history, dataset)
        issue_ns = _issue_times_ns(issue_time_ns, history.shape[0])
        return pred_len, issue_ns


class Persistence(ForecastBaseline):
    name = "Persistence"

    def predict(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any) -> torch.Tensor:
        pred_len, _ = self._inputs(history, issue_time_ns, dataset)
        current = _last_finite_history(history, dataset)
        values = np.repeat(current[:, None, None], pred_len, axis=1)
        return _finish_prediction(values, history)


class SmartPersistence(ForecastBaseline):
    name = "SmartPersistence"

    def predict(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any) -> torch.Tensor:
        pred_len, issue_ns = self._inputs(history, issue_time_ns, dataset)
        current = _last_finite_history(history, dataset)
        future_ns = _future_timestamps(issue_ns, pred_len, dataset)
        query_ns = np.concatenate([issue_ns[:, None], future_ns], axis=1)
        local_times = pd.to_datetime(query_ns.reshape(-1), unit="ns")
        poa = np.asarray(clear_sky_poa(dataset.site_config, local_times), dtype=np.float64).reshape(-1)
        if len(poa) != query_ns.size:
            raise ValueError("clear_sky_poa must return one value per requested timestamp")
        poa = poa.reshape(query_ns.shape)
        denominator = poa[:, :1]
        valid_ratio = (denominator > 1.0) & np.isfinite(denominator)
        ratio = np.zeros((len(current), pred_len), dtype=np.float64)
        np.divide(
            poa[:, 1:],
            denominator,
            out=ratio,
            where=valid_ratio,
        )
        values = current[:, None] * ratio
        return _finish_prediction(values[..., None], history)

    def checkpoint_payload(self) -> dict:
        payload = super().checkpoint_payload()
        payload["clear_sky_model"] = "ineichen"
        payload["clear_sky_denominator_threshold_w_m2"] = 1.0
        return payload


class SeasonalPersistence(ForecastBaseline):
    name = "SeasonalPersistence"

    def predict(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any) -> torch.Tensor:
        pred_len, issue_ns = self._inputs(history, issue_time_ns, dataset)
        current = _last_finite_history(history, dataset)
        future_ns = _future_timestamps(issue_ns, pred_len, dataset)
        previous_day_ns = future_ns - pd.Timedelta(days=1).value
        seasonal = np.asarray(
            dataset.normalized_power_at(previous_day_ns, fallback=current[:, None]),
            dtype=np.float64,
        )
        if seasonal.shape != future_ns.shape:
            seasonal = np.broadcast_to(seasonal, future_ns.shape).copy()
        values = np.where(np.isfinite(seasonal), seasonal, current[:, None])
        return _finish_prediction(values[..., None], history)


class Climatology(ForecastBaseline):
    name = "Climatology"
    window_days = 15

    def __init__(self, train_dataset: Any) -> None:
        super().__init__(train_dataset)
        observations = train_dataset.finite_training_observations
        timestamps = pd.DatetimeIndex(observations["timestamp"])
        values = observations["normalized_power"].to_numpy(dtype=np.float64)
        self.training_mean = _fallback_mean(train_dataset)
        days = _canonical_day_of_year(timestamps)
        minutes = (
            timestamps.hour.to_numpy(dtype=np.int64) * 60
            + timestamps.minute.to_numpy(dtype=np.int64)
        )
        profile = pd.DataFrame({
            "day_of_year": days,
            "minute_of_day": minutes,
            "normalized_power": values,
        }).groupby(["day_of_year", "minute_of_day"], sort=True)["normalized_power"].agg(
            ["sum", "count"]
        ).reset_index()
        self.profile_day = profile["day_of_year"].to_numpy(dtype=np.int64)
        self.profile_minute = profile["minute_of_day"].to_numpy(dtype=np.int64)
        self.profile_sum = profile["sum"].to_numpy(dtype=np.float64)
        self.profile_count = profile["count"].to_numpy(dtype=np.int64)
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Precompute circular-window means for constant-time prediction."""

        lookup = np.full((366, 1440), np.nan, dtype=np.float64)
        kernel = np.ones(2 * self.window_days + 1, dtype=np.float64)
        for minute in np.unique(self.profile_minute):
            same_minute = self.profile_minute == minute
            sums = np.zeros(365, dtype=np.float64)
            counts = np.zeros(365, dtype=np.float64)
            days = self.profile_day[same_minute] - 1
            sums[days] = self.profile_sum[same_minute]
            counts[days] = self.profile_count[same_minute]
            if self.window_days:
                padded_sums = np.concatenate(
                    [sums[-self.window_days :], sums, sums[: self.window_days]]
                )
                padded_counts = np.concatenate(
                    [counts[-self.window_days :], counts, counts[: self.window_days]]
                )
                sums = np.convolve(padded_sums, kernel, mode="valid")
                counts = np.convolve(padded_counts, kernel, mode="valid")
            means = np.divide(
                sums,
                counts,
                out=np.full(365, np.nan, dtype=np.float64),
                where=counts > 0,
            )
            lookup[1:, minute] = means
        self._lookup = lookup

    @classmethod
    def from_checkpoint_payload(cls, payload: dict) -> "Climatology":
        """Restore a fitted climatology without access to its source dataset."""

        if not isinstance(payload, dict) or payload.get("name") != cls.name:
            raise ValueError("checkpoint payload does not describe Climatology")
        profile = payload.get("profile")
        if not isinstance(profile, dict):
            raise ValueError("Climatology checkpoint payload is missing profile")
        try:
            window_days = int(payload["window_days"])
            training_mean = float(payload["training_mean"])
            profile_day = np.asarray(profile["day_of_year"], dtype=np.int64).reshape(-1)
            profile_minute = np.asarray(profile["minute_of_day"], dtype=np.int64).reshape(-1)
            profile_sum = np.asarray(profile["sum_normalized_power"], dtype=np.float64).reshape(-1)
            profile_count = np.asarray(profile["count"], dtype=np.int64).reshape(-1)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Climatology checkpoint payload has an invalid profile") from exc
        lengths = {
            len(profile_day), len(profile_minute), len(profile_sum), len(profile_count),
        }
        if len(lengths) != 1 or window_days < 0 or not np.isfinite(training_mean):
            raise ValueError("Climatology checkpoint payload has an invalid profile")
        if (
            (profile_day < 1).any()
            or (profile_day > 365).any()
            or (profile_minute < 0).any()
            or (profile_minute > 1439).any()
            or (profile_count <= 0).any()
            or not np.isfinite(profile_sum).all()
        ):
            raise ValueError("Climatology checkpoint payload has an invalid profile")

        instance = cls.__new__(cls)
        ForecastBaseline.__init__(instance, None)
        instance.window_days = window_days
        instance.training_mean = training_mean
        instance.profile_day = profile_day
        instance.profile_minute = profile_minute
        instance.profile_sum = profile_sum
        instance.profile_count = profile_count
        instance._build_lookup()
        return instance

    def predict(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any) -> torch.Tensor:
        pred_len, issue_ns = self._inputs(history, issue_time_ns, dataset)
        future_ns = _future_timestamps(issue_ns, pred_len, dataset)
        target_times = pd.DatetimeIndex(
            pd.to_datetime(future_ns.reshape(-1), unit="ns")
        )
        target_day = _canonical_day_of_year(target_times)
        target_minute = (
            target_times.hour.to_numpy(dtype=np.int64) * 60
            + target_times.minute.to_numpy(dtype=np.int64)
        )
        values = self._lookup[target_day, target_minute]
        values = np.where(np.isfinite(values), values, self.training_mean)
        return _finish_prediction(values.reshape(len(issue_ns), pred_len, 1), history)

    def checkpoint_payload(self) -> dict:
        payload = super().checkpoint_payload()
        payload.update(
            {
                "window_days": self.window_days,
                "training_mean": self.training_mean,
                "training_observation_count": int(self.profile_count.sum()),
                "profile": {
                    "day_of_year": self.profile_day.tolist(),
                    "minute_of_day": self.profile_minute.tolist(),
                    "sum_normalized_power": self.profile_sum.tolist(),
                    "count": self.profile_count.tolist(),
                },
            }
        )
        return payload


__all__ = [
    "ForecastBaseline",
    "Persistence",
    "SmartPersistence",
    "SeasonalPersistence",
    "Climatology",
    "clear_sky_poa",
]
