"""Additional classical baselines for power-only photovoltaic forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from .methods import (
    Climatology,
    ForecastBaseline,
    SmartPersistence,
    _fallback_mean,
    _finish_prediction,
    _future_timestamps,
    _last_finite_history,
    clear_sky_poa,
)


CLEAR_SKY_THRESHOLD_W_M2 = 1.0


def _step_ns(dataset: Any) -> int:
    value = int(pd.Timedelta(dataset.step).value)
    if value <= 0:
        raise ValueError("dataset.step must be positive")
    return value


def _forecast_step_ns(dataset: Any) -> int:
    value = int(pd.Timedelta(dataset.forecast_step).value)
    if value <= 0:
        raise ValueError("dataset.forecast_step must be positive")
    return value


def _validate_time_grid(dataset: Any, step_ns: int, forecast_step_ns: int) -> None:
    if _step_ns(dataset) != step_ns or _forecast_step_ns(dataset) != forecast_step_ns:
        raise ValueError("dataset time grid does not match fitted baseline checkpoint")


def _window_points(dataset: Any, window_minutes: int) -> int:
    duration_ns = int(pd.Timedelta(minutes=window_minutes).value)
    return max(1, duration_ns // _step_ns(dataset) + 1)


def _history_timestamps(issue_ns: np.ndarray, history_points: int, dataset: Any) -> np.ndarray:
    offsets = -np.arange(history_points - 1, -1, -1, dtype=np.int64) * _step_ns(dataset)
    return issue_ns[:, None] + offsets[None, :]


def _training_arrays(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    observations = dataset.finite_training_observations
    timestamps = pd.DatetimeIndex(observations["timestamp"])
    values = observations["normalized_power"].to_numpy(dtype=np.float64)
    order = np.argsort(timestamps.asi8)
    return timestamps.asi8[order], values[order]


def _exact_windows(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    anchors_ns: np.ndarray,
    offsets_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query = anchors_ns[:, None] + offsets_ns[None, :]
    positions = np.searchsorted(timestamps_ns, query)
    in_bounds = positions < len(timestamps_ns)
    safe = np.minimum(positions, max(len(timestamps_ns) - 1, 0))
    exact = in_bounds & (timestamps_ns[safe] == query) if len(timestamps_ns) else in_bounds
    result = np.full(query.shape, np.nan, dtype=np.float64)
    if exact.any():
        result[exact] = values[safe[exact]]
    valid = exact.all(axis=1) & np.isfinite(result).all(axis=1)
    return result, valid


class MovingMedian(ForecastBaseline):
    """Repeat the median power observed over the trailing physical hour."""

    name = "MovingMedian"
    window_minutes = 60

    def predict(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any) -> torch.Tensor:
        pred_len, _ = self._inputs(history, issue_time_ns, dataset)
        raw = history.detach().to(dtype=torch.float64).cpu().numpy()[..., 0]
        count = min(raw.shape[1], _window_points(dataset, self.window_minutes))
        fallback = _fallback_mean(dataset)
        medians = np.full(raw.shape[0], fallback, dtype=np.float64)
        for row_index, row in enumerate(raw[:, -count:]):
            finite = row[np.isfinite(row)]
            if len(finite):
                medians[row_index] = float(np.median(finite))
        values = np.repeat(medians[:, None, None], pred_len, axis=1)
        return _finish_prediction(values, history)

    def checkpoint_payload(self) -> dict:
        payload = super().checkpoint_payload()
        payload["window_minutes"] = self.window_minutes
        return payload


class DriftPersistence(ForecastBaseline):
    """Extrapolate the linear drift measured over the trailing physical hour."""

    name = "DriftPersistence"
    window_minutes = 60

    def predict(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any) -> torch.Tensor:
        pred_len, _ = self._inputs(history, issue_time_ns, dataset)
        raw = history.detach().to(dtype=torch.float64).cpu().numpy()[..., 0]
        count = min(raw.shape[1], _window_points(dataset, self.window_minutes))
        recent = raw[:, -count:]
        current = _last_finite_history(history, dataset)
        slope_per_step = np.zeros(raw.shape[0], dtype=np.float64)
        for row_index, row in enumerate(recent):
            finite = np.flatnonzero(np.isfinite(row))
            if len(finite) >= 2:
                first, last = int(finite[0]), int(finite[-1])
                slope_per_step[row_index] = (row[last] - row[first]) / (last - first)
        horizon_scale = float(pd.Timedelta(dataset.forecast_step).value) / _step_ns(dataset)
        horizons = np.arange(1, pred_len + 1, dtype=np.float64) * horizon_scale
        values = current[:, None] + slope_per_step[:, None] * horizons[None, :]
        return _finish_prediction(values[..., None], history)

    def checkpoint_payload(self) -> dict:
        payload = super().checkpoint_payload()
        payload["window_minutes"] = self.window_minutes
        return payload


class ClearSkyEWMA(ForecastBaseline):
    """Forecast an exponentially weighted clear-sky index."""

    name = "ClearSkyEWMA"
    half_life_minutes = 30

    def predict(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any) -> torch.Tensor:
        pred_len, issue_ns = self._inputs(history, issue_time_ns, dataset)
        raw = history.detach().to(dtype=torch.float64).cpu().numpy()[..., 0]
        history_ns = _history_timestamps(issue_ns, raw.shape[1], dataset)
        future_ns = _future_timestamps(issue_ns, pred_len, dataset)
        query_ns = np.concatenate([history_ns, future_ns], axis=1)
        poa = np.asarray(
            clear_sky_poa(dataset.site_config, pd.to_datetime(query_ns.reshape(-1), unit="ns")),
            dtype=np.float64,
        ).reshape(query_ns.shape)
        history_poa = poa[:, : raw.shape[1]]
        future_poa = poa[:, raw.shape[1] :]
        age_minutes = (issue_ns[:, None] - history_ns) / pd.Timedelta(minutes=1).value
        weights = np.exp2(-age_minutes / self.half_life_minutes)
        valid = (
            np.isfinite(raw)
            & np.isfinite(history_poa)
            & (history_poa > CLEAR_SKY_THRESHOLD_W_M2)
        )
        numerator = np.where(valid, raw / np.maximum(history_poa, 1.0) * weights, 0.0).sum(axis=1)
        denominator = np.where(valid, weights, 0.0).sum(axis=1)
        index = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        values = index[:, None] * future_poa
        return _finish_prediction(values[..., None], history)

    def checkpoint_payload(self) -> dict:
        payload = super().checkpoint_payload()
        payload.update(
            {
                "half_life_minutes": self.half_life_minutes,
                "clear_sky_model": "ineichen",
                "clear_sky_denominator_threshold_w_m2": CLEAR_SKY_THRESHOLD_W_M2,
            }
        )
        return payload


class ClearSkyAR(ForecastBaseline):
    """Autoregress the clear-sky index and rescale by future clear-sky POA."""

    name = "ClearSkyAR"
    requested_order = 3
    ridge = 1e-8

    def __init__(self, train_dataset: Any) -> None:
        super().__init__(train_dataset)
        self.step_ns = _step_ns(train_dataset)
        self.forecast_step_ns = _forecast_step_ns(train_dataset)
        timestamps_ns, power = _training_arrays(train_dataset)
        poa = np.asarray(
            clear_sky_poa(
                train_dataset.site_config,
                pd.to_datetime(timestamps_ns, unit="ns"),
            ),
            dtype=np.float64,
        )
        valid = np.isfinite(power) & np.isfinite(poa) & (poa > CLEAR_SKY_THRESHOLD_W_M2)
        index = np.full(len(power), np.nan, dtype=np.float64)
        index[valid] = power[valid] / poa[valid]
        self.training_index_mean = float(np.mean(index[valid])) if valid.any() else 0.0
        self.order, self.intercept, self.coefficients = self._fit(
            timestamps_ns, index, _step_ns(train_dataset)
        )

    def _fit(
        self,
        timestamps_ns: np.ndarray,
        index: np.ndarray,
        step_ns: int,
    ) -> tuple[int, float, np.ndarray]:
        maximum_order = min(self.requested_order, max(len(index) - 1, 1))
        for order in range(maximum_order, 0, -1):
            rows = []
            targets = []
            for position in range(order, len(index)):
                segment = index[position - order : position + 1]
                time_segment = timestamps_ns[position - order : position + 1]
                if np.isfinite(segment).all() and np.all(np.diff(time_segment) == step_ns):
                    rows.append(segment[:-1][::-1])
                    targets.append(segment[-1])
            if rows:
                design = np.column_stack([np.ones(len(rows)), np.asarray(rows)])
                target = np.asarray(targets, dtype=np.float64)
                penalty = np.eye(design.shape[1], dtype=np.float64) * self.ridge
                penalty[0, 0] = 0.0
                fitted = np.linalg.solve(design.T @ design + penalty, design.T @ target)
                return order, float(fitted[0]), np.asarray(fitted[1:], dtype=np.float64)
        return 1, self.training_index_mean, np.zeros(1, dtype=np.float64)

    @classmethod
    def from_checkpoint_payload(cls, payload: dict) -> "ClearSkyAR":
        if not isinstance(payload, dict) or payload.get("name") != cls.name:
            raise ValueError("checkpoint payload does not describe ClearSkyAR")
        try:
            order = int(payload["order"])
            intercept = float(payload["intercept"])
            coefficients = np.asarray(payload["coefficients"], dtype=np.float64).reshape(-1)
            training_index_mean = float(payload["training_index_mean"])
            step_ns = int(payload["step_ns"])
            forecast_step_ns = int(payload["forecast_step_ns"])
            ridge = float(payload["ridge"])
            clear_sky_model = payload["clear_sky_model"]
            threshold = float(payload["clear_sky_denominator_threshold_w_m2"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ClearSkyAR checkpoint payload is invalid") from exc
        if (
            order <= 0
            or len(coefficients) != order
            or not np.isfinite(coefficients).all()
            or not np.isfinite(intercept)
            or not np.isfinite(training_index_mean)
            or step_ns <= 0
            or forecast_step_ns <= 0
            or not np.isfinite(ridge)
            or ridge < 0
            or clear_sky_model != "ineichen"
            or threshold != CLEAR_SKY_THRESHOLD_W_M2
        ):
            raise ValueError("ClearSkyAR checkpoint payload is invalid")
        instance = cls.__new__(cls)
        ForecastBaseline.__init__(instance, None)
        instance.order = order
        instance.intercept = intercept
        instance.coefficients = coefficients
        instance.training_index_mean = training_index_mean
        instance.step_ns = step_ns
        instance.forecast_step_ns = forecast_step_ns
        instance.ridge = ridge
        return instance

    def predict(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any) -> torch.Tensor:
        pred_len, issue_ns = self._inputs(history, issue_time_ns, dataset)
        _validate_time_grid(dataset, self.step_ns, self.forecast_step_ns)
        raw = history.detach().to(dtype=torch.float64).cpu().numpy()[..., 0]
        history_ns = _history_timestamps(issue_ns, raw.shape[1], dataset)
        future_ns = _future_timestamps(issue_ns, pred_len, dataset)
        query_ns = np.concatenate([history_ns, future_ns], axis=1)
        poa = np.asarray(
            clear_sky_poa(dataset.site_config, pd.to_datetime(query_ns.reshape(-1), unit="ns")),
            dtype=np.float64,
        ).reshape(query_ns.shape)
        history_poa = poa[:, : raw.shape[1]]
        future_poa = poa[:, raw.shape[1] :]
        valid = (
            np.isfinite(raw)
            & np.isfinite(history_poa)
            & (history_poa > CLEAR_SKY_THRESHOLD_W_M2)
        )
        history_index = np.divide(
            raw,
            history_poa,
            out=np.full_like(raw, self.training_index_mean),
            where=valid,
        )
        ratio = float(pd.Timedelta(dataset.forecast_step).value) / _step_ns(dataset)
        substeps = int(round(ratio))
        if substeps <= 0 or not np.isclose(ratio, substeps):
            raise ValueError("ClearSkyAR requires forecast_step to be an integer multiple of step")
        predicted_index = np.zeros((len(issue_ns), pred_len), dtype=np.float64)
        for row_index, row in enumerate(history_index):
            state = [self.training_index_mean] * max(0, self.order - len(row)) + row.tolist()
            for horizon in range(pred_len):
                for _ in range(substeps):
                    lags = np.asarray(state[-self.order :][::-1], dtype=np.float64)
                    next_value = self.intercept + float(self.coefficients @ lags)
                    state.append(
                        max(0.0, next_value)
                        if np.isfinite(next_value)
                        else self.training_index_mean
                    )
                predicted_index[row_index, horizon] = state[-1]
        values = predicted_index * future_poa
        return _finish_prediction(values[..., None], history)

    def checkpoint_payload(self) -> dict:
        payload = super().checkpoint_payload()
        payload.update(
            {
                "order": self.order,
                "intercept": self.intercept,
                "coefficients": self.coefficients.tolist(),
                "training_index_mean": self.training_index_mean,
                "step_ns": self.step_ns,
                "forecast_step_ns": self.forecast_step_ns,
                "ridge": self.ridge,
                "clear_sky_model": "ineichen",
                "clear_sky_denominator_threshold_w_m2": CLEAR_SKY_THRESHOLD_W_M2,
            }
        )
        return payload


class SimilarDay(ForecastBaseline):
    """Average the nearest training-day trajectories at the same clock time."""

    name = "SimilarDay"
    window_minutes = 60
    neighbors = 3

    def __init__(self, train_dataset: Any) -> None:
        super().__init__(train_dataset)
        self.step_ns = _step_ns(train_dataset)
        self.forecast_step_ns = _forecast_step_ns(train_dataset)
        timestamps_ns, values = _training_arrays(train_dataset)
        history_points = _window_points(train_dataset, self.window_minutes)
        self.history_points = history_points
        self.pred_len = int(train_dataset.pred_len)
        history_offsets = -np.arange(history_points - 1, -1, -1, dtype=np.int64) * _step_ns(
            train_dataset
        )
        future_offsets = np.arange(1, train_dataset.pred_len + 1, dtype=np.int64) * int(
            pd.Timedelta(train_dataset.forecast_step).value
        )
        candidate_history, history_valid = _exact_windows(
            timestamps_ns, values, timestamps_ns, history_offsets
        )
        candidate_future, future_valid = _exact_windows(
            timestamps_ns, values, timestamps_ns, future_offsets
        )
        valid = history_valid & future_valid
        self.candidate_issue_time_ns = timestamps_ns[valid]
        self.candidate_history = candidate_history[valid]
        self.candidate_future = candidate_future[valid]

    @classmethod
    def from_checkpoint_payload(cls, payload: dict) -> "SimilarDay":
        if not isinstance(payload, dict) or payload.get("name") != cls.name:
            raise ValueError("checkpoint payload does not describe SimilarDay")
        try:
            issue = np.asarray(payload["candidate_issue_time_ns"], dtype=np.int64).reshape(-1)
            history_points = int(payload["history_points"])
            pred_len = int(payload["pred_len"])
            history = np.asarray(payload["candidate_history"], dtype=np.float64).reshape(
                len(issue), history_points
            )
            future = np.asarray(payload["candidate_future"], dtype=np.float64).reshape(
                len(issue), pred_len
            )
            window_minutes = int(payload["window_minutes"])
            neighbors = int(payload["neighbors"])
            step_ns = int(payload["step_ns"])
            forecast_step_ns = int(payload["forecast_step_ns"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("SimilarDay checkpoint payload is invalid") from exc
        if (
            history.ndim != 2
            or future.ndim != 2
            or len(issue) != len(history)
            or len(issue) != len(future)
            or window_minutes <= 0
            or neighbors <= 0
            or history_points <= 0
            or pred_len <= 0
            or step_ns <= 0
            or forecast_step_ns <= 0
            or not np.isfinite(history).all()
            or not np.isfinite(future).all()
        ):
            raise ValueError("SimilarDay checkpoint payload is invalid")
        instance = cls.__new__(cls)
        ForecastBaseline.__init__(instance, None)
        instance.window_minutes = window_minutes
        instance.neighbors = neighbors
        instance.history_points = history_points
        instance.pred_len = pred_len
        instance.step_ns = step_ns
        instance.forecast_step_ns = forecast_step_ns
        instance.candidate_issue_time_ns = issue
        instance.candidate_history = history
        instance.candidate_future = future
        return instance

    def predict(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any) -> torch.Tensor:
        pred_len, issue_ns = self._inputs(history, issue_time_ns, dataset)
        _validate_time_grid(dataset, self.step_ns, self.forecast_step_ns)
        if self.pred_len != pred_len or self.candidate_future.shape[1] != pred_len:
            raise ValueError("SimilarDay checkpoint pred_len does not match dataset")
        raw = history.detach().to(dtype=torch.float64).cpu().numpy()[..., 0]
        current = _last_finite_history(history, dataset)
        values = np.repeat(current[:, None], pred_len, axis=1)
        target_end_offset = pred_len * int(pd.Timedelta(dataset.forecast_step).value)
        candidate_minutes = pd.to_datetime(self.candidate_issue_time_ns, unit="ns").hour * 60
        candidate_minutes = np.asarray(candidate_minutes) + np.asarray(
            pd.to_datetime(self.candidate_issue_time_ns, unit="ns").minute
        )
        issue_times = pd.to_datetime(issue_ns, unit="ns")
        issue_minutes = np.asarray(issue_times.hour * 60 + issue_times.minute)
        for row_index, row in enumerate(raw):
            width = min(len(row), self.candidate_history.shape[1])
            observed = row[-width:]
            candidates = self.candidate_history[:, -width:]
            finite = np.isfinite(observed)
            eligible = (
                (candidate_minutes == issue_minutes[row_index])
                & (self.candidate_issue_time_ns + target_end_offset <= issue_ns[row_index])
            )
            if not finite.any() or not eligible.any():
                continue
            distance = np.mean((candidates[:, finite] - observed[finite]) ** 2, axis=1)
            indices = np.flatnonzero(eligible)
            nearest = indices[np.argsort(distance[indices], kind="stable")[: self.neighbors]]
            if len(nearest):
                values[row_index] = np.mean(self.candidate_future[nearest], axis=0)
        return _finish_prediction(values[..., None], history)

    def checkpoint_payload(self) -> dict:
        payload = super().checkpoint_payload()
        payload.update(
            {
                "window_minutes": self.window_minutes,
                "neighbors": self.neighbors,
                "history_points": self.history_points,
                "pred_len": self.pred_len,
                "step_ns": self.step_ns,
                "forecast_step_ns": self.forecast_step_ns,
                "candidate_issue_time_ns": self.candidate_issue_time_ns.tolist(),
                "candidate_history": self.candidate_history.tolist(),
                "candidate_future": self.candidate_future.tolist(),
            }
        )
        return payload


class PersistenceClimatologyBlend(ForecastBaseline):
    """Blend Smart Persistence and Climatology with fitted horizon weights."""

    name = "PersistenceClimatologyBlend"
    max_fit_samples = 2048

    def __init__(self, train_dataset: Any) -> None:
        super().__init__(train_dataset)
        self.step_ns = _step_ns(train_dataset)
        self.forecast_step_ns = _forecast_step_ns(train_dataset)
        self.smart_persistence = SmartPersistence(train_dataset)
        self.climatology = Climatology(train_dataset)
        self.weights = self._fit_weights(train_dataset)

    def _fit_weights(self, dataset: Any) -> np.ndarray:
        timestamps_ns, values = _training_arrays(dataset)
        history_offsets = -np.arange(dataset.seq_len - 1, -1, -1, dtype=np.int64) * _step_ns(
            dataset
        )
        future_offsets = np.arange(1, dataset.pred_len + 1, dtype=np.int64) * int(
            pd.Timedelta(dataset.forecast_step).value
        )
        histories, history_valid = _exact_windows(
            timestamps_ns, values, timestamps_ns, history_offsets
        )
        targets, target_valid = _exact_windows(timestamps_ns, values, timestamps_ns, future_offsets)
        valid_positions = np.flatnonzero(history_valid & target_valid)
        if len(valid_positions) > self.max_fit_samples:
            selected = np.linspace(
                0, len(valid_positions) - 1, self.max_fit_samples, dtype=np.int64
            )
            valid_positions = valid_positions[selected]
        if not len(valid_positions):
            return np.full(dataset.pred_len, 0.5, dtype=np.float64)
        history_tensor = torch.as_tensor(
            histories[valid_positions, :, None], dtype=torch.float64
        )
        issues = timestamps_ns[valid_positions]
        smart = self.smart_persistence.predict(history_tensor, issues, dataset).numpy()[..., 0]
        climate = self.climatology.predict(history_tensor, issues, dataset).numpy()[..., 0]
        target = targets[valid_positions]
        delta = smart - climate
        numerator = np.sum(delta * (target - climate), axis=0)
        denominator = np.sum(delta * delta, axis=0)
        weights = np.divide(
            numerator,
            denominator,
            out=np.full(dataset.pred_len, 0.5, dtype=np.float64),
            where=denominator > 1e-12,
        )
        return np.clip(weights, 0.0, 1.0)

    @classmethod
    def from_checkpoint_payload(cls, payload: dict) -> "PersistenceClimatologyBlend":
        if not isinstance(payload, dict) or payload.get("name") != cls.name:
            raise ValueError("checkpoint payload does not describe PersistenceClimatologyBlend")
        try:
            weights = np.asarray(payload["weights"], dtype=np.float64).reshape(-1)
            climatology = Climatology.from_checkpoint_payload(payload["climatology"])
            step_ns = int(payload["step_ns"])
            forecast_step_ns = int(payload["forecast_step_ns"])
            smart_payload = payload["smart_persistence"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("PersistenceClimatologyBlend checkpoint payload is invalid") from exc
        smart_valid = (
            isinstance(smart_payload, dict)
            and smart_payload.get("name") == SmartPersistence.name
            and smart_payload.get("clear_sky_model") == "ineichen"
            and smart_payload.get("clear_sky_denominator_threshold_w_m2")
            == CLEAR_SKY_THRESHOLD_W_M2
        )
        if (
            not len(weights)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or (weights > 1).any()
            or step_ns <= 0
            or forecast_step_ns <= 0
            or not smart_valid
        ):
            raise ValueError("PersistenceClimatologyBlend checkpoint payload is invalid")
        instance = cls.__new__(cls)
        ForecastBaseline.__init__(instance, None)
        instance.smart_persistence = SmartPersistence(None)
        instance.climatology = climatology
        instance.weights = weights
        instance.step_ns = step_ns
        instance.forecast_step_ns = forecast_step_ns
        return instance

    def predict(self, history: torch.Tensor, issue_time_ns: Any, dataset: Any) -> torch.Tensor:
        pred_len, _ = self._inputs(history, issue_time_ns, dataset)
        _validate_time_grid(dataset, self.step_ns, self.forecast_step_ns)
        if len(self.weights) != pred_len:
            raise ValueError("blend checkpoint pred_len does not match dataset")
        smart = self.smart_persistence.predict(history, issue_time_ns, dataset)
        climate = self.climatology.predict(history, issue_time_ns, dataset)
        weights = torch.as_tensor(
            self.weights, dtype=smart.dtype, device=smart.device
        ).view(1, -1, 1)
        blended = weights * smart + (1.0 - weights) * climate
        return _finish_prediction(blended.cpu().numpy(), history)

    def checkpoint_payload(self) -> dict:
        payload = super().checkpoint_payload()
        payload.update(
            {
                "weights": self.weights.tolist(),
                "step_ns": self.step_ns,
                "forecast_step_ns": self.forecast_step_ns,
                "fit_sample_limit": self.max_fit_samples,
                "smart_persistence": self.smart_persistence.checkpoint_payload(),
                "climatology": self.climatology.checkpoint_payload(),
            }
        )
        return payload


__all__ = [
    "MovingMedian",
    "DriftPersistence",
    "ClearSkyEWMA",
    "ClearSkyAR",
    "SimilarDay",
    "PersistenceClimatologyBlend",
]
