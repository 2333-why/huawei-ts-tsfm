"""Dedicated power-only Parquet dataset for the pure time-series runner."""

from __future__ import annotations

import json
from pathlib import Path
from numbers import Integral
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Dataset


def load_power_only_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON or YAML power-only dataset configuration."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            config = json.load(handle)
        else:
            config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("power-only dataset config must contain a mapping")
    for section in ("paths", "fields", "time", "power"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"power-only config requires a {section} mapping")
    return config


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class PowerOnlyParquetDataset(Dataset):
    """Return exact, normalized power windows from a two-column Parquet file."""

    def __init__(
        self,
        config_path: Union[str, Path],
        flag: str = "train",
        history_points: Optional[int] = None,
        forecast_steps: Optional[int] = None,
    ) -> None:
        if flag == "valid":
            flag = "val"
        if flag not in {"train", "val", "test"}:
            raise ValueError("flag must be train, val, or test")

        self.config_path = Path(config_path).resolve()
        self.config = load_power_only_config(self.config_path)
        self.flag = flag
        self.paths = self.config["paths"]
        self.fields = self.config["fields"]
        self.time_config = self.config["time"]
        self.power_config = self.config["power"]
        if history_points is None:
            history_points = self.config.get("history_points")
        if forecast_steps is None:
            forecast_steps = self.config.get("forecast_steps")
        if history_points is None or forecast_steps is None:
            raise ValueError("history_points and forecast_steps are required")
        self.seq_len = _positive_integer(history_points, "history_points")
        self.pred_len = _positive_integer(forecast_steps, "forecast_steps")

        self.step = pd.Timedelta(
            minutes=_positive_integer(
                self.time_config["sampling_interval_minutes"],
                "sampling_interval_minutes",
            )
        )
        self.forecast_step = pd.Timedelta(
            minutes=_positive_integer(
                self.time_config["forecast_step_minutes"],
                "forecast_step_minutes",
            )
        )
        self.power_scale = float(self.power_config["power_scale"])
        self.rated_power = float(self.power_config["rated_power"])
        if not np.isfinite(self.power_scale) or self.power_scale <= 0:
            raise ValueError("power_scale must be finite and positive")
        if not np.isfinite(self.rated_power) or self.rated_power <= 0:
            raise ValueError("rated_power must be finite and positive")
        self.units = self.power_config.get("units")
        self.target_floor = (
            float(self.power_config["target_floor"])
            if "target_floor" in self.power_config
            else None
        )
        if self.target_floor is not None and not np.isfinite(self.target_floor):
            raise ValueError("target_floor must be finite when configured")

        parquet_path = Path(self.paths["parquet_file"])
        if not parquet_path.is_absolute():
            parquet_path = self.config_path.parent / parquet_path
        if not parquet_path.is_file():
            raise FileNotFoundError(f"power-only Parquet file not found: {parquet_path}")
        self.parquet_path = parquet_path.resolve()
        timestamp_name = self.fields["timestamp"]
        target_name = self.fields["target"]
        frame = pd.read_parquet(self.parquet_path, columns=[timestamp_name, target_name])
        if timestamp_name not in frame or target_name not in frame:
            raise ValueError("Parquet file must contain configured timestamp and target columns")
        frame[timestamp_name] = pd.to_datetime(frame[timestamp_name], errors="coerce")
        if frame[timestamp_name].isna().any():
            raise ValueError("timestamp must contain finite values")
        if frame[timestamp_name].duplicated().any():
            raise ValueError("timestamp must be unique")
        frame[target_name] = pd.to_numeric(frame[target_name], errors="coerce")
        if self.target_floor is not None:
            finite = np.isfinite(frame[target_name].to_numpy(np.float64, na_value=np.nan))
            values = frame[target_name].to_numpy(np.float64, na_value=np.nan)
            values[finite] = np.maximum(values[finite], self.target_floor)
            frame[target_name] = values
        self.frame = frame.sort_values(timestamp_name).reset_index(drop=True)
        self.timestamps = pd.DatetimeIndex(self.frame[timestamp_name])
        self._timestamp_ns = self.timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self._target_values_raw = self.frame[target_name].to_numpy(np.float64)

        candidates = self._valid_candidates()
        test_start = pd.Timestamp(self.time_config["test_start_timestamp"])
        pre_test_candidates = candidates[candidates < test_start]
        validation_fraction = float(self.time_config["validation_fraction"])
        if not 0.0 <= validation_fraction <= 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        split_at = int(np.floor(len(pre_test_candidates) * (1.0 - validation_fraction)))
        self.validation_start = (
            pre_test_candidates[split_at]
            if len(pre_test_candidates) and split_at < len(pre_test_candidates)
            else test_start
        )

        horizon = self.pred_len * self.forecast_step
        if flag == "train":
            selected = candidates[
                (candidates < self.validation_start)
                & (candidates + horizon < self.validation_start)
            ]
        elif flag == "val":
            selected = candidates[
                (candidates >= self.validation_start)
                & (candidates < test_start)
                & (candidates + horizon < test_start)
            ]
        else:
            test_end = pd.Timestamp(self.time_config["test_end_exclusive_timestamp"])
            selected = candidates[
                (candidates >= test_start)
                & (candidates + horizon < test_end)
            ]
        self.sample_times = selected.to_numpy(dtype="datetime64[ns]")
        self.candidate_audit["selected_sample_count"] = int(len(selected))
        self.train_mean = self._compute_train_mean()

    def _valid_candidates(self) -> pd.DatetimeIndex:
        train_start = pd.Timestamp(self.time_config["train_start_timestamp"])
        test_end = pd.Timestamp(self.time_config["test_end_exclusive_timestamp"])
        observed = self.timestamps[(self.timestamps >= train_start) & (self.timestamps < test_end)]
        issue_ns = observed.to_numpy(dtype="datetime64[ns]").astype(np.int64)

        def exact_windows(offsets: np.ndarray) -> np.ndarray:
            if not len(self._timestamp_ns):
                return np.zeros(len(issue_ns), dtype=bool)
            queries = issue_ns[:, None] + offsets[None, :]
            positions = np.searchsorted(self._timestamp_ns, queries)
            in_bounds = positions < len(self._timestamp_ns)
            safe_positions = np.minimum(positions, len(self._timestamp_ns) - 1)
            return (
                in_bounds
                & (self._timestamp_ns[safe_positions] == queries)
            ).all(axis=1)

        history_offsets = -np.arange(self.seq_len - 1, -1, -1, dtype=np.int64) * self.step.value
        future_offsets = np.arange(1, self.pred_len + 1, dtype=np.int64) * self.forecast_step.value
        valid = exact_windows(history_offsets) & exact_windows(future_offsets)
        valid_candidates = observed[valid]
        theoretical = pd.date_range(
            start=train_start, end=test_end, freq=self.step, inclusive="left"
        )
        self.candidate_audit = {
            "theoretical_issue_count": int(len(theoretical)),
            "observed_issue_count": int(len(observed)),
            "continuous_window_count": int(valid.sum()),
            "valid_candidate_count": int(len(valid_candidates)),
        }
        return pd.DatetimeIndex(valid_candidates)

    def _compute_train_mean(self) -> float:
        train_start = pd.Timestamp(self.time_config["train_start_timestamp"])
        values = self._target_values_raw[
            (self.timestamps >= train_start) & (self.timestamps < self.validation_start)
        ]
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError("training interval has no finite target values")
        return float(finite.mean())

    @staticmethod
    def _fill_history(values: np.ndarray, train_mean: float) -> np.ndarray:
        filled = values.astype(np.float64, copy=True)
        last = np.nan
        for index, value in enumerate(filled):
            if np.isfinite(value):
                last = value
            elif np.isfinite(last):
                filled[index] = last
        filled[~np.isfinite(filled)] = train_mean
        return filled

    def _window_positions(self, issue_ns: int) -> tuple[np.ndarray, np.ndarray]:
        history_offsets = -np.arange(self.seq_len - 1, -1, -1, dtype=np.int64) * self.step.value
        future_offsets = np.arange(1, self.pred_len + 1, dtype=np.int64) * self.forecast_step.value
        history_positions = np.searchsorted(self._timestamp_ns, issue_ns + history_offsets)
        future_positions = np.searchsorted(self._timestamp_ns, issue_ns + future_offsets)
        return history_positions, future_positions

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        issue_ns = int(np.asarray(self.sample_times[item], dtype="datetime64[ns]").astype(np.int64))
        history_positions, future_positions = self._window_positions(issue_ns)
        history_raw = self._target_values_raw[history_positions]
        history_raw = self._fill_history(history_raw, self.train_mean)
        future_raw = self._target_values_raw[future_positions]
        target_mask = np.isfinite(future_raw)
        future_raw = np.where(target_mask, future_raw, 0.0)
        return {
            "history": (history_raw / self.power_scale).astype(np.float32).reshape(self.seq_len, 1),
            "target": (future_raw / self.power_scale).astype(np.float32).reshape(self.pred_len, 1),
            "target_mask": target_mask.astype(bool),
            "issue_time_ns": np.int64(issue_ns),
        }

    def __len__(self) -> int:
        return len(self.sample_times)

    def inverse_transform_power(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.power_scale


__all__ = ["PowerOnlyParquetDataset", "load_power_only_config"]
