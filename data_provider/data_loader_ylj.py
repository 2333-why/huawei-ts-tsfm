"""Configuration-driven, causal YLJ Parquet adapter for FACTS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Dataset


SOURCE_RAW_OR_OWN = np.int8(0)
SOURCE_FORWARD_FILL_OR_ALTERNATIVE = np.int8(1)
SOURCE_PERSISTENCE = np.int8(2)
SOURCE_TRAIN_MEAN = np.int8(3)


def load_ylj_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    fields, dims, timing, power = (
        config["fields"], config["dimensions"], config["time"], config["power"])
    columns = fields["time_series_columns"]
    if config.get("dataset") != "YLJParquet":
        raise ValueError("YLJ config dataset must be YLJParquet")
    if len(columns) != len(set(columns)):
        raise ValueError("YLJ time_series_columns must not contain duplicates")
    if len(columns) != int(dims["time_series_input_dim"]):
        raise ValueError("YLJ field count must equal time_series_input_dim")
    if fields["target"] != columns[0] or "forecast" in fields["target"]:
        raise ValueError("YLJ target must be the first non-forecast channel")
    if fields["history_columns"] + fields["forecast_window_columns"] != columns:
        raise ValueError("YLJ history and forecast field lists must exactly form time_series_columns")
    if int(dims["history_points"]) != int(dims["forecast_steps"]):
        raise ValueError("YLJ shifted forecast layout requires history_points == forecast_steps")
    if int(dims["forecast_steps"]) * int(timing["forecast_step_minutes"]) != int(
        timing["forecast_horizon_minutes"]):
        raise ValueError("YLJ forecast steps and horizon are inconsistent")
    valid_horizons = set(range(
        int(timing["forecast_step_minutes"]),
        int(timing["forecast_horizon_minutes"]) + 1,
        int(timing["forecast_step_minutes"])))
    if not set(map(int, timing["evaluation_horizons_minutes"])).issubset(valid_horizons):
        raise ValueError("YLJ evaluation horizon is not a model output")
    if int(timing["forecast_window_shift_minutes"]) % int(timing["sampling_interval_minutes"]):
        raise ValueError("YLJ forecast shift must be divisible by sampling interval")
    if int(timing["forecast_window_shift_minutes"]) != int(timing["forecast_horizon_minutes"]):
        raise ValueError("YLJ forecast shift must equal the configured future window")
    available = set(columns)
    if set(fields["time_series_scale_factors"]) - available:
        raise ValueError("YLJ scale factor references an unknown field")
    if set(fields["forecast_fallbacks"]) != set(fields["forecast_window_columns"]):
        raise ValueError("YLJ every forecast field must have exactly one fallback mapping")
    for name, fallback in fields["forecast_fallbacks"].items():
        if name not in fields["forecast_window_columns"]:
            raise ValueError(f"fallback key is not a forecast field: {name}")
        if fallback["alternative"] not in available or fallback["persistence"] not in available:
            raise ValueError(f"fallback field is missing for {name}")
    capacity = float(power["rated_power"])
    if not (float(power["power_scale"]) == capacity == float(power["max_power"])):
        raise ValueError("YLJ power_scale, max_power and rated_power must be identical")
    if capacity <= 0:
        raise ValueError("YLJ rated_power must be positive")
    if bool(config["images"]["images_enabled"]):
        raise ValueError("the current YLJ contract requires images_enabled=false")
    privileged = config["privileged_teacher"]
    columns = privileged["future_timeseries_columns"]
    if fields["target"] in columns:
        raise ValueError("YLJ privileged Teacher columns must exclude observe_power")
    if not columns or not set(columns).issubset(fields["history_columns"]):
        raise ValueError("YLJ privileged Teacher columns must be observed exogenous fields")
    if len(columns) != int(dims["weather_placeholder_dim"]):
        raise ValueError("YLJ weather_placeholder_dim must match privileged columns")
    return config


class YLJParquetDataset(Dataset):
    """Return causal YLJ samples with configurable point-count windows."""

    def __init__(self, config_path: Union[str, Path], flag: str = "train",
                 privileged_teacher: bool = False,
                 history_points: Optional[int] = None,
                 forecast_steps: Optional[int] = None,
                 power_only: bool = False):
        if flag == "valid":
            flag = "val"
        if flag not in {"train", "val", "test"}:
            raise ValueError("flag must be train, val, or test")
        self.config_path = Path(config_path).resolve()
        self.config = load_ylj_config(self.config_path)
        self.flag = flag
        self.privileged_teacher = bool(privileged_teacher)
        self.power_only = bool(power_only)
        self.fields = self.config["fields"]
        self.dims = self.config["dimensions"]
        self.timing = self.config["time"]
        self.power = self.config["power"]
        self.seq_len = int(
            self.dims["history_points"] if history_points is None
            else history_points
        )
        self.pred_len = int(
            self.dims["forecast_steps"] if forecast_steps is None
            else forecast_steps
        )
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("history_points and forecast_steps must be positive")
        self.step = pd.Timedelta(minutes=int(self.timing["sampling_interval_minutes"]))
        self.forecast_step = pd.Timedelta(minutes=int(self.timing["forecast_step_minutes"]))
        self.power_scale = float(self.power["power_scale"])
        self.target_floor = float(self.power["target_floor"])
        self.scale_factors = {
            name: float(value) for name, value in self.fields["time_series_scale_factors"].items()}

        parquet_path = Path(self.config["paths"]["parquet_file"])
        if not parquet_path.is_file():
            raise FileNotFoundError(f"YLJ Parquet file not found: {parquet_path}")
        required = {self.fields["timestamp"], *self.fields["time_series_columns"]}
        frame = pd.read_parquet(parquet_path, columns=sorted(required))
        timestamp = self.fields["timestamp"]
        frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce")
        if frame[timestamp].isna().any() or frame[timestamp].duplicated().any():
            raise ValueError("YLJ timestamp must be finite and unique")
        for name in self.fields["time_series_columns"]:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
            if name in self.scale_factors:
                frame[name] = frame[name] * self.scale_factors[name]
        target = self.fields["target"]
        frame[target] = frame[target].clip(lower=self.target_floor)
        self.frame = frame.sort_values(timestamp).set_index(timestamp, drop=False)
        self.timestamps = self.frame.index

        candidates = self._valid_candidates()
        dev_end = pd.Timestamp(self.timing["test_start_timestamp"])
        dev = candidates[candidates < dev_end]
        split_at = int(np.floor(len(dev) * (1.0 - float(self.timing["validation_fraction"]))))
        self.validation_start = dev[split_at] if len(dev) and split_at < len(dev) else dev_end
        horizon = self.pred_len * self.forecast_step
        if flag == "train":
            selected = candidates[(candidates < self.validation_start) & (candidates + horizon < self.validation_start)]
        elif flag == "val":
            selected = candidates[(candidates >= self.validation_start) & (candidates < dev_end) & (candidates + horizon < dev_end)]
        else:
            test_end = pd.Timestamp(self.timing["test_end_exclusive_timestamp"])
            selected = candidates[(candidates >= dev_end) & (candidates + horizon < test_end)]
        self.candidate_audit["selected_sample_count"] = int(len(selected))
        self.sample_times = selected.to_numpy(dtype="datetime64[ns]")
        self._compute_train_statistics()
        self._persistence = {
            name: self.frame[name].ffill() for name in set(
                value["persistence"] for value in self.fields["forecast_fallbacks"].values())}
        self.target_masks = np.stack([self._target_values(pd.Timestamp(issue))[1] for issue in selected]) if len(selected) else np.empty((0, self.pred_len), bool)
        current_raw = self.frame.loc[selected, target].to_numpy(np.float32) if len(selected) else np.empty(0, np.float32)
        self.current_power_valid = np.isfinite(current_raw)
        self.current_powers = np.where(self.current_power_valid, current_raw, self.train_means[0]).astype(np.float32)

    def _valid_candidates(self):
        start = pd.Timestamp(self.timing["train_start_timestamp"])
        end = pd.Timestamp(self.timing["test_end_exclusive_timestamp"])
        observed_issues = self.timestamps[
            (self.timestamps >= start) & (self.timestamps < end)
        ]
        # pandas 3 may store a DatetimeIndex at microsecond resolution.  Cast
        # explicitly so offsets based on Timedelta.value (nanoseconds) match.
        timestamp_ns = np.asarray(
            self.timestamps.astype("datetime64[ns]").astype(np.int64),
            dtype=np.int64,
        )
        issue_ns = np.asarray(
            observed_issues.astype("datetime64[ns]").astype(np.int64),
            dtype=np.int64,
        )

        def exact_positions(offsets: np.ndarray):
            queries = issue_ns[:, None] + offsets[None, :]
            positions = np.searchsorted(timestamp_ns, queries)
            safe_positions = np.minimum(positions, len(timestamp_ns) - 1)
            exact = ((positions < len(timestamp_ns)) &
                     (timestamp_ns[safe_positions] == queries)).all(axis=1)
            return exact, positions

        history_offsets = -np.arange(
            self.seq_len - 1, -1, -1, dtype=np.int64
        ) * self.step.value
        future_offsets = np.arange(
            1, self.pred_len + 1, dtype=np.int64
        ) * self.forecast_step.value
        history_valid, _ = exact_positions(history_offsets)
        future_valid, future_positions = exact_positions(future_offsets)
        continuous = history_valid & future_valid
        safe_future_positions = np.minimum(future_positions, len(timestamp_ns) - 1)
        target_values = self.frame[self.fields["target"]].to_numpy(np.float64)
        target_has_value = np.isfinite(target_values[safe_future_positions]).any(axis=1)
        valid = continuous & target_has_value
        valid_candidates = observed_issues[valid]
        theoretical_issues = pd.date_range(
            start=start, end=end, freq=self.step, inclusive="left"
        )
        self.candidate_audit = {
            "theoretical_issue_count": int(len(theoretical_issues)),
            "observed_issue_count": int(len(observed_issues)),
            "continuous_window_count": int(continuous.sum()),
            "target_all_missing_rejected_count": int(
                (continuous & ~target_has_value).sum()
            ),
            "valid_candidate_count": int(len(valid_candidates)),
        }
        return pd.DatetimeIndex(valid_candidates)

    def _compute_train_statistics(self):
        train_rows = self.frame[
            (self.timestamps >= pd.Timestamp(self.timing["train_start_timestamp"]))
            & (self.timestamps < self.validation_start)]
        means, stds = [], []
        for index, name in enumerate(self.fields["time_series_columns"]):
            values = train_rows[name].to_numpy(np.float64)
            finite = values[np.isfinite(values)]
            if not finite.size:
                raise ValueError(f"YLJ training interval has no finite values for {name}")
            mean, std = float(finite.mean()), float(finite.std())
            means.append(mean)
            stds.append(std if std > float(self.config["missing"]["normalization_epsilon"]) else 1.0)
        self.train_means = np.asarray(means, np.float32)
        self.train_stds = np.asarray(stds, np.float32)

    @staticmethod
    def _time_marks(times: Sequence[pd.Timestamp]):
        return np.asarray([[v.month, v.day, v.weekday(), v.hour, v.minute] for v in times], np.float32)

    def _history(self, times, return_source=False):
        names = self.fields["history_columns"]
        raw = self.frame.loc[list(times), names].to_numpy(np.float32)
        mask = np.isfinite(raw)
        filled = pd.DataFrame(raw).ffill().to_numpy(np.float32).copy()
        source = np.where(
            mask, SOURCE_RAW_OR_OWN, SOURCE_TRAIN_MEAN
        ).astype(np.int8)
        source[~mask & np.isfinite(filled)] = (
            SOURCE_FORWARD_FILL_OR_ALTERNATIVE
        )
        for column, name in enumerate(names):
            full_index = self.fields["time_series_columns"].index(name)
            filled[~np.isfinite(filled[:, column]), column] = self.train_means[full_index]
            if name == self.fields["target"]:
                filled[:, column] = np.maximum(filled[:, column], self.target_floor) / self.power_scale
            else:
                filled[:, column] = (filled[:, column] - self.train_means[full_index]) / self.train_stds[full_index]
        result = (filled, mask)
        return (*result, source) if return_source else result

    def _forecast_window(self, issue, future, return_source=False):
        names = self.fields["forecast_window_columns"]
        values = np.zeros((self.pred_len, len(names)), np.float32)
        mask = np.zeros_like(values, dtype=bool)
        source = np.full(values.shape, SOURCE_TRAIN_MEAN, dtype=np.int8)
        for column, name in enumerate(names):
            fallback = self.fields["forecast_fallbacks"][name]
            own = self.frame.loc[future, name].to_numpy(np.float64)
            alternative = self.frame.loc[future, fallback["alternative"]].to_numpy(np.float64)
            mask[:, column] = np.isfinite(own)
            own_valid = np.isfinite(own)
            alternative_valid = ~own_valid & np.isfinite(alternative)
            source[own_valid, column] = SOURCE_RAW_OR_OWN
            source[alternative_valid, column] = (
                SOURCE_FORWARD_FILL_OR_ALTERNATIVE
            )
            combined = np.where(np.isfinite(own), own, alternative)
            persistence = self._persistence[fallback["persistence"]].loc[issue]
            persistence_valid = ~np.isfinite(combined) & np.isfinite(persistence)
            source[persistence_valid, column] = SOURCE_PERSISTENCE
            combined = np.where(np.isfinite(combined), combined, persistence)
            full_index = self.fields["time_series_columns"].index(name)
            combined = np.where(np.isfinite(combined), combined, self.train_means[full_index])
            values[:, column] = (combined - self.train_means[full_index]) / self.train_stds[full_index]
        result = (values, mask)
        return (*result, source) if return_source else result

    def _target_values(self, issue):
        future = [issue + index * self.forecast_step for index in range(1, self.pred_len + 1)]
        raw = self.frame.loc[future, self.fields["target"]].to_numpy(np.float32)
        mask = np.isfinite(raw)
        raw = np.where(mask, np.maximum(raw, self.target_floor), self.target_floor)
        return (raw / self.power_scale).reshape(-1, 1).astype(np.float32), mask

    def _privileged_timeseries(self, times, return_source=False):
        names = self.config["privileged_teacher"]["future_timeseries_columns"]
        values = self.frame.loc[list(times), names].to_numpy(np.float32).copy()
        mask = np.isfinite(values)
        source = np.where(
            mask, SOURCE_RAW_OR_OWN, SOURCE_TRAIN_MEAN
        ).astype(np.int8)
        for column, name in enumerate(names):
            full_index = self.fields["time_series_columns"].index(name)
            values[~mask[:, column], column] = self.train_means[full_index]
            values[:, column] = (
                values[:, column] - self.train_means[full_index]
            ) / self.train_stds[full_index]
        result = (values.astype(np.float32), mask)
        return (*result, source) if return_source else result

    def inverse_transform_power(self, values):
        return np.asarray(values) * self.power_scale

    def __getitem__(self, item):
        issue = pd.Timestamp(self.sample_times[item])
        history_times = [issue - index * self.step for index in range(self.seq_len - 1, -1, -1)]
        future_times = [issue + index * self.forecast_step for index in range(1, self.pred_len + 1)]
        history, history_mask, history_source = self._history(
            history_times, return_source=True
        )
        if self.power_only:
            seq_x = history[:, :1].astype(np.float32)
            timeseries_mask = history_mask[:, :1]
            timeseries_source = history_source[:, :1].astype(np.int8)
            forecast_mask = np.empty((self.pred_len, 0), dtype=bool)
            forecast_source = np.empty((self.pred_len, 0), dtype=np.int8)
        else:
            forecasts, forecast_mask, forecast_source = self._forecast_window(
                issue, future_times, return_source=True
            )
            seq_x = np.concatenate([history, forecasts], axis=1).astype(np.float32)
            timeseries_mask = np.concatenate(
                [history_mask, forecast_mask], axis=1
            )
            timeseries_source = np.concatenate(
                [history_source, forecast_source], axis=1
            ).astype(np.int8)
        seq_y, target_mask = self._target_values(issue)
        dims = self.dims
        images = np.full((dims["image_history_frames"], dims["image_channels"], dims["image_height"], dims["image_width"]), float(self.config["images"]["pad_value"]), np.float32)
        future_images = np.zeros((self.pred_len, dims["image_channels"], dims["image_height"], dims["image_width"]), np.float32)
        weather_dim = int(dims["weather_placeholder_dim"])
        if self.privileged_teacher:
            (
                teacher_history,
                teacher_history_mask,
                teacher_history_source,
            ) = self._privileged_timeseries(history_times, return_source=True)
            (
                teacher_future,
                teacher_future_mask,
                teacher_future_source,
            ) = self._privileged_timeseries(future_times, return_source=True)
        else:
            teacher_history = np.zeros((self.seq_len, weather_dim), np.float32)
            teacher_future = np.zeros((self.pred_len, weather_dim), np.float32)
            teacher_history_mask = np.zeros_like(teacher_history, dtype=bool)
            teacher_future_mask = np.zeros_like(teacher_future, dtype=bool)
            teacher_history_source = np.full(
                teacher_history.shape, SOURCE_TRAIN_MEAN, dtype=np.int8
            )
            teacher_future_source = np.full(
                teacher_future.shape, SOURCE_TRAIN_MEAN, dtype=np.int8
            )
        metadata = {
            "image_mask": np.zeros(int(dims["image_history_frames"]), bool),
            "image_age_minutes": np.zeros(
                int(dims["image_history_frames"]), np.float32),
            "history_images_enabled": np.bool_(False),
            "timeseries_mask": timeseries_mask,
            "timeseries_source": timeseries_source,
            "history_source": (
                history_source[:, :1] if self.power_only else history_source
            ),
            "power_history_mask": history_mask[:, 0],
            "forecast_mask": forecast_mask,
            "forecast_source": forecast_source,
            "target_mask": target_mask,
            "current_power_valid": np.bool_(history_mask[-1, 0]),
            "teacher_history_timeseries_mask": teacher_history_mask,
            "teacher_future_timeseries_mask": teacher_future_mask,
            "teacher_history_timeseries_source": teacher_history_source,
            "teacher_future_timeseries_source": teacher_future_source,
            "teacher_future_image_mask": np.zeros(self.pred_len, bool),
            "privileged_teacher_enabled": np.bool_(self.privileged_teacher),
            "issue_time_ns": np.int64(issue.value),
        }
        return (
            seq_x, seq_y, self._time_marks(history_times), self._time_marks(future_times),
            images, future_images,
            teacher_history, teacher_future,
            np.float32(1.0), metadata,
        )

    def __len__(self):
        return len(self.sample_times)
