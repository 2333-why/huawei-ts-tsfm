"""Configuration-driven Luoyang Parquet adapter for FACTS.

Only this adapter knows the Parquet schema.  It preserves the legacy FACTS
nine-item batch prefix and appends explicit modality masks as item ten.
"""

from __future__ import annotations

import ast
from bisect import bisect_right
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


SOURCE_RAW = np.int8(0)
SOURCE_FORWARD_FILL = np.int8(1)
SOURCE_TRAIN_MEAN = np.int8(3)
_IMAGE_TIMELINE_CACHE: Dict[str, List[Tuple[pd.Timestamp, str]]] = {}


def load_luoyang_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    fields = config["fields"]
    dims = config["dimensions"]
    time = config["time"]
    power = config["power"]
    columns = fields["timeseries_columns"]
    if config.get("dataset") != "LuoyangParquet":
        raise ValueError("Luoyang config dataset must be LuoyangParquet")
    if len(columns) != int(dims["timeseries_input_dim"]):
        raise ValueError("timeseries column count must equal timeseries_input_dim")
    if columns[0] != fields["target"]:
        raise ValueError("the target must be the first time-series channel")
    if int(dims["forecast_steps"]) * int(time["forecast_step_minutes"]) != int(
        time["forecast_horizon_minutes"]
    ):
        raise ValueError("forecast steps, step minutes, and horizon are inconsistent")
    capacity = float(power["rated_power"])
    if not (
        float(power["power_scale"]) == capacity == float(power["max_power"])
    ):
        raise ValueError("power_scale, max_power, and rated_power must be identical")
    if capacity <= 0:
        raise ValueError("rated power must be positive")
    prediction_floor = float(config["output"]["prediction_floor"])
    if not np.isfinite(prediction_floor) or prediction_floor < 0:
        raise ValueError("Luoyang prediction_floor must be finite and non-negative")
    if power.get("model_input_normalization") != "divide_by_power_scale":
        raise ValueError(
            "Luoyang model_input_normalization must be divide_by_power_scale"
        )
    image_config = config["images"]
    if image_config["sampling_strategy"] not in {
        "uniform", "latest", "history_aligned_ffill"
    }:
        raise ValueError(
            "image sampling_strategy must be uniform, latest, or "
            "history_aligned_ffill"
        )
    if image_config["sampling_strategy"] == "history_aligned_ffill":
        if int(dims["image_history_frames"]) != int(dims["history_points"]):
            raise ValueError(
                "history_aligned_ffill requires image_history_frames to equal "
                "history_points"
            )
        max_gap = float(image_config["image_ffill_max_gap_minutes"])
        if max_gap <= 0:
            raise ValueError("image_ffill_max_gap_minutes must be positive")
    if time["history_order"] != "past_first":
        raise ValueError("Luoyang history_order must be past_first")
    privileged = config["privileged_teacher"]
    privileged_columns = privileged["future_timeseries_columns"]
    if fields["target"] in privileged_columns:
        raise ValueError("Teacher privileged time series must not contain the power target")
    if not privileged_columns or not set(privileged_columns).issubset(columns):
        raise ValueError("Teacher privileged columns must be non-empty configured time-series fields")
    if int(dims["weather_placeholder_dim"]) != len(privileged_columns):
        raise ValueError("weather_placeholder_dim must match privileged future time-series columns")
    return config


def _as_list(value: Any) -> List[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
                return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
        return [stripped]
    return [value]


class LuoyangParquetDataset(Dataset):
    """Five-minute Luoyang samples with configurable point-count windows."""

    def __init__(self, config_path: Union[str, Path], flag: str = "train",
                 load_images: bool = True, privileged_teacher: bool = False,
                 history_points: Optional[int] = None,
                 forecast_steps: Optional[int] = None):
        if flag == "valid":
            flag = "val"
        if flag not in {"train", "val", "test"}:
            raise ValueError("flag must be train, val, or test")
        self.config_path = Path(config_path).resolve()
        self.config = load_luoyang_config(self.config_path)
        self.flag = flag
        self.load_images = bool(load_images)
        self.privileged_teacher = bool(privileged_teacher)
        self.fields = self.config["fields"]
        self.dims = self.config["dimensions"]
        self.time_config = self.config["time"]
        self.power_config = self.config["power"]
        self.power_scale = float(self.power_config["power_scale"])
        self.image_config = self.config["images"]
        self.paths = self.config["paths"]
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
        self.image_frames = (
            self.seq_len if history_points is not None
            else int(self.dims["image_history_frames"])
        )
        self.step = pd.Timedelta(minutes=int(self.time_config["sampling_interval_minutes"]))
        self.forecast_step = pd.Timedelta(minutes=int(self.time_config["forecast_step_minutes"]))
        self.image_root = Path(self.paths["image_root"])

        parquet_path = Path(self.paths["parquet_file"])
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        required = {
            self.fields["timestamp"], self.fields["target"],
            self.fields["image_paths"], self.fields["image_timestamps"],
            *self.fields["timeseries_columns"],
        }
        frame = pd.read_parquet(parquet_path, columns=sorted(required))
        timestamp = self.fields["timestamp"]
        frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce")
        if frame[timestamp].isna().any():
            raise ValueError("timestamp contains missing or invalid values")
        if frame[timestamp].duplicated().any():
            raise ValueError("timestamp must be unique")
        target = self.fields["target"]
        if not np.isfinite(pd.to_numeric(frame[target], errors="coerce")).all():
            raise ValueError("final_power target contains missing/non-finite values")
        self.frame = frame.sort_values(timestamp).set_index(timestamp, drop=False)
        self.timestamps = self.frame.index
        self.image_timeline = self._build_image_timeline() if self.load_images else []
        self.image_timeline_timestamps = [item[0] for item in self.image_timeline]

        candidates = self._valid_candidates()
        dev_end = pd.Timestamp(self.time_config["test_start_timestamp"])
        dev_candidates = candidates[candidates < dev_end]
        split_at = int(np.floor(len(dev_candidates) * (1.0 - float(self.time_config["validation_fraction"]))))
        if len(dev_candidates) and split_at < len(dev_candidates):
            self.validation_start = dev_candidates[split_at]
        else:
            self.validation_start = dev_end
        horizon = self.pred_len * self.forecast_step
        if flag == "train":
            selected = candidates[(candidates < self.validation_start) & (candidates + horizon < self.validation_start)]
        elif flag == "val":
            selected = candidates[(candidates >= self.validation_start) & (candidates < dev_end) & (candidates + horizon < dev_end)]
        else:
            test_end = pd.Timestamp(self.time_config["test_end_exclusive_timestamp"])
            selected = candidates[(candidates >= dev_end) & (candidates + horizon < test_end)]
        self.candidate_audit["selected_sample_count"] = int(len(selected))
        self.sample_times = selected.to_numpy(dtype="datetime64[ns]")
        self._compute_train_statistics()
        self.current_powers = self.frame.loc[pd.DatetimeIndex(selected), target].to_numpy(np.float32)

    def _valid_candidates(self) -> pd.DatetimeIndex:
        start = pd.Timestamp(self.time_config["train_start_timestamp"])
        end = pd.Timestamp(self.time_config["test_end_exclusive_timestamp"])
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

        def exact_windows(offsets: np.ndarray) -> np.ndarray:
            queries = issue_ns[:, None] + offsets[None, :]
            positions = np.searchsorted(timestamp_ns, queries)
            safe_positions = np.minimum(positions, len(timestamp_ns) - 1)
            return ((positions < len(timestamp_ns)) &
                    (timestamp_ns[safe_positions] == queries)).all(axis=1)

        history_offsets = -np.arange(
            self.seq_len - 1, -1, -1, dtype=np.int64
        ) * self.step.value
        future_offsets = np.arange(
            1, self.pred_len + 1, dtype=np.int64
        ) * self.forecast_step.value
        valid = exact_windows(history_offsets) & exact_windows(future_offsets)
        valid_candidates = observed_issues[valid]
        theoretical_issues = pd.date_range(
            start=start, end=end, freq=self.step, inclusive="left"
        )
        self.candidate_audit = {
            "theoretical_issue_count": int(len(theoretical_issues)),
            "observed_issue_count": int(len(observed_issues)),
            "continuous_window_count": int(valid.sum()),
            "target_all_missing_rejected_count": 0,
            "valid_candidate_count": int(len(valid_candidates)),
        }
        return pd.DatetimeIndex(valid_candidates)

    def _compute_train_statistics(self) -> None:
        columns = self.fields["timeseries_columns"]
        train_rows = self.frame[
            (self.timestamps >= pd.Timestamp(self.time_config["train_start_timestamp"]))
            & (self.timestamps < self.validation_start)
        ]
        means = []
        stds = []
        for index, name in enumerate(columns):
            values = pd.to_numeric(train_rows[name], errors="coerce").to_numpy(np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(f"training interval has no finite values for {name}")
            means.append(0.0 if index == 0 else float(finite.mean()))
            std = 1.0 if index == 0 else float(finite.std())
            stds.append(std if std > float(self.config["missing"]["normalization_epsilon"]) else 1.0)
        self.train_means = np.asarray(means, dtype=np.float32)
        self.train_stds = np.asarray(stds, dtype=np.float32)

    @staticmethod
    def _time_marks(times: Sequence[pd.Timestamp]) -> np.ndarray:
        return np.asarray(
            [[v.month, v.day, v.weekday(), v.hour, v.minute] for v in times], dtype=np.float32
        )

    def _timeseries(
        self, history_times: Sequence[pd.Timestamp], return_source: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        columns = self.fields["timeseries_columns"]
        raw = self.frame.loc[list(history_times), columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        mask = np.isfinite(raw)
        if not mask[:, 0].all():
            raise ValueError("historical final_power cannot be missing")
        filled = raw.copy()
        source = np.where(mask, SOURCE_RAW, SOURCE_TRAIN_MEAN).astype(np.int8)
        for column in range(1, filled.shape[1]):
            last = np.nan
            for row in range(filled.shape[0]):
                if np.isfinite(filled[row, column]):
                    last = filled[row, column]
                elif np.isfinite(last):
                    filled[row, column] = last
                    source[row, column] = SOURCE_FORWARD_FILL
            filled[~np.isfinite(filled[:, column]), column] = self.train_means[column]
            filled[:, column] = (filled[:, column] - self.train_means[column]) / self.train_stds[column]
        filled[:, 0] = filled[:, 0] / self.power_scale
        result = (filled.astype(np.float32), mask)
        return (*result, source) if return_source else result

    def _privileged_timeseries(
        self, times: Sequence[pd.Timestamp], return_source: bool = False
    ):
        columns = self.config["privileged_teacher"]["future_timeseries_columns"]
        indices = [self.fields["timeseries_columns"].index(name) for name in columns]
        raw = self.frame.loc[list(times), columns].apply(
            pd.to_numeric, errors="coerce").to_numpy(np.float32)
        mask = np.isfinite(raw)
        source = np.where(mask, SOURCE_RAW, SOURCE_TRAIN_MEAN).astype(np.int8)
        filled = raw.copy()
        for column, full_index in enumerate(indices):
            filled[~mask[:, column], column] = self.train_means[full_index]
            filled[:, column] = (
                filled[:, column] - self.train_means[full_index]
            ) / self.train_stds[full_index]
        result = (filled.astype(np.float32), mask)
        return (*result, source) if return_source else result

    def inverse_transform_power(self, values: np.ndarray) -> np.ndarray:
        """Restore model-space power values to the configured original unit."""
        return np.asarray(values) * self.power_scale

    def _build_image_timeline(self):
        """Build the causal image timeline once from configured parquet fields."""
        cache_key = "|".join([
            str(Path(self.paths["parquet_file"]).resolve()),
            self.fields["image_paths"], self.fields["image_timestamps"],
        ])
        cached = _IMAGE_TIMELINE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        pairs = {}
        path_values = self.frame[self.fields["image_paths"]].to_numpy()
        timestamp_values = self.frame[self.fields["image_timestamps"]].to_numpy()
        for row_time, path_value, timestamp_value in zip(
            self.frame.index, path_values, timestamp_values
        ):
            paths = _as_list(path_value)
            stamps = _as_list(timestamp_value)
            if len(paths) != len(stamps):
                raise ValueError(f"image path/timestamp lengths differ at {row_time}")
            for path, stamp in zip(paths, stamps):
                if path is None or stamp is None or not str(path).strip() or not str(stamp).strip():
                    continue
                image_time = pd.to_datetime(stamp, errors="coerce")
                if pd.isna(image_time):
                    continue
                pairs[image_time] = str(path)
        timeline = sorted(pairs.items())
        _IMAGE_TIMELINE_CACHE[cache_key] = timeline
        return timeline

    def _image_pairs(self, history_times: Sequence[pd.Timestamp], issue: pd.Timestamp):
        if self.image_config["sampling_strategy"] == "history_aligned_ffill":
            aligned = []
            max_gap = float(self.image_config["image_ffill_max_gap_minutes"])
            for slot_time in history_times:
                source_index = bisect_right(
                    self.image_timeline_timestamps, slot_time
                ) - 1
                if source_index < 0:
                    aligned.append(None)
                    continue
                source_time, relative_path = self.image_timeline[source_index]
                age_minutes = float(
                    (slot_time - source_time).total_seconds() / 60.0
                )
                if source_time > issue or age_minutes > max_gap:
                    aligned.append(None)
                    continue
                aligned.append((source_time, relative_path))
            return aligned

        start = history_times[0]
        pairs = {}
        for row_time in history_times:
            row = self.frame.loc[row_time]
            paths = _as_list(row[self.fields["image_paths"]])
            stamps = _as_list(row[self.fields["image_timestamps"]])
            if len(paths) != len(stamps):
                raise ValueError(f"image path/timestamp lengths differ at {row_time}")
            for path, stamp in zip(paths, stamps):
                if path is None or stamp is None or not str(path).strip() or not str(stamp).strip():
                    continue
                image_time = pd.to_datetime(stamp, errors="coerce")
                if pd.isna(image_time) or image_time < start or image_time > issue:
                    continue
                pairs[image_time] = str(path)
        ordered = sorted(pairs.items())
        if len(ordered) > self.image_frames:
            if self.image_config["sampling_strategy"] == "latest":
                ordered = ordered[-self.image_frames :]
            else:
                positions = np.linspace(0, len(ordered) - 1, self.image_frames).round().astype(int)
                ordered = [ordered[position] for position in positions]
        return ordered

    def _images(self, history_times: Sequence[pd.Timestamp], issue: pd.Timestamp):
        height = int(self.dims["image_height"])
        width = int(self.dims["image_width"])
        channels = int(self.dims["image_channels"])
        output = np.zeros((self.image_frames, channels, height, width), dtype=np.float32)
        mask = np.zeros(self.image_frames, dtype=bool)
        offsets = np.zeros(self.image_frames, dtype=np.float32)
        if not self.load_images:
            return output, mask, offsets
        decoded = {}
        for slot, pair in enumerate(self._image_pairs(history_times, issue)):
            if pair is None:
                continue
            stamp, relative_path = pair
            path = self.image_root / relative_path
            if path not in decoded:
                decoded[path] = None
                if path.is_file():
                    try:
                        with Image.open(path) as image:
                            image = image.convert("RGB").resize(
                                (width, height), Image.Resampling.BILINEAR
                            )
                            decoded[path] = np.moveaxis(
                                np.asarray(image, dtype=np.float32) / 255.0, -1, 0
                            )
                    except (OSError, ValueError):
                        pass
            if decoded[path] is None:
                continue
            output[slot] = decoded[path]
            mask[slot] = True
            offsets[slot] = float((stamp - issue).total_seconds() / 60.0)
        return output, mask, offsets

    def _teacher_future_images(self, future_times: Sequence[pd.Timestamp]):
        height = int(self.dims["image_height"])
        width = int(self.dims["image_width"])
        channels = int(self.dims["image_channels"])
        output = np.zeros((self.pred_len, channels, height, width), np.float32)
        mask = np.zeros(self.pred_len, dtype=bool)
        if not self.privileged_teacher or not self.load_images:
            return output, mask
        max_gap = float(
            self.config["privileged_teacher"]["future_image_ffill_max_gap_minutes"])
        decoded = {}
        for slot, slot_time in enumerate(future_times):
            source_index = bisect_right(self.image_timeline_timestamps, slot_time) - 1
            if source_index < 0:
                continue
            source_time, relative_path = self.image_timeline[source_index]
            age = float((slot_time - source_time).total_seconds() / 60.0)
            if source_time <= future_times[0] - self.forecast_step or age > max_gap:
                continue
            path = self.image_root / relative_path
            if path not in decoded:
                decoded[path] = None
                if path.is_file():
                    try:
                        with Image.open(path) as image:
                            image = image.convert("RGB").resize(
                                (width, height), Image.Resampling.BILINEAR)
                            decoded[path] = np.moveaxis(
                                np.asarray(image, np.float32) / 255.0, -1, 0)
                    except (OSError, ValueError):
                        pass
            if decoded[path] is not None:
                output[slot] = decoded[path]
                mask[slot] = True
        return output, mask

    def __getitem__(self, item: int):
        issue = pd.Timestamp(self.sample_times[item])
        history_times = [issue - i * self.step for i in range(self.seq_len - 1, -1, -1)]
        future_times = [issue + i * self.forecast_step for i in range(1, self.pred_len + 1)]
        seq_x, timeseries_mask, timeseries_source = self._timeseries(
            history_times, return_source=True
        )
        seq_y = self.frame.loc[future_times, self.fields["target"]].to_numpy(np.float32).reshape(-1, 1)
        target_mask = np.isfinite(seq_y[:, 0])
        seq_y = seq_y / self.power_scale
        seq_x_img, image_mask, image_offsets = self._images(history_times, issue)
        slot_offsets = np.asarray([
            (slot_time - issue).total_seconds() / 60.0
            for slot_time in history_times
        ], dtype=np.float32)
        image_age_minutes = np.where(
            image_mask, slot_offsets - image_offsets, 0.0).astype(np.float32)
        seq_y_img, teacher_future_image_mask = self._teacher_future_images(future_times)
        weather_dim = int(self.dims["weather_placeholder_dim"])
        if self.privileged_teacher:
            (
                seq_x_weather,
                teacher_history_timeseries_mask,
                teacher_history_timeseries_source,
            ) = self._privileged_timeseries(history_times, return_source=True)
            (
                seq_y_weather,
                teacher_future_timeseries_mask,
                teacher_future_timeseries_source,
            ) = self._privileged_timeseries(future_times, return_source=True)
        else:
            seq_x_weather = np.zeros((self.seq_len, weather_dim), dtype=np.float32)
            seq_y_weather = np.zeros((self.pred_len, weather_dim), dtype=np.float32)
            teacher_history_timeseries_mask = np.zeros_like(seq_x_weather, dtype=bool)
            teacher_future_timeseries_mask = np.zeros_like(seq_y_weather, dtype=bool)
            teacher_history_timeseries_source = np.full(
                seq_x_weather.shape, SOURCE_TRAIN_MEAN, dtype=np.int8
            )
            teacher_future_timeseries_source = np.full(
                seq_y_weather.shape, SOURCE_TRAIN_MEAN, dtype=np.int8
            )
        metadata = {
            "image_mask": image_mask,
            "timeseries_mask": timeseries_mask,
            "timeseries_source": timeseries_source,
            "target_mask": target_mask,
            "image_minute_offsets": image_offsets,
            "image_age_minutes": image_age_minutes,
            "history_images_enabled": np.bool_(self.load_images),
            "teacher_future_image_mask": teacher_future_image_mask,
            "teacher_history_timeseries_mask": teacher_history_timeseries_mask,
            "teacher_future_timeseries_mask": teacher_future_timeseries_mask,
            "teacher_history_timeseries_source": teacher_history_timeseries_source,
            "teacher_future_timeseries_source": teacher_future_timeseries_source,
            "privileged_teacher_enabled": np.bool_(self.privileged_teacher),
            "issue_time_ns": np.int64(issue.value),
        }
        return (
            seq_x, seq_y, self._time_marks(history_times), self._time_marks(future_times),
            seq_x_img, seq_y_img,
            seq_x_weather, seq_y_weather,
            np.float32(1.0), metadata,
        )

    def __len__(self) -> int:
        return len(self.sample_times)
