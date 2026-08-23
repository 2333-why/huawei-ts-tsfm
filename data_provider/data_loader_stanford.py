import os
import datetime as dt

import h5py
import numpy as np
from torch.utils.data import Dataset

from utils.metrics import STANFORD_CLOUDY_DATES
from data_provider.stanford_sunset_preprocessing import validate_sunset_processed_data


STANFORD_PHASE_PATH_MINUTES = np.arange(1, 16, dtype=np.int64)
STANFORD_PHASE_LEADS = np.asarray([1, 3, 5, 10, 15], dtype=np.int64)


class StanfordSolarForecastDataset(Dataset):
    """Loader for Stanford SKIPP'D forecast samples.

    Expected files:
      - Original forecast HDF5: trainval/test images_log, pv_log, pv_pred
      - Or benchmark nowcast HDF5: trainval/test images_log, pv_log
      - times_trainval.npy
      - times_test.npy

    Returned sample shape follows Stanford forecast samples and FACTS' loop:
      seq_x:         [16, 1] raw PV history (pv_log)
      seq_y:         [1, 1] raw PV target at forecast_horizon_minutes (pv_pred)
      seq_x_img:     [16, H, W] grayscale history sky images normalized to [0, 1]
                     or [16, 3, H, W] RGB images when image_mode=rgb
      seq_y_img:     [1, H, W] / [1, 3, H, W] future sky image at pv_pred time
                     when available; zero only for forecast HDF5 files without
                     a stored future image.
      weather fields are zero placeholders so FACTS can disable weather fusion.

    The legacy interface returns nine items (the original eight modalities plus
    ``sample_weight``).  With ``return_phase_path=True``, dynamic-nowcast samples
    append a tenth, label-only dictionary:

      pv_path:       [15, 1] PV(t+1), ..., PV(t+15), zero where unavailable
      valid_mask:    [15] boolean mask for available path targets
      lead_marks:    [5, 5] calendar marks at leads 1/3/5/10/15 minutes
      leads:         [5] fixed lead minutes [1, 3, 5, 10, 15]
      episode_weight: scalar neutral weight (1.0; reserved for episode sampling)

    This dictionary contains supervision/evaluation labels only.  It is never
    merged into ``seq_x``, ``seq_x_img``, or any other Student input.
    """

    def __init__(
        self,
        root_path,
        data_path="2017_2019_images_pv_processed.hdf5",
        flag="train",
        seq_len=48,
        label_len=0,
        pred_len=24,
        features="S",
        target="pv_log",
        scale=False,
        timeenc=0,
        freq="t",
        train_ratio=0.85,
        split_strategy="day_block",
        num_fold=10,
        fold_index=0,
        forecast_horizon_minutes=15,
        sample_interval_minutes=1,
        history_order="current_first",
        times_trainval_path="times_trainval.npy",
        times_test_path="times_test.npy",
        weather_feature_dim=1,
        image_mode="gray",
        stanford_capacity_kw=30.1,
        sample_weight_mode="none",
        sample_weight_scale=1.0,
        sample_weight_clip=3.0,
        ramp_weight_5=1.5,
        ramp_weight_8=2.0,
        load_images=True,
        load_future_images=True,
        return_phase_path=False,
    ):
        if flag == "valid":
            flag = "val"
        if flag not in ["train", "val", "test"]:
            raise ValueError("flag must be one of train, val, test")

        self.root_path = root_path
        self.data_path = self._resolve_path(root_path, data_path)
        self.times_trainval_path = self._resolve_path(root_path, times_trainval_path)
        self.times_test_path = self._resolve_path(root_path, times_test_path)
        self.flag = flag
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.features = features
        self.target = target
        self.scale = False
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = float(train_ratio)
        self.split_strategy = split_strategy
        self.num_fold = int(num_fold)
        self.fold_index = int(fold_index)
        self.forecast_horizon = np.timedelta64(int(forecast_horizon_minutes * 60), "s")
        self.sample_interval = np.timedelta64(int(sample_interval_minutes * 60), "s")
        self.history_order = history_order
        self.weather_feature_dim = int(weather_feature_dim)
        self.image_mode = image_mode
        self.stanford_capacity_kw = float(stanford_capacity_kw)
        self.sample_weight_mode = sample_weight_mode
        self.sample_weight_scale = float(sample_weight_scale)
        self.sample_weight_clip = float(sample_weight_clip)
        self.ramp_weight_5 = float(ramp_weight_5)
        self.ramp_weight_8 = float(ramp_weight_8)
        self.load_images = bool(load_images)
        self.load_future_images = bool(load_future_images) and self.load_images
        self.return_phase_path = bool(return_phase_path)
        self.total_y_len = self.label_len + self.pred_len
        self.minute = np.timedelta64(60, "s")

        if self.label_len != 0 or self.pred_len != 1:
            raise ValueError("Stanford forecast samples require label_len=0 and pred_len=1")
        if self.history_order not in ["current_first", "past_first"]:
            raise ValueError("history_order must be current_first or past_first")
        if self.split_strategy not in ["day_block", "ratio"]:
            raise ValueError("split_strategy must be day_block or ratio")
        if self.image_mode not in ["gray", "rgb"]:
            raise ValueError("image_mode must be gray or rgb")
        if self.sample_weight_mode not in [
            "none", "pv_ramp", "ramp_gt5", "ramp_gt8", "ramp_piecewise", "cloudy_dates"]:
            raise ValueError(
                "sample_weight_mode must be none, pv_ramp, ramp_gt5, ramp_gt8, ramp_piecewise, or cloudy_dates"
            )
        if self.stanford_capacity_kw <= 0:
            raise ValueError("stanford_capacity_kw must be positive")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.data_path}")
        if not os.path.exists(self.times_trainval_path):
            raise FileNotFoundError(f"times_trainval file not found: {self.times_trainval_path}")
        if not os.path.exists(self.times_test_path):
            raise FileNotFoundError(f"times_test file not found: {self.times_test_path}")

        self.group_name = "test" if self.flag == "test" else "trainval"
        self.raw_times, self.times = self._load_times(
            self.times_test_path if self.flag == "test" else self.times_trainval_path
        )
        self.precomputed_forecast = self._has_precomputed_forecast_samples()
        self.future_image_dataset_name = self._find_future_image_dataset_name() if self.precomputed_forecast else None
        self.sunset_cleaning_audit = self._validate_sunset_cleaning_contract()

        if self.precomputed_forecast:
            self.sample_indices = np.arange(len(self.times), dtype=np.int64)
            self.sample_times = self.times.copy()
        else:
            self.time_to_index = self._build_time_index(self.times)
            self.sample_indices, self.sample_times = self._build_sample_indices(self.times)

        if self.flag in ["train", "val"]:
            self.sample_indices, self.sample_times = self._split_trainval(self.sample_indices, self.sample_times)

        self.pv_values = None if self.precomputed_forecast else self._load_pv_values()
        if self.return_phase_path and self.precomputed_forecast:
            raise ValueError(
                "return_phase_path requires the dynamic nowcast HDF5; "
                "precomputed forecast samples do not contain intermediate PV targets"
            )
        self.phase_path_indices = (
            self._build_phase_path_indices() if self.return_phase_path else None
        )
        self.cloud_event_day_mask, self.cloud_event_dates = self._build_cloud_event_day_mask()
        self._h5 = None

        print(
            f"Stanford {self.flag}: group={self.group_name}, mode={self._mode_name()}, "
            f"samples={len(self.sample_indices)}, image_log={self._image_shape_text()}, "
            f"pv_log=({self.seq_len},), pv_pred=(), pv_scale=raw, image_scale=/255, "
            f"future_image={self._future_image_source_text()}, "
            f"forecast_horizon_minutes={forecast_horizon_minutes}, history_order={self.history_order}, "
            f"split_strategy={self.split_strategy}, sample_weight_mode={self.sample_weight_mode}, "
            f"load_images={self.load_images}, load_future_images={self.load_future_images}, "
            f"phase_path_labels={self.return_phase_path}, "
            f"cloud_event_day_samples={int(self.cloud_event_day_mask.sum())}, "
            f"sunset_cleaning=validated"
        )

    def _image_shape_text(self):
        if self.image_mode == "rgb":
            return f"({self.seq_len},3,64,64)"
        return f"({self.seq_len},64,64)"

    @staticmethod
    def _resolve_path(root_path, path):
        if os.path.isabs(path):
            return path
        return os.path.join(root_path, path)

    @staticmethod
    def _load_times(path):
        raw_times = np.load(path, allow_pickle=True)
        return raw_times, np.array(raw_times, dtype="datetime64[s]")

    def _validate_sunset_cleaning_contract(self):
        """Ensure FACTS only consumes artifacts from SUNSET's cleaning domain.

        The released nowcast HDF5 is already downstream of the original raw-PV
        notebook.  Re-interpolating its aligned values would change the data, so
        the loader validates all invariants that remain observable after that
        preprocessing: naive time, 10-second alignment, PV > 0, and exclusion
        of all 22 manual invalid intervals.
        """

        with h5py.File(self.data_path, "r") as h5_file:
            group = h5_file[self.group_name]
            pv_parts = [group["pv_log"][...]]
            if "pv_pred" in group:
                pv_parts.append(group["pv_pred"][...])
            pv_values = np.concatenate([np.asarray(part).reshape(-1) for part in pv_parts])
        return validate_sunset_processed_data(
            self.raw_times,
            pv_values,
            context=f"Stanford {self.group_name}",
        )

    def _has_precomputed_forecast_samples(self):
        with h5py.File(self.data_path, "r") as h5_file:
            group = h5_file[self.group_name]
            has_forecast_target = "pv_pred" in group
            pv_log_is_sampled = group["pv_log"].ndim == 2
            image_log_is_sampled = group["images_log"].ndim == 5
            if has_forecast_target and (not pv_log_is_sampled or not image_log_is_sampled):
                raise ValueError("Stanford forecast HDF5 must store pv_log as [N,16] and images_log as [N,16,H,W,3]")
            if has_forecast_target and len(self.times) != group["pv_pred"].shape[0]:
                raise ValueError(
                    f"Timestamp count {len(self.times)} does not match pv_pred count {group['pv_pred'].shape[0]}"
                )
            return has_forecast_target

    def _mode_name(self):
        return "forecast_hdf5" if self.precomputed_forecast else "dynamic_from_nowcast_hdf5"

    def _find_future_image_dataset_name(self):
        candidates = [
            "images_pred",
            "image_pred",
            "images_y",
            "image_y",
            "images_future",
            "future_images",
        ]
        with h5py.File(self.data_path, "r") as h5_file:
            group = h5_file[self.group_name]
            for name in candidates:
                if name in group:
                    return name
        return None

    def _future_image_source_text(self):
        if not self.load_future_images:
            return "disabled"
        if self.precomputed_forecast:
            if self.future_image_dataset_name is not None:
                return self.future_image_dataset_name
            return "zero_missing_in_forecast_hdf5"
        return "images_log[target_time]"

    @staticmethod
    def _time_marks(times):
        marks = np.zeros((len(times), 5), dtype=np.float32)
        for i, value in enumerate(times):
            timestamp = value.astype(dt.datetime)
            marks[i] = [
                timestamp.month,
                timestamp.day,
                timestamp.weekday(),
                timestamp.hour,
                timestamp.minute,
            ]
        return marks

    @staticmethod
    def _build_time_index(times):
        return {time_point: idx for idx, time_point in enumerate(times)}

    def _find_time_index(self, time_point):
        return self.time_to_index.get(time_point)

    def _build_sample_indices(self, times):
        samples = []
        sample_times = []
        last_selected_time = None
        offsets = np.arange(self.seq_len, dtype=np.int64) * self.minute

        # The released times_test.npy is grouped by sunny/cloudy dates and is not globally sorted.
        # Stanford's forecast preprocessing created samples before the test split, from chronological times.
        for current_time in np.sort(times):
            if last_selected_time is not None and current_time - last_selected_time < self.sample_interval:
                continue

            history_times = current_time - offsets
            if self.history_order == "past_first":
                history_times = history_times[::-1]

            history_indices = []
            valid = True
            for history_time in history_times:
                history_idx = self._find_time_index(history_time)
                if history_idx is None:
                    valid = False
                    break
                history_indices.append(history_idx)
            if not valid:
                continue

            pred_idx = self._find_time_index(current_time + self.forecast_horizon)
            if pred_idx is None:
                continue

            samples.append(history_indices + [pred_idx])
            sample_times.append(current_time)
            last_selected_time = current_time

        if not samples:
            return np.empty((0, self.seq_len + 1), dtype=np.int64), np.empty((0,), dtype="datetime64[s]")
        return np.asarray(samples, dtype=np.int64), np.asarray(sample_times, dtype="datetime64[s]")

    def _build_phase_path_indices(self):
        """Index t+1..t+15 labels without changing forecast sample eligibility."""

        indices = np.full(
            (len(self.sample_times), len(STANFORD_PHASE_PATH_MINUTES)),
            -1,
            dtype=np.int64,
        )
        for column, lead in enumerate(STANFORD_PHASE_PATH_MINUTES):
            delta = int(lead) * self.minute
            indices[:, column] = np.fromiter(
                (
                    self.time_to_index.get(current_time + delta, -1)
                    for current_time in self.sample_times
                ),
                dtype=np.int64,
                count=len(self.sample_times),
            )
        return indices

    def _split_trainval(self, samples, sample_times):
        if self.split_strategy == "ratio":
            split = int(len(samples) * self.train_ratio)
            selected = np.arange(split) if self.flag == "train" else np.arange(split, len(samples))
        else:
            train_positions, val_positions = self._day_block_cv_indices(sample_times)
            selected = train_positions if self.flag == "train" else val_positions
        return samples[selected], sample_times[selected]

    def _day_block_cv_indices(self, sample_times):
        if self.num_fold <= 0:
            raise ValueError("num_fold must be positive")

        fold_index = self.fold_index % self.num_fold
        dates = sample_times.astype("datetime64[D]")
        blocks = [np.flatnonzero(dates == value) for value in np.unique(dates)]
        if self.num_fold > len(blocks):
            raise ValueError(
                f"num_fold={self.num_fold} exceeds distinct date count={len(blocks)}"
            )

        block_rng = np.random.RandomState(1)
        block_rng.shuffle(blocks)

        folds = np.array_split(np.arange(len(blocks)), self.num_fold)
        validation_block_ids = set(folds[fold_index].tolist())
        train_blocks = [
            block for index, block in enumerate(blocks)
            if index not in validation_block_ids
        ]
        train_positions = (
            np.concatenate(train_blocks)
            if train_blocks else np.empty((0,), dtype=np.int64)
        )
        val_positions = np.concatenate([
            blocks[index] for index in folds[fold_index]
        ])

        split_rng = np.random.RandomState(fold_index)
        split_rng.shuffle(train_positions)
        split_rng.shuffle(val_positions)

        return train_positions, val_positions

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.data_path, "r")
        return self._h5

    def _load_pv_values(self):
        with h5py.File(self.data_path, "r") as h5_file:
            return h5_file[self.group_name]["pv_log"][...].astype(np.float32)

    def _build_cloud_event_day_mask(self):
        """Identify internally volatile days without using weather/test labels.

        Stanford's Sunny/Cloudy metric is defined at the date level, while the
        training split has no weather labels.  A day is treated as an event day
        when at least 2% of its forecast windows (and at least five windows)
        contain a 5 kW or larger 15-minute PV change.  Requiring multiple
        events prevents a single bad point from determining the day label.
        """

        if self.precomputed_forecast:
            with h5py.File(self.data_path, "r") as h5_file:
                group = h5_file[self.group_name]
                selected = np.asarray(self.sample_indices, dtype=np.int64)
                history = np.asarray(group["pv_log"])[selected]
                current_index = 0 if self.history_order == "current_first" else -1
                current = history[:, current_index]
                target = np.asarray(group["pv_pred"]).reshape(-1)[selected]
                if len(history) != len(self.sample_times) or len(target) != len(self.sample_times):
                    raise ValueError("precomputed power rows do not align with selected sample times")
        else:
            current = self.pv_values[self.sample_indices[:, 0]]
            target = self.pv_values[self.sample_indices[:, -1]]

        absolute_ramp = np.abs(target - current)
        dates = self.sample_times.astype("datetime64[D]")
        event_dates = []
        for date in np.unique(dates):
            day_mask = dates == date
            minimum_events = max(5, int(np.ceil(0.02 * int(day_mask.sum()))))
            if int((absolute_ramp[day_mask] >= 5.0).sum()) >= minimum_events:
                event_dates.append(date)

        event_dates = np.asarray(event_dates, dtype="datetime64[D]")
        return np.isin(dates, event_dates), event_dates

    @staticmethod
    def _prepare_pv(values):
        return values.astype(np.float32).reshape(-1, 1)

    def _prepare_images(self, images):
        images = images.astype(np.float32)
        if images.ndim == 4 and images.shape[-1] == 3:
            if self.image_mode == "rgb":
                return np.moveaxis(images, -1, -3) / 255.0
            return images.mean(axis=-1) / 255.0
        return images / 255.0

    def _prepare_future_images(self, images):
        if images.ndim in [2, 3]:
            images = images[np.newaxis, ...]
        return self._prepare_images(images)

    def _zero_images(self):
        if self.image_mode == "rgb":
            return np.zeros((self.seq_len, 3, 64, 64), dtype=np.float32)
        return np.zeros((self.seq_len, 64, 64), dtype=np.float32)

    def _zero_future_images(self, seq_x_img):
        return np.zeros((self.pred_len,) + seq_x_img.shape[1:], dtype=np.float32)

    def _sample_weight(self, seq_x, seq_y, current_time):
        if self.sample_weight_mode == "none":
            return np.float32(1.0)

        if self.sample_weight_mode == "cloudy_dates":
            current_date = np.asarray(current_time, dtype="datetime64[s]").astype("datetime64[D]")
            weight = 1.0 + self.sample_weight_scale if np.isin(current_date, STANFORD_CLOUDY_DATES) else 1.0
        elif self.sample_weight_mode == "pv_ramp":
            current_pv = seq_x[0, 0] if self.history_order == "current_first" else seq_x[-1, 0]
            target_pv = seq_y[-1, 0]
            ramp = abs(float(target_pv) - float(current_pv)) / self.stanford_capacity_kw
            weight = 1.0 + self.sample_weight_scale * ramp
        elif self.sample_weight_mode == "ramp_piecewise":
            current_pv = seq_x[0, 0] if self.history_order == "current_first" else seq_x[-1, 0]
            target_pv = seq_y[-1, 0]
            ramp = abs(float(target_pv) - float(current_pv))
            if ramp >= 8.0:
                weight = self.ramp_weight_8
            elif ramp >= 5.0:
                weight = self.ramp_weight_5
            else:
                weight = 1.0
        else:
            current_pv = seq_x[0, 0] if self.history_order == "current_first" else seq_x[-1, 0]
            target_pv = seq_y[-1, 0]
            threshold = 5.0 if self.sample_weight_mode == "ramp_gt5" else 8.0
            weight = 1.0 + self.sample_weight_scale if abs(float(target_pv) - float(current_pv)) > threshold else 1.0

        if self.sample_weight_clip > 0:
            weight = min(weight, self.sample_weight_clip)
        return np.float32(weight)

    def _phase_path_labels(self, item, current_time):
        indices = self.phase_path_indices[item]
        valid_mask = indices >= 0
        pv_path = np.zeros((len(indices), 1), dtype=np.float32)
        pv_path[valid_mask, 0] = self.pv_values[indices[valid_mask]]
        lead_times = (
            np.asarray(current_time, dtype="datetime64[s]")
            + STANFORD_PHASE_LEADS * self.minute
        )
        return {
            "pv_path": pv_path,
            "valid_mask": valid_mask,
            "lead_marks": self._time_marks(lead_times),
            "leads": STANFORD_PHASE_LEADS.copy(),
            # Neutral until an episode definition is selected and audited.
            "episode_weight": np.float32(1.0),
        }

    @staticmethod
    def _read_h5_indices(dataset, indices):
        order = np.argsort(indices)
        sorted_indices = indices[order]
        data = dataset[sorted_indices]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return data[inverse]

    def __getitem__(self, item):
        sample = self.sample_indices[item]
        h5_file = self._file()
        group = h5_file[self.group_name]

        if self.precomputed_forecast:
            sample_idx = int(sample)
            current_time = self.times[sample_idx]
            seq_x = self._prepare_pv(group["pv_log"][sample_idx])
            seq_y = self._prepare_pv(np.asarray([group["pv_pred"][sample_idx]], dtype=np.float32))
            if self.load_images:
                seq_x_img = self._prepare_images(group["images_log"][sample_idx])
                if self.load_future_images and self.future_image_dataset_name is not None:
                    seq_y_img = self._prepare_future_images(group[self.future_image_dataset_name][sample_idx])
                else:
                    seq_y_img = self._zero_future_images(seq_x_img)
            else:
                seq_x_img = self._zero_images()
                seq_y_img = self._zero_future_images(seq_x_img)

            offsets = np.arange(self.seq_len, dtype=np.int64) * self.minute
            history_times = current_time - offsets
            if self.history_order == "past_first":
                history_times = history_times[::-1]
            pred_time = current_time + self.forecast_horizon
            seq_x_mark = self._time_marks(history_times)
            seq_y_mark = self._time_marks(np.asarray([pred_time], dtype="datetime64[s]"))
        else:
            history_indices = sample[: self.seq_len]
            pred_idx = int(sample[-1])

            seq_x = self._prepare_pv(self.pv_values[history_indices])
            seq_y = self._prepare_pv(np.asarray([self.pv_values[pred_idx]], dtype=np.float32))

            if self.load_images:
                if self.load_future_images:
                    image_indices = np.concatenate([history_indices, np.asarray([pred_idx], dtype=np.int64)])
                    images = self._read_h5_indices(group["images_log"], image_indices)
                    seq_x_img = self._prepare_images(images[: self.seq_len])
                    seq_y_img = self._prepare_future_images(images[self.seq_len :])
                else:
                    images = self._read_h5_indices(group["images_log"], history_indices)
                    seq_x_img = self._prepare_images(images)
                    seq_y_img = self._zero_future_images(seq_x_img)
            else:
                seq_x_img = self._zero_images()
                seq_y_img = self._zero_future_images(seq_x_img)

            seq_x_mark = self._time_marks(self.times[history_indices])
            seq_y_mark = self._time_marks(np.asarray([self.times[pred_idx]], dtype="datetime64[s]"))
            current_time = self.sample_times[item]

        seq_x_weather = np.zeros((self.seq_len, self.weather_feature_dim), dtype=np.float32)
        seq_y_weather = np.zeros((1, self.weather_feature_dim), dtype=np.float32)
        sample_weight = self._sample_weight(seq_x, seq_y, current_time)

        sample = (
            seq_x.astype(np.float32),
            seq_y.astype(np.float32),
            seq_x_mark.astype(np.float32),
            seq_y_mark.astype(np.float32),
            seq_x_img.astype(np.float32),
            seq_y_img.astype(np.float32),
            seq_x_weather,
            seq_y_weather,
            sample_weight,
        )
        if not self.return_phase_path:
            return sample
        return (*sample, self._phase_path_labels(item, current_time))

    def __len__(self):
        return len(self.sample_indices)

    def __del__(self):
        h5_file = getattr(self, "_h5", None)
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception:
                pass
