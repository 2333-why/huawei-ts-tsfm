import h5py
import numpy as np

from data_provider.data_loader_stanford import StanfordSolarForecastDataset


def test_day_block_fold_keeps_each_date_atomic():
    dataset = object.__new__(StanfordSolarForecastDataset)
    dataset.num_fold = 3
    dataset.fold_index = 1
    counts = [5, 7, 4, 8]
    times = np.concatenate([
        np.arange(count, dtype="timedelta64[m]")
        + np.datetime64(f"2019-01-{day:02d}")
        for day, count in enumerate(counts, start=1)
    ])

    train, validation = dataset._day_block_cv_indices(times)

    assert set(times[train].astype("datetime64[D]")).isdisjoint(
        set(times[validation].astype("datetime64[D]"))
    )
    assert set(train).isdisjoint(set(validation))
    assert sorted(np.concatenate([train, validation]).tolist()) == list(range(len(times)))


def test_day_block_fold_with_one_fold_returns_empty_int64_train_indices():
    dataset = object.__new__(StanfordSolarForecastDataset)
    dataset.num_fold = 1
    dataset.fold_index = 0
    times = np.asarray([
        np.datetime64("2019-01-01"),
        np.datetime64("2019-01-01T00:01"),
        np.datetime64("2019-01-02"),
    ])

    train, validation = dataset._day_block_cv_indices(times)

    assert train.dtype == np.int64
    assert train.tolist() == []
    assert sorted(validation.tolist()) == [0, 1, 2]


def test_precomputed_cloud_events_use_selected_rows(tmp_path):
    path = tmp_path / "forecast.h5"
    selected = np.asarray([0, 2, 4, 6, 8, 9])
    with h5py.File(path, "w") as handle:
        group = handle.create_group("trainval")
        history = np.ones((10, 2), dtype=np.float32)
        target = np.ones(10, dtype=np.float32)
        target[selected] = 7.0
        group.create_dataset("pv_log", data=history)
        group.create_dataset("pv_pred", data=target)

    dataset = object.__new__(StanfordSolarForecastDataset)
    dataset.precomputed_forecast = True
    dataset.data_path = str(path)
    dataset.group_name = "trainval"
    dataset.history_order = "current_first"
    dataset.sample_indices = selected
    dataset.sample_times = np.asarray([
        np.datetime64("2019-01-01") + np.timedelta64(i, "m") for i in range(6)
    ])

    mask, dates = dataset._build_cloud_event_day_mask()

    assert mask.tolist() == [True] * 6
    assert dates.tolist() == [np.datetime64("2019-01-01")]
