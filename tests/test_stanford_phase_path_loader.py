from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data_provider import data_factory
from data_provider.data_loader_stanford import StanfordSolarForecastDataset


@pytest.fixture
def dynamic_nowcast_files(tmp_path):
    start = np.datetime64("2019-06-01T10:00:00", "s")
    minute_numbers = np.asarray(
        [minute for minute in range(41) if minute not in {8, 31}],
        dtype=np.int64,
    )
    times = start + minute_numbers * np.timedelta64(1, "m")
    pv = (minute_numbers + 1).astype(np.float32)
    images = np.zeros((len(times), 4, 4, 3), dtype=np.uint8)
    images[..., 0] = minute_numbers[:, None, None]

    h5_path = tmp_path / "tiny_nowcast.h5"
    with h5py.File(h5_path, "w") as h5_file:
        for group_name in ["trainval", "test"]:
            group = h5_file.create_group(group_name)
            group.create_dataset("pv_log", data=pv)
            group.create_dataset("images_log", data=images)

    np.save(tmp_path / "times_trainval.npy", times)
    np.save(tmp_path / "times_test.npy", times)
    return tmp_path, start


def _dynamic_dataset(root, flag, return_phase_path):
    return StanfordSolarForecastDataset(
        root_path=str(root),
        data_path="tiny_nowcast.h5",
        flag=flag,
        seq_len=2,
        label_len=0,
        pred_len=1,
        train_ratio=0.5,
        split_strategy="ratio",
        forecast_horizon_minutes=15,
        sample_interval_minutes=1,
        history_order="current_first",
        times_trainval_path="times_trainval.npy",
        times_test_path="times_test.npy",
        load_images=False,
        load_future_images=False,
        return_phase_path=return_phase_path,
    )


@pytest.mark.parametrize("flag", ["train", "val", "test"])
def test_phase_path_opt_in_does_not_change_existing_sample_filtering(
    dynamic_nowcast_files, flag
):
    root, _ = dynamic_nowcast_files
    legacy = _dynamic_dataset(root, flag, return_phase_path=False)
    phase = _dynamic_dataset(root, flag, return_phase_path=True)

    np.testing.assert_array_equal(phase.sample_indices, legacy.sample_indices)
    np.testing.assert_array_equal(phase.sample_times, legacy.sample_times)
    assert len(legacy[0]) == 9
    assert len(phase[0]) == 10


def test_phase_path_values_mask_marks_and_endpoint_are_exact(dynamic_nowcast_files):
    root, start = dynamic_nowcast_files
    legacy = _dynamic_dataset(root, "train", return_phase_path=False)
    phase = _dynamic_dataset(root, "train", return_phase_path=True)
    current_time = start + np.timedelta64(1, "m")
    item = int(np.flatnonzero(phase.sample_times == current_time)[0])

    legacy_sample = legacy[item]
    phase_sample = phase[item]
    for old_value, new_value in zip(legacy_sample, phase_sample[:9]):
        np.testing.assert_array_equal(new_value, old_value)

    labels = phase_sample[9]
    assert set(labels) == {
        "pv_path", "valid_mask", "lead_marks", "leads", "episode_weight"
    }
    assert labels["pv_path"].shape == (15, 1)
    assert labels["valid_mask"].shape == (15,)
    assert labels["valid_mask"].dtype == np.bool_
    assert labels["lead_marks"].shape == (5, 5)
    np.testing.assert_array_equal(labels["leads"], [1, 3, 5, 10, 15])
    assert float(labels["episode_weight"]) == pytest.approx(1.0)

    # Minute 8 is absent, so only t+7 is unavailable for the t=minute-1 sample.
    expected_valid = np.ones(15, dtype=bool)
    expected_valid[6] = False
    np.testing.assert_array_equal(labels["valid_mask"], expected_valid)
    expected_path = np.arange(3, 18, dtype=np.float32)
    expected_path[6] = 0.0
    np.testing.assert_array_equal(labels["pv_path"][:, 0], expected_path)

    # The legacy endpoint remains authoritative and is not replaced by the path.
    assert phase_sample[1].shape == (1, 1)
    assert labels["pv_path"][-1, 0] == phase_sample[1][0, 0]
    np.testing.assert_array_equal(
        labels["lead_marks"][:, 4], np.asarray([2, 4, 6, 11, 16])
    )
    np.testing.assert_array_equal(labels["lead_marks"][:, :2], [[6, 1]] * 5)


def test_phase_path_dictionary_collates_as_label_only_tensors(dynamic_nowcast_files):
    root, _ = dynamic_nowcast_files
    dataset = _dynamic_dataset(root, "test", return_phase_path=True)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert len(batch) == 10
    labels = batch[9]
    assert labels["pv_path"].shape == (2, 15, 1)
    assert labels["valid_mask"].shape == (2, 15)
    assert labels["valid_mask"].dtype == torch.bool
    assert labels["lead_marks"].shape == (2, 5, 5)
    assert labels["leads"].shape == (2, 5)
    assert labels["episode_weight"].shape == (2,)


@pytest.mark.parametrize(
    "has_kd_attribute,privileged,flag,strategy,expected",
    [
        (True, True, "train", "causal_solar_token_tail", True),
        (True, True, "val", "causal_solar_token_tail", False),
        (True, True, "test", "causal_solar_token_tail", False),
        (True, False, "train", None, False),
        (False, False, "val", None, True),
        (False, False, "test", "causal_img", False),
    ],
)
def test_future_image_policy_distinguishes_kd_from_standalone_teacher(
    has_kd_attribute, privileged, flag, strategy, expected
):
    args = SimpleNamespace(fuse_strategy=strategy)
    if has_kd_attribute:
        args.stanford_privileged_teacher = privileged
    assert data_factory._stanford_load_future_images(args, flag) is expected


def test_factory_passes_phase_opt_in_and_train_only_kd_images(monkeypatch):
    captured = []

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def __len__(self):
            return 2

        def __getitem__(self, item):
            return item

    monkeypatch.setattr(data_factory, "_get_data_class", lambda name: TinyDataset)
    args = SimpleNamespace(
        data="Stanford",
        embed="timeF",
        batch_size=2,
        freq="t",
        task_name="long_term_forecast",
        root_path=".",
        data_path="unused.h5",
        seq_len=16,
        label_len=0,
        pred_len=1,
        features="S",
        target="pv_log",
        train_ratio=0.85,
        forecast_horizon_minutes=15,
        sample_interval_minutes=1,
        history_order="current_first",
        times_trainval_path="unused_train.npy",
        times_test_path="unused_test.npy",
        weather_feature_dim=1,
        fuse_strategy="causal_solar_token_tail",
        stanford_privileged_teacher=True,
        stanford_return_phase_path=True,
        num_workers=0,
        use_gpu=False,
        gpu_type="cuda",
    )

    data_factory.data_provider(args, "train")
    data_factory.data_provider(args, "val")
    data_factory.data_provider(args, "test")

    assert [item["load_future_images"] for item in captured] == [True, False, False]
    assert all(item["return_phase_path"] is True for item in captured)

