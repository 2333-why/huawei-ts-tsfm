from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from data_provider.data_loader_luoyang import LuoyangParquetDataset, load_luoyang_config
from scripts.build_skippd_luoyang_parquet import build


def _write_source(tmp_path: Path) -> tuple[Path, Path, pd.Timestamp, dict[int, list[float]]]:
    base = pd.Timestamp("2020-01-01 00:00:00")
    rows = []
    pv_by_minute: dict[int, list[float]] = {}
    total_minutes = 2700
    raw_index = 0
    for minute in range(total_minutes):
        if 650 <= minute <= 654:  # a real five-minute source outage
            continue
        value = 1.0 + minute * 0.01
        timestamp = base + pd.Timedelta(minutes=minute, seconds=20)
        if minute == 700:  # the only row for this minute is outside tolerance
            timestamp = base + pd.Timedelta(minutes=minute, seconds=30)
        rows.append((timestamp.to_pydatetime(), value, raw_index))
        pv_by_minute.setdefault(minute, []).append(value)
        raw_index += 1
        if minute == 10:  # duplicate canonical minute; exact row wins image choice
            rows.append((base + pd.Timedelta(minutes=minute), 9.9, raw_index))
            pv_by_minute[minute].append(9.9)
            raw_index += 1

    times = np.asarray([row[0] for row in rows], dtype=object)
    images = np.zeros((len(rows), 64, 64, 3), dtype=np.uint8)
    pv = np.asarray([row[1] for row in rows], dtype=np.float64)
    for position, (_, _, source_index) in enumerate(rows):
        images[position, :, :, :] = source_index % 251
    # Make the deterministic duplicate winner visually unambiguous.
    duplicate_position = next(position for position, row in enumerate(rows) if row[2] == 11)
    images[duplicate_position, :, :, :] = 200

    hdf5 = tmp_path / "source.hdf5"
    with h5py.File(hdf5, "w") as handle:
        group = handle.create_group("trainval")
        group.create_dataset("images_log", data=images)
        group.create_dataset("pv_log", data=pv)
        handle.create_group("test")
    times_path = tmp_path / "times.npy"
    np.save(times_path, times, allow_pickle=True)
    return hdf5, times_path, base, pv_by_minute


def _temp_config(tmp_path: Path, parquet: Path, image_root: Path, base: pd.Timestamp) -> Path:
    config_path = Path(__file__).parents[1] / "configs" / "datasets" / "skippd_luoyang.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["paths"]["parquet_file"] = str(parquet)
    config["paths"]["image_root"] = str(image_root)
    config["time"].update(
        {
            "train_start_timestamp": str(base + pd.Timedelta(minutes=10)),
            "test_start_timestamp": str(base + pd.Timedelta(minutes=2100)),
            "test_end_exclusive_timestamp": str(base + pd.Timedelta(minutes=2500)),
        }
    )
    output = tmp_path / "config.json"
    output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output


def test_skippd_conversion_contract_and_dataset(tmp_path):
    hdf5, times, base, pv_by_minute = _write_source(tmp_path)
    output_root = tmp_path / "luoyang"
    manifest = build(hdf5, times, output_root)
    parquet_path = output_root / "skippd_luoyang.parquet"
    frame = pd.read_parquet(parquet_path)

    assert list(frame.columns) == [
        "timestamp", "final_power", "pv_ramp_5min", "asi_path", "asi_path_timestamps"
    ]
    assert frame["timestamp"].is_unique
    assert frame["timestamp"].is_monotonic_increasing
    assert manifest["filter"]["dropped_by_tolerance_rows"] == 1
    assert manifest["filter"]["duplicate_rows_collapsed"] == 1
    assert manifest["aggregation"]["complete_bins"] == len(frame)
    assert manifest["output"]["image_count"] == len(frame)

    duplicate_bin = frame.loc[frame["timestamp"] == base + pd.Timedelta(minutes=10)].iloc[0]
    expected_minutes = [6, 7, 8, 9, 10]
    expected_power = np.mean([
        *[1.0 + minute * 0.01 for minute in expected_minutes[:-1]],
        np.mean(pv_by_minute[10]),
    ])
    assert duplicate_bin["final_power"] == pytest.approx(expected_power)
    duplicate_path = output_root / "images" / json.loads(duplicate_bin["asi_path"])[0]
    with Image.open(duplicate_path) as image:
        assert image.mode == "RGB" and image.size == (64, 64)
        assert np.asarray(image)[0, 0, 0] == 200

    for _, row in frame.iterrows():
        paths = json.loads(row["asi_path"])
        timestamps = json.loads(row["asi_path_timestamps"])
        assert len(paths) == len(timestamps) == 1
        image_timestamp = pd.Timestamp(timestamps[0])
        assert image_timestamp <= row["timestamp"]
        assert (output_root / "images" / paths[0]).is_file()
    gap_positions = np.flatnonzero(frame["timestamp"].diff().dt.total_seconds().to_numpy() > 300)
    assert len(gap_positions) >= 1
    assert np.isnan(frame.iloc[gap_positions[0]]["pv_ramp_5min"])

    config = _temp_config(tmp_path, parquet_path, output_root / "images", base)
    loaded = load_luoyang_config(config)
    assert loaded["fields"]["timeseries_columns"] == ["final_power", "pv_ramp_5min"]
    datasets = [LuoyangParquetDataset(config, flag) for flag in ("train", "val", "test")]
    assert all(len(dataset) > 0 for dataset in datasets)
    for dataset in datasets:
        sample = dataset[0]
        seq_x, seq_y, _, _, images, future_images, _, _, _, metadata = sample
        assert seq_x.shape == (16, 2)
        assert seq_y.shape == (48, 1)
        assert images.shape == (16, 3, 64, 64)
        assert future_images.shape == (48, 3, 64, 64)
        assert np.isfinite(seq_x).all() and np.isfinite(seq_y).all()
        assert metadata["image_mask"].any()
        assert np.all(metadata["image_minute_offsets"][metadata["image_mask"]] <= 0)
        assert float(seq_y.max()) <= 1.0

    teacher = LuoyangParquetDataset(config, "train", privileged_teacher=True)
    teacher_sample = teacher[0]
    assert teacher_sample[6].shape == (16, 1)
    assert teacher_sample[7].shape == (48, 1)
    assert teacher_sample[-1]["teacher_future_image_mask"].any()
    assert np.isfinite(teacher_sample[6]).all() and np.isfinite(teacher_sample[7]).all()

    # A second run must reuse valid PNGs and preserve the same selected rows.
    second = build(hdf5, times, output_root)
    assert second["output"]["images_rewritten"] == 0
    pd.testing.assert_frame_equal(frame, pd.read_parquet(parquet_path))


def test_skippd_invalid_shape_and_timezone_are_rejected(tmp_path):
    hdf5 = tmp_path / "invalid.hdf5"
    with h5py.File(hdf5, "w") as handle:
        group = handle.create_group("trainval")
        group.create_dataset("images_log", data=np.zeros((2, 64, 64, 1), dtype=np.uint8))
        group.create_dataset("pv_log", data=np.ones(2, dtype=np.float64))
    times = tmp_path / "times.npy"
    np.save(times, np.asarray([datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 1)], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="shape"):
        build(hdf5, times, tmp_path / "out")

    valid_hdf5 = tmp_path / "valid.hdf5"
    with h5py.File(valid_hdf5, "w") as handle:
        group = handle.create_group("trainval")
        group.create_dataset("images_log", data=np.zeros((2, 64, 64, 3), dtype=np.uint8))
        group.create_dataset("pv_log", data=np.ones(2, dtype=np.float64))
    np.save(
        times,
        np.asarray([datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, 0, 1)], dtype=object),
        allow_pickle=True,
    )
    with pytest.raises(ValueError, match="naive"):
        build(valid_hdf5, times, tmp_path / "out2")
