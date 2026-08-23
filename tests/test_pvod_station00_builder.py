from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from data_provider.data_loader_ylj import YLJParquetDataset, load_ylj_config
from scripts.build_pvod_station00_parquet import (
    FORECAST_COLUMNS,
    HISTORY_COLUMNS,
    build_parquet,
)


def _source_csv(tmp_path: Path) -> Path:
    timestamps = pd.date_range(
        "2019-12-31 20:00:00", "2020-01-21 00:00:00", freq="15min", inclusive="left"
    )
    rows = {
        "date_time": timestamps,
        "nwp_globalirrad": np.arange(len(timestamps), dtype=float),
        "nwp_directirrad": np.arange(len(timestamps), dtype=float) + 1,
        "nwp_temperature": np.arange(len(timestamps), dtype=float) + 2,
        "nwp_humidity": np.arange(len(timestamps), dtype=float) + 3,
        "nwp_windspeed": np.arange(len(timestamps), dtype=float) + 4,
        "nwp_winddirection": np.arange(len(timestamps), dtype=float) + 5,
        "nwp_pressure": np.arange(len(timestamps), dtype=float) + 6,
        "lmd_totalirrad": np.arange(len(timestamps), dtype=float) + 7,
        "lmd_diffuseirrad": np.arange(len(timestamps), dtype=float) + 8,
        "lmd_temperature": np.arange(len(timestamps), dtype=float) + 9,
        "lmd_pressure": np.arange(len(timestamps), dtype=float) + 10,
        "lmd_winddirection": np.arange(len(timestamps), dtype=float) + 11,
        "lmd_windspeed": np.arange(len(timestamps), dtype=float) + 12,
        "power": np.arange(len(timestamps), dtype=float) / 100.0,
    }
    frame = pd.DataFrame(rows)
    duplicate_time = pd.Timestamp("2020-01-01 01:15:00")
    duplicate = frame.loc[frame["date_time"].eq(duplicate_time)].copy()
    duplicate.loc[:, frame.columns[1:]] += 10.0
    frame = pd.concat([frame.iloc[::-1], duplicate], ignore_index=True)
    source = tmp_path / "station00.csv"
    frame.to_csv(source, index=False)
    return source


def test_station00_builder_maps_and_collapses_duplicates(tmp_path):
    source = _source_csv(tmp_path)
    output = tmp_path / "station00_ylj.parquet"
    summary = build_parquet(source, output)
    converted = pd.read_parquet(output)

    assert summary["source_rows"] == len(pd.read_csv(source))
    assert summary["duplicate_rows_collapsed"] == 1
    assert summary["output_rows"] == len(converted)
    assert converted["timestamp"].is_monotonic_increasing
    assert converted["timestamp"].is_unique
    assert converted.columns.tolist() == ["timestamp", *HISTORY_COLUMNS, *FORECAST_COLUMNS]

    row = converted.set_index("timestamp").loc[pd.Timestamp("2020-01-01 01:15:00")]
    original = pd.read_csv(source)
    values = original.loc[original["date_time"] == "2020-01-01 01:15:00", "power"]
    assert row["observe_power"] == pytest.approx(values.mean())
    assert row["lmd_temperature"] == pytest.approx(
        original.loc[original["date_time"] == "2020-01-01 01:15:00", "lmd_temperature"].mean()
    )
    assert row["nwp_globalirrad"] == pytest.approx(
        original.loc[original["date_time"] == "2020-01-01 01:15:00", "nwp_globalirrad"].mean()
    )
    assert summary["gap_count"] == 0


def test_station00_config_and_loader_splits(tmp_path):
    source = _source_csv(tmp_path)
    output = tmp_path / "station00_ylj.parquet"
    build_parquet(source, output)

    config_path = Path(__file__).parents[1] / "configs/datasets/pvod_station00_ylj.yaml"
    config = load_ylj_config(config_path)
    assert config["fields"]["history_columns"] == HISTORY_COLUMNS
    assert config["fields"]["forecast_window_columns"] == FORECAST_COLUMNS
    assert config["fields"]["time_series_columns"] == HISTORY_COLUMNS + FORECAST_COLUMNS
    assert config["dimensions"]["weather_placeholder_dim"] == 6
    assert config["power"]["power_scale"] == config["power"]["max_power"] == config["power"]["rated_power"] == 6.6
    assert config["runtime"]["gpu_devices"] == "0"
    assert config["runtime"]["precision"] == "fp32"

    config["paths"]["parquet_file"] = str(output)
    config["time"].update({
        "train_start_timestamp": "2020-01-01 00:00:00",
        "test_start_timestamp": "2020-01-15 00:00:00",
        "test_end_exclusive_timestamp": "2020-01-21 00:00:00",
    })
    temp_config = tmp_path / "pvod_station00_ylj.yaml"
    temp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    datasets = [YLJParquetDataset(temp_config, flag) for flag in ("train", "val", "test")]
    assert all(len(dataset) > 0 for dataset in datasets)
    for dataset in datasets:
        sample = dataset[0]
        assert sample[0].shape == (16, 14)
        assert sample[1].shape == (16, 1)
        assert np.isfinite(sample[0]).all()
        assert np.isfinite(sample[1]).all()
        assert not sample[-1]["image_mask"].any()
        assert not sample[-1]["teacher_future_image_mask"].any()
