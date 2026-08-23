import json

import numpy as np
import pandas as pd

from data_provider.data_loader_luoyang import load_luoyang_config
from data_provider.data_loader_ylj import load_ylj_config
from scripts.build_folsom_parquet import _aggregate, _read_source, _write_luoyang, _write_ylj


def _source_csv(tmp_path):
    timestamps = pd.date_range("2014-01-01", "2014-01-04", freq="1min", inclusive="left")
    index = np.arange(len(timestamps), dtype=np.float64)
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "ghi": 100.0 + index,
        "dni": 200.0 + index,
        "dhi": 300.0 + index,
        "air_temp": 10.0 + index / 100.0,
        "relhum": 50.0,
        "press": 1000.0,
        "windsp": 1.0,
        "winddir": 180.0,
        "max_windsp": 2.0,
    })
    path = tmp_path / "folsom.csv"
    frame.to_csv(path, index=False)
    return path


def test_folsom_builder_preserves_adapter_layout_and_causal_forecasts(tmp_path):
    source = _source_csv(tmp_path)
    raw = _read_source(source)
    luoyang_frame = _aggregate(raw, 5)
    ylj_frame = _aggregate(raw, 15)

    luoyang_config_path = _write_luoyang(
        luoyang_frame, tmp_path / "luoyang", source, image_cache=None, image_quality=85
    )
    ylj_config_path = _write_ylj(ylj_frame, tmp_path / "ylj", source)

    luoyang_config = load_luoyang_config(luoyang_config_path)
    ylj_config = load_ylj_config(ylj_config_path)
    luoyang = pd.read_parquet(luoyang_config["paths"]["parquet_file"])
    ylj = pd.read_parquet(ylj_config["paths"]["parquet_file"])

    assert list(luoyang.columns) == [
        "timestamp", "final_power", "dni", "dhi", "air_temp", "relhum",
        "press", "windsp", "winddir", "max_windsp", "asi_path",
        "asi_path_timestamps",
    ]
    assert luoyang["asi_path"].map(json.loads).map(len).eq(0).all()
    assert ylj.columns[1:].tolist() == ylj_config["fields"]["time_series_columns"]

    issue = pd.Timestamp("2014-01-02 12:00:00")
    row = ylj.set_index("timestamp").loc[issue]
    four_hour_source = ylj.set_index("timestamp").loc[issue - pd.Timedelta(hours=4), "GHI_observe"]
    one_day_source = ylj.set_index("timestamp").loc[issue - pd.Timedelta(hours=24), "GHI_observe"]
    assert row["GHI_forecast_4hour"] == four_hour_source
    assert row["GHI_forecast_1day"] == one_day_source

    for name in ylj_config["fields"]["forecast_window_columns"]:
        assert name in ylj.columns
    assert ylj_config["dimensions"]["time_series_input_dim"] == 22
    assert ylj_config["power"]["power_scale"] == ylj_config["power"]["rated_power"]
