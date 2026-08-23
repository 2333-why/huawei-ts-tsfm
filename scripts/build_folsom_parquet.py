#!/usr/bin/env python3
"""Build Luoyang-compatible and YLJ-compatible Parquet files from Folsom CSV.

The source CSV is minute-level but has outages.  Each output is an aggregate
on a regular grid, and rows belonging to empty source bins are omitted.  The
loaders then reject windows crossing those gaps instead of silently inventing
targets.  YLJ forecast columns are causal lagged products: a value at time u
is copied from u-24h or u-4h, so an issue time t never reads weather after t.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


SOURCE_COLUMNS = [
    "ghi", "dni", "dhi", "air_temp", "relhum", "press", "windsp", "winddir", "max_windsp"
]
LUOYANG_EXOGENOUS = SOURCE_COLUMNS[1:]
YLJ_EXOGENOUS = [
    ("GHI", "ghi"),
    ("TEMP", "air_temp"),
    ("WS", "windsp"),
    ("WD", "winddir"),
    ("RELHUM", "relhum"),
    ("PRESS", "press"),
    ("MAX_WINDSP", "max_windsp"),
]


def _read_source(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path, usecols=["timestamp", *SOURCE_COLUMNS])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("Folsom source contains invalid timestamps")
    for column in SOURCE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Folsom source timestamps must be unique")
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    return frame.reset_index(drop=True)


def _aggregate(frame: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    indexed = frame.set_index("timestamp")
    # Right-labelled bins make each row the end of its aggregation interval.
    # This keeps the first row causal for the following forecast interval.
    result = indexed[SOURCE_COLUMNS].resample(
        f"{interval_minutes}min", label="right", closed="right"
    ).mean()
    result = result.dropna(how="all").reset_index()
    result = result.dropna(subset=SOURCE_COLUMNS).reset_index(drop=True)
    if result.empty:
        raise ValueError("resampling produced no complete rows")
    return result


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _load_image_cache(cache_path: str | Path):
    cache = Path(cache_path)
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata.get("image_size", -1)) != 32:
        raise ValueError("Folsom image cache must contain 32x32 images")
    minutes = np.load(cache / "minutes.npy", mmap_mode="r", allow_pickle=False)
    rgb = np.load(cache / "rgb.npy", mmap_mode="r", allow_pickle=False)
    if minutes.dtype != np.int64 or rgb.dtype != np.uint8 or rgb.shape != (len(minutes), 32, 32, 3):
        raise ValueError("invalid Folsom image cache arrays")
    return minutes, rgb


def _export_images(
    output_root: Path,
    frame_times: pd.Series,
    cache_path: str | Path,
    interval_minutes: int,
    quality: int = 85,
) -> tuple[list[str], list[str]]:
    """Export at most one cached image in each output interval.

    The selected capture is at or before the row timestamp.  Its real capture
    timestamp is retained in asi_path_timestamps, allowing the loader's
    history-aligned forward fill to enforce the image causality boundary.
    """
    minutes, rgb = _load_image_cache(cache_path)
    frame_ns = pd.to_datetime(frame_times).astype("int64").to_numpy()
    frame_minutes = frame_ns // 60_000_000_000
    cache_positions = np.searchsorted(minutes, frame_minutes, side="right") - 1
    # Do not use a stale image from an earlier output interval.
    ages = frame_minutes - minutes[np.maximum(cache_positions, 0)]
    valid = (cache_positions >= 0) & (ages >= 0) & (ages < interval_minutes)

    image_root = output_root / "images"
    paths: list[str] = []
    timestamps: list[str] = []
    written: dict[int, str] = {}
    for row_index, is_valid in enumerate(valid):
        if not is_valid:
            paths.append(_json_list([]))
            timestamps.append(_json_list([]))
            continue
        position = int(cache_positions[row_index])
        minute = int(minutes[position])
        relative = written.get(minute)
        if relative is None:
            capture = pd.Timestamp(minute, unit="m")
            relative = capture.strftime("%Y/%m/%d/%Y%m%d_%H%M%S.jpg")
            path = image_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(rgb[position])).save(path, format="JPEG", quality=quality)
            written[minute] = relative
        capture = pd.Timestamp(minute, unit="m").strftime("%Y-%m-%d %H:%M:%S")
        paths.append(_json_list([relative]))
        timestamps.append(_json_list([capture]))
    return paths, timestamps


def _power_scale(frame: pd.DataFrame) -> float:
    maximum = float(pd.to_numeric(frame["ghi"], errors="coerce").max())
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Folsom GHI has no positive finite maximum")
    return maximum


def _base_metadata(source: Path, interval: int, output: pd.DataFrame) -> dict:
    return {
        "source_csv": str(source.resolve()),
        "sampling_interval_minutes": interval,
        "aggregation": "mean over right-closed, right-labelled bins; empty bins omitted",
        "rows": int(len(output)),
        "timestamp_timezone": "naive UTC",
        "target_source_column": "ghi",
        "target_max_after_aggregation": float(output["ghi"].max()),
    }


def _write_luoyang(
    frame: pd.DataFrame,
    output_root: Path,
    source: Path,
    image_cache: str | Path | None,
    image_quality: int,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    output = frame.rename(columns={"ghi": "final_power"}).copy()
    if image_cache:
        paths, timestamps = _export_images(
            output_root, output["timestamp"], image_cache, 5, image_quality
        )
    else:
        paths = [_json_list([])] * len(output)
        timestamps = [_json_list([])] * len(output)
    output["asi_path"] = paths
    output["asi_path_timestamps"] = timestamps
    parquet_path = output_root / "folsom_luoyang.parquet"
    output.to_parquet(parquet_path, index=False)
    scale = _power_scale(frame)
    config = {
        "schema_version": 1,
        "dataset": "LuoyangParquet",
        "paths": {
            "parquet_file": str(parquet_path.resolve()),
            "image_root": str((output_root / "images").resolve()),
            "checkpoints_root": str((output_root / "checkpoints_luoyang").resolve()),
            "results_root": str((output_root / "results_luoyang").resolve()),
            "logs_root": str((output_root / "logs_luoyang").resolve()),
            "teacher_checkpoint": str((output_root / "checkpoints_luoyang/teacher/checkpoint.pth").resolve()),
            "student_checkpoint": str((output_root / "checkpoints_luoyang/student/checkpoint.pth").resolve()),
        },
        "fields": {
            "timestamp": "timestamp", "target": "final_power",
            "image_paths": "asi_path", "image_timestamps": "asi_path_timestamps",
            "timeseries_columns": ["final_power", *LUOYANG_EXOGENOUS],
        },
        "dimensions": {
            "timeseries_input_dim": 9, "history_points": 16,
            "image_history_frames": 16, "image_channels": 3,
            "image_height": 32, "image_width": 32, "forecast_steps": 48,
            "target_dim": 1, "weather_placeholder_dim": 8,
        },
        "privileged_teacher": {
            "enabled": True, "future_timeseries_columns": LUOYANG_EXOGENOUS,
            "future_images_enabled": bool(image_cache),
            "future_image_sampling_strategy": "forecast_aligned_ffill",
            "future_image_ffill_max_gap_minutes": 30.0,
            "exclude_target_from_inputs": True,
        },
        "time": {
            "sampling_interval_minutes": 5, "history_order": "past_first",
            "forecast_step_minutes": 5, "forecast_horizon_minutes": 240,
            "evaluation_horizons_minutes": [15, 240],
            "train_start_timestamp": "2014-01-01 00:00:00",
            "test_start_timestamp": "2016-01-01 00:00:00",
            "test_end_exclusive_timestamp": "2017-01-01 00:00:00",
            "validation_fraction": 0.15,
        },
        "power": {
            "power_scale": scale, "max_power": scale, "rated_power": scale,
            "model_input_normalization": "divide_by_power_scale",
            "units": "W/m^2", "hard_delta_fraction": 0.10,
            "direction_min_delta_fraction": 0.02,
        },
        "images": {
            "sampling_strategy": "history_aligned_ffill",
            "image_ffill_max_gap_minutes": 30.0, "reject_future_images": True,
            "pad_value": 0.0, "resize_mode": "bilinear",
        },
        "missing": {
            "exogenous_fill": "causal_forward_fill_then_train_mean",
            "exogenous_normalization": "train_zscore", "normalization_epsilon": 1e-6,
        },
        "runtime": {
            "gpu_devices": "0", "num_workers": 0, "prefetch_factor": 2,
            "use_amp": False, "teacher_batch_size": 64, "student_batch_size": 64,
            "teacher_epochs": 40, "student_epochs": 40, "patience": 10,
            "learning_rate": 0.0001,
        },
        "monitoring": {
            "enabled": True, "level": "standard", "aggregate_only": True,
            "epoch_interval": 1, "gradient_interval_steps": 50,
            "min_slice_count": 100, "min_slice_days": 5,
            "include_test_during_training": False, "save_raw_predictions": False,
            "heavy_diagnostics": ["initial", "best", "final"],
        },
        "model": {
            "d_model": 512, "n_heads": 8, "e_layers": 2, "d_layers": 1,
            "d_ff": 2048, "factor": 3, "image_encoder_type": "cnn",
            "image_cnn_dim": 64, "image_temporal_hidden_dim": 64,
            "teacher_fuse_strategy": "full" if image_cache else "no_img",
            "student_fuse_strategy": "causal_img" if image_cache else "ts_only",
            "kd_type": "inter_causal", "kd_similarity_weight": 0.001,
            "kd_intervention_weight": 0.1, "kd_intervention_min_feature_delta": 1e-8,
            "modality_dropout_rate": 0.0,
        },
        "output": {"prediction_floor": 0.0, "predictions_file": "official_test_predictions.csv", "metrics_file": "official_test_metrics.json"},
        "conversion": _base_metadata(source, 5, frame),
    }
    config_path = output_root / "folsom_luoyang.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path


def _write_ylj(frame: pd.DataFrame, output_root: Path, source: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(index=frame["timestamp"])
    output.index.name = "timestamp"
    output["observe_power"] = frame["ghi"].to_numpy()
    for name, source_name in YLJ_EXOGENOUS:
        output[f"{name}_observe"] = frame[source_name].to_numpy()
    base = output.copy()
    history_columns = ["observe_power", *[f"{name}_observe" for name, _ in YLJ_EXOGENOUS]]
    forecast_columns: list[str] = []
    for lag_name, lag in (("1day", pd.Timedelta(hours=24)), ("4hour", pd.Timedelta(hours=4))):
        for name, _ in YLJ_EXOGENOUS:
            output_name = f"{name}_forecast_{lag_name}"
            source_values = base[f"{name}_observe"].reindex(base.index - lag).to_numpy()
            output[output_name] = source_values
            forecast_columns.append(output_name)
    output = output.reset_index()
    parquet_path = output_root / "folsom_ylj.parquet"
    output.to_parquet(parquet_path, index=False)
    scale = _power_scale(frame)
    all_columns = history_columns + forecast_columns
    fallback = {}
    for name, _ in YLJ_EXOGENOUS:
        fallback[f"{name}_forecast_1day"] = {"alternative": f"{name}_forecast_4hour", "persistence": f"{name}_observe"}
        fallback[f"{name}_forecast_4hour"] = {"alternative": f"{name}_forecast_1day", "persistence": f"{name}_observe"}
    config = {
        "schema_version": 1, "dataset": "YLJParquet",
        "paths": {
            "parquet_file": str(parquet_path.resolve()),
            "checkpoints_root": str((output_root / "checkpoints_ylj").resolve()),
            "results_root": str((output_root / "results_ylj").resolve()),
            "logs_root": str((output_root / "logs_ylj").resolve()),
            "teacher_checkpoint": str((output_root / "checkpoints_ylj/teacher/checkpoint.pth").resolve()),
            "student_checkpoint": str((output_root / "checkpoints_ylj/student/checkpoint.pth").resolve()),
        },
        "fields": {
            "timestamp": "timestamp", "target": "observe_power",
            "time_series_columns": all_columns, "history_columns": history_columns,
            "forecast_window_columns": forecast_columns, "time_series_scale_factors": {},
            "forecast_fallbacks": fallback,
        },
        "dimensions": {
            "time_series_input_dim": len(all_columns), "history_points": 16,
            "forecast_steps": 16, "target_dim": 1, "image_history_frames": 16,
            "image_channels": 3, "image_height": 32, "image_width": 32,
            "weather_placeholder_dim": len(YLJ_EXOGENOUS),
        },
        "privileged_teacher": {
            "enabled": True,
            "future_timeseries_columns": [f"{name}_observe" for name, _ in YLJ_EXOGENOUS],
            "future_images_enabled": False, "exclude_target_from_inputs": True,
        },
        "time": {
            "sampling_interval_minutes": 15, "forecast_step_minutes": 15,
            "forecast_horizon_minutes": 240, "forecast_window_shift_minutes": 240,
            "evaluation_horizons_minutes": [15, 240], "history_order": "past_first",
            "train_start_timestamp": "2014-01-01 00:00:00",
            "test_start_timestamp": "2016-01-01 00:00:00",
            "test_end_exclusive_timestamp": "2017-01-01 00:00:00",
            "validation_fraction": 0.15,
        },
        "power": {
            "power_scale": scale, "max_power": scale, "rated_power": scale,
            "units": "W/m^2", "model_input_normalization": "divide_by_power_scale",
            "target_floor": 0.0, "allow_missing_target": True,
            "hard_delta_fraction": 0.10, "direction_min_delta_fraction": 0.02,
        },
        "images": {"images_enabled": False, "pad_value": 0.0},
        "missing": {
            "history_fill": "causal_forward_fill_then_train_mean",
            "forecast_fill": "paired_product_then_issue_time_persistence_then_train_mean",
            "normalization": "train_zscore", "normalization_epsilon": 1e-6,
        },
        "runtime": {
            "gpu_devices": "0", "precision": "fp32", "num_workers": 0,
            "prefetch_factor": 2, "pin_memory": False, "teacher_batch_size": 64,
            "student_batch_size": 64, "teacher_epochs": 40, "student_epochs": 40,
            "patience": 10, "learning_rate": 0.0001,
        },
        "monitoring": {
            "enabled": True, "level": "standard", "aggregate_only": True,
            "epoch_interval": 1, "gradient_interval_steps": 50,
            "min_slice_count": 100, "min_slice_days": 5,
            "include_test_during_training": False, "save_raw_predictions": False,
            "heavy_diagnostics": ["initial", "best", "final"],
        },
        "model": {
            "d_model": 512, "n_heads": 8, "e_layers": 2, "d_layers": 1,
            "d_ff": 2048, "factor": 3, "image_encoder_type": "cnn",
            "image_cnn_dim": 64, "image_temporal_hidden_dim": 64,
            "teacher_fuse_strategy": "no_img", "student_fuse_strategy": "ts_only",
            "kd_type": "inter_causal", "kd_similarity_weight": 0.001,
            "kd_intervention_weight": 0.1, "kd_intervention_min_feature_delta": 1e-8,
            "modality_dropout_rate": 0.0,
        },
        "output": {"predictions_file": "official_test_predictions.csv", "metrics_file": "official_test_metrics.json"},
        "conversion": {
            **_base_metadata(source, 15, frame),
            "forecast_products": {"1day": "value at u-24h", "4hour": "value at u-4h"},
            "forecast_product_is_causal": True,
            "source_feature_mapping": {f"{name}_observe": source_name for name, source_name in YLJ_EXOGENOUS},
        },
    }
    config_path = output_root / "folsom_ylj.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def build(source: str | Path, output_root: str | Path, image_cache: str | Path | None = None, image_quality: int = 85) -> dict[str, str]:
    source_path = Path(source).resolve()
    output_path = Path(output_root).resolve()
    raw = _read_source(source_path)
    luoyang_frame = _aggregate(raw, 5)
    ylj_frame = _aggregate(raw, 15)
    luoyang_config = _write_luoyang(luoyang_frame, output_path / "luoyang", source_path, image_cache, image_quality)
    ylj_config = _write_ylj(ylj_frame, output_path / "ylj", source_path)
    manifest = {
        "source": str(source_path), "output_root": str(output_path),
        "luoyang_config": str(luoyang_config), "ylj_config": str(ylj_config),
        "image_cache": str(Path(image_cache).resolve()) if image_cache else None,
    }
    (output_path / "conversion_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (output_path / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--image-cache", type=Path)
    parser.add_argument("--image-quality", type=int, default=85)
    args = parser.parse_args()
    if not 1 <= args.image_quality <= 100:
        raise SystemExit("--image-quality must be between 1 and 100")
    print(json.dumps(build(args.source, args.output_root, args.image_cache, args.image_quality), indent=2))


if __name__ == "__main__":
    main()
