import csv
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from utils.training_monitor import DataQualityAccumulator, TrainingMonitor


class _Dataset:
    seq_len = 4
    pred_len = 2
    image_frames = 3

    def __init__(self):
        self.candidate_audit = {
            "theoretical_issue_count": 220,
            "observed_issue_count": 210,
            "valid_candidate_count": 200,
            "selected_sample_count": 200,
        }
        january = np.repeat(np.array([
            f"2026-01-{day:02d}T07:00" for day in range(1, 6)
        ], dtype="datetime64[ns]"), 20)
        february = np.repeat(np.array([
            f"2026-02-{day:02d}T07:00" for day in range(1, 6)
        ], dtype="datetime64[ns]"), 20)
        self.sample_times = np.concatenate([january, february])
        self.config = {
            "dataset": "SyntheticPrivateDataset",
            "dimensions": {"history_points": 4, "forecast_steps": 2},
            "time": {"forecast_step_minutes": 15},
            "power": {"rated_power": 100.0, "units": "MW"},
            "paths": {"parquet_file": "/private/not-for-monitor/data.parquet"},
        }

    def __len__(self):
        return 200


def _args(**overrides):
    values = {
        "monitor_enabled": True,
        "monitor_dir": "",
        "monitor_rated_power": 100.0,
        "monitor_units": "MW",
        "monitor_min_slice_count": 100,
        "monitor_min_slice_days": 5,
        "monitor_reservoir_size": 64,
        "monitor_forecast_step_minutes": 15,
        "monitor_clip_min": 0.0,
        "monitor_clip_max": 1.0,
        "monitor_direction_threshold": 0.02,
        "model": "TinyModel",
        "batch_size": 3,
        "root_path": "/private/secret-root",
        "api_token": "must-not-be-written",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_monitor_writes_strict_aggregate_artifacts_and_metrics(tmp_path):
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    monitor = TrainingMonitor.from_args(_args(), tmp_path / "run", "teacher", model=model)
    assert monitor.enabled
    assert monitor.run_dir == tmp_path / "run" / "monitor"
    monitor.start_run(model, {"train": _Dataset(), "val": _Dataset()})
    manifest = json.loads(
        (monitor.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["datasets"]["train"]["candidate_audit"][
        "selected_sample_count"
    ] == 200
    assert manifest["datasets"]["train"]["time_coverage"][
        "month_of_year_histogram"
    ]["02"] == 100
    assert manifest["datasets"]["train"]["time_coverage"][
        "hour_of_day_histogram"
    ]["07"] == 200
    assert manifest["arguments"]["monitor_dir"]["redacted"] is True
    assert "basename" not in manifest["arguments"]["monitor_dir"]
    assert manifest["arguments"]["monitor_direction_threshold"] == pytest.approx(
        0.02)
    monitor.register_datasets({"test": _Dataset()})
    manifest = json.loads(
        (monitor.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["datasets"]["test"]["candidate_audit"][
        "selected_sample_count"
    ] == 200

    accumulator = monitor.new_data_quality_accumulator("train", ["power", "weather"])
    repeat = 100
    quality_issue_times = (
        np.datetime64("2026-01-01T00:00", "ns")
        + np.arange(300) * np.timedelta64(1, "h")
    ).astype(np.int64)
    accumulator.update(
        features=np.tile(np.array(
            [[[0.1, 1.0], [0.2, np.nan]], [[0.3, 3.0], [0.4, 4.0]], [[0.5, 5.0], [0.6, 6.0]]]
        ), (repeat, 1, 1)),
        feature_mask=np.tile(np.array(
            [[[1, 1], [1, 0]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=bool
        ), (repeat, 1, 1)),
        target=np.tile(np.array([[0.2, 0.4], [0.3, 0.5], [0.4, 0.6]]), (repeat, 1)),
        current_power=np.tile(np.array([0.1, 0.2, 0.3]), repeat).reshape(-1, 1, 1),
        target_mask=np.tile(np.array([[1, 1], [1, 0], [1, 1]], dtype=bool), (repeat, 1)),
        image_mask=np.tile(np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=bool), (repeat, 1)),
        image_age_minutes=np.tile(np.array([[0, 0, 0], [2, 0, 5], [1, 2, 3]]), (repeat, 1)),
        source_codes={"ghi": np.tile(np.array([[0, 1], [2, 2], [3, 0]]), (repeat, 1))},
        issue_time_ns=quality_issue_times,
        current_power_valid=np.tile(np.array([1, 0, 1], dtype=bool), repeat),
    )
    quality = accumulator.summary()
    assert quality["samples_seen"] == 300
    assert quality["features"]["weather"]["missing_count"] == 100
    assert "min" not in quality["features"]["weather"]
    assert "max" not in quality["features"]["weather"]
    assert quality["target"]["count"] == 500
    assert quality["current_power"]["count"] == 200
    assert quality["target"]["original_units"]["mean"] == pytest.approx(38.0)
    assert quality["images"]["none_count"] == 100
    assert quality["images"]["partial_count"] == 100
    assert quality["images"]["full_count"] == 100
    assert quality["privacy_support"]["aggregate_allowed"] is True
    assert len(quality["horizon_label_profile"]) == 2
    assert quality["horizon_label_profile"]["lead_001"]["horizon_minutes"] == 15
    assert quality["missingness_by_position"]["timeseries"]["shape"] == [2, 2]
    assert quality["sample_completeness"]["input_valid_fraction"]["mean"] is not None
    monitor.record_data_quality("train", quality)
    shifted_quality = copy.deepcopy(quality)
    shifted_quality["features"]["weather"]["mean"] += 1.0
    monitor.record_data_quality("val", shifted_quality)
    quality_file = json.loads(
        (monitor.run_dir / "data_quality.json").read_text(encoding="utf-8")
    )
    assert quality_file["schema_version"] == 1
    assert quality_file["privacy"] == "aggregate_only"
    assert quality_file["comparisons_to_train"]["val"]["features"]["weather"][
        "mean_delta"
    ] == pytest.approx(1.0)

    snapshot = monitor.start_parameter_snapshot(model)
    with torch.no_grad():
        model[0].weight.add_(0.01)
    update = monitor.finish_parameter_snapshot(model, snapshot)
    assert update["update_norm"] > 0
    assert update["branches"]["0"]["update_norm"] > 0
    assert update["branches"]["2"]["update_norm"] == 0

    monitor.record_epoch({
        "epoch": 1,
        "train_loss": 0.2,
        "val_loss": float("nan"),
        "gradient_norm": 0.5,
        "parameter_update": update,
        "predictions": np.array([0.1, 0.2]),
        "batch_losses": np.array([0.1, 0.3]),
        "issue_time_ns": 987654321,
        "checkpoint_paths": ["/private/one.pth", "/private/two.pth"],
    })

    predictions = np.tile(
        np.array([[-0.1, 0.6], [0.3, 1.2], [0.5, 0.7]]), (repeat, 1))
    targets = np.tile(
        np.array([[0.0, 0.5], [0.2, 1.0], [0.4, np.nan]]), (repeat, 1))
    current = np.tile(np.array([0.0, 0.4, 0.45]), repeat)
    target_mask = np.ones_like(targets, dtype=bool)
    issue_times = (
        np.datetime64("2026-01-01T00:00", "ns")
        + np.arange(predictions.shape[0]) * np.timedelta64(1, "h")
    ).astype(np.int64)
    forecast = monitor.record_forecast(
        epoch=1,
        split="val",
        role="teacher",
        predictions=predictions,
        targets=targets,
        current_power=current,
        target_mask=target_mask,
        metadata={
            "image_mask": np.tile(
                np.array([[0, 0], [1, 0], [1, 1]], dtype=bool), (repeat, 1)),
            "timeseries_mask": np.tile(
                np.array([[1, 1], [1, 0], [1, 1]], dtype=bool), (repeat, 1)),
            "issue_time_ns": issue_times,
            "image_paths": ["/private/a.png", "/private/b.png", "/private/c.png"],
        },
    )
    assert forecast["valid_target_count"] == 500
    assert forecast["overall"]["rmse"] == pytest.approx(np.sqrt(0.016) * 100.0)
    assert forecast["overall"]["nrmse"] == pytest.approx(np.sqrt(0.016))
    assert forecast["overall"]["bias"] == pytest.approx(8.0)
    assert forecast["overall_clipped"]["rmse"] == pytest.approx(np.sqrt(0.006) * 100.0)
    assert forecast["below_floor_rate"] == pytest.approx(0.2)
    assert forecast["above_capacity_rate"] == pytest.approx(0.2)
    assert forecast["out_of_bounds_rate"] == pytest.approx(0.4)

    monitor.record_distillation({
        "epoch": 1,
        "teacher_advantage_rate": 0.7,
        "embeddings": np.arange(12).reshape(3, 4),
    })
    result = monitor.finalize({"best_epoch": 1, "best_score": float("inf")})
    assert result["epochs_recorded"] == 1
    # Finalization is repeatable and does not prevent subsequent records.
    monitor.record_epoch({"epoch": 2, "train_loss": 0.1})
    assert monitor.finalize({"best_epoch": 2})["epochs_recorded"] == 2

    expected = {
        "run_manifest.json", "data_quality.json", "epochs.jsonl",
        "horizon_metrics.csv", "slice_metrics.csv", "distillation.jsonl",
        "summary.json",
    }
    assert {path.name for path in monitor.run_dir.iterdir()} == expected

    for name in ("run_manifest.json", "data_quality.json", "summary.json"):
        json.loads((monitor.run_dir / name).read_text(encoding="utf-8"), parse_constant=lambda value: pytest.fail(value))
    for name in ("epochs.jsonl", "distillation.jsonl"):
        for line in (monitor.run_dir / name).read_text(encoding="utf-8").splitlines():
            json.loads(line, parse_constant=lambda value: pytest.fail(value))

    all_output = "\n".join(path.read_text(encoding="utf-8") for path in monitor.run_dir.iterdir())
    assert "must-not-be-written" not in all_output
    assert "/private/" not in all_output
    assert "111111111" not in all_output
    assert "987654321" not in all_output
    assert "a.png" not in all_output
    assert "one.pth" not in all_output
    assert '"val_loss":null' in (monitor.run_dir / "epochs.jsonl").read_text(encoding="utf-8")
    epoch_record = json.loads(
        (monitor.run_dir / "epochs.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert epoch_record["batch_losses"] == {
        "count": 2,
        "finite_count": 2,
        "privacy_suppressed": True,
    }
    assert '"embeddings":{"count":12,"suppressed":true}' in (
        monitor.run_dir / "distillation.jsonl"
    ).read_text(encoding="utf-8")

    with (monitor.run_dir / "horizon_metrics.csv").open(encoding="utf-8", newline="") as handle:
        horizon_rows = list(csv.DictReader(handle))
    assert len(horizon_rows) == 2
    assert float(horizon_rows[0]["rmse"]) == pytest.approx(10.0)
    with (monitor.run_dir / "slice_metrics.csv").open(encoding="utf-8", newline="") as handle:
        slice_rows = list(csv.DictReader(handle))
    assert slice_rows
    assert all(int(row["count"]) >= 2 for row in slice_rows)
    assert any(
        row["slice_dimension"] == "period" and row["slice_value"] == "night"
        for row in slice_rows
    )


def test_data_quality_reservoir_is_bounded_and_mapping_batches_work():
    accumulator = DataQualityAccumulator(
        "val", feature_names=["x"], reservoir_size=32, rated_power=10.0, seed=7
    )
    for start in range(0, 1000, 100):
        accumulator.update({
            "seq_x": np.arange(start, start + 100, dtype=float).reshape(10, 10, 1),
            "timeseries_mask": np.ones((10, 10, 1), dtype=bool),
            "seq_y": np.full((10, 2), 0.5),
            "target_mask": np.ones((10, 2), dtype=bool),
            "current_power": np.full(10, 0.4),
        })
    summary = accumulator.finalize()
    assert summary["samples_seen"] == 100
    assert summary["features"]["x"]["count"] == 1000
    assert summary["features"]["x"]["mean"] == pytest.approx(499.5)
    assert len(accumulator.feature_distributions[0].reservoir) == 32
    assert summary["ramp"]["original_units"]["mean"] == pytest.approx(1.0)


def test_timeseries_source_mapping_is_split_by_feature():
    accumulator = DataQualityAccumulator(
        "train", feature_names=["power", "ghi"], reservoir_size=8
    )
    accumulator.update(
        features=np.ones((1, 2, 2)),
        source_codes={
            "timeseries_source": np.array([[[0, 1], [2, 1]]]),
            "forecast_source": np.array([[3, 4]]),
            "teacher_future_timeseries_source": np.array([[3, 3]]),
        },
    )
    sources = accumulator.summary()["source_codes"]
    assert sources["timeseries:power"] == {"0": 1, "2": 1}
    assert sources["timeseries:ghi"] == {"1": 2}
    assert sources["forecast_source"] == {"3": 1, "other": 1}
    assert sources["teacher_future_timeseries_source"] == {"3": 2}
    rates = accumulator.summary()["source_code_rates"]
    assert rates["timeseries:power"] == {"0": 0.5, "2": 0.5}
    assert rates["timeseries:ghi"] == {"1": 1.0}


def test_disabled_monitor_is_a_noop(tmp_path):
    monitor = TrainingMonitor.from_args(
        _args(monitor_enabled=False), tmp_path / "disabled", "student"
    )
    monitor.start_run(None, {})
    monitor.record_epoch({"epoch": 1})
    assert monitor.record_forecast(1, "val", "student", [0.0], [0.0], [0.0]) == {
        "enabled": False
    }
    assert monitor.finalize({}) == {"enabled": False}
    assert not monitor.run_dir.exists()


def test_explicit_monitor_directory_takes_precedence(tmp_path):
    explicit = tmp_path / "shared-monitor"
    monitor = TrainingMonitor.from_args(
        _args(monitor_dir=str(explicit)), tmp_path / "ignored", "student"
    )
    assert monitor.run_dir == explicit


def test_monitor_cli_alias_enables_monitor(tmp_path):
    args = _args()
    del args.monitor_enabled
    args.monitor = True
    monitor = TrainingMonitor.from_args(args, tmp_path / "run", "student")
    assert monitor.enabled


def test_minimum_distinct_days_is_enforced_for_slices(tmp_path):
    args = _args()
    monitor = TrainingMonitor.from_args(args, tmp_path / "run", "student")
    monitor.start_run(None, {})
    predictions = np.tile(
        np.array([[-0.1, 0.6], [0.3, 1.2], [0.5, 0.7]]), (60, 1))
    targets = np.tile(
        np.array([[0.0, 0.5], [0.2, 1.0], [0.4, np.nan]]), (60, 1))
    current = np.tile(np.array([0.0, 0.4, 0.45]), 60)
    issue_times = (
        np.datetime64("2026-01-01T07:00", "ns")
        + np.arange(predictions.shape[0]) * np.timedelta64(1, "h")
    ).astype(np.int64)
    monitor.record_forecast(
        1, "val", "student", predictions, targets, current,
        metadata={"issue_time_ns": issue_times},
    )
    with (monitor.run_dir / "slice_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(int(row["distinct_day_count"]) >= 5 for row in rows)


def test_training_rerun_resets_but_test_only_preserves_and_merges(tmp_path):
    output = tmp_path / "monitor"
    training_args = _args(monitor_dir=str(output), is_training=1)
    first = TrainingMonitor.from_args(training_args, tmp_path / "ignored", "teacher")
    first.start_run(None, {})
    first.record_epoch({"epoch": 1, "train_loss": 1.0})
    first.finalize({"status": "trained", "best_epoch": 1})

    second = TrainingMonitor.from_args(training_args, tmp_path / "ignored", "teacher")
    second.start_run(None, {})
    assert (output / "epochs.jsonl").read_text(encoding="utf-8") == ""
    second.record_epoch({"epoch": 1, "train_loss": 0.5})
    second.finalize({"status": "trained", "best_epoch": 1, "run_marker": "new"})

    test_args = _args(monitor_dir=str(output), is_training=0)
    test_monitor = TrainingMonitor.from_args(test_args, tmp_path / "ignored", "teacher")
    test_monitor.start_run(None, {"test": _Dataset()})
    assert test_monitor.result_value("best_epoch") == 1
    summary = test_monitor.finalize({"status": "tested", "final_test_recorded": True})
    assert summary["epochs_recorded"] == 1
    assert summary["result"]["best_epoch"] == 1
    assert summary["result"]["run_marker"]["count"] == 1
    assert summary["result"]["run_marker"]["suppressed"] is True
    assert summary["result"]["final_test_recorded"] is True
    assert summary["status"] == "tested"


def test_finalize_emits_optimization_alerts(tmp_path):
    monitor = TrainingMonitor.from_args(_args(learning_rate=1e-3), tmp_path, "student")
    monitor.start_run(None, {})
    monitor.record_epoch({
        "epoch": 1,
        "validation_loss": 0.10,
        "learning_rate": 1e-3,
        "gradient_norm_mean": 0.0,
        "nonfinite_gradient_steps": 1,
        "amp_skipped_steps": 1,
        "parameter_updates": {"relative_update_norm": 0.0},
    })
    monitor.record_epoch({
        "epoch": 2,
        "validation_loss": 0.13,
        "learning_rate": 1e-8,
        "gradient_norm_mean": 1.0,
    })
    codes = {alert["code"] for alert in monitor.finalize({})["alerts"]}
    assert {
        "vanishing_gradient",
        "nonfinite_gradient",
        "amp_steps_skipped",
        "negligible_parameter_update",
        "validation_degraded_after_best",
        "learning_rate_near_zero",
    }.issubset(codes)


def test_low_count_values_are_suppressed_at_the_write_boundary(tmp_path):
    monitor = TrainingMonitor.from_args(_args(), tmp_path / "run", "student")
    monitor.start_run(None, {})

    quality = monitor.new_data_quality_accumulator("val", ["power"])
    quality.update(
        features=np.array([[[0.25]]]),
        target=np.array([[0.4]]),
        current_power=np.array([0.3]),
        source_codes={"timeseries_source": np.array([[[0]]])},
    )
    monitor.record_data_quality("val", quality.summary())
    written_quality = json.loads(
        (monitor.run_dir / "data_quality.json").read_text(encoding="utf-8"))
    assert written_quality["val"]["target"]["mean"] is None
    assert written_quality["val"]["target"]["p50"] is None
    assert written_quality["val"]["target"]["missing_rate"] is None
    assert written_quality["val"]["target"]["privacy_suppressed"] is True
    assert written_quality["val"]["source_codes"]["timeseries:power"][
        "privacy_suppressed"] is True

    issue_times = np.array(["2026-01-01T07:00"], dtype="datetime64[ns]").astype(
        np.int64)
    forecast = monitor.record_forecast(
        1, "val", "student", np.array([[0.7]]), np.array([[0.4]]),
        np.array([0.3]), metadata={"issue_time_ns": issue_times})
    assert forecast["privacy_suppressed"] is True
    assert forecast["overall"]["rmse"] is None
    assert forecast["prediction_mean_normalized"] is None
    with (monitor.run_dir / "horizon_metrics.csv").open(
        encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["privacy_suppressed"] == "True"
    assert rows[0]["rmse"] == ""
    with (monitor.run_dir / "slice_metrics.csv").open(
        encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []


def test_data_quality_requires_independent_sample_and_day_support(tmp_path):
    monitor = TrainingMonitor.from_args(_args(), tmp_path / "run", "student")
    monitor.start_run(None, {})

    def update_quality(accumulator, sample_count, day_count):
        dates = (
            np.datetime64("2026-01-01", "ns")
            + (np.arange(sample_count) % day_count) * np.timedelta64(1, "D")
        ).astype(np.int64)
        accumulator.update(
            features=np.ones((sample_count, 16, 1), dtype=np.float32),
            feature_mask=np.ones((sample_count, 16, 1), dtype=bool),
            target=np.full((sample_count, 2, 1), 0.5, dtype=np.float32),
            target_mask=np.ones((sample_count, 2), dtype=bool),
            current_power=np.full((sample_count, 1, 1), 0.4, dtype=np.float32),
            source_codes={
                "timeseries_source": np.zeros((sample_count, 16, 1), np.int8)
            },
            issue_time_ns=dates,
        )

    one_day = monitor.new_data_quality_accumulator("val", ["power"])
    update_quality(one_day, 100, 1)
    one_day_summary = one_day.summary()
    assert one_day_summary["features"]["power"]["count"] == 1600
    assert one_day_summary["features"]["power"]["support_sample_count"] == 100
    assert one_day_summary["features"]["power"]["mean"] is None
    assert one_day_summary["features"]["power"]["privacy_suppressed"] is True

    too_few_samples = monitor.new_data_quality_accumulator("val", ["power"])
    update_quality(too_few_samples, 99, 5)
    assert too_few_samples.summary()["features"]["power"]["mean"] is None

    sufficient = monitor.new_data_quality_accumulator("val", ["power"])
    update_quality(sufficient, 100, 5)
    summary = sufficient.summary()
    assert summary["privacy_support"]["aggregate_allowed"] is True
    assert summary["features"]["power"]["mean"] == pytest.approx(1.0)
    assert summary["horizon_label_profile"]["lead_001"]["target"]["mean"] == pytest.approx(0.5)
    assert summary["horizon_label_profile"]["lead_001"]["persistence_baseline"][
        "mae_normalized"] == pytest.approx(0.1)
    position = summary["missingness_by_position"]["timeseries"]["positions"][
        "position_0000"]
    assert position["valid_rate"] == pytest.approx(1.0)
    assert summary["source_codes"]["timeseries:power"] == {"0": 1600}


def test_data_quality_suppresses_a_rare_source_category(tmp_path):
    monitor = TrainingMonitor.from_args(_args(), tmp_path / "run", "student")
    monitor.start_run(None, {})
    quality = monitor.new_data_quality_accumulator("train", ["power"])
    dates = (
        np.datetime64("2026-01-01", "ns")
        + (np.arange(100) % 5) * np.timedelta64(1, "D")
    ).astype(np.int64)
    source = np.zeros((100, 1, 1), np.int8)
    source[0, 0, 0] = 1
    quality.update(
        features=np.ones((100, 1, 1)),
        source_codes={"timeseries_source": source},
        issue_time_ns=dates,
    )
    summary = quality.summary()["source_codes"]["timeseries:power"]
    assert summary["privacy_suppressed"] is True
    assert "counts" not in summary
    assert "0" not in summary and "1" not in summary


def test_rare_missingness_has_no_cross_profile_bypass(tmp_path):
    monitor = TrainingMonitor.from_args(_args(), tmp_path / "run", "student")
    monitor.start_run(None, {})
    quality = monitor.new_data_quality_accumulator("train", ["power"])
    dates = (
        np.datetime64("2026-01-01", "ns")
        + (np.arange(100) % 5) * np.timedelta64(1, "D")
    ).astype(np.int64)
    feature_mask = np.ones((100, 2, 1), dtype=bool)
    feature_mask[0, 0, 0] = False
    target_mask = np.ones((100, 2), dtype=bool)
    target_mask[0, 0] = False
    image_mask = np.ones((100, 2), dtype=bool)
    image_mask[0, 0] = False
    quality.update(
        features=np.ones((100, 2, 1)),
        feature_mask=feature_mask,
        target=np.full((100, 2, 1), 0.5),
        target_mask=target_mask,
        current_power=np.full((100, 1, 1), 0.4),
        image_mask=image_mask,
        issue_time_ns=dates,
    )
    summary = quality.summary()
    assert summary["features"]["power"]["missing_rate"] is None
    assert summary["horizon_label_profile"]["lead_001"]["target"][
        "missing_rate"] is None
    assert summary["images"]["coverage"]["mean"] is None
    assert summary["sample_completeness"]["input_valid_fraction"]["mean"] is None
    assert summary["sample_completeness"]["target_valid_fraction"]["mean"] is None


def test_full_dataset_profile_dimensions_survive_safe_serialization(tmp_path):
    monitor = TrainingMonitor.from_args(_args(), tmp_path / "run", "student")
    monitor.start_run(None, {})
    feature_names = [f"feature_{index}" for index in range(22)]
    quality = monitor.new_data_quality_accumulator("train", feature_names)
    dates = (
        np.datetime64("2026-01-01", "ns")
        + (np.arange(100) % 5) * np.timedelta64(1, "D")
    ).astype(np.int64)
    quality.update(
        features=np.ones((100, 16, 22), dtype=np.float32),
        feature_mask=np.ones((100, 16, 22), dtype=bool),
        target=np.full((100, 48, 1), 0.5, dtype=np.float32),
        target_mask=np.ones((100, 48), dtype=bool),
        current_power=np.full((100, 1, 1), 0.4, dtype=np.float32),
        issue_time_ns=dates,
    )
    monitor.record_data_quality("train", quality.summary())
    written = json.loads(
        (monitor.run_dir / "data_quality.json").read_text(encoding="utf-8"))
    profile = written["train"]
    assert len(profile["horizon_label_profile"]) == 48
    assert len(profile["missingness_by_position"]["timeseries"]["positions"]) == 352
    assert profile["horizon_label_profile"]["lead_048"][
        "horizon_minutes"] == 720


def test_payload_aliases_and_short_sequences_cannot_reach_disk(tmp_path):
    monitor = TrainingMonitor.from_args(_args(), tmp_path / "run", "student")
    monitor.start_run(None, {})
    monitor.record_epoch({
        "epoch": 1,
        "prediction": 0.123456789,
        "preds": [0.234567891],
        "outputs": np.array([0.345678912]),
        "labels": "private-label-value",
        "short_values": [0.456789123],
        "warnings": ["safe_aggregate_warning"],
    })
    monitor.record_distillation({
        "epoch": 1,
        "embedding": "private-embedding-value",
        "trues": [0.567891234],
    })
    summary = monitor.finalize({"y_true": 0.678912345})
    assert summary["result"]["y_true"]["suppressed"] is True

    epoch = json.loads(
        (monitor.run_dir / "epochs.jsonl").read_text(encoding="utf-8"))
    assert epoch["prediction"]["suppressed"] is True
    assert epoch["preds"]["suppressed"] is True
    assert epoch["outputs"]["suppressed"] is True
    assert epoch["labels"]["suppressed"] is True
    assert epoch["short_values"]["privacy_suppressed"] is True
    all_output = "\n".join(
        path.read_text(encoding="utf-8") for path in monitor.run_dir.iterdir())
    for private_value in (
        "0.123456789", "0.234567891", "0.345678912", "0.456789123",
        "0.567891234", "0.678912345", "private-label-value",
        "private-embedding-value",
    ):
        assert private_value not in all_output


def test_flat_strata_metrics_follow_their_own_privacy_counts(tmp_path):
    monitor = TrainingMonitor.from_args(_args(), tmp_path / "run", "student")
    monitor.start_run(None, {})
    monitor.record_epoch({
        "epoch": 1,
        "processed_samples": 200,
        "valid_privileged_interventions": 175,
        "skipped_privileged_interventions": 25,
        "early_stop_counter": 2,
        "gradient_norm_max": 0.4,
        "direction_aux_loss": 0.05,
        "direction_threshold_fraction": 0.02,
        "negative_cosine_count": 1,
        "negative_cosine_mean": 0.198765432,
        "train_kd_intervention_loss_weighted_count": 1,
        "train_kd_intervention_loss_weighted": 0.187654321,
        "loss_component_fraction": {
            "kd_intervention_count": 1,
            "kd_intervention": 0.176543219,
        },
        "amp": {"scale_final": 1024.0},
        "checkpoint": {
            "saved": True,
            "best_epoch": 1,
            "early_stop_counter": 0,
        },
        "checkpoint_eligible": True,
        "checkpoint_decision": "validation_improved",
        "checkpoint_paths": [
            "/private/checkpoints/one.pth",
            "/private/checkpoints/two.pth",
        ],
        "validation_forecast": {
            "valid_target_count": 200,
            "prediction_mean_normalized": 0.35,
            "prediction_std_normalized": 0.12,
            "target_mean_normalized": 0.34,
            "target_std_normalized": 0.11,
        },
        "validation_metrics": {
            "count": 200,
            "distinct_day_count": 5,
            "rmse": 0.2,
            "ramp_ge5_count": 150,
            "ramp_ge5_rmse": 0.3,
            "no_path_count": 150,
            "no_path_distinct_day_count": 5,
            "no_path_rmse": 0.25,
            "ramp_ge8_count": 1,
            "ramp_ge8_rmse": 0.987654321,
            "parent_ramp_ge8_rmse": 0.876543219,
            "cloud_event_count": 2,
            "cloud_event_rmse": 0.765432198,
            "cloud_event_ramp_ge8_count": 1,
            "cloud_event_ramp_ge8_rmse": 0.654321987,
            "phase_gate_ge5_count": 1,
            "phase_gate_ge5_mean": 0.543219876,
            "phase_gate_no_path_count": 150,
            "phase_gate_no_path_distinct_day_count": 1,
            "phase_gate_no_path_mean": 0.532198764,
            "phase_route_support_ge5_count": 1,
            "phase_route_support_ge5_mean": 0.432198765,
            "phase_correction_direction_count": 1,
            "phase_correction_direction_bacc": 0.321987654,
            "phase_endpoint_active_recall_count": 1,
            "phase_endpoint_active_recall": 0.219876543,
        },
    })
    record = json.loads(
        (monitor.run_dir / "epochs.jsonl").read_text(encoding="utf-8"))
    metrics = record["validation_metrics"]
    assert record["valid_privileged_interventions"] == 175
    assert record["skipped_privileged_interventions"] == 25
    assert record["early_stop_counter"] == 2
    assert record["gradient_norm_max"] == pytest.approx(0.4)
    assert record["direction_aux_loss"] == pytest.approx(0.05)
    assert record["direction_threshold_fraction"] == pytest.approx(0.02)
    assert record["negative_cosine_mean"] is None
    assert record["train_kd_intervention_loss_weighted"] is None
    assert record["loss_component_fraction"]["kd_intervention"] is None
    assert record["amp"]["scale_final"] == pytest.approx(1024.0)
    assert record["checkpoint"]["saved"] is True
    assert record["checkpoint"]["best_epoch"] == 1
    assert record["checkpoint_eligible"] is True
    assert record["checkpoint_decision"] == "validation_improved"
    assert record["checkpoint_paths"]["redacted"] is True
    assert record["validation_forecast"][
        "prediction_mean_normalized"] == pytest.approx(0.35)
    assert record["validation_forecast"][
        "target_std_normalized"] == pytest.approx(0.11)
    assert metrics["rmse"] == pytest.approx(0.2)
    assert "distinct_day_privacy_suppressed" not in metrics
    assert metrics["ramp_ge5_rmse"] == pytest.approx(0.3)
    assert metrics["no_path_rmse"] == pytest.approx(0.25)
    assert metrics["ramp_ge8_rmse"] is None
    assert metrics["parent_ramp_ge8_rmse"] is None
    assert metrics["cloud_event_rmse"] is None
    assert metrics["cloud_event_ramp_ge8_rmse"] is None
    assert metrics["phase_gate_ge5_mean"] is None
    assert metrics["phase_gate_no_path_mean"] is None
    assert metrics["phase_route_support_ge5_mean"] is None
    assert metrics["phase_correction_direction_bacc"] is None
    assert metrics["phase_endpoint_active_recall"] is None
    assert metrics["ramp_ge8_privacy_suppressed"] is True
    output = (monitor.run_dir / "epochs.jsonl").read_text(encoding="utf-8")
    for private_value in (
        "0.987654321", "0.876543219", "0.765432198", "0.654321987",
        "0.543219876", "0.432198765", "0.321987654", "0.219876543",
        "0.532198764",
        "0.198765432", "0.187654321", "0.176543219",
        "/private/checkpoints/one.pth", "/private/checkpoints/two.pth",
    ):
        assert private_value not in output
    assert monitor.finalize({"epochs_completed": 3})["result"][
        "epochs_completed"] == 3


def test_privacy_thresholds_and_dedicated_directory_are_mandatory(tmp_path):
    with pytest.raises(ValueError, match="at least 100"):
        TrainingMonitor.from_args(
            _args(monitor_min_slice_count=99), tmp_path / "run", "student")
    with pytest.raises(ValueError, match="at least 5"):
        TrainingMonitor.from_args(
            _args(monitor_min_slice_days=4), tmp_path / "run", "student")
    with pytest.raises(ValueError, match="dedicated directory"):
        TrainingMonitor.from_args(
            _args(monitor_dir=str(tmp_path)), tmp_path / "run", "student")

    occupied = tmp_path / "occupied-monitor"
    occupied.mkdir()
    (occupied / "phase_validation_private.npz").write_bytes(b"private")
    monitor = TrainingMonitor.from_args(
        _args(monitor_dir=str(occupied)), tmp_path / "other-run", "student")
    with pytest.raises(ValueError, match="only monitor artifacts"):
        monitor.start_run(None, {})


def test_preserved_monitor_rejects_symlinks_and_legacy_manifest(tmp_path):
    symlink_dir = tmp_path / "symlink-monitor"
    symlink_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (symlink_dir / "run_manifest.json").symlink_to(outside)
    symlink_monitor = TrainingMonitor.from_args(
        _args(monitor_dir=str(symlink_dir), is_training=0),
        tmp_path / "ignored", "student",
    )
    with pytest.raises(ValueError, match="regular non-symlink"):
        symlink_monitor.start_run(None, {})

    legacy_dir = tmp_path / "legacy-monitor"
    legacy_dir.mkdir()
    (legacy_dir / "run_manifest.json").write_text(
        json.dumps({"schema_version": 0, "raw_predictions": [0.1]}) + "\n",
        encoding="utf-8",
    )
    legacy_monitor = TrainingMonitor.from_args(
        _args(monitor_dir=str(legacy_dir), is_training=0),
        tmp_path / "ignored", "student",
    )
    with pytest.raises(ValueError, match="aggregate_only"):
        legacy_monitor.start_run(None, {})


def test_teacher_epoch_validation_requires_a_day_safe_forecast():
    from exp.exp_long_term_forecasting_teacher import Exp_Long_Term_Forecast

    raw = {
        "split": "val",
        "phase": "epoch",
        "valid_target_count": 200,
        "total_loss": 0.123456789,
    }
    suppressed = Exp_Long_Term_Forecast._privacy_safe_validation_diagnostics(raw)
    assert suppressed["privacy_suppressed"] is True
    assert "total_loss" not in suppressed
    assert Exp_Long_Term_Forecast._privacy_safe_validation_value(
        0.123456789, suppressed) is None

    raw["forecast"] = {
        "valid_target_count": 200,
        "distinct_day_count": 5,
        "privacy_suppressed": False,
    }
    assert Exp_Long_Term_Forecast._privacy_safe_validation_diagnostics(raw) is raw
    assert Exp_Long_Term_Forecast._privacy_safe_validation_value(
        0.123456789, raw) == pytest.approx(0.123456789)

    similarity_stats = {"count": 200, "sum": 100.0, "sum_sq": 60.0}
    assert Exp_Long_Term_Forecast._privacy_safe_similarity(
        similarity_stats, similarity_stats, None
    ) == {"privacy_suppressed": True}
    safe_similarity = Exp_Long_Term_Forecast._privacy_safe_similarity(
        similarity_stats, similarity_stats, raw["forecast"])
    assert safe_similarity["image"]["mean"] == pytest.approx(0.5)
