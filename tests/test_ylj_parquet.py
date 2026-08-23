from pathlib import Path
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import yaml

from data_provider.data_loader_ylj import YLJParquetDataset, load_ylj_config
from utils.ylj_outputs import build_ylj_outputs
from utils.ylj_checkpoint_contract import (
    validate_ylj_checkpoint_contract,
    write_ylj_checkpoint_contract,
)


def test_ts_only_student_skips_unavailable_phase_prior_statistics():
    from exp.exp_long_term_forecasting_kd import Exp_Long_Term_Forecast

    class TsOnlyStudent:
        def phase_winner_prior_statistics(self):
            raise AssertionError("ts_only must not read phase-only buffers")

    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    experiment.model = TsOnlyStudent()
    assert experiment._phase_winner_prior_statistics() is None


def test_kd_monitor_distillation_winners_are_finite_and_mutually_exclusive():
    from exp.exp_long_term_forecasting_kd import Exp_Long_Term_Forecast

    student_payload = {
        "predictions": np.array([[[np.nan]], [[0.8]], [[0.500000001]]]),
        "targets": np.array([[[0.0]], [[0.5]], [[0.5]]]),
        "target_mask": np.ones((3, 1), dtype=bool),
    }
    teacher_payload = {
        "predictions": np.array([[[0.0]], [[0.6]], [[0.5]]]),
    }

    summary = Exp_Long_Term_Forecast._distillation_monitor_statistics(
        student_payload, teacher_payload)

    assert summary["valid_count"] == 2
    assert summary["teacher_better_fraction"] == pytest.approx(0.5)
    assert summary["student_better_fraction"] == pytest.approx(0.0)
    assert summary["equal_error_fraction"] == pytest.approx(0.5)
    assert (
        summary["teacher_better_fraction"]
        + summary["student_better_fraction"]
        + summary["equal_error_fraction"]
    ) == pytest.approx(1.0)
    suppressed = Exp_Long_Term_Forecast._distillation_monitor_statistics(
        student_payload, teacher_payload, min_count=3)
    assert suppressed == {
        "valid_count": 2,
        "distinct_day_count": None,
        "privacy_suppressed": True,
    }


@pytest.fixture(scope="module")
def synthetic_ylj(tmp_path_factory):
    root = tmp_path_factory.mktemp("ylj")
    config = load_ylj_config(Path(__file__).parents[1] / "configs" / "datasets" / "ylj.yaml")
    timestamps = pd.date_range("2023-12-31 20:15", "2024-01-07", freq="15min", inclusive="left")
    count = len(timestamps)
    frame = pd.DataFrame({"timestamp": timestamps})
    for index, name in enumerate(config["fields"]["time_series_columns"]):
        if name == "observe_power":
            frame[name] = 100.0 + 20.0 * np.sin(np.arange(count) / 20.0)
        elif name.startswith("PWAT"):
            frame[name] = 20.0 + index
        else:
            frame[name] = index * 10.0 + np.arange(count) / 100.0

    issue = pd.Timestamp("2024-01-05 00:00:00")
    row = frame.index[frame["timestamp"] == issue][0]
    frame.loc[row, "observe_power"] = -3.0
    frame.loc[row - 1, "observe_power"] = np.nan
    frame.loc[row - 15, "GHI_observe"] = np.nan
    frame.loc[row, "GHI_observe"] = 123.0
    frame.loc[row + 1, "GHI_forecast_1day"] = np.nan
    frame.loc[row + 1, "GHI_forecast_4hour"] = 321.0
    frame.loc[row + 2, ["GHI_forecast_1day", "GHI_forecast_4hour"]] = np.nan
    frame.loc[row + 3, "observe_power"] = np.nan
    frame.loc[row + 4, "observe_power"] = -8.0
    frame.loc[row + 16, "observe_power"] = np.nan
    parquet = root / "ylj.parquet"
    frame.to_parquet(parquet, index=False)

    config["paths"]["parquet_file"] = str(parquet)
    config["time"].update({
        "train_start_timestamp": "2024-01-01 00:00:00",
        "test_start_timestamp": "2024-01-05 00:00:00",
        "test_end_exclusive_timestamp": "2024-01-07 00:00:00",
    })
    config_path = root / "ylj.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_ylj_config_contract():
    config = load_ylj_config(Path(__file__).parents[1] / "configs" / "datasets" / "ylj.yaml")
    assert len(config["fields"]["time_series_columns"]) == 22
    assert config["dimensions"]["time_series_input_dim"] == 22
    assert config["dimensions"]["history_points"] == 16
    assert config["dimensions"]["forecast_steps"] == 16
    assert config["power"]["rated_power"] == 468.0
    assert not config["images"]["images_enabled"]
    assert config["time"]["train_start_timestamp"] == "2024-01-01 00:00:00"
    assert config["time"]["test_start_timestamp"] == "2025-01-01 00:00:00"
    assert config["time"]["test_end_exclusive_timestamp"] == "2025-06-01 00:00:00"
    assert config["dimensions"]["weather_placeholder_dim"] == 7
    assert "observe_power" not in config["privileged_teacher"]["future_timeseries_columns"]


def test_ylj_privileged_teacher_receives_future_observed_exogenous(synthetic_ylj):
    dataset = YLJParquetDataset(
        synthetic_ylj, "test", privileged_teacher=True)
    sample = dataset[0]
    history_weather, future_weather = sample[6], sample[7]
    metadata = sample[-1]
    assert history_weather.shape == future_weather.shape == (16, 7)
    assert np.isfinite(history_weather).all() and np.isfinite(future_weather).all()
    assert not np.allclose(future_weather, 0.0)
    assert metadata["privileged_teacher_enabled"]
    assert metadata["teacher_future_timeseries_mask"].shape == (16, 7)
    assert metadata["teacher_history_timeseries_source"].shape == (16, 7)
    assert metadata["teacher_future_timeseries_source"].shape == (16, 7)
    assert metadata["teacher_history_timeseries_source"][0, 0] == 3
    assert not metadata["teacher_history_timeseries_mask"][0, 0]
    assert not metadata["teacher_future_image_mask"].any()


def test_ylj_default_student_contract_hides_privileged_future(synthetic_ylj):
    sample = YLJParquetDataset(synthetic_ylj, "test")[0]
    assert not sample[-1]["privileged_teacher_enabled"]
    assert not sample[-1]["history_images_enabled"]
    assert not sample[-1]["image_mask"].any()
    assert np.count_nonzero(sample[7]) == 0
    assert (sample[-1]["teacher_history_timeseries_source"] == 3).all()
    assert (sample[-1]["teacher_future_timeseries_source"] == 3).all()


def test_privileged_intervention_derangement_has_no_fixed_points():
    from exp.exp_long_term_forecasting_kd import Exp_Long_Term_Forecast

    indices = Exp_Long_Term_Forecast._privileged_derangement(
        8, torch.device("cpu"))
    assert indices is not None
    assert torch.all(indices != torch.arange(8))
    assert Exp_Long_Term_Forecast._privileged_derangement(
        1, torch.device("cpu")) is None


def test_ylj_checkpoint_contract_ignores_artifact_path_changes(
    synthetic_ylj, tmp_path
):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    write_ylj_checkpoint_contract(checkpoint, synthetic_ylj)
    config = load_ylj_config(synthetic_ylj)
    config["paths"]["student_checkpoint"] = "/new/location/checkpoint.pth"
    moved_path_config = tmp_path / "moved-path.yaml"
    moved_path_config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    contract = validate_ylj_checkpoint_contract(checkpoint, moved_path_config)
    assert contract["schema_version"] == 3


def test_ylj_causal_fill_forecast_alignment_and_masks(synthetic_ylj):
    train = YLJParquetDataset(synthetic_ylj, "train")
    validation = YLJParquetDataset(synthetic_ylj, "val")
    dataset = YLJParquetDataset(synthetic_ylj, "test")
    sample = dataset[0]
    seq_x, seq_y, _, y_marks, images, future_images, _, _, _, metadata = sample
    assert seq_x.shape == (16, 22)
    assert seq_y.shape == (16, 1)
    assert images.shape == (16, 3, 64, 64)
    assert future_images.shape == (16, 3, 64, 64)
    assert not metadata["image_mask"].any()
    assert np.isfinite(seq_x).all() and np.isfinite(seq_y).all()
    assert y_marks[0, 4] == 15 and y_marks[-1, 4] == 0

    issue = pd.Timestamp(dataset.sample_times[0])
    assert issue == pd.Timestamp("2024-01-05 00:00:00")
    assert pd.Timestamp(train.sample_times[-1]) + pd.Timedelta(minutes=240) < train.validation_start
    assert pd.Timestamp(validation.sample_times[0]) >= validation.validation_start
    assert pd.Timestamp(validation.sample_times[-1]) + pd.Timedelta(minutes=240) < issue
    assert pd.Timestamp(dataset.sample_times[-1]) + pd.Timedelta(minutes=240) < pd.Timestamp("2024-01-07")
    previous = dataset.frame.loc[issue - pd.Timedelta(minutes=30), "observe_power"]
    assert seq_x[-2, 0] == pytest.approx(max(previous, 0.0) / 468.0)
    assert not metadata["power_history_mask"][-2]
    assert seq_x[-1, 0] == 0.0

    ghi_1day_index = dataset.fields["time_series_columns"].index("GHI_forecast_1day")
    recovered_first = seq_x[0, ghi_1day_index] * dataset.train_stds[ghi_1day_index] + dataset.train_means[ghi_1day_index]
    recovered_second = seq_x[1, ghi_1day_index] * dataset.train_stds[ghi_1day_index] + dataset.train_means[ghi_1day_index]
    assert recovered_first == pytest.approx(321.0)
    assert recovered_second == pytest.approx(123.0)
    assert not metadata["forecast_mask"][0, 0]
    assert not metadata["forecast_mask"][1, 0]
    assert metadata["forecast_source"][0, 0] == 1
    assert metadata["forecast_source"][1, 0] == 2
    assert metadata["forecast_source"][2, 0] == 0
    assert not metadata["target_mask"][2]
    assert metadata["target_mask"][3] and seq_y[3, 0] == 0.0

    oldest_ghi = dataset.fields["time_series_columns"].index("GHI_observe")
    assert seq_x[0, oldest_ghi] == pytest.approx(0.0, abs=1e-6)
    assert not metadata["timeseries_mask"][0, oldest_ghi]
    assert metadata["history_source"][0, 1] == 3
    assert metadata["history_source"][-2, 0] == 1
    assert metadata["timeseries_source"].shape == seq_x.shape
    assert metadata["timeseries_source"].dtype == np.int8
    np.testing.assert_array_equal(
        metadata["timeseries_source"][:, :8], metadata["history_source"]
    )
    np.testing.assert_array_equal(
        metadata["timeseries_source"][:, 8:], metadata["forecast_source"]
    )
    pwat_index = dataset.fields["time_series_columns"].index("PWAT_observe")
    raw_pwat_cm = seq_x[-1, pwat_index] * dataset.train_stds[pwat_index] + dataset.train_means[pwat_index]
    assert raw_pwat_cm == pytest.approx(2.0 + pwat_index * 0.1)

    audit = dataset.candidate_audit
    expected_issues = pd.date_range(
        "2024-01-01", "2024-01-07", freq="15min", inclusive="left"
    )
    assert audit["theoretical_issue_count"] == len(expected_issues)
    assert audit["observed_issue_count"] == len(expected_issues)
    assert audit["continuous_window_count"] == audit["valid_candidate_count"]
    assert audit["target_all_missing_rejected_count"] == 0
    assert audit["selected_sample_count"] == len(dataset)


def test_ylj_dataset_accepts_point_count_overrides(synthetic_ylj):
    dataset = YLJParquetDataset(
        synthetic_ylj,
        "test",
        history_points=48,
        forecast_steps=12,
        power_only=True,
    )

    seq_x, seq_y = dataset[0][:2]
    assert (dataset.seq_len, dataset.pred_len) == (48, 12)
    assert seq_x.shape == (48, 1)
    assert seq_y.shape == (12, 1)


def test_ylj_forecast_train_mean_is_last_fallback(synthetic_ylj):
    dataset = YLJParquetDataset(synthetic_ylj, "test")
    issue = pd.Timestamp(dataset.sample_times[0])
    future = [
        issue + index * dataset.forecast_step
        for index in range(1, dataset.pred_len + 1)
    ]
    name = "TEMP_forecast_1day"
    column = dataset.fields["forecast_window_columns"].index(name)
    fallback = dataset.fields["forecast_fallbacks"][name]
    dataset.frame.loc[future[0], [name, fallback["alternative"]]] = np.nan
    persistence = dataset._persistence[fallback["persistence"]].copy()
    persistence.loc[issue] = np.nan
    dataset._persistence[fallback["persistence"]] = persistence

    values, mask, source = dataset._forecast_window(
        issue, future, return_source=True
    )
    assert not mask[0, column]
    assert source[0, column] == 3
    assert values[0, column] == pytest.approx(0.0)


def test_ylj_candidate_audit_counts_all_missing_targets(
    synthetic_ylj, tmp_path
):
    config = load_ylj_config(synthetic_ylj)
    frame = pd.read_parquet(config["paths"]["parquet_file"])
    issue = pd.Timestamp("2024-01-05 00:00:00")
    target_times = pd.date_range(
        issue + pd.Timedelta(minutes=15), periods=16, freq="15min"
    )
    frame.loc[frame["timestamp"].isin(target_times), "observe_power"] = np.nan
    parquet = tmp_path / "all-missing-target-window.parquet"
    frame.to_parquet(parquet, index=False)
    config["paths"]["parquet_file"] = str(parquet)
    config_path = tmp_path / "all-missing-target-window.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    dataset = YLJParquetDataset(config_path, "test")
    audit = dataset.candidate_audit
    assert audit["target_all_missing_rejected_count"] == 1
    assert (
        audit["continuous_window_count"] - audit["valid_candidate_count"]
        == 1
    )
    assert issue.to_datetime64() not in dataset.sample_times


def test_masked_losses_ignore_invalid_extremes():
    from exp.exp_long_term_forecasting_teacher import Exp_Long_Term_Forecast as TeacherExp
    from exp.exp_long_term_forecasting_kd import Exp_Long_Term_Forecast as KDExp

    target = torch.zeros(2, 16, 1)
    outputs = target.clone()
    outputs[:, 3] = 1e6
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[:, 3] = False

    teacher = TeacherExp.__new__(TeacherExp)
    teacher.loss_type = "MSE"
    teacher.model = nn.Module()
    teacher.device = torch.device("cpu")
    teacher.args = SimpleNamespace(
        ramp_aux_weight=0.0, ramp_direction_aux_weight=0.0, fuse_strategy="no_img")
    assert teacher._compute_loss(outputs, target, target_mask=mask).item() == 0.0

    kd = KDExp.__new__(KDExp)
    assert kd._criterion_loss(outputs, target, nn.MSELoss(), target_mask=mask).item() == 0.0


class _TinyTime(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, values):
        return values[:, -1:, :].repeat(1, self.pred_len, 1) * self.weight


class _TinyImage(nn.Module):
    def __init__(self, args, *unused):
        super().__init__()
        self.pred_len = args.pred_len
        self.frame_encoder = nn.Sequential(nn.Conv2d(3, 1, 1))

    def forward(self, values):
        return torch.zeros(values.shape[0], self.pred_len, 1, device=values.device)


def test_ylj_all_false_image_mask_is_finite(monkeypatch):
    from models import MTS_31, MTS_31F

    monkeypatch.setattr(MTS_31, "encoder", _TinyTime)
    monkeypatch.setattr(MTS_31, "encoder_img", _TinyTime)
    monkeypatch.setattr(MTS_31, "StanfordImageSequenceEncoder", _TinyImage)
    monkeypatch.setattr(MTS_31F, "encoder", _TinyTime)
    monkeypatch.setattr(MTS_31F, "encoder_img", _TinyTime)
    monkeypatch.setattr(MTS_31F, "StanfordImageSequenceEncoder", _TinyImage)
    args = SimpleNamespace(
        data="YLJParquet", seq_len=16, pred_len=16, enc_in=22, c_out=1,
        image_encoder_type="cnn", tors="student", modality_dropout_rate=0.0,
        residual_correction_clip_kw=1.0, expert_gate_bias=-1.0)
    image_args = SimpleNamespace(
        pred_len=16, enc_in=192, c_out=192, image_cnn_dim=64,
        image_temporal_hidden_dim=64, dropout=0.1, history_order="past_first",
        stanford_image_mode="rgb")
    weather_args = SimpleNamespace(pred_len=16, enc_in=1, c_out=1)
    x = torch.rand(2, 16, 22)
    images = torch.zeros(2, 16, 3, 64, 64)
    future_images = torch.zeros_like(images)
    weather = torch.zeros(2, 16, 1)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    teacher = MTS_31F.Model(args, image_args, weather_args)
    student = MTS_31.Model(args, image_args, weather_args)
    teacher_output = teacher(x, images, future_images, weather, weather, "ts_only", image_mask=mask)[0]
    student_output = student(x, images, future_images, weather, weather, "ts_only", image_mask=mask)[0]
    assert teacher_output.shape == student_output.shape == (2, 16, 1)
    assert torch.isfinite(teacher_output).all() and torch.isfinite(student_output).all()


def test_ylj_teacher_uses_future_weather_while_student_does_not(monkeypatch):
    from models import MTS_31, MTS_31F

    monkeypatch.setattr(MTS_31, "encoder", _TinyTime)
    monkeypatch.setattr(MTS_31, "encoder_img", _TinyTime)
    monkeypatch.setattr(MTS_31, "StanfordImageSequenceEncoder", _TinyImage)
    monkeypatch.setattr(MTS_31F, "encoder", _TinyTime)
    monkeypatch.setattr(MTS_31F, "encoder_img", _TinyTime)
    monkeypatch.setattr(MTS_31F, "StanfordImageSequenceEncoder", _TinyImage)
    args = SimpleNamespace(
        data="YLJParquet", seq_len=16, pred_len=16, enc_in=22, c_out=1,
        image_encoder_type="cnn", tors="student", modality_dropout_rate=0.0,
        residual_correction_clip_kw=1.0, expert_gate_bias=-1.0)
    image_args = SimpleNamespace(pred_len=16, enc_in=1, c_out=1)
    weather_args = SimpleNamespace(pred_len=16, enc_in=7, c_out=7)
    x = torch.rand(2, 16, 22)
    images = torch.zeros(2, 16, 3, 8, 8)
    future_images = torch.zeros_like(images)
    weather_h = torch.zeros(2, 16, 7)
    weather_a = torch.zeros(2, 16, 7)
    weather_b = torch.ones(2, 16, 7)
    teacher = MTS_31F.Model(args, image_args, weather_args)
    student = MTS_31.Model(args, image_args, weather_args)
    teacher_a = teacher(
        x, images, future_images, weather_h, weather_a, "no_img")[0]
    teacher_b = teacher(
        x, images, future_images, weather_h, weather_b, "no_img")[0]
    student_a = student(
        x, images, future_images, weather_h, weather_a, "ts_only")[0]
    student_b = student(
        x, images, future_images, weather_h, weather_b, "ts_only")[0]
    assert not torch.equal(teacher_a, teacher_b)
    assert torch.equal(student_a, student_b)


def test_ylj_official_outputs_are_masked_and_in_mw(synthetic_ylj, tmp_path):
    dataset = YLJParquetDataset(synthetic_ylj, "test")
    count = 3
    model_truth = np.stack([dataset[index][1][:, 0] for index in range(count)])
    truth = dataset.inverse_transform_power(model_truth)
    pred = truth + 46.8
    pred[0, 0] = -5.0
    pred[0, -1] = 1e9
    dataset.sample_times = dataset.sample_times[:count]
    dataset.target_masks = dataset.target_masks[:count]
    dataset.current_powers = dataset.current_powers[:count]
    dataset.current_power_valid = dataset.current_power_valid[:count]
    csv_path, metrics_path = build_ylj_outputs(pred, truth, dataset, tmp_path, synthetic_ylj)
    frame = pd.read_csv(csv_path)
    metrics = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))
    assert frame.groupby("issue_time").size().eq(16).all()
    assert list(frame["horizon_minutes"].unique()) == list(range(15, 241, 15))
    assert frame["y_pred"].min() >= 0.0
    assert frame.loc[~frame["target_valid"], "y_true"].isna().all()
    assert metrics["rated_power"] == 468.0
    assert metrics["horizons"]["240_minutes"]["overall"]["NMAE"] == pytest.approx(0.1)


def test_official_output_verifier_accepts_ylj_artifacts(synthetic_ylj, tmp_path):
    from scripts.verify_official_outputs import verify

    config = load_ylj_config(synthetic_ylj)
    config["paths"]["results_root"] = str(tmp_path)
    config_path = tmp_path / "verify_ylj.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    dataset = YLJParquetDataset(config_path, "test")
    count = 2
    dataset.sample_times = dataset.sample_times[:count]
    dataset.target_masks = dataset.target_masks[:count]
    dataset.current_powers = dataset.current_powers[:count]
    dataset.current_power_valid = dataset.current_power_valid[:count]
    truth = np.stack([dataset[index][1][:, 0] for index in range(count)])
    truth = dataset.inverse_transform_power(truth)
    build_ylj_outputs(truth, truth, dataset, tmp_path / "run", config_path)
    verify("ylj", config_path)


def test_ylj_pipeline_resolves_only_teacher_checkpoint(synthetic_ylj, tmp_path):
    from scripts.ylj_pipeline import resolve_teacher

    config = load_ylj_config(synthetic_ylj)
    root = tmp_path / "checkpoints"
    teacher = root / "experiment_YLJ_teacher" / "checkpoint.pth"
    teacher.parent.mkdir(parents=True)
    teacher.write_bytes(b"teacher")
    write_ylj_checkpoint_contract(teacher, synthetic_ylj)
    student = root / "experiment_YLJ_student" / "checkpoint.pth"
    student.parent.mkdir(parents=True)
    student.write_bytes(b"student")
    write_ylj_checkpoint_contract(student, synthetic_ylj)
    config["paths"]["checkpoints_root"] = str(root)
    config["paths"]["teacher_checkpoint"] = str(root / "missing.pth")
    assert resolve_teacher(config, synthetic_ylj) == teacher


def test_ylj_pipeline_exposes_future_observations_only_outside_student_test(
    synthetic_ylj,
):
    from scripts.ylj_pipeline import command

    config = load_ylj_config(synthetic_ylj)
    teacher = command(config, Path(synthetic_ylj), "teacher")
    student = command(config, Path(synthetic_ylj), "student")
    test = command(config, Path(synthetic_ylj), "test")
    assert "--privileged_teacher" in teacher
    assert "--privileged_teacher" in student
    assert "--privileged_teacher" not in test
    assert teacher[teacher.index("--fuse_strategy") + 1] == "no_img"


def test_ylj_pipeline_student_only_has_no_teacher_dependency(synthetic_ylj):
    from scripts.ylj_pipeline import command

    config = load_ylj_config(synthetic_ylj)
    values = command(config, Path(synthetic_ylj), "student-only")
    assert "--student_only" in values
    assert "--privileged_teacher" not in values
    assert "--teacher_model" not in values
    assert "--teacher_path" not in values
    assert values[values.index("--kd_loss_weight_sim") + 1] == "0.0"
    assert values[values.index("--kd_loss_weight_inter") + 1] == "0.0"


def test_ylj_pipeline_resolves_only_student_checkpoint(synthetic_ylj, tmp_path):
    from scripts.ylj_pipeline import resolve_student

    config = load_ylj_config(synthetic_ylj)
    root = tmp_path / "checkpoints"
    teacher = root / "experiment_YLJ_teacher" / "checkpoint.pth"
    teacher.parent.mkdir(parents=True)
    teacher.write_bytes(b"teacher")
    write_ylj_checkpoint_contract(teacher, synthetic_ylj)
    student = root / "experiment_YLJ_student_kd" / "checkpoint.pth"
    student.parent.mkdir(parents=True)
    student.write_bytes(b"student")
    write_ylj_checkpoint_contract(student, synthetic_ylj)
    config["paths"]["checkpoints_root"] = str(root)
    config["paths"]["student_checkpoint"] = str(root / "missing.pth")
    assert resolve_student(config, synthetic_ylj) == student


def test_ylj_pipeline_can_prefer_latest_checkpoint(synthetic_ylj, tmp_path):
    from scripts.ylj_pipeline import resolve_student

    config = load_ylj_config(synthetic_ylj)
    root = tmp_path / "checkpoints"
    configured = root / "configured_student" / "checkpoint.pth"
    newest = root / "new_student" / "checkpoint.pth"
    for checkpoint in (configured, newest):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"weights")
        write_ylj_checkpoint_contract(checkpoint, synthetic_ylj)
    old_time = 1700000000
    new_time = 1700000100
    os.utime(configured, (old_time, old_time))
    os.utime(newest, (new_time, new_time))
    config["paths"].update({
        "checkpoints_root": str(root),
        "student_checkpoint": str(configured),
    })

    assert resolve_student(config, synthetic_ylj) == configured
    assert resolve_student(config, synthetic_ylj, prefer_latest=True) == newest
