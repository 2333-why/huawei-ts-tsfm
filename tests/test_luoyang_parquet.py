import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from data_provider.data_loader_luoyang import LuoyangParquetDataset, load_luoyang_config
from utils.luoyang_outputs import build_official_outputs
from utils.checkpoint_contract import write_checkpoint_contract, validate_checkpoint_contract
from utils.tools import EarlyStopping


@pytest.fixture(scope="module")
def synthetic_luoyang(tmp_path_factory):
    root = tmp_path_factory.mktemp("luoyang")
    image_root = root / "images"
    image_root.mkdir()
    timestamps = pd.date_range("2026-04-04 22:45:00", "2026-06-12 00:00:00", freq="5min", inclusive="left")
    count = len(timestamps)
    values = np.linspace(1000.0, 30000.0, count, dtype=np.float64)
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "final_power": values,
        "GHI_mean_observe": values / 50.0,
        "GHI_mean_forecast": values / 49.0,
        "msl_forecast": 100000.0 + np.sin(np.arange(count)),
        "t2m_forecast": 280.0 + np.cos(np.arange(count)),
        "u10_forecast": 1.0,
        "v10_forecast": 2.0,
        "u100_forecast": 3.0,
        "v100_forecast": 4.0,
        "asi_path": "[]",
        "asi_path_timestamps": "[]",
    })
    issue = pd.Timestamp("2026-05-11 00:00:00")
    for name, color in (("old.png", 64), ("now.png", 128), ("future.png", 255)):
        Image.fromarray(np.full((64, 64, 3), color, dtype=np.uint8)).save(image_root / name)
    row = frame.index[frame["timestamp"] == issue][0]
    frame.loc[row, "asi_path"] = json.dumps(["old.png", "missing.png", "now.png", "future.png"])
    frame.loc[row, "asi_path_timestamps"] = json.dumps([
        "2026-05-10 22:50:00", "2026-05-10 23:55:00",
        "2026-05-11 00:00:00", "2026-05-11 00:05:00",
    ])
    frame.loc[row - 1, "u10_forecast"] = np.nan
    frame.loc[row - 15, "v10_forecast"] = np.nan
    frame.loc[row + 2, "GHI_mean_observe"] = np.nan
    parquet = root / "synthetic.parquet"
    frame.to_parquet(parquet, index=False)
    config = json.loads((Path(__file__).parents[1] / "configs" / "luoyang_parquet.json").read_text(encoding="utf-8"))
    config["paths"]["parquet_file"] = str(parquet)
    config["paths"]["image_root"] = str(image_root)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def test_config_contract(synthetic_luoyang):
    config = load_luoyang_config(synthetic_luoyang)
    assert config["fields"]["timeseries_columns"] == [
        "final_power", "GHI_mean_observe", "GHI_mean_forecast", "msl_forecast",
        "t2m_forecast", "u10_forecast", "v10_forecast", "u100_forecast", "v100_forecast",
    ]
    assert config["dimensions"]["timeseries_input_dim"] == 9
    assert config["power"]["power_scale"] == config["power"]["max_power"] == config["power"]["rated_power"] == 48629.73
    assert config["output"]["prediction_floor"] == 0.0
    assert config["runtime"]["teacher_batch_size"] >= len(config["runtime"]["gpu_devices"].split(","))


def test_dataset_shapes_boundaries_and_masks(synthetic_luoyang):
    train = LuoyangParquetDataset(synthetic_luoyang, "train")
    val = LuoyangParquetDataset(synthetic_luoyang, "val")
    test = LuoyangParquetDataset(synthetic_luoyang, "test")
    sample = test[0]
    seq_x, seq_y, _, y_marks, images, future_images, _, _, _, metadata = sample
    assert seq_x.shape == (16, 9)
    assert seq_y.shape == (48, 1)
    assert images.shape == (16, 3, 64, 64)
    assert future_images.shape == (48, 3, 64, 64)
    # Missing five-minute slots causally reuse the latest earlier image, as in
    # PVMMoE. The 30-minute cap prevents stale images crossing a long outage.
    assert metadata["image_mask"].sum() == 8
    np.testing.assert_array_equal(images[1], images[7])
    assert metadata["image_mask"][1:8].all()
    assert not metadata["image_mask"][8:15].any()
    assert images[15].mean() == pytest.approx(128.0 / 255.0)
    assert images.max() < 1.0  # the t+5 future image must never be selected
    assert metadata["image_minute_offsets"][1] == pytest.approx(-70.0)
    assert metadata["image_minute_offsets"][7] == pytest.approx(-70.0)
    assert metadata["history_images_enabled"]
    assert metadata["image_age_minutes"][1] == pytest.approx(0.0)
    assert metadata["image_age_minutes"][7] == pytest.approx(30.0)
    assert (metadata["image_age_minutes"][metadata["image_mask"]] >= 0).all()
    assert not metadata["timeseries_mask"][-2, 5]
    assert metadata["timeseries_source"].shape == seq_x.shape
    assert metadata["timeseries_source"].dtype == np.int8
    assert metadata["timeseries_source"][-2, 5] == 1
    assert metadata["timeseries_source"][0, 6] == 3
    assert metadata["timeseries_source"][-1, 0] == 0
    assert metadata["target_mask"].all()
    assert np.isfinite(seq_x).all()
    assert 0.0 <= seq_x[:, 0].min() <= seq_x[:, 0].max() <= 1.0
    assert 0.0 <= seq_y.min() <= seq_y.max() <= 1.0
    raw_target = test.frame.loc[
        pd.Timestamp(test.sample_times[0]) + pd.Timedelta(minutes=5), "final_power"
    ]
    assert test.inverse_transform_power(seq_y[0, 0]) == pytest.approx(raw_target)
    assert metadata["image_minute_offsets"][metadata["image_mask"]].max() <= 0
    assert y_marks[2, 4] == 15
    assert pd.Timestamp(test.sample_times[0]) + pd.Timedelta(minutes=15) == pd.Timestamp("2026-05-11 00:15:00")
    assert pd.Timestamp(test.sample_times[0]) + pd.Timedelta(minutes=240) == pd.Timestamp("2026-05-11 04:00:00")
    assert pd.Timestamp(train.sample_times[-1]) + pd.Timedelta(minutes=240) < train.validation_start
    assert pd.Timestamp(val.sample_times[0]) >= val.validation_start
    assert pd.Timestamp(val.sample_times[0]) - pd.Timedelta(minutes=75) < val.validation_start
    assert pd.Timestamp(val.sample_times[-1]) + pd.Timedelta(minutes=240) < pd.Timestamp("2026-05-11")
    assert pd.Timestamp(test.sample_times[-1]) + pd.Timedelta(minutes=240) < pd.Timestamp("2026-06-12")
    empty_metadata = test[32][-1]
    assert not empty_metadata["image_mask"].any()

    audit = test.candidate_audit
    expected_issues = pd.date_range(
        "2026-04-05", "2026-06-12", freq="5min", inclusive="left"
    )
    assert audit["theoretical_issue_count"] == len(expected_issues)
    assert audit["observed_issue_count"] == len(expected_issues)
    assert audit["continuous_window_count"] == audit["valid_candidate_count"]
    assert audit["target_all_missing_rejected_count"] == 0
    assert audit["selected_sample_count"] == len(test)


def test_luoyang_dataset_accepts_point_count_overrides(synthetic_luoyang):
    dataset = LuoyangParquetDataset(
        synthetic_luoyang,
        "test",
        load_images=False,
        history_points=24,
        forecast_steps=1,
    )

    seq_x, seq_y = dataset[0][:2]
    assert (dataset.seq_len, dataset.pred_len) == (24, 1)
    assert seq_x.shape[0] == 24
    assert seq_y.shape == (1, 1)


def test_history_aligned_ffill_config_contract(synthetic_luoyang):
    config = load_luoyang_config(synthetic_luoyang)
    assert config["images"]["sampling_strategy"] == "history_aligned_ffill"
    assert config["images"]["image_ffill_max_gap_minutes"] == 30.0
    assert config["dimensions"]["image_history_frames"] == config["dimensions"]["history_points"]


def test_luoyang_privileged_teacher_receives_future_exogenous_and_images(
    synthetic_luoyang,
):
    dataset = LuoyangParquetDataset(
        synthetic_luoyang, "test", privileged_teacher=True)
    sample = dataset[0]
    future_images = sample[5]
    history_weather, future_weather = sample[6], sample[7]
    metadata = sample[-1]
    assert history_weather.shape == (16, 1)
    assert future_weather.shape == (48, 1)
    assert np.isfinite(history_weather).all() and np.isfinite(future_weather).all()
    assert not np.allclose(future_weather, 0.0)
    assert metadata["privileged_teacher_enabled"]
    assert metadata["teacher_future_image_mask"][0]
    assert future_images[0].mean() == pytest.approx(1.0)
    assert metadata["teacher_future_timeseries_mask"].shape == (48, 1)
    assert metadata["teacher_history_timeseries_source"].shape == (16, 1)
    assert metadata["teacher_future_timeseries_source"].shape == (48, 1)
    assert metadata["teacher_future_timeseries_source"][1, 0] == 3
    assert not metadata["teacher_future_timeseries_mask"][1, 0]


def test_luoyang_default_student_contract_hides_privileged_future(
    synthetic_luoyang,
):
    sample = LuoyangParquetDataset(synthetic_luoyang, "test")[0]
    assert not sample[-1]["privileged_teacher_enabled"]
    assert not sample[-1]["teacher_future_image_mask"].any()
    assert np.count_nonzero(sample[5]) == 0
    assert np.count_nonzero(sample[7]) == 0
    assert (sample[-1]["teacher_history_timeseries_source"] == 3).all()
    assert (sample[-1]["teacher_future_timeseries_source"] == 3).all()


def test_official_export_and_metrics(synthetic_luoyang, tmp_path):
    dataset = LuoyangParquetDataset(synthetic_luoyang, "test", load_images=False)
    model_truth = np.stack([dataset[index][1][:, 0] for index in range(3)], axis=0)
    truth = dataset.inverse_transform_power(model_truth)
    pred = truth + 100.0
    dataset.sample_times = dataset.sample_times[:3]
    dataset.current_powers = dataset.current_powers[:3]
    csv_path, metrics_path = build_official_outputs(pred, truth, dataset, tmp_path, synthetic_luoyang)
    frame = pd.read_csv(csv_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert list(frame.columns) == ["issue_time", "target_time", "horizon_minutes", "y_true", "y_pred"]
    assert frame.groupby("issue_time").size().eq(48).all()
    assert frame["horizon_minutes"].min() == 5 and frame["horizon_minutes"].max() == 240
    assert metrics["rated_power"] == 48629.73
    assert metrics["forecast_steps"] == 48
    assert metrics["horizons"]["15_minutes"]["overall"]["RMSE"] == pytest.approx(100.0, abs=1e-3)
    assert metrics["horizons"]["240_minutes"]["overall"]["NMAE"] == pytest.approx(100.0 / 48629.73, abs=1e-7)


def test_official_export_clips_predictions_to_configured_power_floor(
    synthetic_luoyang, tmp_path,
):
    dataset = LuoyangParquetDataset(synthetic_luoyang, "test", load_images=False)
    count = 2
    model_truth = np.stack([dataset[index][1][:, 0] for index in range(count)])
    truth = dataset.inverse_transform_power(model_truth)
    pred = truth.copy()
    pred[0, 2] = -123.0
    dataset.sample_times = dataset.sample_times[:count]
    dataset.current_powers = dataset.current_powers[:count]

    csv_path, metrics_path = build_official_outputs(
        pred, truth, dataset, tmp_path, synthetic_luoyang)
    frame = pd.read_csv(csv_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert frame["y_pred"].min() >= 0.0
    expected_rmse = truth[0, 2] / np.sqrt(count)
    assert metrics["horizons"]["15_minutes"]["overall"]["count"] == count
    assert metrics["horizons"]["15_minutes"]["overall"]["RMSE"] == pytest.approx(
        expected_rmse)


def test_official_output_verifier_accepts_luoyang_artifacts(
    synthetic_luoyang, tmp_path,
):
    from scripts.verify_official_outputs import verify

    config = json.loads(Path(synthetic_luoyang).read_text(encoding="utf-8"))
    config["paths"]["results_root"] = str(tmp_path)
    config_path = tmp_path / "verify_luoyang.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    dataset = LuoyangParquetDataset(config_path, "test", load_images=False)
    dataset.sample_times = dataset.sample_times[:2]
    dataset.current_powers = dataset.current_powers[:2]
    truth = np.stack([dataset[index][1][:, 0] for index in range(2)])
    truth = dataset.inverse_transform_power(truth)
    predictions, _ = build_official_outputs(
        truth, truth, dataset, tmp_path / "run", config_path)
    verify("luoyang", config_path)
    corrupted = pd.read_csv(predictions)
    corrupted.loc[0, "y_pred"] = np.nan
    corrupted.to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="NaN or infinity"):
        verify("luoyang", config_path)
    corrupted.loc[0, "y_pred"] = -1.0
    corrupted.to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="below prediction_floor"):
        verify("luoyang", config_path)


def test_checkpoint_contract_rejects_legacy_checkpoint(synthetic_luoyang, tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    with pytest.raises(FileNotFoundError):
        validate_checkpoint_contract(checkpoint, synthetic_luoyang)
    sidecar = write_checkpoint_contract(checkpoint, synthetic_luoyang)
    assert sidecar.is_file()
    assert validate_checkpoint_contract(checkpoint, synthetic_luoyang)["pred_len"] == 48


def test_early_stopping_atomically_writes_checkpoint_metadata(
    synthetic_luoyang, tmp_path
):
    model = nn.Linear(2, 1)
    stopper = EarlyStopping(
        patience=2,
        on_save=lambda checkpoint: write_checkpoint_contract(
            checkpoint, synthetic_luoyang
        ),
    )
    stopper(1.0, model, str(tmp_path))
    checkpoint = tmp_path / "checkpoint.pth"
    sidecar = tmp_path / "checkpoint.pth.metadata.json"
    assert checkpoint.is_file()
    assert sidecar.is_file()
    assert not (tmp_path / "checkpoint.pth.tmp").exists()
    assert not (tmp_path / "checkpoint.pth.metadata.json.tmp").exists()
    assert validate_checkpoint_contract(checkpoint, synthetic_luoyang)["schema_version"] == 3


def test_checkpoint_contract_ignores_artifact_paths_but_rejects_semantics(
    synthetic_luoyang, tmp_path
):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    write_checkpoint_contract(checkpoint, synthetic_luoyang)
    config = json.loads(Path(synthetic_luoyang).read_text(encoding="utf-8"))

    config["paths"]["student_checkpoint"] = "/new/location/checkpoint.pth"
    config["model"]["kd_intervention_weight"] = 0.25
    config["output"]["prediction_floor"] = 5.0
    moved_path_config = tmp_path / "moved-path.json"
    moved_path_config.write_text(json.dumps(config), encoding="utf-8")
    assert validate_checkpoint_contract(checkpoint, moved_path_config)["pred_len"] == 48

    config["dimensions"]["forecast_steps"] = 24
    config["time"]["forecast_horizon_minutes"] = 120
    incompatible_config = tmp_path / "incompatible.json"
    incompatible_config.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_checkpoint_contract(checkpoint, incompatible_config)


def test_pipeline_resolves_dynamic_teacher_checkpoint(synthetic_luoyang, tmp_path):
    from scripts.luoyang_pipeline import resolve_teacher_checkpoint

    config = load_luoyang_config(synthetic_luoyang)
    checkpoint_root = tmp_path / "checkpoints"
    teacher = checkpoint_root / "long_term_forecast_Luoyang_teacher_16_48" / "checkpoint.pth"
    teacher.parent.mkdir(parents=True)
    teacher.write_bytes(b"teacher")
    write_checkpoint_contract(teacher, synthetic_luoyang)

    student = checkpoint_root / "long_term_forecast_Luoyang_student_16_48" / "checkpoint.pth"
    student.parent.mkdir(parents=True)
    student.write_bytes(b"student")
    write_checkpoint_contract(student, synthetic_luoyang)

    config["paths"]["checkpoints_root"] = str(checkpoint_root)
    config["paths"]["teacher_checkpoint"] = str(tmp_path / "missing" / "checkpoint.pth")
    assert resolve_teacher_checkpoint(config, synthetic_luoyang) == teacher


def test_luoyang_pipeline_exposes_privileged_data_only_outside_student_test(
    synthetic_luoyang,
):
    from scripts.luoyang_pipeline import command

    config = load_luoyang_config(synthetic_luoyang)
    teacher = command(config, Path(synthetic_luoyang), "teacher")
    student = command(config, Path(synthetic_luoyang), "student")
    test = command(config, Path(synthetic_luoyang), "test")
    assert "--privileged_teacher" in teacher
    assert "--privileged_teacher" in student
    assert "--privileged_teacher" not in test
    assert teacher[teacher.index("--fuse_strategy") + 1] == "full"


def test_luoyang_pipeline_student_only_has_no_teacher_dependency(
    synthetic_luoyang,
):
    from scripts.luoyang_pipeline import command

    config = load_luoyang_config(synthetic_luoyang)
    values = command(config, Path(synthetic_luoyang), "student-only")
    assert "--student_only" in values
    assert "--privileged_teacher" not in values
    assert "--teacher_model" not in values
    assert "--teacher_path" not in values
    assert values[values.index("--kd_loss_weight_sim") + 1] == "0.0"
    assert values[values.index("--kd_loss_weight_inter") + 1] == "0.0"


def test_luoyang_tuned_config_only_changes_training_hyperparameters():
    from scripts.luoyang_pipeline import command

    root = Path(__file__).parents[1]
    baseline_path = root / "configs" / "luoyang_parquet.json"
    tuned_path = root / "configs" / "luoyang_tuned.json"
    baseline = load_luoyang_config(baseline_path)
    tuned = load_luoyang_config(tuned_path)
    baseline_teacher = command(baseline, baseline_path, "teacher")
    baseline_student = command(baseline, baseline_path, "student")
    tuned_teacher = command(tuned, tuned_path, "teacher")
    tuned_student = command(tuned, tuned_path, "student")

    def option(values, name):
        return values[values.index(name) + 1]

    assert option(baseline_teacher, "--learning_rate") == "0.0001"
    assert option(baseline_student, "--learning_rate") == "0.0001"
    assert option(baseline_student, "--val_selection_metric") == "loss"
    assert option(tuned_teacher, "--learning_rate") == "0.0001"
    assert option(tuned_student, "--learning_rate") == "5e-05"
    assert option(tuned_student, "--kd_loss_weight_sim") == "0.0005"
    assert option(tuned_student, "--kd_loss_weight_inter") == "0.02"
    assert option(tuned_student, "--modality_dropout_rate") == "0.05"
    assert option(tuned_student, "--val_selection_metric") == "rmse"
    assert option(tuned_student, "--student_fuse_strategy") == "causal_img"
    assert option(tuned_student, "--teacher_fuse_strategy") == "full"
    assert option(tuned_student, "--kd_type") == "inter_causal"


def test_pipeline_resolves_dynamic_student_checkpoint(synthetic_luoyang, tmp_path):
    from scripts.luoyang_pipeline import resolve_student_checkpoint

    config = load_luoyang_config(synthetic_luoyang)
    checkpoint_root = tmp_path / "checkpoints"
    teacher = checkpoint_root / "teacher_run" / "checkpoint.pth"
    teacher.parent.mkdir(parents=True)
    teacher.write_bytes(b"teacher")
    write_checkpoint_contract(teacher, synthetic_luoyang)
    student = checkpoint_root / "student_kd_run" / "checkpoint.pth"
    student.parent.mkdir(parents=True)
    student.write_bytes(b"student")
    write_checkpoint_contract(student, synthetic_luoyang)
    config["paths"]["checkpoints_root"] = str(checkpoint_root)
    config["paths"]["student_checkpoint"] = str(tmp_path / "missing" / "checkpoint.pth")
    assert resolve_student_checkpoint(config, synthetic_luoyang) == student


def test_pipeline_can_prefer_latest_checkpoint_for_full_rerun(
    synthetic_luoyang, tmp_path
):
    from scripts.luoyang_pipeline import (
        resolve_student_checkpoint,
        resolve_teacher_checkpoint,
    )

    config = load_luoyang_config(synthetic_luoyang)
    root = tmp_path / "checkpoints"
    old_teacher = root / "configured_teacher" / "checkpoint.pth"
    new_teacher = root / "new_teacher" / "checkpoint.pth"
    old_student = root / "configured_student" / "checkpoint.pth"
    new_student = root / "new_student" / "checkpoint.pth"
    for checkpoint in (old_teacher, new_teacher, old_student, new_student):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"weights")
        write_checkpoint_contract(checkpoint, synthetic_luoyang)
    old_time = 1700000000
    new_time = 1700000100
    os.utime(old_teacher, (old_time, old_time))
    os.utime(old_student, (old_time, old_time))
    os.utime(new_teacher, (new_time, new_time))
    os.utime(new_student, (new_time, new_time))
    config["paths"].update({
        "checkpoints_root": str(root),
        "teacher_checkpoint": str(old_teacher),
        "student_checkpoint": str(old_student),
    })

    assert resolve_teacher_checkpoint(config, synthetic_luoyang) == old_teacher
    assert resolve_student_checkpoint(config, synthetic_luoyang) == old_student
    assert resolve_teacher_checkpoint(
        config, synthetic_luoyang, prefer_latest=True
    ) == new_teacher
    assert resolve_student_checkpoint(
        config, synthetic_luoyang, prefer_latest=True
    ) == new_student


def test_luoyang_metadata_does_not_enable_phase_validation():
    from exp.exp_long_term_forecasting_kd import Exp_Long_Term_Forecast

    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    experiment.args = SimpleNamespace(student_fuse_strategy="causal_img")
    metadata = {"image_mask": torch.ones(2, 16, dtype=torch.bool)}
    assert not experiment._uses_phase_validation_targets(metadata)

    experiment.args.student_fuse_strategy = "causal_phase_field_tail"
    assert experiment._uses_phase_validation_targets(metadata)


class _TinyEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.projection = nn.Linear(args.enc_in, args.c_out)

    def forward(self, value):
        return self.projection(value[:, -1:, :]).repeat(1, self.pred_len, 1)


class _TinyTimeEncoder(nn.Module):
    """Match FACTS Crossformer: its output width is enc_in, not c_out."""

    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, value):
        return value[:, -1:, :].repeat(1, self.pred_len, 1) * self.scale


class _TinyImageEncoder(nn.Module):
    def __init__(self, args, *unused):
        super().__init__()
        self.pred_len = args.pred_len
        self.scale = nn.Parameter(torch.ones(1))
        self.frame_encoder = nn.Sequential(nn.Conv2d(3, 1, kernel_size=1))

    def forward(self, value):
        return value.flatten(1).mean(1)[:, None, None].repeat(1, self.pred_len, 1) * self.scale

    def encode_spatial_sequence(self, value, all_steps=False):
        return value.mean(dim=2, keepdim=True)


def _model_args():
    return SimpleNamespace(
        data="LuoyangParquet", seq_len=16, pred_len=48, enc_in=9, c_out=1,
        image_encoder_type="cnn", tors="student", modality_dropout_rate=0.0,
        residual_correction_clip_kw=10000.0, expert_gate_bias=-1.0,
        expert_gate_hidden_dim=8, stanford_capacity_kw=48629.73,
    )


def test_teacher_student_48_step_forward_and_backward(monkeypatch):
    from models import MTS_31, MTS_31F
    monkeypatch.setattr(MTS_31, "encoder", _TinyTimeEncoder)
    monkeypatch.setattr(MTS_31, "encoder_img", _TinyEncoder)
    monkeypatch.setattr(MTS_31, "StanfordImageSequenceEncoder", _TinyImageEncoder)
    monkeypatch.setattr(MTS_31F, "encoder", _TinyTimeEncoder)
    monkeypatch.setattr(MTS_31F, "encoder_img", _TinyEncoder)
    monkeypatch.setattr(MTS_31F, "StanfordImageSequenceEncoder", _TinyImageEncoder)
    args = _model_args()
    image_args = SimpleNamespace(pred_len=48, enc_in=1, c_out=1)
    weather_args = SimpleNamespace(pred_len=48, enc_in=1, c_out=1)
    student = MTS_31.Model(args, image_args, weather_args)
    teacher = MTS_31F.Model(args, image_args, weather_args)
    x = torch.randn(2, 16, 9)
    images = torch.randn(2, 16, 3, 8, 8)
    future_images = torch.zeros(2, 48, 3, 8, 8)
    weather_h = torch.zeros(2, 16, 1)
    weather_f = torch.zeros(2, 48, 1)
    mask = torch.tensor([[True] * 16, [False] * 16])
    student_output = student(x, images, future_images, weather_h, weather_f, "causal_img", image_mask=mask)[0]
    teacher_output = teacher(x, images, future_images, weather_h, weather_f, "no_weather", image_mask=mask)[0]
    target = torch.randn(2, 48, 1)
    loss = nn.MSELoss()(student_output, target) + nn.MSELoss()(teacher_output, target)
    loss.backward()
    assert student_output.shape == teacher_output.shape == target.shape == (2, 48, 1)
    assert student.ts_projection.in_features == teacher.ts_projection.in_features == 9
    assert student.ts_projection.out_features == teacher.ts_projection.out_features == 1
    assert any(parameter.grad is not None for parameter in student.parameters())
