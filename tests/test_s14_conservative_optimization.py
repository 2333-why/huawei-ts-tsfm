from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from data_provider import data_factory
from exp.exp_long_term_forecasting_kd import (
    ConservativeCheckpointGuard,
    Exp_Long_Term_Forecast,
)
from data_provider.data_loader_stanford import StanfordSolarForecastDataset
from models import MTS_31


def test_conservative_guard_requires_score_improvement_and_ordinary_guard():
    guard = ConservativeCheckpointGuard(
        input_score=10.0,
        input_ordinary_rmse=2.0,
        patience=3,
        min_improvement=0.01,
        ordinary_guard=True,
        ordinary_relative_tolerance=0.0,
    )

    rejected = guard.consider(epoch=1, score=9.8, ordinary_rmse=2.01)
    assert rejected["accepted"] is False
    assert rejected["reason"] == "ordinary_guard_failed"
    assert guard.best_epoch == 0

    accepted = guard.consider(epoch=2, score=9.8, ordinary_rmse=2.0)
    assert accepted["accepted"] is True
    assert guard.best_epoch == 2
    assert guard.best_score == pytest.approx(9.8)

    unchanged = guard.consider(epoch=3, score=9.8, ordinary_rmse=1.9)
    assert unchanged["accepted"] is False
    assert unchanged["reason"] == "no_validation_improvement"


def test_conservative_guard_patience_can_restart_after_gate_pretraining():
    guard = ConservativeCheckpointGuard(
        input_score=3.0,
        input_ordinary_rmse=2.0,
        patience=2,
    )
    guard.consider(epoch=1, score=3.1, ordinary_rmse=2.0)
    assert guard.counter == 1
    guard.reset_patience()
    assert guard.counter == 0
    assert guard.early_stop is False
    guard.consider(epoch=2, score=3.1, ordinary_rmse=2.0)
    assert guard.early_stop is False


def test_balanced_binary_loss_gives_rare_and_common_classes_equal_mass():
    per_sample = torch.tensor([10.0, 1.0, 1.0, 1.0])
    positive = torch.tensor([True, False, False, False])
    loss = Exp_Long_Term_Forecast._balanced_binary_loss(
        per_sample, positive)
    assert loss.item() == pytest.approx(5.5)


def test_tail_mask_requires_both_large_ramp_and_large_parent_error():
    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    experiment.model = nn.Module()
    experiment.model.last_tail_parent_pred = torch.tensor([
        [[0.0]], [[6.0]], [[0.0]],
    ])
    experiment.args = SimpleNamespace(
        expert_gate_threshold_kw=5.0,
        tail_error_threshold_kw=3.0,
    )
    target = torch.tensor([[[6.0]], [[6.0]], [[4.0]]])
    current = torch.zeros_like(target)

    mask, correction = experiment._tail_supervision_targets(target, current)
    torch.testing.assert_close(mask, torch.tensor([True, False, False]))
    torch.testing.assert_close(correction, torch.tensor([
        [[6.0]], [[0.0]], [[4.0]],
    ]))


def test_tail_up_down_heads_are_supervised_separately_in_capacity_units():
    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    up_raw = torch.tensor([[[0.0]], [[100.0]], [[100.0]]], requires_grad=True)
    down_raw = torch.tensor([[[-100.0]], [[0.0]], [[100.0]]], requires_grad=True)
    experiment.model = nn.Module()
    experiment.model.last_tail_parent_pred = torch.tensor([
        [[0.0]], [[0.0]], [[5.5]],
    ])
    experiment.model.last_stable_pred = experiment.model.last_tail_parent_pred
    experiment.model.last_tail_expert_correction = torch.zeros(3, 1, 1)
    experiment.model.last_tail_up_raw_correction = up_raw
    experiment.model.last_tail_down_raw_correction = down_raw
    experiment.args = SimpleNamespace(
        student_fuse_strategy="causal_solar_token_tail",
        ramp_aux_weight=0.0,
        ramp_direction_aux_weight=0.0,
        residual_aux_weight=0.0,
        stable_correction_aux_weight=0.0,
        expert_gate_aux_weight=0.0,
        event_state_aux_weight=0.0,
        event_residual_aux_weight=1.0,
        expert_gate_threshold_kw=5.0,
        tail_error_threshold_kw=3.0,
        clear_sky_aux_weight=0.0,
        stanford_capacity_kw=10.0,
    )
    target = torch.tensor([[[6.0]], [[-6.0]], [[6.0]]])
    current = torch.zeros_like(target)

    loss = experiment._auxiliary_loss(target, current)
    assert loss.item() == pytest.approx(0.36)
    loss.backward()
    assert up_raw.grad[0].item() < 0
    assert up_raw.grad[1:].abs().sum().item() == 0
    assert down_raw.grad[1].item() > 0
    assert down_raw.grad[[0, 2]].abs().sum().item() == 0


def test_clear_sky_objective_is_capacity_normalized_pinball_loss():
    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    experiment.model = nn.Module()
    experiment.model.last_clear_sky_kw = torch.tensor([[[0.0]], [[10.0]]])
    experiment.args = SimpleNamespace(
        student_fuse_strategy="ts",
        ramp_aux_weight=0.0,
        ramp_direction_aux_weight=0.0,
        residual_aux_weight=0.0,
        stable_correction_aux_weight=0.0,
        expert_gate_aux_weight=0.0,
        event_state_aux_weight=0.0,
        event_residual_aux_weight=0.0,
        clear_sky_aux_weight=1.0,
        clear_sky_quantile=0.9,
        stanford_capacity_kw=10.0,
    )
    target = torch.tensor([[[10.0]], [[0.0]]])
    current = torch.zeros_like(target)

    loss = experiment._auxiliary_loss(target, current)
    assert loss.item() == pytest.approx(0.5)


def test_stable_correction_loss_uses_strict_lt5_mask_and_raw_correction():
    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    correction = torch.tensor([[[2.0]], [[3.0]], [[4.0]]], requires_grad=True)
    experiment.model = nn.Module()
    experiment.model.last_raw_applied_correction = correction
    experiment.args = SimpleNamespace(
        ramp_aux_weight=0.0,
        ramp_direction_aux_weight=0.0,
        residual_aux_weight=0.0,
        stable_correction_aux_weight=1.0,
        stable_ramp_threshold_kw=5.0,
        expert_gate_aux_weight=0.0,
    )
    target = torch.tensor([[[4.9]], [[5.0]], [[8.0]]])
    current = torch.zeros_like(target)

    loss = experiment._auxiliary_loss(target, current)
    assert loss.item() == pytest.approx(4.0)
    loss.backward()
    torch.testing.assert_close(
        correction.grad,
        torch.tensor([[[4.0]], [[0.0]], [[0.0]]]),
    )


def test_event_residual_loss_is_conditioned_on_true_transition_samples():
    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    correction = torch.tensor([[[100.0]], [[3.0]], [[4.0]]], requires_grad=True)
    experiment.model = nn.Module()
    experiment.model.last_event_correction = correction
    experiment.model.last_stable_pred = torch.zeros_like(correction)
    experiment.args = SimpleNamespace(
        student_fuse_strategy="causal_solar_event",
        ramp_aux_weight=0.0,
        ramp_direction_aux_weight=0.0,
        residual_aux_weight=0.0,
        stable_correction_aux_weight=0.0,
        expert_gate_aux_weight=0.0,
        event_state_aux_weight=0.0,
        event_residual_aux_weight=1.0,
        expert_gate_threshold_kw=5.0,
    )
    target = torch.tensor([[[4.9]], [[5.0]], [[8.0]]])
    current = torch.zeros_like(target)

    loss = experiment._auxiliary_loss(target, current)
    assert loss.item() == pytest.approx(10.0)
    loss.backward()
    torch.testing.assert_close(
        correction.grad,
        torch.tensor([[[0.0]], [[-2.0]], [[-4.0]]]),
    )


@pytest.mark.parametrize(
    "ramp, expected_weight",
    [(4.999, 1.0), (5.0, 1.1), (7.999, 1.1), (8.0, 1.25)],
)
def test_conservative_piecewise_ramp_weights(ramp, expected_weight):
    dataset = StanfordSolarForecastDataset.__new__(StanfordSolarForecastDataset)
    dataset.sample_weight_mode = "ramp_piecewise"
    dataset.history_order = "current_first"
    dataset.ramp_weight_5 = 1.1
    dataset.ramp_weight_8 = 1.25
    dataset.sample_weight_clip = 3.0
    dataset.sample_weight_scale = 1.0
    dataset.stanford_capacity_kw = 30.1
    seq_x = np.asarray([[0.0]], dtype=np.float32)
    seq_y = np.asarray([[ramp]], dtype=np.float32)
    weight = dataset._sample_weight(seq_x, seq_y, np.datetime64("2019-01-01"))
    assert float(weight) == pytest.approx(expected_weight)


def test_stanford_validation_loader_is_complete_and_deterministic(monkeypatch):
    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, **kwargs):
            self.rows = list(range(3))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, item):
            return self.rows[item]

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
        split_strategy="day_block",
        num_fold=10,
        fold_index=0,
        forecast_horizon_minutes=15,
        sample_interval_minutes=1,
        history_order="current_first",
        times_trainval_path="unused_train.npy",
        times_test_path="unused_test.npy",
        weather_feature_dim=1,
        stanford_image_mode="gray",
        stanford_capacity_kw=30.1,
        stanford_sample_weight_mode="none",
        stanford_sample_weight_scale=1.0,
        stanford_sample_weight_clip=3.0,
        stanford_ramp_weight_5=1.1,
        stanford_ramp_weight_8=1.25,
        fuse_strategy="no_weather",
        stanford_privileged_teacher=True,
        num_workers=0,
        use_gpu=False,
        gpu_type="cuda",
    )

    _, validation_loader = data_factory.data_provider(args, "val")
    _, training_loader = data_factory.data_provider(args, "train")
    assert isinstance(validation_loader.sampler, SequentialSampler)
    assert validation_loader.drop_last is False
    assert len(list(validation_loader)) == 2
    assert isinstance(training_loader.sampler, RandomSampler)
    assert training_loader.drop_last is True


class _DummyTimeEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, x):
        return x[:, :1, :1]


class _DummyImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, x):
        return torch.zeros(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)


class _DummyDualEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, x):
        appearance = x[:, :1, :1].mean(dim=2, keepdim=True)
        motion = x[:, -1:, -1:].mean(dim=2, keepdim=True)
        energy = torch.ones_like(appearance)
        return appearance, motion, energy


def test_motion_and_softgate_are_identity_children_of_appearance(monkeypatch):
    monkeypatch.setattr(MTS_31, "encoder", _DummyTimeEncoder)
    monkeypatch.setattr(MTS_31, "encoder_img", _DummyImageEncoder)
    monkeypatch.setattr(MTS_31, "StanfordDualAppearanceMotionEncoder", _DummyDualEncoder)
    args = SimpleNamespace(
        data="Stanford",
        image_encoder_type="cnn_dual_motion",
        c_out=1,
        tors="student",
        modality_dropout_rate=0.0,
        residual_correction_clip_kw=10.0,
        expert_gate_bias=-2.0,
        expert_gate_hidden_dim=4,
        stanford_capacity_kw=30.1,
        seq_len=2,
    )
    args_img = SimpleNamespace(c_out=1)
    args_weather = SimpleNamespace(c_out=1)
    model = MTS_31.Model(args, args_img, args_weather).eval()
    with torch.no_grad():
        model.appearance_correction_head.weight.fill_(0.5)
        model.appearance_correction_head.bias.fill_(0.25)
        model.motion_correction_head.weight.zero_()
        model.motion_correction_head.bias.zero_()

    x_ts = torch.tensor([[[3.0], [2.5]], [[4.0], [3.5]]])
    x_img = torch.tensor([[[1.0], [2.0]], [[2.0], [4.0]]])
    x_weather = torch.zeros(2, 1, 1)
    future_img = torch.full((2, 1, 1), 999.0)

    def output(strategy):
        return model(x_ts, x_img, future_img, x_weather, x_weather, strategy)[0]

    appearance = output("causal_residual_appearance")
    torch.testing.assert_close(output("causal_residual_motion"), appearance, rtol=0, atol=0)
    torch.testing.assert_close(output("causal_soft_gate"), appearance, rtol=0, atol=0)


class _TrainingPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dual_image_encoder = True
        self.branch_ts = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.branch_img = nn.Module()
        self.branch_img.appearance = nn.Linear(2, 2)
        self.branch_img.motion = nn.Linear(2, 2)
        self.branch_weather = nn.Linear(2, 2)
        self.appearance_correction_head = nn.Linear(2, 1)
        self.motion_correction_head = nn.Linear(4, 1)
        self.soft_expert_gate = nn.Linear(2, 1)


def test_image_residual_policy_excludes_and_keeps_ts_encoder_in_eval_mode():
    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    experiment.model = _TrainingPolicyModel()
    experiment.args = SimpleNamespace(
        train_image_residual_only=True,
        freeze_ts_permanently=True,
        student_fuse_strategy="causal_soft_gate",
        fuse_strategy="causal_soft_gate",
        learning_rate=1e-3,
    )
    experiment._apply_student_training_policy()

    assert experiment.model.branch_ts.training is False
    assert all(not parameter.requires_grad for parameter in experiment.model.branch_ts.parameters())
    assert all(not parameter.requires_grad for parameter in experiment.model.branch_weather.parameters())
    assert all(parameter.requires_grad for parameter in experiment.model.branch_img.parameters())

    optimizer = experiment._select_optimizer()
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert not any(id(parameter) in optimized_ids for parameter in experiment.model.branch_ts.parameters())

    experiment.model.train()
    experiment._enforce_frozen_ts_mode()
    assert experiment.model.branch_ts.training is False


class _TinyResidualStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.dual_image_encoder = True
        self.branch_ts = nn.Linear(1, 1, bias=False)
        self.branch_img = nn.Linear(1, 1, bias=False)
        self.branch_weather = nn.Linear(1, 1, bias=False)
        self.appearance_correction_head = nn.Linear(1, 1)
        self.motion_correction_head = nn.Linear(2, 1)
        self.soft_expert_gate = nn.Linear(1, 1)
        with torch.no_grad():
            self.branch_ts.weight.fill_(1.0)
            self.appearance_correction_head.weight.zero_()
            self.appearance_correction_head.bias.zero_()
        self.last_raw_applied_correction = None

    def forward(self, x_ts, x_img_h, x_img_f, x_weather_h, x_weather_f, strategy):
        stable = self.branch_ts(x_ts[:, :1, :])
        image_feature = self.branch_img(x_img_h[:, :1, :])
        correction = self.appearance_correction_head(image_feature)
        output = stable + correction
        self.last_raw_applied_correction = correction
        zeros = torch.zeros_like(output)
        if self.training:
            one = torch.tensor(1.0, device=output.device)
            zero = torch.tensor(0.0, device=output.device)
            return output, zeros, zeros, one, zero, output
        return output, zeros, zeros, output


def test_full_conservative_train_rolls_back_to_epoch_zero(tmp_path):
    sample_rows = []
    for current in [1.0, 2.0]:
        x = torch.tensor([[current]])
        y = torch.tensor([[current + 1.0]])
        mark = torch.zeros_like(x)
        image = torch.tensor([[0.5]])
        weather = torch.zeros_like(x)
        sample_rows.append((x, y, mark, mark, image, image, weather, weather, torch.tensor(1.0)))
    loader = DataLoader(sample_rows, batch_size=2, shuffle=False)

    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    experiment.model = _TinyResidualStudent()
    experiment.model_teacher = _TinyResidualStudent()
    experiment.device = torch.device("cpu")
    experiment.loss_type = "MSE"
    experiment.args = SimpleNamespace(
        checkpoints=str(tmp_path),
        conservative_checkpointing=True,
        student_init_path=str(tmp_path / "parent.pth"),
        train_image_residual_only=True,
        freeze_ts_permanently=True,
        freeze_ts_epochs=0,
        student_fuse_strategy="causal_residual_appearance",
        fuse_strategy="causal_residual_appearance",
        teacher_fuse_strategy="ts",
        parent_fuse_strategy="ts",
        learning_rate=0.0,
        patience=1,
        train_epochs=1,
        use_amp=False,
        kd_type="response",
        kd_loss_weight_sim=0.0,
        kd_loss_weight_inter=0.0,
        kd_extreme_response_weight=0.0,
        data="Stanford",
        features="S",
        pred_len=1,
        history_order="current_first",
        ramp_aux_weight=0.0,
        ramp_direction_aux_weight=0.0,
        residual_aux_weight=0.0,
        stable_correction_aux_weight=0.0,
        expert_gate_aux_weight=0.0,
        val_selection_metric="rmse",
        val_extreme_metric_weight=0.25,
        val_min_improvement=0.0,
        val_ordinary_guard=True,
        val_ordinary_relative_tolerance=0.0,
        lradj="type3",
        strict_causal_student=False,
        stanford_privileged_teacher=False,
        seed=2021,
        stanford_sample_weight_mode="none",
        stanford_ramp_weight_5=1.1,
        stanford_ramp_weight_8=1.25,
        stable_ramp_threshold_kw=5.0,
        extra_tag="rollback_fixture",
        model_id="rollback_fixture",
    )
    experiment._get_data = lambda flag: (sample_rows, loader)

    experiment.train("rollback_fixture_setting")

    manifest_path = tmp_path / "rollback_fixture_setting" / "checkpoint_selection.json"
    import json
    manifest = json.loads(manifest_path.read_text())
    assert manifest["accepted"] is False
    assert manifest["selected_source"] == "input_rollback"
    assert manifest["selected"]["epoch"] == 0
    assert manifest["encoder_unchanged"] is True
    assert manifest["encoder_unchanged_during_training"] is True
    assert (tmp_path / "rollback_fixture_setting" / "input_checkpoint.pth").is_file()
