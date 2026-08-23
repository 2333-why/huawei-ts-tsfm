import inspect
from types import SimpleNamespace

import torch
import torch.nn as nn

from exp.exp_long_term_forecasting_kd import Exp_Long_Term_Forecast
from models import MTS_31
from models.image_sequence_encoder import (
    StanfordImageSequenceEncoder,
    StanfordSolarAdvectionEncoder,
    StanfordSolarTokenEncoder,
)


def legacy_config(history_order="current_first", image_mode="gray"):
    return SimpleNamespace(
        c_out=8,
        pred_len=1,
        history_order=history_order,
        stanford_image_mode=image_mode,
        image_cnn_dim=8,
        image_temporal_hidden_dim=8,
        dropout=0.0,
    )


def advection_config(history_order="current_first", input_dim=8):
    return SimpleNamespace(
        image_cnn_dim=input_dim,
        advection_output_dim=64,
        advection_motion_dim=input_dim,
        advection_correlation_temperature=0.05,
        advection_recent_flow_steps=5,
        history_order=history_order,
        forecast_horizon_minutes=15,
        sample_interval_minutes=1,
        stanford_capacity_kw=30.1,
    )


def solar_token_config(history_order="current_first"):
    return SimpleNamespace(
        history_order=history_order,
        stanford_capacity_kw=30.1,
        solar_token_dim=32,
        solar_token_conv_dim=16,
        solar_token_hypotheses=3,
        solar_token_heads=4,
    )


def chronological_marks(batch, steps):
    minute = torch.arange(steps, dtype=torch.float32)
    marks = torch.stack([
        torch.full_like(minute, 6.0),
        torch.full_like(minute, 24.0),
        torch.full_like(minute, 5.0),
        torch.full_like(minute, 10.0),
        minute,
    ], dim=-1)
    return marks.unsqueeze(0).repeat(batch, 1, 1)


def target_marks(batch, hour=10.0, minute=30.0):
    return torch.tensor(
        [6.0, 24.0, 5.0, hour, minute], dtype=torch.float32
    ).view(1, 1, 5).repeat(batch, 1, 1)


def test_legacy_spatial_sequence_is_compatible_and_chronological():
    torch.manual_seed(1)
    current_first = StanfordImageSequenceEncoder(
        legacy_config("current_first"), "cnn_latest").eval()
    past_first = StanfordImageSequenceEncoder(
        legacy_config("past_first"), "cnn_latest").eval()
    past_first.load_state_dict(current_first.state_dict(), strict=True)

    chronological = torch.randn(2, 16, 64, 64)
    current_first_input = torch.flip(chronological, dims=[1])
    spatial_current_first = current_first.encode_spatial_sequence(
        current_first_input, all_steps=True)
    spatial_past_first = past_first.encode_spatial_sequence(
        chronological, all_steps=True)

    assert spatial_current_first.shape == (2, 16, 8, 8, 8)
    torch.testing.assert_close(
        spatial_current_first, spatial_past_first, rtol=0, atol=0)

    latest_spatial = current_first.encode_spatial_sequence(
        current_first_input, all_steps=False)
    pooled = current_first.frame_encoder[-1](
        latest_spatial.reshape(-1, 8, 8, 8)).flatten(1).unsqueeze(1)
    temporal, _ = current_first.temporal_encoder(pooled)
    expected = current_first.head(temporal[:, -1]).unsqueeze(1)
    torch.testing.assert_close(
        current_first(current_first_input), expected, rtol=0, atol=0)

    legacy_keys = set(current_first.state_dict())
    assert not any("spatial" in key or "advection" in key for key in legacy_keys)


def test_advection_output_shapes_and_causal_interface():
    torch.manual_seed(2)
    model = StanfordSolarAdvectionEncoder(advection_config()).eval()
    spatial = torch.randn(3, 6, 8, 8, 8)
    marks = torch.flip(chronological_marks(3, 6), dims=[1])
    pv = torch.flip(torch.linspace(2.0, 5.0, 6).view(1, 6, 1), dims=[1])
    pv = pv.repeat(3, 1, 1)
    output = model(spatial, marks, target_marks(3), pv)

    assert model.uses_future_images is False
    assert list(inspect.signature(model.forward).parameters) == [
        "spatial_sequence", "x_mark_h", "y_mark", "pv_history"
    ]
    assert output["event_feature"].shape == (3, 1, 64)
    assert output["future_spatial_map"].shape == (3, 8, 8, 8)
    assert output["pairwise_flow"].shape == (3, 5, 2, 8, 8)
    assert output["pairwise_confidence"].shape == (3, 5, 1, 8, 8)
    assert output["advection_velocity"].shape == (3, 2)
    assert output["advection_displacement"].shape == (3, 2)
    assert output["advection_confidence"].shape == (3, 1)
    assert output["advection_grid"].shape == (3, 8, 8, 2)
    assert output["advection_validity"].shape == (3, 1, 8, 8)
    assert output["history_attention"].shape == (3, 6, 1, 8, 8)
    assert output["future_attention"].shape == (3, 1, 8, 8)
    torch.testing.assert_close(
        output["history_attention"].sum(dim=(-1, -2)),
        torch.ones(3, 6, 1),
    )
    torch.testing.assert_close(
        output["future_attention"].sum(dim=(-1, -2)),
        torch.ones(3, 1),
    )
    assert output["pairwise_confidence"].min() >= 0
    assert output["pairwise_confidence"].max() <= 1


def test_local_correlation_recovers_rightward_motion():
    model = StanfordSolarAdvectionEncoder(
        advection_config(history_order="past_first", input_dim=1)).eval()
    with torch.no_grad():
        model.motion_projection.weight.fill_(1.0)

    spatial = torch.zeros(1, 2, 1, 8, 8)
    spatial[0, 0, 0, 4, 3] = 1.0
    spatial[0, 1, 0, 4, 4] = 1.0
    flow, confidence = model._local_correlation_flow(spatial)

    assert flow[0, 0, 0, 4, 3] > 0.95
    assert flow[0, 0, 1, 4, 3].abs() < 0.05
    assert confidence[0, 0, 0, 4, 3] > 0.9


def test_current_first_auxiliary_order_matches_past_first():
    torch.manual_seed(3)
    current_first = StanfordSolarAdvectionEncoder(
        advection_config("current_first")).eval()
    past_first = StanfordSolarAdvectionEncoder(
        advection_config("past_first")).eval()
    past_first.load_state_dict(current_first.state_dict(), strict=True)

    spatial = torch.randn(2, 6, 8, 8, 8)
    marks = chronological_marks(2, 6)
    pv = torch.linspace(1.0, 8.0, 6).view(1, 6, 1).repeat(2, 1, 1)
    y_mark = target_marks(2)
    output_current_first = current_first(
        spatial, torch.flip(marks, [1]), y_mark, torch.flip(pv, [1]))
    output_past_first = past_first(spatial, marks, y_mark, pv)

    for key in output_current_first:
        torch.testing.assert_close(
            output_current_first[key], output_past_first[key], rtol=0, atol=0)


def test_solar_advection_is_time_and_space_sensitive():
    torch.manual_seed(4)
    model = StanfordSolarAdvectionEncoder(
        advection_config(history_order="past_first")).eval()
    spatial_left = torch.zeros(1, 6, 8, 8, 8)
    spatial_left[:, :, :, 3, 2] = torch.linspace(0.5, 1.2, 8)
    spatial_right = torch.zeros_like(spatial_left)
    spatial_right[:, :, :, 3, 5] = spatial_left[:, :, :, 3, 2]
    marks = chronological_marks(1, 6)
    pv = torch.linspace(2.0, 4.0, 6).view(1, 6, 1)

    morning = model(spatial_left, marks, target_marks(1, hour=9.0), pv)
    afternoon = model(spatial_left, marks, target_marks(1, hour=15.0), pv)
    shifted = model(spatial_right, marks, target_marks(1, hour=9.0), pv)

    assert not torch.allclose(
        morning["event_feature"], afternoon["event_feature"], atol=1e-6)
    assert not torch.allclose(
        morning["future_attention"], afternoon["future_attention"], atol=1e-6)
    assert not torch.allclose(
        morning["event_feature"], shifted["event_feature"], atol=1e-6)
    assert not torch.allclose(
        morning["history_attention"], shifted["history_attention"], atol=1e-6)


def test_solar_advection_has_end_to_end_gradients():
    torch.manual_seed(5)
    model = StanfordSolarAdvectionEncoder(
        advection_config(history_order="past_first"))
    spatial = torch.randn(2, 6, 8, 8, 8, requires_grad=True)
    marks = chronological_marks(2, 6)
    y_mark = target_marks(2).requires_grad_()
    pv = torch.linspace(2.0, 5.0, 6).view(1, 6, 1).repeat(2, 1, 1)
    output = model(spatial, marks, y_mark, pv)
    loss = (
        output["event_feature"].square().mean()
        + output["future_spatial_map"].square().mean()
        + output["pairwise_flow"].square().mean()
    )
    loss.backward()

    assert spatial.grad is not None
    assert spatial.grad.abs().sum() > 0
    assert y_mark.grad is not None
    assert y_mark.grad.abs().sum() > 0
    assert model.motion_projection.weight.grad is not None
    assert model.motion_projection.weight.grad.abs().sum() > 0
    assert model.coordinate_projection.weight.grad is not None
    assert model.coordinate_projection.weight.grad.abs().sum() > 0


def test_solar_token_encoder_is_causal_and_preserves_local_trajectories():
    torch.manual_seed(6)
    model = StanfordSolarTokenEncoder(solar_token_config()).eval()
    images = torch.rand(2, 16, 3, 64, 64)
    marks = torch.flip(chronological_marks(2, 16), dims=[1])
    target = target_marks(2)
    pv = torch.flip(
        torch.linspace(3.0, 12.0, 16).view(1, 16, 1).repeat(2, 1, 1),
        dims=[1],
    )
    output = model(images, marks, target, pv)

    assert model.uses_future_images is False
    assert list(inspect.signature(model.forward).parameters) == [
        "images", "x_mark_h", "y_mark", "pv_history"
    ]
    assert output["event_feature"].shape == (2, 1, 32)
    assert output["transition_feature"].shape == (2, 1, 32)
    assert output["future_solar_token"].shape == (2, 32)
    assert output["solar_location_logits"].shape == (2, 64)
    assert output["solar_location"].shape == (2, 64)
    assert output["clear_sky_kw"].shape == (2, 1, 1)
    assert output["trajectory_hypotheses"].shape == (2, 3, 32)
    assert output["trajectory_attention"].shape == (2, 3, 16 * 16 * 16)
    torch.testing.assert_close(
        output["solar_location"].sum(dim=-1), torch.ones(2))
    assert output["clear_sky_kw"].min() >= 0
    assert output["clear_sky_kw"].max() <= 30.1

    shifted = images.clone()
    shifted[:, :, :, 12:28, 8:24] += 1.0
    shifted_output = model(shifted, marks, target, pv)
    assert not torch.allclose(
        output["future_solar_token"], shifted_output["future_solar_token"])


def test_solar_token_encoder_order_and_gradients():
    torch.manual_seed(7)
    current_first = StanfordSolarTokenEncoder(
        solar_token_config("current_first"))
    past_first = StanfordSolarTokenEncoder(
        solar_token_config("past_first"))
    past_first.load_state_dict(current_first.state_dict(), strict=True)

    images = torch.rand(2, 16, 3, 64, 64, requires_grad=True)
    marks = chronological_marks(2, 16)
    pv = torch.linspace(2.0, 9.0, 16).view(1, 16, 1).repeat(2, 1, 1)
    target = target_marks(2)
    current_output = current_first(
        torch.flip(images, [1]), torch.flip(marks, [1]), target,
        torch.flip(pv, [1]))
    past_output = past_first(images, marks, target, pv)
    for key in [
        "event_feature", "future_solar_token", "solar_location_logits",
        "transition_feature",
        "clear_sky_kw", "trajectory_hypotheses", "trajectory_attention",
        "roi_sequence",
    ]:
        torch.testing.assert_close(
            current_output[key], past_output[key], rtol=0, atol=0)

    loss = (
        current_output["event_feature"].square().mean()
        + current_output["future_solar_token"].square().mean()
        + current_output["clear_sky_kw"].square().mean()
    )
    loss.backward()
    assert images.grad is not None
    assert images.grad.abs().sum() > 0
    assert current_first.hypothesis_queries.grad is not None
    assert current_first.hypothesis_queries.grad.abs().sum() > 0


class _TinyTimeEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

    def forward(self, values):
        return values[:, :1, :1]


class _TinyWeatherEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

    def forward(self, values):
        return torch.zeros(
            values.shape[0], 1, 1, device=values.device, dtype=values.dtype)


def test_solar_event_integration_preserves_appearance_and_training_boundary(monkeypatch):
    monkeypatch.setattr(MTS_31, "encoder", _TinyTimeEncoder)
    monkeypatch.setattr(MTS_31, "encoder_img", _TinyWeatherEncoder)
    args = SimpleNamespace(
        data="Stanford",
        image_encoder_type="cnn_solar_advection",
        c_out=1,
        tors="student",
        modality_dropout_rate=0.0,
        residual_correction_clip_kw=10.0,
        expert_gate_bias=-2.0,
        expert_gate_hidden_dim=8,
        stanford_capacity_kw=30.1,
        history_order="current_first",
        event_routing_mode="hard",
        seq_len=16,
    )
    args_img = SimpleNamespace(
        c_out=64,
        pred_len=1,
        history_order="current_first",
        stanford_image_mode="rgb",
        image_cnn_dim=64,
        image_temporal_hidden_dim=64,
        dropout=0.0,
        forecast_horizon_minutes=15,
        sample_interval_minutes=1,
        stanford_capacity_kw=30.1,
    )
    args_weather = SimpleNamespace(c_out=1)
    model = MTS_31.Model(args, args_img, args_weather).eval()

    x_ts = torch.linspace(8.0, 4.0, 16).view(1, 16, 1)
    images = torch.rand(1, 16, 3, 64, 64)
    future_a = torch.zeros(1, 1, 3, 64, 64)
    future_b = torch.full_like(future_a, 1000.0)
    weather = torch.zeros(1, 16, 1)
    marks = torch.flip(chronological_marks(1, 16), dims=[1])
    target = target_marks(1)

    with torch.no_grad():
        first = model(
            x_ts, images, future_a, weather, weather,
            "causal_solar_event", marks, target)[0]
        second = model(
            x_ts, images, future_b, weather, weather,
            "causal_solar_event", marks, target)[0]
    torch.testing.assert_close(first, x_ts[:, :1], rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert model.last_spatial_forecast.shape == (1, 64, 8, 8)
    assert model.last_event_state_logits.shape == (1, 1, 3)

    with torch.no_grad():
        model.event_up_correction_head.bias.fill_(5.0)
        stable_routed = model(
            x_ts, images, future_a, weather, weather,
            "causal_solar_event", marks, target)[0]
        model.event_state_head.bias.copy_(torch.tensor([-2.0, -2.0, 2.0]))
        event_routed = model(
            x_ts, images, future_a, weather, weather,
            "causal_solar_event", marks, target)[0]
    torch.testing.assert_close(stable_routed, first, rtol=0, atol=0)
    torch.testing.assert_close(event_routed, first + 5.0, rtol=0, atol=0)

    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    experiment.model = model
    experiment.args = SimpleNamespace(
        train_image_residual_only=True,
        freeze_ts_permanently=True,
        student_fuse_strategy="causal_solar_event",
        fuse_strategy="causal_solar_event",
        learning_rate=1e-4,
    )
    experiment._apply_student_training_policy()
    assert all(
        not parameter.requires_grad
        for parameter in model.branch_img.appearance.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in model.appearance_correction_head.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.solar_event_encoder.parameters())


def test_solar_token_tail_is_exact_frozen_parent_child(monkeypatch):
    monkeypatch.setattr(MTS_31, "encoder", _TinyTimeEncoder)
    monkeypatch.setattr(MTS_31, "encoder_img", _TinyWeatherEncoder)
    args = SimpleNamespace(
        data="Stanford",
        image_encoder_type="cnn_solar_advection",
        c_out=1,
        tors="student",
        modality_dropout_rate=0.0,
        residual_correction_clip_kw=30.1,
        expert_gate_bias=-2.0,
        expert_gate_hidden_dim=8,
        stanford_capacity_kw=30.1,
        history_order="current_first",
        event_routing_mode="soft",
        tail_routing_mode="confidence",
        tail_confidence_threshold=0.5,
        seq_len=16,
    )
    args_img = SimpleNamespace(
        c_out=64,
        pred_len=1,
        history_order="current_first",
        stanford_image_mode="rgb",
        image_cnn_dim=64,
        image_temporal_hidden_dim=64,
        dropout=0.0,
        forecast_horizon_minutes=15,
        sample_interval_minutes=1,
        stanford_capacity_kw=30.1,
    )
    model = MTS_31.Model(args, args_img, SimpleNamespace(c_out=1)).eval()
    with torch.no_grad():
        model.appearance_correction_head.bias.fill_(0.4)
        model.event_state_head.bias.copy_(torch.tensor([0.5, -0.5, -0.5]))
        model.event_up_correction_head.bias.fill_(1.2)
        model.event_down_correction_head.bias.fill_(-0.8)

    x_ts = torch.linspace(8.0, 4.0, 16).view(1, 16, 1)
    images = torch.rand(1, 16, 3, 64, 64)
    future = torch.rand(1, 1, 3, 64, 64)
    weather = torch.zeros(1, 16, 1)
    marks = torch.flip(chronological_marks(1, 16), dims=[1])
    target = target_marks(1)
    with torch.no_grad():
        parent = model(
            x_ts, images, future, weather, weather,
            "causal_solar_event", marks, target)[0]
        child_epoch0 = model(
            x_ts, images, future * 1000.0, weather, weather,
            "causal_solar_token_tail", marks, target)[0]
    torch.testing.assert_close(child_epoch0, parent, rtol=0, atol=0)
    assert model.last_solar_token.shape == (1, 64)
    assert model.last_solar_attention_logits.shape == (1, 64)
    assert model.last_tail_scale is None
    torch.testing.assert_close(
        model.last_tail_up_raw_correction, torch.zeros(1, 1, 1))
    torch.testing.assert_close(
        model.last_tail_down_raw_correction, torch.zeros(1, 1, 1))
    torch.testing.assert_close(
        model.last_tail_routing_strength, torch.zeros(1, 1, 1))
    assert model.last_tail_event_logit.shape == (1, 1, 1)
    assert model.last_tail_direction_logit.shape == (1, 1, 1)
    assert model.last_tail_expert_correction.shape == (1, 1, 1)

    with torch.no_grad():
        model.tail_event_head.weight.zero_()
        model.tail_event_head.bias.fill_(torch.logit(torch.tensor(0.75)))
        model.tail_direction_head.weight.zero_()
        model.tail_direction_head.bias.fill_(torch.logit(torch.tensor(0.8)))
        model.tail_up_head[-1].weight.zero_()
        model.tail_up_head[-1].bias.fill_(0.1)
        model.tail_down_head[-1].weight.zero_()
        model.tail_down_head[-1].bias.fill_(-0.2)
        model.eval()
        model(
            x_ts, images, future, weather, weather,
            "causal_solar_token_tail", marks, target)
        eval_strength = model.last_tail_routing_strength.clone()
        eval_correction = model.last_event_correction.clone()
        model.train()
        model(
            x_ts, images, future * 1000.0, weather, weather,
            "causal_solar_token_tail", marks, target)
        train_strength = model.last_tail_routing_strength.clone()
        train_correction = model.last_event_correction.clone()

    torch.testing.assert_close(
        eval_strength, torch.full_like(eval_strength, 0.5))
    torch.testing.assert_close(train_strength, eval_strength, rtol=0, atol=0)
    # 0.5 * (0.8 * 3.01 + 0.2 * -6.02) = 0.602 kW.
    torch.testing.assert_close(
        eval_correction, torch.full_like(eval_correction, 0.602),
        rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(train_correction, eval_correction, rtol=0, atol=0)

    model.zero_grad(set_to_none=True)
    model.train()
    task_output = model(
        x_ts, images, future, weather, weather,
        "causal_solar_token_tail", marks, target)[0]
    task_output.sum().backward()
    assert all(
        parameter.grad is None
        for parameter in model.tail_up_head.parameters())
    assert all(
        parameter.grad is None
        for parameter in model.tail_down_head.parameters())

    with torch.no_grad():
        model.tail_routing_mode = "posterior"
        model.tail_prior_event_positive.fill_(2.0)
        model.tail_prior_event_total.fill_(100.0)
        model.tail_prior_up_positive.fill_(3.0)
        model.tail_prior_direction_total.fill_(10.0)
        model.eval()
        model(
            x_ts, images, future, weather, weather,
            "causal_solar_token_tail", marks, target)
        posterior_strength = model.last_tail_routing_strength.clone()
        posterior_up = model.last_tail_direction_probability.clone()
        posterior_correction = model.last_event_correction.clone()

    expected_posterior = (0.02 * 0.75) / (0.02 * 0.75 + 0.98 * 0.25)
    expected_event = (expected_posterior - 0.02) / (1.0 - 0.02)
    expected_up = (0.3 * 0.8) / (0.3 * 0.8 + 0.7 * 0.2)
    expected_mix = expected_up * 3.01 + (1.0 - expected_up) * -6.02
    torch.testing.assert_close(
        posterior_strength,
        torch.full_like(posterior_strength, expected_event),
        rtol=1e-5, atol=1e-6)
    assert posterior_strength.dtype == x_ts.dtype
    torch.testing.assert_close(
        model.last_tail_event_probability,
        torch.full_like(model.last_tail_event_probability, expected_posterior),
        rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        posterior_up, torch.full_like(posterior_up, expected_up),
        rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        posterior_correction,
        torch.full_like(
            posterior_correction, expected_event * expected_mix),
        rtol=1e-5, atol=1e-6)

    model.train()
    model.update_tail_priors(
        torch.tensor([True, False, True]),
        torch.tensor([True, True, False]),
    )
    assert model.tail_prior_event_positive.item() == 4.0
    assert model.tail_prior_event_total.item() == 103.0
    assert model.tail_prior_up_positive.item() == 4.0
    assert model.tail_prior_direction_total.item() == 12.0
    model.eval()
    model.update_tail_priors(
        torch.tensor([True]), torch.tensor([True]))
    assert model.tail_prior_event_total.item() == 103.0
    model.freeze_tail_priors()
    assert model.tail_prior_frozen.item() is True
    model.train()
    model.update_tail_priors(
        torch.tensor([True]), torch.tensor([True]))
    assert model.tail_prior_event_total.item() == 103.0

    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    experiment.model = model
    experiment.args = SimpleNamespace(
        train_image_residual_only=True,
        freeze_ts_permanently=True,
        student_fuse_strategy="causal_solar_token_tail",
        fuse_strategy="causal_solar_token_tail",
        event_residual_aux_weight=1.0,
        learning_rate=1e-4,
    )
    experiment._apply_student_training_policy()
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith((
        "solar_token_encoder.",
        "tail_event_head.",
        "tail_direction_head.",
        "tail_up_head.",
        "tail_down_head.",
    )) for name in trainable)
    assert not any(name.startswith("solar_event_encoder.") for name in trainable)
    assert not any(name.startswith("appearance_correction_head.") for name in trainable)

    experiment._set_tail_magnitude_trainable(False)
    assert all(
        not parameter.requires_grad
        for parameter in model.tail_up_head.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in model.tail_down_head.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.tail_event_head.parameters())
    experiment._set_tail_magnitude_trainable(True)
    assert all(
        parameter.requires_grad
        for parameter in model.tail_up_head.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.tail_down_head.parameters())
