import inspect
from types import SimpleNamespace

import pytest
import torch

from models.image_sequence_encoder import (
    StanfordCausalPhaseFieldEncoder,
    hierarchical_phase_probabilities,
)


LEADS = (1, 3, 5, 10, 15)


def phase_config(history_order="current_first"):
    return SimpleNamespace(
        history_order=history_order,
        stanford_capacity_kw=30.1,
        sample_interval_minutes=1.0,
        phase_field_dim=8,
        phase_field_stem_dim=8,
        phase_field_motion_dim=4,
        phase_field_correlation_radius=1,
        phase_field_correlation_temperature=0.01,
        phase_field_temporal_decay=2.0,
    )


def multi_hypothesis_config(history_order="current_first"):
    config = phase_config(history_order)
    config.phase_transport_mode = "multi_hypothesis"
    config.phase_transport_hypotheses = 3
    return config


def set_attention_config(history_order="current_first"):
    config = multi_hypothesis_config(history_order)
    config.phase_transport_mode = "set_attention"
    return config


def time_marks(batch, minute_values):
    minute_values = torch.as_tensor(minute_values, dtype=torch.float32)
    hour = torch.floor(minute_values / 60.0)
    minute = torch.remainder(minute_values, 60.0)
    marks = torch.stack([
        torch.full_like(minute, 6.0),
        torch.full_like(minute, 24.0),
        torch.full_like(minute, 5.0),
        hour,
        minute,
    ], dim=-1)
    return marks.unsqueeze(0).repeat(batch, 1, 1)


def lead_marks(batch, current_minute):
    return time_marks(batch, [current_minute + lead for lead in LEADS])


def test_phase_field_shapes_and_strictly_causal_student_boundary():
    torch.manual_seed(21)
    model = StanfordCausalPhaseFieldEncoder(phase_config()).eval()
    chronological_images = torch.rand(2, 4, 3, 32, 32)
    chronological_marks = time_marks(2, [600, 601, 602, 603])
    chronological_pv = torch.linspace(3.0, 7.0, 4).view(1, 4, 1).repeat(2, 1, 1)

    output = model(
        torch.flip(chronological_images, dims=[1]),
        torch.flip(chronological_marks, dims=[1]),
        lead_marks(2, 603),
        torch.flip(chronological_pv, dims=[1]),
    )

    assert model.uses_future_images is False
    assert list(inspect.signature(model.forward).parameters) == [
        "images", "x_mark_h", "lead_marks", "pv_history"
    ]
    assert "seq_y_img" not in inspect.getsource(model.forward)
    with pytest.raises(TypeError):
        model(
            chronological_images, chronological_marks, lead_marks(2, 603),
            chronological_pv, torch.rand(2, 5, 3, 32, 32))

    assert output["hazard_features"].shape == (2, 5, 8)
    assert output["phase_features"].shape == (2, 5, 8)
    assert output["path_features"].shape == (2, 5, 8)
    assert output["hazard_logits"].shape == (2, 5)
    assert output["phase_logits"].shape == (2, 5, 5)
    assert output["phase_factor_logits"].shape == (2, 5, 3)
    assert output["phase_probability"].shape == (2, 5, 5)
    assert output["class_path_delta_kw"].shape == (2, 5, 5, 1)
    assert output["path_delta_kw"].shape == (2, 5, 1)
    torch.testing.assert_close(
        output["phase_probability"].sum(dim=-1), torch.ones(2, 5))
    assert output["event_feature"].shape == (2, 1, 8)
    torch.testing.assert_close(
        output["lead_minutes"], torch.tensor(LEADS, dtype=torch.float32))

    # 32x32 inputs become 8x8 fields; all flow tensors retain spatial axes.
    assert output["correspondence_cost_volume"].shape == (2, 3, 9, 8, 8)
    assert output["correspondence_probabilities"].shape == (2, 3, 9, 8, 8)
    assert output["pairwise_flow_field"].shape == (2, 3, 2, 8, 8)
    assert output["local_flow_field"].shape == (2, 2, 8, 8)
    assert output["transported_feature_fields"].shape == (2, 5, 8, 8, 8)
    assert output["path_sampling_grids"].shape == (2, 5, 8, 8, 2)
    assert output["historical_sun_stream"].shape == (2, 4, 8)
    assert output["target_sun_field_stream"].shape == (2, 5, 8)
    assert output["motion_consistency_loss"].shape == ()
    assert torch.isfinite(output["motion_consistency_loss"])
    assert "advection_velocity" not in output

    torch.testing.assert_close(
        output["correspondence_probabilities"].sum(dim=2),
        torch.ones(2, 3, 8, 8),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output["historical_sun_attention"].sum(dim=-1),
        torch.ones(2, 4),
    )
    torch.testing.assert_close(
        output["target_sun_attention"].sum(dim=-1),
        torch.ones(2, 5),
    )


def test_motion_consistency_trains_historical_correspondence_features():
    torch.manual_seed(29)
    model = StanfordCausalPhaseFieldEncoder(
        phase_config(history_order="past_first")).train()
    images = torch.rand(2, 4, 3, 32, 32)
    marks = time_marks(2, [600, 601, 602, 603])
    pv = torch.linspace(3.0, 7.0, 4).view(1, 4, 1).repeat(2, 1, 1)

    output = model(images, marks, lead_marks(2, 603), pv)
    output["motion_consistency_loss"].backward()

    assert model.motion_projection.weight.grad is not None
    assert model.motion_projection.weight.grad.abs().sum().item() > 0
    assert model.frame_stem[0].weight.grad is not None
    assert model.frame_stem[0].weight.grad.abs().sum().item() > 0


def test_hierarchical_phase_factorization_matches_endpoint_semantics():
    logits = torch.tensor([
        [-10.0, 0.0, 0.0],
        [10.0, 10.0, -10.0],
        [10.0, 10.0, 10.0],
        [10.0, -10.0, -10.0],
        [10.0, -10.0, 10.0],
    ])

    probability = hierarchical_phase_probabilities(logits)

    assert probability.argmax(dim=-1).tolist() == [0, 1, 2, 3, 4]
    torch.testing.assert_close(
        probability.sum(dim=-1), torch.ones(5), rtol=0, atol=1e-6)


def test_dense_cost_volume_preserves_opposite_local_motion():
    model = StanfordCausalPhaseFieldEncoder(
        phase_config(history_order="past_first")).eval()

    # Two distinguishable local structures cross in opposite directions. A
    # single global velocity would cancel them; the dense field must retain both.
    features = torch.zeros(1, 2, 2, 7, 7)
    features[0, 0, 0, 2, 1] = 1.0
    features[0, 1, 0, 2, 2] = 1.0  # channel 0 moves right
    features[0, 0, 1, 4, 5] = 1.0
    features[0, 1, 1, 4, 4] = 1.0  # channel 1 moves left

    cost, probability, flow, confidence = model._local_cost_volume(features)

    assert cost.shape == (1, 1, 9, 7, 7)
    assert probability.shape == (1, 1, 9, 7, 7)
    assert flow.shape == (1, 1, 2, 7, 7)
    assert flow[0, 0, 0, 2, 2] > 0.99
    assert flow[0, 0, 0, 4, 4] < -0.99
    assert flow[0, 0, 1, 2, 2].abs() < 1e-4
    assert flow[0, 0, 1, 4, 4].abs() < 1e-4
    assert confidence[0, 0, 0, 2, 2] > 0.99
    assert confidence[0, 0, 0, 4, 4] > 0.99
    assert flow[0, 0, 0].amax() - flow[0, 0, 0].amin() > 1.98


def test_multi_hypothesis_transport_preserves_normalized_path_mixture():
    torch.manual_seed(26)
    model = StanfordCausalPhaseFieldEncoder(
        multi_hypothesis_config(history_order="past_first")).eval()
    images = torch.rand(2, 4, 3, 32, 32)
    marks = time_marks(2, [600, 601, 602, 603])
    targets = lead_marks(2, 603)
    pv = torch.linspace(3.0, 7.0, 4).view(1, 4, 1).repeat(2, 1, 1)

    output = model(images, marks, targets, pv)

    assert output["transport_hypothesis_weights"].shape == (2, 3, 5)
    assert output["transport_hypothesis_source_coordinates"].shape == (2, 3, 5, 2)
    assert output["transport_source_dispersion"].shape == (2, 5, 1)
    assert output["target_sun_field_std"].shape == (2, 5, 8)
    assert output["target_phase_std"].shape == (2, 5, 8)
    torch.testing.assert_close(
        output["transport_hypothesis_weights"].sum(dim=1),
        torch.ones(2, 5), rtol=1e-5, atol=1e-6)
    assert torch.isfinite(output["transport_source_dispersion"]).all()
    assert output["transport_source_dispersion"].min().item() >= 0.0


def test_hypothesis_set_attention_is_permutation_invariant():
    torch.manual_seed(27)
    model = StanfordCausalPhaseFieldEncoder(set_attention_config()).eval()
    inputs = torch.randn(
        4, 3, model.hypothesis_token_projection[0].in_features)
    tokens = model.hypothesis_token_projection(inputs)
    queries = model.hypothesis_set_queries.expand(4, -1, -1)

    original, _ = model.hypothesis_set_attention(
        queries, tokens, tokens, need_weights=False)
    permuted, _ = model.hypothesis_set_attention(
        queries, tokens[:, [2, 0, 1]], tokens[:, [2, 0, 1]],
        need_weights=False)

    torch.testing.assert_close(original, permuted, rtol=1e-5, atol=1e-6)


def test_each_timestamp_uses_its_own_historical_and_target_sun_location():
    torch.manual_seed(22)
    model = StanfordCausalPhaseFieldEncoder(
        phase_config(history_order="past_first")).eval()
    images = torch.rand(1, 4, 3, 32, 32)
    # Widely separated known timestamps make accidental reuse of the target-sun
    # coordinate for every historical frame directly observable.
    history_marks = time_marks(1, [8 * 60, 10 * 60, 12 * 60, 14 * 60])
    targets = lead_marks(1, 14 * 60)
    pv = torch.linspace(2.0, 8.0, 4).view(1, 4, 1)

    output = model(images, history_marks, targets, pv)
    historical_coordinates = output["historical_sun_coordinates"][0]
    target_coordinates = output["target_sun_coordinates"][0]
    historical_peaks = output["historical_sun_attention"][0].argmax(dim=-1)

    assert historical_coordinates.shape == (4, 2)
    assert target_coordinates.shape == (5, 2)
    assert torch.unique(historical_peaks).numel() >= 3
    assert (historical_coordinates[1:] - historical_coordinates[:-1]).abs().sum() > 0.2
    assert not torch.allclose(historical_coordinates[-1], target_coordinates[-1])
    assert not torch.allclose(
        output["historical_sun_stream"][:, :1],
        output["historical_sun_stream"][:, -1:],
    )


def test_current_first_and_past_first_are_identical():
    torch.manual_seed(23)
    current_first = StanfordCausalPhaseFieldEncoder(
        phase_config("current_first")).eval()
    past_first = StanfordCausalPhaseFieldEncoder(
        phase_config("past_first")).eval()
    past_first.load_state_dict(current_first.state_dict(), strict=True)

    images = torch.rand(2, 4, 3, 32, 32)
    marks = time_marks(2, [600, 601, 602, 603])
    targets = lead_marks(2, 603)
    pv = torch.linspace(4.0, 9.0, 4).view(1, 4, 1).repeat(2, 1, 1)

    current_output = current_first(
        torch.flip(images, dims=[1]),
        torch.flip(marks, dims=[1]),
        targets,
        torch.flip(pv, dims=[1]),
    )
    past_output = past_first(images, marks, targets, pv)

    assert current_output.keys() == past_output.keys()
    for key in current_output:
        torch.testing.assert_close(
            current_output[key], past_output[key], rtol=0, atol=0)


def test_phase_and_path_predictions_use_explicit_pv_history():
    torch.manual_seed(24)
    model = StanfordCausalPhaseFieldEncoder(
        phase_config(history_order="past_first")).eval()
    images = torch.rand(1, 16, 3, 32, 32)
    marks = time_marks(1, list(range(600, 616)))
    targets = lead_marks(1, 615)
    flat_pv = torch.full((1, 16, 1), 8.0)
    volatile_pv = flat_pv.clone()
    volatile_pv[:, -6:, 0] = torch.tensor([3.0, 10.0, 4.0, 12.0, 5.0, 14.0])

    flat = model(images, marks, targets, flat_pv)
    volatile = model(images, marks, targets, volatile_pv)

    assert not torch.equal(flat["pv_statistics"], volatile["pv_statistics"])
    assert not torch.equal(flat["phase_logits"], volatile["phase_logits"])
    assert not torch.equal(flat["path_delta_kw"], volatile["path_delta_kw"])


def test_shift_cost_volume_matches_unfold_reference():
    torch.manual_seed(25)
    model = StanfordCausalPhaseFieldEncoder(
        phase_config(history_order="past_first")).eval()
    features = torch.randn(2, 4, 3, 7, 6)
    cost, probability, flow, confidence = model._local_cost_volume(features)

    normalized = torch.nn.functional.normalize(features, dim=2, eps=1e-6)
    previous = normalized[:, :-1].reshape(-1, 3, 7, 6)
    current = normalized[:, 1:].reshape(-1, 3, 7, 6)
    patches = torch.nn.functional.unfold(
        previous, kernel_size=3, padding=1).reshape(-1, 3, 9, 7, 6)
    reference_cost = (current.unsqueeze(2) * patches).sum(dim=1)
    reference_valid = torch.nn.functional.unfold(
        torch.ones(previous.shape[0], 1, 7, 6),
        kernel_size=3, padding=1).reshape(-1, 9, 7, 6).bool()
    reference_cost = reference_cost.masked_fill(
        ~reference_valid, torch.finfo(reference_cost.dtype).min)
    reference_cost = reference_cost.reshape(2, 3, 9, 7, 6)

    torch.testing.assert_close(cost, reference_cost, rtol=0, atol=0)
    assert torch.isfinite(probability).all()
    assert torch.isfinite(flow).all()
    assert torch.isfinite(confidence).all()
