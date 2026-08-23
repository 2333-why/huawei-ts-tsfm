from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from exp.exp_long_term_forecasting_kd import (
    ConservativeCheckpointGuard,
    Exp_Long_Term_Forecast,
    solve_phase_family_calibration,
)
from models.MTS_31 import (
    conditional_phase_physical_support,
    conditional_phase_risk_route_support,
    conditional_phase_route_support,
    confidence_margin_support,
    conjunctive_event_support,
    endpoint_magnitude_support,
    hard_phase_winner_support,
    phase_path_correction,
    phase_regime_support,
    winner_conjunctive_support,
)


def phase_labels(paths):
    paths = torch.as_tensor(paths, dtype=torch.float32).unsqueeze(-1)
    batch = paths.shape[0]
    return {
        "pv_path": paths,
        "valid_mask": torch.ones(batch, 15, dtype=torch.bool),
        "episode_weight": torch.ones(batch),
    }


def phase_experiment(stage, **weights):
    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    experiment.model = nn.Module()
    experiment._phase_training_stage = stage
    defaults = dict(
        student_fuse_strategy="causal_phase_field_tail",
        phase_state_aux_weight=0.0,
        phase_path_aux_weight=0.0,
        phase_benefit_aux_weight=0.0,
        phase_benefit_mode="winner",
        event_residual_aux_weight=0.0,
        stable_correction_aux_weight=0.0,
        tail_error_threshold_kw=3.0,
        expert_gate_threshold_kw=5.0,
        stanford_capacity_kw=30.1,
    )
    defaults.update(weights)
    experiment.args = SimpleNamespace(**defaults)
    return experiment


def test_phase_family_calibration_finds_guarded_stable_active_projection():
    count = 8
    diagnostics = {
        'target_kw': torch.ones(count).numpy(),
        'parent_kw': torch.zeros(count).numpy(),
        'class_route_support': torch.zeros(count, 5).numpy(),
        'class_scale_probability': torch.ones(count, 5).numpy(),
        'class_raw_correction_kw': torch.zeros(count, 5).numpy(),
        'signed_ramp_kw': torch.tensor(
            [0, 0, 0, 0, 8, -8, 6, -6], dtype=torch.float32).numpy(),
        'cloud_event_day': torch.ones(count, dtype=torch.bool).numpy(),
        'no_path': torch.tensor(
            [1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool).numpy(),
        'round_trip': torch.tensor(
            [0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.bool).numpy(),
    }
    diagnostics['class_route_support'][:4, 0] = 1.0
    diagnostics['class_raw_correction_kw'][:4, 0] = 1.0
    diagnostics['class_route_support'][4:, 2] = 1.0
    diagnostics['class_raw_correction_kw'][4:, 2] = 1.0

    result = solve_phase_family_calibration(diagnostics, grid_size=21)

    assert result['feasible'] is True
    assert result['stable_scale'] == 1.0
    assert result['active_scale'] == 1.0
    assert all(delta < 0 for delta in result['rmse_deltas'].values())


def test_phase_path_correction_is_physical_endpoint_minus_parent():
    current = torch.tensor([[[10.0]], [[20.0]]])
    parent = torch.tensor([[[12.0]], [[18.0]]])
    class_path = torch.tensor([
        [[0.0, -6.0, 6.0, -2.0, 2.0]],
        [[0.0, -8.0, 8.0, -3.0, 3.0]],
    ])

    correction = phase_path_correction(current, parent, class_path)

    torch.testing.assert_close(
        parent + correction,
        current + class_path,
    )


def test_state_checkpoint_score_tracks_representation_not_parent_loss():
    weak = Exp_Long_Term_Forecast._phase_pretrain_selection_score(
        'state', 1.0,
        {'phase_hazard_event_auprc': 0.70, 'phase_macro_accuracy': 0.50})
    strong = Exp_Long_Term_Forecast._phase_pretrain_selection_score(
        'state', 2.0,
        {'phase_hazard_event_auprc': 0.75, 'phase_macro_accuracy': 0.55})

    assert strong < weak
    assert Exp_Long_Term_Forecast._phase_pretrain_selection_score(
        'magnitude', 2.0, {}) == 2.0


def test_conjunctive_event_support_requires_joint_confidence():
    path = torch.tensor([0.1, 0.4, 0.8, 1.0])
    phase = torch.tensor([0.9, 0.8, 0.7, 1.0])

    support = conjunctive_event_support(path, phase)

    assert torch.allclose(support, torch.tensor([0.0, 0.2, 0.5, 1.0]))


def test_winner_conjunctive_support_requires_all_three_signals():
    path = torch.tensor([1.0, 0.9, 0.7, 1.0])
    active = torch.tensor([1.0, 0.8, 0.8, 0.5])
    winner = torch.tensor([1.0, 0.7, 0.6, 0.5])

    support = winner_conjunctive_support(path, active, winner)

    assert torch.allclose(support, torch.tensor([1.0, 0.4, 0.1, 0.0]))


def test_hard_phase_winner_support_rejects_return_and_weak_evidence():
    path = torch.tensor([[[0.9]], [[0.9]], [[0.4]], [[0.9]]])
    phases = torch.tensor([
        [[[0.1, 0.6, 0.1, 0.1, 0.1]]],
        [[[0.1, 0.1, 0.1, 0.6, 0.1]]],
        [[[0.1, 0.6, 0.1, 0.1, 0.1]]],
        [[[0.1, 0.1, 0.6, 0.1, 0.1]]],
    ]).squeeze(1)
    winner = torch.tensor([[[0.9]], [[0.9]], [[0.9]], [[0.4]]])

    support = hard_phase_winner_support(path, phases, winner)

    assert support.reshape(-1).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_confidence_margin_support_uses_only_excess_evidence():
    path = torch.tensor([[[0.75]], [[0.75]]])
    phases = torch.tensor([
        [[0.1, 0.6, 0.1, 0.1, 0.1]],
        [[0.4, 0.3, 0.1, 0.1, 0.1]],
    ])
    winner = torch.tensor([[[0.75]], [[0.75]]])

    support = confidence_margin_support(path, phases, winner)

    assert support[0].item() == pytest.approx(0.125)
    assert support[1].item() == 0.0

    prior_support = confidence_margin_support(
        path[:1], phases[:1], torch.tensor([[[0.25]]]),
        winner_prior=torch.tensor(0.1))
    assert prior_support[0].item() == pytest.approx(1.0 / 24.0)


def test_endpoint_magnitude_support_uses_physical_event_boundary():
    delta = torch.tensor([-7.0, -2.5, 0.0, 5.0])
    assert torch.allclose(
        endpoint_magnitude_support(delta),
        torch.tensor([1.0, 0.5, 0.0, 1.0]))


def test_phase_regime_support_distinguishes_active_and_return():
    path = torch.tensor([[[0.8]], [[0.8]], [[0.2]]])
    endpoint = torch.tensor([[[0.8]], [[0.2]], [[0.2]]])
    phases = torch.tensor([
        [[0.05, 0.8, 0.05, 0.05, 0.05]],
        [[0.05, 0.05, 0.05, 0.8, 0.05]],
        [[0.8, 0.05, 0.05, 0.05, 0.05]],
    ])
    winner = torch.ones(3, 1, 1)

    support, physical = phase_regime_support(
        path, phases, endpoint, winner, winner_prior=torch.tensor(0.0))

    assert torch.allclose(
        physical.reshape(-1), torch.tensor([0.64, 0.64, 0.64]))
    assert torch.allclose(
        support.reshape(-1), torch.tensor([0.48, 0.48, 0.48]))


def test_conditional_phase_support_enforces_signed_endpoint_physics():
    path = torch.tensor([[[0.8]]])
    class_path = torch.tensor([[[0.0, -7.0, 7.0, -2.0, 2.0]]])

    physical = conditional_phase_physical_support(path, class_path)

    torch.testing.assert_close(
        physical,
        torch.tensor([[[0.2, 0.8, 0.8, 0.48, 0.48]]]),
        rtol=0,
        atol=1e-6,
    )
    wrong_sign = conditional_phase_physical_support(
        path, torch.tensor([[[0.0, 7.0, -7.0, -2.0, 2.0]]]))
    assert wrong_sign[..., 1:3].abs().sum().item() == 0


def test_conditional_phase_route_masks_unaccepted_expert_families():
    path = torch.tensor([[[1.0]]])
    phase = torch.full((1, 1, 5), 0.2)
    class_path = torch.tensor([[[0.0, -7.0, 7.0, 0.0, 0.0]]])
    winner = torch.full((1, 1, 5), 0.75)
    prior = torch.full((5,), 0.5)

    route, _ = conditional_phase_route_support(
        path, phase, class_path, winner, prior,
        class_mask=torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0]))

    torch.testing.assert_close(
        route, torch.tensor([[[0.0, 0.1, 0.1, 0.0, 0.0]]]))


def test_conditional_phase_risk_route_abstains_on_nonnegative_risk():
    path = torch.tensor([[[1.0]]])
    phase = torch.full((1, 1, 5), 0.2)
    class_path = torch.tensor([[[0.0, -7.0, 7.0, 0.0, 0.0]]])
    risk = torch.tensor([[[0.0, -0.01, 0.02, -0.01, -0.01]]])

    route, _ = conditional_phase_risk_route_support(
        path, phase, class_path, risk, normalized_risk_scale=0.01,
        class_mask=torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0]))

    torch.testing.assert_close(
        route, torch.tensor([[[0.0, 0.2, 0.0, 0.0, 0.0]]]))


def test_magnitude_loss_does_not_update_benefit_gate():
    experiment = phase_experiment("magnitude", event_residual_aux_weight=1.0)
    model = experiment.model
    model.last_phase_parent_pred = torch.full((2, 1, 1), 10.0)
    model.last_phase_class_raw_correction = torch.zeros(
        2, 1, 5, requires_grad=True)
    model.last_phase_benefit_logit = torch.zeros(2, 1, 1, requires_grad=True)
    model.last_phase_scale_logit = torch.zeros(2, 1, 1, requires_grad=True)
    model.last_phase_candidate_correction = torch.zeros(2, 1, 1)
    current = torch.full((2, 1, 1), 10.0)
    target = torch.tensor([[[16.0]], [[4.0]]])
    paths = phase_labels([
        [10, 10, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        [10, 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ])

    loss = experiment._auxiliary_loss(target, current, phase_targets=paths)
    loss.backward()

    assert model.last_phase_class_raw_correction.grad is not None
    assert model.last_phase_class_raw_correction.grad[..., 1:3].abs().sum() > 0
    assert model.last_phase_class_raw_correction.grad[..., [0, 3, 4]].abs().sum() == 0
    assert model.last_phase_benefit_logit.grad is None


def test_phase_magnitude_heads_receive_only_matching_phase_gradients():
    experiment = phase_experiment("magnitude", event_residual_aux_weight=1.0)
    model = experiment.model
    model.last_phase_parent_pred = torch.full((2, 1, 1), 10.0)
    raw = torch.full((2, 1, 5), 3.0, requires_grad=True)
    model.last_phase_class_raw_correction = raw
    current = torch.full((2, 1, 1), 10.0)
    target = torch.tensor([[[16.0]], [[4.0]]])
    paths = phase_labels([
        [10, 10, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        [10, 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ])

    loss = experiment._auxiliary_loss(target, current, phase_targets=paths)
    loss.backward()

    assert raw.grad[0, 0, 2].abs().item() > 0
    assert raw.grad[1, 0, 1].abs().item() > 0
    assert raw.grad[0, 0, [0, 1, 3, 4]].abs().sum().item() == 0
    assert raw.grad[1, 0, [0, 2, 3, 4]].abs().sum().item() == 0


def test_benefit_loss_does_not_update_magnitude_candidate():
    experiment = phase_experiment("benefit", phase_benefit_aux_weight=1.0)
    model = experiment.model
    model.last_phase_parent_pred = torch.full((2, 1, 1), 10.0)
    class_raw = torch.tensor([
        [[0.0, 0.0, 6.0, 0.0, 0.0]],
        [[0.0, -6.0, 0.0, 0.0, 0.0]],
    ], requires_grad=True)
    model.last_phase_class_raw_correction = class_raw
    candidate = torch.tensor([[[6.0]], [[-6.0]]], requires_grad=True)
    model.last_phase_candidate_correction = candidate
    model.last_phase_benefit_logit = torch.zeros(2, 1, 1, requires_grad=True)
    model.last_phase_scale_logit = torch.zeros(2, 1, 1, requires_grad=True)
    model.last_phase_class_benefit_logits = torch.zeros(
        2, 1, 5, requires_grad=True)
    model.last_phase_class_scale_logits = torch.zeros(
        2, 1, 5, requires_grad=True)
    model.last_phase_route_support = torch.ones(2, 1, 1)
    current = torch.full((2, 1, 1), 10.0)
    target = torch.tensor([[[16.0]], [[4.0]]])
    paths = phase_labels([
        [10, 10, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        [10, 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ])

    loss = experiment._auxiliary_loss(target, current, phase_targets=paths)
    loss.backward()

    assert model.last_phase_class_benefit_logits.grad.abs().sum().item() > 0
    assert model.last_phase_class_scale_logits.grad.abs().sum().item() > 0
    assert model.last_phase_benefit_logit.grad is None
    assert model.last_phase_scale_logit.grad is None
    assert candidate.grad is None
    assert model.last_phase_class_raw_correction.grad is None


def test_net_risk_loss_regresses_signed_candidate_cost_without_magnitude_grad():
    experiment = phase_experiment(
        "benefit", phase_benefit_aux_weight=1.0,
        phase_benefit_mode="net_risk")
    model = experiment.model
    model.last_phase_parent_pred = torch.full((2, 1, 1), 10.0)
    class_raw = torch.tensor([
        [[0.0, -6.0, 6.0, 0.0, 0.0]],
        [[0.0, -6.0, 6.0, 0.0, 0.0]],
    ], requires_grad=True)
    model.last_phase_class_raw_correction = class_raw
    model.last_phase_candidate_correction = torch.zeros(2, 1, 1)
    model.last_phase_benefit_logit = torch.zeros(2, 1, 1, requires_grad=True)
    model.last_phase_scale_logit = torch.zeros(2, 1, 1, requires_grad=True)
    risk = torch.zeros(2, 1, 5, requires_grad=True)
    scale = torch.zeros(2, 1, 5, requires_grad=True)
    model.last_phase_class_benefit_logits = risk
    model.last_phase_class_scale_logits = scale
    model.last_phase_route_support = torch.ones(2, 1, 1)
    current = torch.full((2, 1, 1), 10.0)
    target = torch.tensor([[[16.0]], [[10.0]]])
    paths = phase_labels([
        [10, 10, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        [10] * 15,
    ])

    loss = experiment._auxiliary_loss(target, current, phase_targets=paths)
    loss.backward()

    assert risk.grad.abs().sum().item() > 0
    assert scale.grad.abs().sum().item() > 0
    assert class_raw.grad is None


def test_constrained_stack_updates_group_duals_from_mean_excess_risk():
    experiment = phase_experiment(
        "benefit",
        phase_benefit_aux_weight=1.0,
        phase_benefit_mode="constrained_stack",
        phase_constraint_dual_lr=0.05,
        phase_constraint_penalty=1.0,
    )
    model = experiment.model
    model.train()
    model.last_phase_parent_pred = torch.full((2, 1, 1), 10.0)
    model.last_phase_class_raw_correction = torch.zeros(2, 1, 5)
    model.last_phase_candidate_correction = torch.zeros(2, 1, 1)
    model.last_phase_benefit_logit = torch.zeros(2, 1, 1)
    model.last_phase_scale_logit = torch.zeros(2, 1, 1)
    model.last_phase_class_benefit_logits = torch.zeros(2, 1, 5)
    model.last_phase_class_scale_logits = torch.zeros(2, 1, 5)
    model.last_phase_route_support = torch.ones(2, 1, 1)
    correction = torch.full((2, 1, 1), 3.0, requires_grad=True)
    model.last_phase_safety_correction = correction
    current = torch.full((2, 1, 1), 10.0)
    target = torch.full((2, 1, 1), 11.0)
    paths = phase_labels([[10] * 15, [10] * 15])

    loss = experiment._auxiliary_loss(target, current, phase_targets=paths)
    loss.backward()

    assert experiment._last_phase_constraint_violations["ordinary"] == pytest.approx(3.0)
    assert experiment._last_phase_constraint_violations["no_path"] == pytest.approx(3.0)
    assert experiment._phase_constraint_duals["ordinary"] == pytest.approx(0.15)
    assert experiment._phase_constraint_duals["no_path"] == pytest.approx(0.15)
    assert correction.grad.abs().sum().item() > 0


def test_constrained_stack_allows_group_improvement_despite_one_harmed_sample():
    experiment = phase_experiment(
        "benefit",
        phase_benefit_aux_weight=1.0,
        phase_benefit_mode="constrained_stack",
        phase_constraint_dual_lr=0.05,
        phase_constraint_penalty=1.0,
    )
    model = experiment.model
    model.train()
    model.last_phase_parent_pred = torch.full((2, 1, 1), 10.0)
    model.last_phase_class_raw_correction = torch.zeros(2, 1, 5)
    model.last_phase_candidate_correction = torch.zeros(2, 1, 1)
    model.last_phase_benefit_logit = torch.zeros(2, 1, 1)
    model.last_phase_scale_logit = torch.zeros(2, 1, 1)
    model.last_phase_class_benefit_logits = torch.zeros(2, 1, 5)
    model.last_phase_class_scale_logits = torch.zeros(2, 1, 5)
    model.last_phase_route_support = torch.ones(2, 1, 1)
    model.last_phase_safety_correction = torch.tensor(
        [[[3.0]], [[1.0]]], requires_grad=True)
    current = torch.full((2, 1, 1), 10.0)
    target = torch.tensor([[[12.0]], [[10.0]]])
    paths = phase_labels([[10] * 15, [10] * 15])

    experiment._auxiliary_loss(target, current, phase_targets=paths)

    # The second sample is harmed, but aggregate ordinary/no-path risk is 50%
    # below the frozen parent, so the non-degradation constraints stay inactive.
    assert experiment._last_phase_constraint_violations["ordinary"] == pytest.approx(-0.5)
    assert experiment._last_phase_constraint_violations["no_path"] == pytest.approx(-0.5)
    assert experiment._phase_constraint_duals["ordinary"] == 0.0
    assert experiment._phase_constraint_duals["no_path"] == 0.0


def test_constrained_stack_guard_risk_ignores_episode_training_weights():
    experiment = phase_experiment(
        "benefit",
        phase_benefit_aux_weight=1.0,
        phase_benefit_mode="constrained_stack",
        phase_constraint_dual_lr=0.05,
        phase_constraint_penalty=1.0,
    )
    model = experiment.model
    model.train()
    model.last_phase_parent_pred = torch.full((2, 1, 1), 10.0)
    model.last_phase_class_raw_correction = torch.zeros(2, 1, 5)
    model.last_phase_candidate_correction = torch.zeros(2, 1, 1)
    model.last_phase_benefit_logit = torch.zeros(2, 1, 1)
    model.last_phase_scale_logit = torch.zeros(2, 1, 1)
    model.last_phase_class_benefit_logits = torch.zeros(2, 1, 5)
    model.last_phase_class_scale_logits = torch.zeros(2, 1, 5)
    model.last_phase_route_support = torch.ones(2, 1, 1)
    model.last_phase_safety_correction = torch.tensor(
        [[[1.0]], [[3.0]]], requires_grad=True)
    current = torch.full((2, 1, 1), 10.0)
    target = torch.full((2, 1, 1), 11.0)
    paths = phase_labels([[10] * 15, [10] * 15])
    paths["episode_weight"] = torch.tensor([100.0, 1.0])

    experiment._auxiliary_loss(target, current, phase_targets=paths)

    # Unweighted guard risk is (0 + 4) / 2 versus parent risk 1.
    assert experiment._last_phase_constraint_violations["ordinary"] == pytest.approx(1.0)
    assert experiment._last_phase_constraint_violations["no_path"] == pytest.approx(1.0)


def test_ordinary_regret_updates_gate_for_no_path_and_round_trip():
    experiment = phase_experiment("active", stable_correction_aux_weight=1.0)
    model = experiment.model
    model.last_phase_parent_pred = torch.full((2, 1, 1), 10.0)
    model.last_phase_candidate_correction = torch.full((2, 1, 1), 2.0)
    gate_logit = torch.zeros(2, 1, 1, requires_grad=True)
    model.last_phase_benefit_logit = gate_logit
    model.last_phase_scale_logit = torch.zeros(2, 1, 1, requires_grad=True)
    model.last_phase_route_support = torch.sigmoid(gate_logit)
    model.last_phase_safety_strength = (
        torch.sigmoid(gate_logit) * torch.sigmoid(model.last_phase_scale_logit))
    current = torch.full((2, 1, 1), 10.0)
    target = current.clone()
    paths = phase_labels([
        [10] * 15,
        [10, 16, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    ])

    loss = experiment._auxiliary_loss(target, current, phase_targets=paths)
    loss.backward()

    assert loss.item() == pytest.approx(0.25)
    assert gate_logit.grad[0].item() > 0
    assert gate_logit.grad[1].item() > 0
    assert experiment._last_phase_path_targets.no_path_mask.tolist() == [True, False]
    assert experiment._last_phase_path_targets.round_trip_mask.tolist() == [False, True]


def test_phase_guard_requires_cloud_tail_improvement_and_extreme_safety():
    guard = ConservativeCheckpointGuard(
        input_score=3.0,
        input_ordinary_rmse=1.4,
        input_no_path_rmse=1.2,
        patience=3,
        ordinary_guard=True,
        no_path_guard=True,
        metric_guards={
            "cloud_event_ramp_ge5_rmse": {
                "baseline": 7.0,
                "mode": "improve",
            },
            "cloud_event_ramp_ge8_rmse": {
                "baseline": 8.5,
                "mode": "non_degrade",
            },
        },
    )

    rejected = guard.consider(
        1, 2.9, 1.3, 1.1,
        {
            "cloud_event_ramp_ge5_rmse": 6.9,
            "cloud_event_ramp_ge8_rmse": 8.6,
        },
    )
    assert rejected["accepted"] is False
    assert rejected["reason"] == "cloud_event_ramp_ge8_rmse_guard_failed"

    accepted = guard.consider(
        2, 2.9, 1.3, 1.1,
        {
            "cloud_event_ramp_ge5_rmse": 6.9,
            "cloud_event_ramp_ge8_rmse": 8.4,
        },
    )
    assert accepted["accepted"] is True
