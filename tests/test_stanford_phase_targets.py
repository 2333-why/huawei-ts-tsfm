import math

import pytest
import torch
import torch.nn.functional as F

from utils.stanford_phase import (
    ACTIVE_DOWN,
    ACTIVE_UP,
    NO_EVENT,
    PHASE_IGNORE_INDEX,
    RETURN_FROM_DOWN,
    RETURN_FROM_UP,
    build_stanford_path_targets,
    capacity_normalized_path_huber,
    discrete_hazard_survival_nll,
    episode_inverse_weights,
    fixed_lead_path_huber_loss,
    macro_balanced_phase_loss,
    window_inverse_weights,
)


def path(values):
    return torch.tensor(values, dtype=torch.float32).view(1, 15, 1)


def valid(value=True):
    return torch.full((1, 15, 1), value, dtype=torch.bool)


def test_no_path_masks_every_target_and_losses_are_differentiable_zero():
    future = path([0.0] * 15)
    targets = build_stanford_path_targets(future, valid(False))

    assert targets.no_path_mask.tolist() == [False]
    assert targets.round_trip_mask.tolist() == [False]
    assert targets.first_crossing_bin.tolist() == [-1]
    assert not targets.hazard_at_risk_mask.any()
    assert not targets.hazard_event_mask.any()
    assert torch.equal(
        targets.phase,
        torch.full((1, 5), PHASE_IGNORE_INDEX, dtype=torch.long),
    )

    hazard_logits = torch.zeros(1, 5, requires_grad=True)
    hazard_loss = discrete_hazard_survival_nll(
        hazard_logits,
        targets.first_crossing_bin,
        targets.hazard_at_risk_mask,
    )
    assert hazard_loss.item() == 0.0
    hazard_loss.backward()
    torch.testing.assert_close(hazard_logits.grad, torch.zeros_like(hazard_logits))

    prediction = torch.zeros_like(future, requires_grad=True)
    path_loss = capacity_normalized_path_huber(
        prediction, future, valid(False), capacity_kw=10.0
    )
    assert path_loss.item() == 0.0
    path_loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_active_up_and_down_include_the_five_kw_boundary_and_hazard_nll():
    future = torch.cat([
        path([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10]),
        path([0, -1, -5, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6]),
    ])
    targets = build_stanford_path_targets(
        future, torch.ones_like(future, dtype=torch.bool)
    )

    assert targets.phase[0].tolist() == [NO_EVENT, NO_EVENT, ACTIVE_UP, ACTIVE_UP, ACTIVE_UP]
    assert targets.phase[1].tolist() == [NO_EVENT, ACTIVE_DOWN, ACTIVE_DOWN, ACTIVE_DOWN, ACTIVE_DOWN]
    assert targets.first_crossing_bin.tolist() == [2, 1]
    assert targets.hazard_event_mask[0].tolist() == [False, False, True, False, False]
    assert targets.hazard_at_risk_mask[0].tolist() == [True, True, True, False, False]
    assert not targets.round_trip_mask.any()

    logits = torch.zeros(2, 5)
    nll = discrete_hazard_survival_nll(
        logits, targets.first_crossing_bin, targets.hazard_at_risk_mask,
        reduction="none",
    )
    torch.testing.assert_close(
        nll,
        torch.tensor([3.0 * math.log(2.0), 2.0 * math.log(2.0)]),
    )


def test_complete_stable_path_is_no_path_but_missing_path_is_unknown():
    future = path([0.0] * 15)
    complete = build_stanford_path_targets(future, valid())
    incomplete_mask = valid()
    incomplete_mask[:, 7] = False
    incomplete = build_stanford_path_targets(future, incomplete_mask)

    assert complete.no_path_mask.tolist() == [True]
    assert complete.round_trip_mask.tolist() == [False]
    assert incomplete.no_path_mask.tolist() == [False]


def test_return_then_active_at_endpoint_is_not_round_trip_endpoint_ordinary():
    future = path([0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6])
    targets = build_stanford_path_targets(future, valid())
    assert targets.round_trip_mask.tolist() == [False]


def test_return_from_down_and_up_are_labeled_from_the_latest_excursion():
    down_return = path([0, -3, -6, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    up_return = path([0, 3, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    future = torch.cat([down_return, up_return])
    targets = build_stanford_path_targets(
        future, torch.ones_like(future, dtype=torch.bool)
    )

    assert targets.phase[0].tolist() == [NO_EVENT, ACTIVE_DOWN, RETURN_FROM_DOWN, RETURN_FROM_DOWN, RETURN_FROM_DOWN]
    assert targets.phase[1].tolist() == [NO_EVENT, ACTIVE_UP, RETURN_FROM_UP, RETURN_FROM_UP, RETURN_FROM_UP]
    assert targets.round_trip_mask.tolist() == [True, True]
    assert targets.first_crossing_bin.tolist() == [1, 1]


def test_multiple_crossings_update_return_direction_without_losing_first_crossing():
    future = path([
        0, 6, 2, -6, -1,
        0, 0, 7, 7, 7,
        0, 0, 0, 0, 0,
    ])
    targets = build_stanford_path_targets(future, valid())

    assert targets.phase.tolist() == [[
        NO_EVENT,
        RETURN_FROM_UP,
        RETURN_FROM_DOWN,
        ACTIVE_UP,
        RETURN_FROM_UP,
    ]]
    assert targets.first_crossing_bin.tolist() == [1]
    assert targets.round_trip_mask.tolist() == [True]


def test_missing_minute_censors_later_state_and_survival_but_not_earlier_lead():
    future = path([0, float("nan"), 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
    mask = valid()
    targets = build_stanford_path_targets(future, mask)

    assert targets.lead_observation_mask.tolist() == [[True, True, True, True, True]]
    assert targets.lead_valid_mask.tolist() == [[True, False, False, False, False]]
    assert targets.phase.tolist() == [[
        NO_EVENT,
        PHASE_IGNORE_INDEX,
        PHASE_IGNORE_INDEX,
        PHASE_IGNORE_INDEX,
        PHASE_IGNORE_INDEX,
    ]]
    assert targets.first_crossing_bin.tolist() == [-1]
    assert targets.hazard_at_risk_mask.tolist() == [[True, False, False, False, False]]
    assert targets.no_path_mask.tolist() == [False]

    logits = torch.zeros(1, 5)
    nll = discrete_hazard_survival_nll(
        logits, targets.first_crossing_bin, targets.hazard_at_risk_mask
    )
    assert nll.item() == pytest.approx(math.log(2.0))


def test_crossing_inside_a_fixed_lead_interval_is_assigned_to_its_next_bin():
    future = path([0, 0, 0, 0, 0, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    targets = build_stanford_path_targets(future, valid())
    assert targets.first_crossing_bin.tolist() == [3]
    assert targets.hazard_event_mask.tolist() == [[False, False, False, True, False]]
    assert targets.round_trip_mask.tolist() == [True]


def test_macro_phase_loss_balances_classes_not_element_counts():
    targets = torch.tensor([[NO_EVENT, NO_EVENT, NO_EVENT, ACTIVE_UP]])
    logits = torch.zeros(1, 4, 5)
    logits[0, :3, NO_EVENT] = math.log(4.0)
    logits[0, 3, ACTIVE_UP] = -math.log(4.0)

    per_element = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1), reduction="none")
    expected = 0.5 * (per_element[:3].mean() + per_element[3])
    actual = macro_balanced_phase_loss(logits, targets)
    torch.testing.assert_close(actual, expected)


def test_capacity_normalized_huber_and_fixed_lead_masking():
    prediction = torch.zeros(1, 15, 1, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, 0] = 5.0
    target[:, 1] = 100.0  # Not a fixed lead and must not affect fixed-lead loss.
    mask = torch.ones_like(prediction, dtype=torch.bool)

    fixed_loss = fixed_lead_path_huber_loss(
        prediction, target, mask, capacity_kw=10.0, delta=1.0
    )
    # Normalized error at lead 1 is 0.5: Huber=0.5*0.5^2, averaged over 5 leads.
    assert fixed_loss.item() == pytest.approx(0.025)

    mask[:, 0] = False
    assert fixed_lead_path_huber_loss(
        prediction, target, mask, capacity_kw=10.0
    ).item() == 0.0


def test_episode_and_window_inverse_weights_have_unit_mass():
    mask = torch.tensor([[True, True, False, True, True, True]])
    episode = episode_inverse_weights(mask)
    torch.testing.assert_close(
        episode,
        torch.tensor([[0.5, 0.5, 0.0, 1 / 3, 1 / 3, 1 / 3]]),
    )
    assert episode[0, :2].sum().item() == pytest.approx(1.0)
    assert episode[0, 3:].sum().item() == pytest.approx(1.0)

    window = window_inverse_weights(mask)
    torch.testing.assert_close(
        window,
        torch.tensor([[0.2, 0.2, 0.0, 0.2, 0.2, 0.2]]),
    )
    assert window.sum().item() == pytest.approx(1.0)
