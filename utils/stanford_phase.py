"""Causal path targets and losses for Stanford 15-minute PV transitions.

``future_path`` is a signed path relative to the forecast origin, in kW.  The
module observes all 15 one-minute values but emits supervision at the fixed
lead times 1, 3, 5, 10 and 15 minutes.  A prefix-valid mask is used for state
labels: after a missing minute, first-crossing and phase history are censored
rather than silently treating the missing interval as stable.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


FIXED_LEADS: Tuple[int, ...] = (1, 3, 5, 10, 15)
PHASE_NAMES: Tuple[str, ...] = (
    "no_event",
    "active_down",
    "active_up",
    "return_from_down",
    "return_from_up",
)
NO_EVENT = 0
ACTIVE_DOWN = 1
ACTIVE_UP = 2
RETURN_FROM_DOWN = 3
RETURN_FROM_UP = 4
NUM_PHASES = len(PHASE_NAMES)
PHASE_IGNORE_INDEX = -100


@dataclass(frozen=True)
class StanfordPathTargets:
    """Fixed-lead labels derived from an observed minute-level future path."""

    lead_values: torch.Tensor
    lead_observation_mask: torch.Tensor
    lead_valid_mask: torch.Tensor
    phase: torch.Tensor
    first_crossing_bin: torch.Tensor
    hazard_event_mask: torch.Tensor
    hazard_at_risk_mask: torch.Tensor
    no_path_mask: torch.Tensor
    round_trip_mask: torch.Tensor


def _normalize_path_inputs(
    future_path: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(future_path, torch.Tensor):
        raise TypeError("future_path must be a torch.Tensor")
    if future_path.ndim != 3 or future_path.shape[1:] != (15, 1):
        raise ValueError(
            "future_path must have shape [B,15,1], "
            f"got {tuple(future_path.shape)}"
        )
    if not future_path.is_floating_point():
        raise TypeError("future_path must have a floating-point dtype")
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if valid_mask.ndim == 3 and valid_mask.shape[-1] == 1:
        valid_mask = valid_mask[..., 0]
    if valid_mask.shape != future_path.shape[:2]:
        raise ValueError(
            "valid_mask must have shape [B,15] or [B,15,1], "
            f"got {tuple(valid_mask.shape)}"
        )
    path = future_path[..., 0]
    valid = valid_mask.to(device=path.device, dtype=torch.bool)
    valid = valid & torch.isfinite(path)
    return path, valid


def _lead_indices(device: torch.device, leads: Sequence[int] = FIXED_LEADS):
    if tuple(leads) != FIXED_LEADS:
        raise ValueError(f"Stanford phase leads must be exactly {FIXED_LEADS}")
    return torch.tensor([lead - 1 for lead in leads], device=device, dtype=torch.long)


def select_fixed_leads(
    future_path: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return values and pointwise observation masks at ``FIXED_LEADS``."""

    path, valid = _normalize_path_inputs(future_path, valid_mask)
    indices = _lead_indices(path.device)
    values = path.index_select(1, indices)
    observed = valid.index_select(1, indices)
    return values, observed


def build_stanford_path_targets(
    future_path: torch.Tensor,
    valid_mask: torch.Tensor,
    boundary_kw: float = 5.0,
    ignore_index: int = PHASE_IGNORE_INDEX,
) -> StanfordPathTargets:
    """Build first-crossing, phase, no-path and round-trip targets.

    Phase is a state at each fixed lead:

    * ``no_event``: no observed boundary crossing in the valid prefix;
    * ``active_down`` / ``active_up``: the lead value is at or beyond -/+5 kW;
    * ``return_from_down`` / ``return_from_up``: the path is back inside the
      boundary after its most recent down/up excursion.

    Processing every minute allows repeated active/return episodes.  A new
    excursion replaces the remembered direction used by subsequent returns.
    """

    if boundary_kw <= 0:
        raise ValueError("boundary_kw must be positive")
    path, observed = _normalize_path_inputs(future_path, valid_mask)
    batch = path.shape[0]
    device = path.device
    lead_indices = _lead_indices(device)

    # State-based labels are only identifiable while the full prefix is valid.
    prefix_valid = observed.to(torch.int64).cumprod(dim=1).to(torch.bool)
    lead_values = path.index_select(1, lead_indices)
    lead_observation_mask = observed.index_select(1, lead_indices)
    lead_valid_mask = prefix_valid.index_select(1, lead_indices)

    minute_phase = torch.full(
        (batch, 15), int(ignore_index), device=device, dtype=torch.long
    )
    last_direction = torch.zeros(batch, device=device, dtype=torch.int8)
    ever_active = torch.zeros(batch, device=device, dtype=torch.bool)
    round_trip_mask = torch.zeros(batch, device=device, dtype=torch.bool)

    for minute in range(15):
        valid = prefix_valid[:, minute]
        value = path[:, minute]
        active_down = valid & (value <= -boundary_kw)
        active_up = valid & (value >= boundary_kw)
        inside = valid & ~active_down & ~active_up

        phase_at_minute = minute_phase[:, minute]
        phase_at_minute[inside & ~ever_active] = NO_EVENT
        phase_at_minute[
            inside & ever_active & (last_direction < 0)
        ] = RETURN_FROM_DOWN
        phase_at_minute[
            inside & ever_active & (last_direction > 0)
        ] = RETURN_FROM_UP
        phase_at_minute[active_down] = ACTIVE_DOWN
        phase_at_minute[active_up] = ACTIVE_UP

        returned = inside & ever_active
        round_trip_mask |= returned
        last_direction = torch.where(
            active_down, torch.full_like(last_direction, -1), last_direction
        )
        last_direction = torch.where(
            active_up, torch.full_like(last_direction, 1), last_direction
        )
        ever_active |= active_down | active_up

    phase = minute_phase.index_select(1, lead_indices)

    crossing = prefix_valid & (path.abs() >= boundary_kw)
    event_observed = crossing.any(dim=1)
    first_crossing_minute = crossing.to(torch.int64).argmax(dim=1) + 1
    first_crossing_bin = torch.full(
        (batch,), -1, device=device, dtype=torch.long
    )
    if bool(event_observed.any().item()):
        fixed_leads = torch.tensor(FIXED_LEADS, device=device, dtype=torch.long)
        first_crossing_bin[event_observed] = torch.bucketize(
            first_crossing_minute[event_observed], fixed_leads, right=False
        )

    bins = torch.arange(len(FIXED_LEADS), device=device).unsqueeze(0)
    hazard_at_risk_mask = lead_valid_mask.clone()
    if bool(event_observed.any().item()):
        # An observed event censors its bin immediately, even if a later minute
        # in that fixed-lead interval is missing.
        hazard_at_risk_mask[event_observed] = (
            bins <= first_crossing_bin[event_observed, None]
        )
    hazard_event_mask = torch.zeros(
        (batch, len(FIXED_LEADS)), device=device, dtype=torch.bool
    )
    if bool(event_observed.any().item()):
        hazard_event_mask[event_observed, first_crossing_bin[event_observed]] = True

    full_path_valid = prefix_valid[:, -1]
    no_path_mask = full_path_valid & ~event_observed
    round_trip_mask = (
        full_path_valid
        & event_observed
        & (path[:, -1].abs() < boundary_kw)
    )
    return StanfordPathTargets(
        lead_values=lead_values,
        lead_observation_mask=lead_observation_mask,
        lead_valid_mask=lead_valid_mask,
        phase=phase,
        first_crossing_bin=first_crossing_bin,
        hazard_event_mask=hazard_event_mask,
        hazard_at_risk_mask=hazard_at_risk_mask,
        no_path_mask=no_path_mask,
        round_trip_mask=round_trip_mask,
    )


def discrete_hazard_survival_nll(
    hazard_logits: torch.Tensor,
    first_crossing_bin: torch.Tensor,
    at_risk_mask: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """First-crossing discrete hazard/survival negative log-likelihood.

    For an event in bin ``k``, this is ``-log(h_k)`` plus survival terms
    ``-log(1-h_j)`` for all observed bins ``j < k``.  A censored path contains
    survival terms for every observed at-risk bin.  Samples with no at-risk bin
    contribute zero and are excluded from the mean denominator.
    """

    if hazard_logits.ndim != 2 or hazard_logits.shape[1] != len(FIXED_LEADS):
        raise ValueError(
            f"hazard_logits must have shape [B,{len(FIXED_LEADS)}]"
        )
    batch = hazard_logits.shape[0]
    first_crossing_bin = first_crossing_bin.to(
        device=hazard_logits.device, dtype=torch.long
    ).view(-1)
    at_risk_mask = at_risk_mask.to(
        device=hazard_logits.device, dtype=torch.bool
    )
    if first_crossing_bin.shape != (batch,) or at_risk_mask.shape != hazard_logits.shape:
        raise ValueError("hazard target shapes must align with hazard_logits")
    if bool(((first_crossing_bin < -1) | (first_crossing_bin >= len(FIXED_LEADS))).any().item()):
        raise ValueError("first_crossing_bin must be -1 or a fixed-lead bin index")

    event_target = torch.zeros_like(hazard_logits)
    observed_event = first_crossing_bin >= 0
    if bool(observed_event.any().item()):
        rows = torch.arange(batch, device=hazard_logits.device)[observed_event]
        columns = first_crossing_bin[observed_event]
        if not bool(at_risk_mask[rows, columns].all().item()):
            raise ValueError("the first-crossing bin must be marked at risk")
        event_target[rows, columns] = 1.0

    per_bin = F.binary_cross_entropy_with_logits(
        hazard_logits, event_target, reduction="none"
    )
    per_sample = (per_bin * at_risk_mask.to(per_bin.dtype)).sum(dim=1)
    observed_path = at_risk_mask.any(dim=1)

    if reduction == "none":
        return per_sample * observed_path.to(per_sample.dtype)
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be none, mean or sum")
    weights = observed_path.to(per_sample.dtype)
    if sample_weight is not None:
        sample_weight = sample_weight.to(
            device=per_sample.device, dtype=per_sample.dtype
        ).view(-1)
        if sample_weight.shape != (batch,):
            raise ValueError("sample_weight must have shape [B]")
        weights = weights * sample_weight
    weighted = per_sample * weights
    if reduction == "sum":
        return weighted.sum()
    if not bool((weights.sum() > 0).item()):
        return hazard_logits.sum() * 0.0
    return weighted.sum() / weights.sum()


def macro_balanced_phase_loss(
    phase_logits: torch.Tensor,
    phase_target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    element_weight: Optional[torch.Tensor] = None,
    ignore_index: int = PHASE_IGNORE_INDEX,
) -> torch.Tensor:
    """Cross entropy with equal mass for each phase present in the batch."""

    if phase_logits.ndim < 2 or phase_logits.shape[-1] != NUM_PHASES:
        raise ValueError(f"phase_logits must end with {NUM_PHASES} classes")
    expected_shape = phase_logits.shape[:-1]
    if phase_target.shape != expected_shape:
        raise ValueError("phase_target must match phase_logits without its class axis")
    logits = phase_logits.reshape(-1, NUM_PHASES)
    target = phase_target.to(device=logits.device, dtype=torch.long).reshape(-1)
    eligible = target != int(ignore_index)
    if valid_mask is not None:
        if valid_mask.shape != expected_shape:
            raise ValueError("valid_mask must match phase_target")
        eligible &= valid_mask.to(device=logits.device, dtype=torch.bool).reshape(-1)
    if bool((eligible & ((target < 0) | (target >= NUM_PHASES))).any().item()):
        raise ValueError("phase_target contains an invalid class index")

    safe_target = target.clone()
    safe_target[~eligible] = 0
    per_element = F.cross_entropy(logits, safe_target, reduction="none")
    weights = torch.ones_like(per_element)
    if element_weight is not None:
        if element_weight.shape != expected_shape:
            raise ValueError("element_weight must match phase_target")
        weights = element_weight.to(device=logits.device, dtype=logits.dtype).reshape(-1)
        if bool((weights < 0).any().item()):
            raise ValueError("element_weight must be non-negative")

    class_losses = []
    for phase_class in range(NUM_PHASES):
        mask = eligible & (target == phase_class) & (weights > 0)
        if bool(mask.any().item()):
            class_losses.append(
                (per_element[mask] * weights[mask]).sum() / weights[mask].sum()
            )
    if not class_losses:
        return phase_logits.sum() * 0.0
    return torch.stack(class_losses).mean()


def capacity_normalized_path_huber(
    predicted_path: torch.Tensor,
    target_path: torch.Tensor,
    valid_mask: torch.Tensor,
    capacity_kw: float = 30.1,
    delta: float = 1.0,
    element_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Huber loss on path errors expressed as a fraction of PV capacity."""

    if predicted_path.shape != target_path.shape:
        raise ValueError("predicted_path and target_path must have identical shapes")
    if not predicted_path.is_floating_point() or not target_path.is_floating_point():
        raise TypeError("path tensors must be floating point")
    if capacity_kw <= 0:
        raise ValueError("capacity_kw must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if valid_mask.shape == predicted_path.shape[:-1] and predicted_path.shape[-1:] == (1,):
        valid_mask = valid_mask.unsqueeze(-1)
    if valid_mask.shape != predicted_path.shape:
        raise ValueError("valid_mask must match the path shape")
    valid = valid_mask.to(device=predicted_path.device, dtype=torch.bool)
    valid &= torch.isfinite(predicted_path) & torch.isfinite(target_path)
    per_element = F.huber_loss(
        predicted_path / capacity_kw,
        target_path.to(predicted_path) / capacity_kw,
        reduction="none",
        delta=delta,
    )
    weights = valid.to(per_element.dtype)
    if element_weight is not None:
        if element_weight.shape == predicted_path.shape[:-1] and predicted_path.shape[-1:] == (1,):
            element_weight = element_weight.unsqueeze(-1)
        if element_weight.shape != predicted_path.shape:
            raise ValueError("element_weight must match the path shape")
        element_weight = element_weight.to(
            device=predicted_path.device, dtype=predicted_path.dtype
        )
        if bool((element_weight < 0).any().item()):
            raise ValueError("element_weight must be non-negative")
        weights = weights * element_weight
    weighted = per_element * weights
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    if reduction != "mean":
        raise ValueError("reduction must be none, mean or sum")
    if not bool((weights.sum() > 0).item()):
        return predicted_path.sum() * 0.0
    return weighted.sum() / weights.sum()


def fixed_lead_path_huber_loss(
    predicted_path: torch.Tensor,
    target_path: torch.Tensor,
    valid_mask: torch.Tensor,
    capacity_kw: float = 30.1,
    delta: float = 1.0,
    element_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Capacity-normalized Huber loss at ``FIXED_LEADS`` only."""

    predicted_values, predicted_valid = select_fixed_leads(predicted_path, valid_mask)
    target_values, target_valid = select_fixed_leads(target_path, valid_mask)
    lead_valid = predicted_valid & target_valid
    lead_weight = None
    if element_weight is not None:
        if element_weight.ndim == 3 and element_weight.shape[-1] == 1:
            element_weight = element_weight[..., 0]
        if element_weight.shape != valid_mask.shape[:2]:
            raise ValueError("element_weight must have shape [B,15] or [B,15,1]")
        lead_weight = element_weight.index_select(
            1, _lead_indices(element_weight.device)
        )
    return capacity_normalized_path_huber(
        predicted_values,
        target_values,
        lead_valid,
        capacity_kw=capacity_kw,
        delta=delta,
        element_weight=lead_weight,
    )


def window_inverse_weights(valid_mask: torch.Tensor) -> torch.Tensor:
    """Give every non-empty window total weight one across its valid positions."""

    valid = valid_mask.to(dtype=torch.bool)
    counts = valid.flatten(start_dim=1).sum(dim=1)
    view_shape = (valid.shape[0],) + (1,) * (valid.ndim - 1)
    denominator = counts.clamp_min(1).view(view_shape).to(torch.float32)
    return valid.to(torch.float32) / denominator


def episode_inverse_weights(
    episode_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Give each contiguous true episode total weight one.

    Invalid positions split episodes.  This is a local building block; callers
    choose whether an episode consists only of active states or also includes
    return states.
    """

    if episode_mask.ndim < 2:
        raise ValueError("episode_mask must have a batch and time dimension")
    active = episode_mask.to(dtype=torch.bool)
    if valid_mask is not None:
        if valid_mask.shape != episode_mask.shape:
            raise ValueError("valid_mask must match episode_mask")
        active &= valid_mask.to(device=active.device, dtype=torch.bool)
    flat = active.flatten(start_dim=1)
    weights = torch.zeros_like(flat, dtype=torch.float32)
    for row in range(flat.shape[0]):
        starts = flat[row] & ~F.pad(flat[row, :-1], (1, 0), value=False)
        episode_ids = starts.to(torch.long).cumsum(dim=0)
        for episode_id in episode_ids[flat[row]].unique():
            mask = flat[row] & (episode_ids == episode_id)
            weights[row, mask] = 1.0 / mask.sum().to(torch.float32)
    return weights.reshape(active.shape)


# Short aliases for integration code.
build_phase_targets = build_stanford_path_targets
path_huber_loss = capacity_normalized_path_huber
