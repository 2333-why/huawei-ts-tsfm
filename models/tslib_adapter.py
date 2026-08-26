"""Adapters for normalized power batches and the copied TSLib models.

The only model input is historical normalized power with shape
``[batch, seq_len, 1]``. Future targets are used by the loss/evaluation path,
never to construct model input.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Tuple

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class PowerBatch:
    """The power history, power target, and valid-target mask for one batch."""

    x: Tensor
    y: Tensor
    mask: Tensor


def _as_float_tensor(value: Any, *, name: str, device: torch.device) -> Tensor:
    try:
        return torch.as_tensor(value, device=device, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be convertible to a floating-point tensor") from exc


def _validate_history(x: Tensor) -> Tensor:
    if not torch.is_tensor(x):
        try:
            x = torch.as_tensor(x)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("x must be convertible to a tensor") from exc
    if x.ndim != 3:
        raise ValueError(
            f"expected x with shape [batch, seq_len, channels], got {tuple(x.shape)}"
        )
    if x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != 1:
        raise ValueError(
            "x must have positive batch/sequence dimensions and exactly one power channel, "
            f"got {tuple(x.shape)}"
        )
    return x


def _nonnegative_length(value: int, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _target_mask(batch: Mapping, target: Tensor, device: torch.device) -> Tensor:
    metadata = batch
    value = metadata.get("target_mask")
    if value is None:
        return torch.ones(target.shape[:2], dtype=torch.bool, device=device)

    mask = torch.as_tensor(value, device=device, dtype=torch.bool)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2 or tuple(mask.shape) != tuple(target.shape[:2]):
        raise ValueError(
            "target_mask must have shape "
            f"[batch, target_len] matching {tuple(target.shape[:2])}, got {tuple(mask.shape)}"
        )
    return mask


def power_only_batch(batch, device: torch.device) -> PowerBatch:
    """Extract normalized power tensors from a collated mapping.

    The dedicated dataset emits ``history``, ``target``, ``target_mask``, and
    ``issue_time_ns``. Only this mapping contract is accepted.
    """

    if not isinstance(batch, Mapping):
        raise ValueError("dataset batch must be a mapping")
    if "history" not in batch or "target" not in batch:
        raise ValueError("dataset batch mapping must contain history and target")
    raw_x, raw_y = batch["history"], batch["target"]

    batch_x = _as_float_tensor(raw_x, name="history", device=device)
    batch_y = _as_float_tensor(raw_y, name="target", device=device)
    if batch_x.ndim != 3 or batch_x.shape[-1] != 1:
        raise ValueError(
            "expected [B, seq_len, 1] history with one power channel, "
            f"got {tuple(batch_x.shape)}"
        )
    if batch_y.ndim != 3 or batch_y.shape[-1] != 1:
        raise ValueError(
            "expected [B, pred_len, 1] target with one power channel, "
            f"got {tuple(batch_y.shape)}"
        )
    if batch_x.shape[0] != batch_y.shape[0]:
        raise ValueError(
            "history and target batch sizes must match, got "
            f"{batch_x.shape[0]} and {batch_y.shape[0]}"
        )

    return PowerBatch(
        x=batch_x,
        y=batch_y,
        mask=_target_mask(batch, batch_y, device),
    )


def decoder_from_history(x: Tensor, label_len: int, pred_len: int) -> Tensor:
    """Build a decoder window from trailing history followed by zero targets."""

    x = _validate_history(x)
    label_len = _nonnegative_length(label_len, name="label_len", allow_zero=True)
    pred_len = _nonnegative_length(pred_len, name="pred_len")
    if label_len > x.shape[1]:
        raise ValueError(
            f"label_len={label_len} exceeds available history length {x.shape[1]}"
        )

    history = x[:, -label_len:, :] if label_len else x[:, :0, :]
    zeros = x.new_zeros((x.shape[0], pred_len, x.shape[2]))
    return torch.cat((history, zeros), dim=1)


def prepare_model_inputs(
    model_name: str, x: Tensor, label_len: int, pred_len: int
) -> Tuple[Tensor, Any, Tensor, Any]:
    """Prepare the four positional inputs used by TSLib forecasting models.

    The only source for both encoder and decoder values is ``x``. All models
    receive the same power-only positional input contract.
    """

    x = _validate_history(x)
    label_len = _nonnegative_length(label_len, name="label_len", allow_zero=True)
    pred_len = _nonnegative_length(pred_len, name="pred_len")

    x_dec = decoder_from_history(x, label_len, pred_len)
    return x, None, x_dec, None


def normalize_forecast(
    model_name: str, output: Any, batch_size: int, pred_len: int
) -> Tensor:
    """Normalize a model result to the strict ``[B, pred_len, 1]`` contract."""

    pred_len = _nonnegative_length(pred_len, name="pred_len")
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError(f"{model_name} returned an empty output tuple")
        output = output[0]
    if not torch.is_tensor(output):
        raise ValueError(f"{model_name} output must contain a tensor")
    if output.ndim != 3:
        raise ValueError(
            f"{model_name} output must be 3-D [batch, time, channels], "
            f"got {tuple(output.shape)}"
        )
    if output.shape[0] != batch_size:
        raise ValueError(
            f"{model_name} output batch size {output.shape[0]} does not match "
            f"expected batch size {batch_size}"
        )
    if output.shape[-1] != 1:
        raise ValueError(
            f"{model_name} output must have one output channel, got {output.shape[-1]}"
        )
    if output.shape[1] < pred_len:
        raise ValueError(
            f"{model_name} output must contain at least {pred_len} time steps, "
            f"got {output.shape[1]}"
        )
    return output[:, -pred_len:, :]


def forward_power_model(
    model_name: str, model, x: Tensor, label_len: int, pred_len: int
) -> Tensor:
    """Run one model with power-only inputs and normalize its forecast."""

    x = _validate_history(x)
    model_inputs = prepare_model_inputs(model_name, x, label_len, pred_len)
    output = model(*model_inputs)
    return normalize_forecast(model_name, output, x.shape[0], pred_len)


__all__ = [
    "PowerBatch",
    "decoder_from_history",
    "forward_power_model",
    "normalize_forecast",
    "power_only_batch",
    "prepare_model_inputs",
]
