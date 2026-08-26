"""Power-only input and output adapters for the copied TSLib models.

The dataset batches contain several modalities, but a forecasting model in
this integration receives only the normalized power history.  In particular,
the model-input helpers below deliberately have no target argument: future
targets are used by the loss/evaluation path, never to construct model input.
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
    if x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError(
            f"x must have positive batch, sequence, and channel dimensions, got {tuple(x.shape)}"
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


def _target_mask(batch, target: Tensor, device: torch.device) -> Tensor:
    if isinstance(batch, Mapping):
        metadata = batch
    else:
        metadata = batch[9] if len(batch) > 9 and isinstance(batch[9], Mapping) else {}
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
    ``issue_time_ns``.  The legacy tuple layout remains accepted for focused
    compatibility tests, but no tuple-only data is created by the new loader.
    """

    if isinstance(batch, Mapping):
        if "history" not in batch or "target" not in batch:
            raise ValueError("dataset batch must contain history and target")
        raw_x, raw_y = batch["history"], batch["target"]
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        raw_x, raw_y = batch[0], batch[1]
    else:
        raise ValueError("dataset batch must contain at least seq_x and seq_y")

    batch_x = _as_float_tensor(raw_x, name="history", device=device)
    batch_y = _as_float_tensor(raw_y, name="target", device=device)
    if batch_x.ndim != 3 or batch_x.shape[-1] < 1:
        raise ValueError(
            f"expected [B, seq_len, channels] input, got {tuple(batch_x.shape)}"
        )
    if batch_y.ndim != 3 or batch_y.shape[-1] < 1:
        raise ValueError(
            f"expected [B, pred_len, channels] target, got {tuple(batch_y.shape)}"
        )
    if batch_x.shape[0] != batch_y.shape[0]:
        raise ValueError(
            "history and target batch sizes must match, got "
            f"{batch_x.shape[0]} and {batch_y.shape[0]}"
        )

    return PowerBatch(
        x=batch_x[..., :1],
        y=batch_y[..., :1],
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

    The only source for both encoder and decoder values is ``x``.  For the
    two models with structural mark/padding requirements, all synthetic
    compatibility values are zeros or repeated *historical* power values.
    """

    x = _validate_history(x)
    label_len = _nonnegative_length(label_len, name="label_len", allow_zero=True)
    pred_len = _nonnegative_length(pred_len, name="pred_len")

    # Enforce the power-only contract even when a caller bypasses
    # ``power_only_batch`` and supplies extra historical channels directly.
    x_model = x if x.shape[-1] == 1 else x[..., :1]
    if model_name == "MultiPatchFormer":
        model_seq_len = 32
        if x_model.shape[1] < model_seq_len:
            pad_len = model_seq_len - x_model.shape[1]
            left_pad = x_model[:, :1, :].expand(-1, pad_len, -1)
            x_model = torch.cat((left_pad, x_model), dim=1)
        elif x_model.shape[1] > model_seq_len:
            x_model = x_model[:, -model_seq_len:, :]
        label_len = min(label_len, model_seq_len)

    x_dec = decoder_from_history(x_model, label_len, pred_len)

    if model_name == "TemporalFusionTransformer":
        x_mark_enc = x_model.new_zeros((x_model.shape[0], x_model.shape[1], 4))
        x_mark_dec = x_model.new_zeros((x_model.shape[0], label_len + pred_len, 4))
        return x_model, x_mark_enc, x_dec, x_mark_dec

    return x_model, None, x_dec, None


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
