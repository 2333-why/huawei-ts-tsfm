"""Shared validation, normalization, and generated-backend helpers."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Optional, Tuple

import torch

from .base import ensure_forecast_shape
from .registry import FoundationModelSpec
from .trainability import configure_trainable


def _validate_history(history: Any) -> torch.Tensor:
    if not torch.is_tensor(history):
        raise ValueError("history must be a torch.Tensor")
    if not history.is_floating_point():
        raise ValueError("history must use a floating-point dtype")
    if history.ndim != 3 or history.shape[-1] != 1:
        raise ValueError("history must have shape [batch, history_points, 1]")
    if history.shape[0] < 1 or history.shape[1] < 1:
        raise ValueError("history must have positive batch and time dimensions")
    if not bool(torch.isfinite(history).all()):
        raise ValueError("history must contain only finite values")
    return history


def _validate_pred_len(pred_len: Any) -> int:
    if isinstance(pred_len, bool) or not isinstance(pred_len, Integral):
        raise ValueError("pred_len must be a positive integer")
    pred_len = int(pred_len)
    if pred_len <= 0:
        raise ValueError("pred_len must be a positive integer")
    return pred_len


def _normalise_window(
    history: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = history.mean(dim=1, keepdim=True).detach()
    centered = history.sub(means)
    stdev = torch.sqrt(
        torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
    ).detach()
    normalized = centered.div(stdev)
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("normalized history must contain only finite values")
    return normalized, means, stdev


def _model_dtype(model: Any, fallback: torch.dtype) -> torch.dtype:
    """Return a floating model dtype, falling back to the input dtype."""

    if hasattr(model, "parameters"):
        for parameter in model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
    if hasattr(model, "buffers"):
        for buffer in model.buffers():
            if buffer.is_floating_point():
                return buffer.dtype
    return fallback


def _validate_training_inputs(
    history: Any,
    target: Any,
    target_mask: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and normalize a training batch before touching the model."""

    history = _validate_history(history)
    if not torch.is_tensor(target):
        raise ValueError("target must be a torch.Tensor")
    if not target.is_floating_point():
        raise ValueError("target must use a floating-point dtype")
    if target.ndim != 3 or target.shape[-1] != 1:
        raise ValueError("target must have shape [batch, horizon, 1]")
    if target.shape[0] < 1 or target.shape[1] < 1:
        raise ValueError("target must have positive batch and horizon dimensions")
    if target.shape[0] != history.shape[0]:
        raise ValueError(
            "history and target must have the same positive batch dimension"
        )
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target must contain only finite values")

    if not torch.is_tensor(target_mask):
        raise ValueError("target_mask must be a torch.Tensor")
    if target_mask.ndim == 2:
        expected_mask_shape = (target.shape[0], target.shape[1])
        if tuple(target_mask.shape) != expected_mask_shape:
            raise ValueError(
                "target_mask must have shape [batch, horizon] or [batch, horizon, 1]"
            )
        target_mask = target_mask.unsqueeze(-1)
    elif target_mask.ndim == 3:
        expected_mask_shape = (target.shape[0], target.shape[1], 1)
        if tuple(target_mask.shape) != expected_mask_shape:
            raise ValueError(
                "target_mask must have shape [batch, horizon] or [batch, horizon, 1]"
            )
    else:
        raise ValueError(
            "target_mask must have shape [batch, horizon] or [batch, horizon, 1]"
        )
    if target_mask.is_complex():
        raise ValueError("target_mask must be boolean or real numeric")
    if not bool(torch.isfinite(target_mask).all()):
        raise ValueError("target_mask must contain only finite values")
    binary = (target_mask == 0) | (target_mask == 1)
    if not bool(binary.all()):
        raise ValueError("target_mask must contain only binary 0/1 values")
    valid = target_mask.to(device=device, dtype=torch.bool)
    if not bool(valid.any()):
        raise ValueError("target_mask contains no valid targets")

    history = history.to(device=device, dtype=dtype)
    target = target.to(device=device, dtype=dtype)
    normalized, means, stdev = _normalise_window(history)
    safe_target = torch.where(valid, target, means.expand_as(target))
    normalized_target = (safe_target - means) / stdev
    normalized_target = torch.where(
        valid, normalized_target, torch.zeros_like(normalized_target)
    )
    if not bool(torch.isfinite(normalized_target).all()):
        raise ValueError("normalized target must contain only finite values")
    return normalized, normalized_target, valid


def _extract_loss(output: Any) -> Any:
    if isinstance(output, Mapping):
        return output.get("loss")
    return getattr(output, "loss", None)


def _validate_loss(loss: Any) -> torch.Tensor:
    if not torch.is_tensor(loss):
        raise ValueError("native training loss must be a torch.Tensor")
    if loss.ndim != 0:
        raise ValueError("native training loss must be a scalar tensor")
    if not bool(torch.isfinite(loss).all()):
        raise ValueError("native training loss must be finite")
    if not loss.requires_grad:
        raise ValueError("native training loss must require gradients")
    return loss


def _positive_config_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"model config {name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"model config {name} must be a positive integer")
    return value


def _generated_tensor(output: Any) -> torch.Tensor:
    if not torch.is_tensor(output):
        raise ValueError("model.generate must return a torch.Tensor")
    return output


def _denormalise(
    normalized_forecast: torch.Tensor,
    means: torch.Tensor,
    stdev: torch.Tensor,
    batch: int,
    pred_len: int,
) -> torch.Tensor:
    normalized_forecast = ensure_forecast_shape(
        normalized_forecast, batch=batch, pred_len=pred_len
    )
    forecast = normalized_forecast * stdev[:, :1, :] + means[:, :1, :]
    return ensure_forecast_shape(forecast, batch=batch, pred_len=pred_len)


def _load_pretrained(model_id: str, revision: str) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "foundation-model backends require Transformers; install the optional "
            "dependency before constructing a backend"
        ) from exc
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )


class _GenerateBackend:
    """Shared construction, validation, and trainability for generate APIs."""

    model_name = ""

    def __init__(
        self,
        spec: FoundationModelSpec,
        device: Any,
        revision: Optional[str] = None,
    ) -> None:
        self.spec = spec
        self.model_name = spec.name
        self.model_id = spec.model_id
        self.revision = spec.revision if revision is None else revision
        self.device = torch.device(device)
        model = _load_pretrained(self.model_id, self.revision)
        moved = model.to(device)
        self.model = model if moved is None else moved
        self.model.eval()

    def _inputs(self, history: Any, pred_len: Any):
        history = _validate_history(history)
        pred_len = _validate_pred_len(pred_len)
        if history.device != self.device:
            history = history.to(self.device)
        return history, pred_len, _normalise_window(history)

    def configure_trainable(self, mode: str, lora_settings: Any = None) -> Any:
        """Apply the shared trainability policy and set the model phase."""

        report = configure_trainable(
            self.model,
            self.spec,
            mode,
            lora_settings,
        )
        if mode == "zero_shot":
            self.model.eval()
        elif mode in ("adapter", "full", "last_layer"):
            self.model.train()
        return report

    def _training_inputs(
        self,
        history: Any,
        target: Any,
        target_mask: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_tensor = _validate_history(history)
        dtype = _model_dtype(self.model, history_tensor.dtype)
        return _validate_training_inputs(
            history_tensor,
            target,
            target_mask,
            self.device,
            dtype,
        )


__all__ = [
    "_GenerateBackend",
    "_denormalise",
    "_extract_loss",
    "_generated_tensor",
    "_load_pretrained",
    "_model_dtype",
    "_normalise_window",
    "_positive_config_int",
    "_validate_history",
    "_validate_loss",
    "_validate_pred_len",
    "_validate_training_inputs",
]
