"""Shared protocol and output-shape checks for foundation-model backends."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class FoundationBackend(Protocol):
    """Common interface consumed by the foundation-model runner."""

    model_name: str
    model_id: str
    model: torch.nn.Module

    def predict(self, history: torch.Tensor, pred_len: int) -> torch.Tensor:
        """Return a finite floating-point forecast with shape ``[B, H, 1]``."""

    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the model-native training loss."""

    def configure_trainable(self, mode: str, lora: Any) -> Any:
        """Apply a trainability policy and return its audit report."""

    def save(self, path: Path) -> None:
        """Save backend state to ``path``."""


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def ensure_forecast_shape(
    tensor: torch.Tensor, batch: int, pred_len: int
) -> torch.Tensor:
    """Validate and canonicalize a forecast to ``[batch, pred_len, 1]``.

    The only accepted input shapes are ``[B, H]`` and ``[B, H, 1]``. No
    coercion from arrays, integer tensors, or extra channels is performed.
    """

    batch = _positive_integer(batch, "batch")
    pred_len = _positive_integer(pred_len, "pred_len")
    if not torch.is_tensor(tensor):
        raise ValueError("forecast output must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise ValueError("forecast output must use a floating-point dtype")
    if tensor.ndim == 2:
        if tuple(tensor.shape) != (batch, pred_len):
            raise ValueError(
                "forecast output must have shape "
                f"[{batch}, {pred_len}] or [{batch}, {pred_len}, 1], "
                f"got {tuple(tensor.shape)}"
            )
        tensor = tensor.unsqueeze(-1)
    elif tensor.ndim == 3:
        if tuple(tensor.shape) != (batch, pred_len, 1):
            raise ValueError(
                "forecast output must have shape "
                f"[{batch}, {pred_len}, 1], got {tuple(tensor.shape)}"
            )
    else:
        raise ValueError(
            "forecast output must have rank 2 or 3, " f"got rank {tensor.ndim}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("forecast output must contain only finite values")
    return tensor


__all__ = ["FoundationBackend", "ensure_forecast_shape"]
