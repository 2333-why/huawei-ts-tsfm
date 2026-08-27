"""Hugging Face generate-based backends for the supported models.

The optional Transformers dependency is imported only when a backend is
constructed. Importing this module therefore remains safe for model listing
and command-line parsing.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Optional, Tuple

import torch

from .base import ensure_forecast_shape
from .registry import FoundationModelSpec


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
    stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
    normalized = centered.div(stdev)
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("normalized history must contain only finite values")
    return normalized, means, stdev


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


class _GenerateBackend:
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


class SundialBackend(_GenerateBackend):
    """Sundial backend using its 20-sample generate API."""

    model_name = "Sundial"

    def predict(self, history: torch.Tensor, pred_len: int) -> torch.Tensor:
        with torch.inference_mode():
            history, pred_len, (normalized, means, stdev) = self._inputs(
                history, pred_len
            )
            output = _generated_tensor(
                self.model.generate(
                    normalized[..., 0],
                    max_new_tokens=pred_len,
                    num_samples=20,
                )
            )
            if output.ndim < 2:
                raise ValueError(
                    "Sundial model.generate output must include a sample axis"
                )
            sampled = output.mean(dim=1)
            return _denormalise(sampled, means, stdev, history.shape[0], pred_len)


class TimeMoEBackend(_GenerateBackend):
    """TimeMoE backend cropping the final requested generated horizon."""

    model_name = "TimeMoE"

    def predict(self, history: torch.Tensor, pred_len: int) -> torch.Tensor:
        with torch.inference_mode():
            history, pred_len, (normalized, means, stdev) = self._inputs(
                history, pred_len
            )
            output = _generated_tensor(
                self.model.generate(
                    normalized[..., 0],
                    max_new_tokens=pred_len,
                )
            )
            if output.ndim < 2:
                raise ValueError(
                    "TimeMoE model.generate output must include a time axis"
                )
            cropped = output[:, -pred_len:]
            return _denormalise(cropped, means, stdev, history.shape[0], pred_len)


__all__ = ["SundialBackend", "TimeMoEBackend"]
