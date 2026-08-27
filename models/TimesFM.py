"""TimesFM 2.5 Transformers foundation-model backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch

from .base import ensure_forecast_shape
from .common import (
    _model_dtype,
    _validate_history,
    _validate_loss,
    _validate_pred_len,
    _validate_training_batch,
)
from .registry import FoundationModelSpec, validate_model_mode
from .trainability import configure_trainable


def _optional_timesfm_class() -> Any:
    try:
        from transformers import TimesFm2_5ModelForPrediction
    except Exception as exc:
        raise ImportError(
            "TimesFM requires Transformers with TimesFm2_5ModelForPrediction and "
            "Python 3.10 or newer; install a compatible optional dependency"
        ) from exc
    return TimesFm2_5ModelForPrediction


def _mean_value(output: Any) -> Any:
    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        for key in ("mean_predictions", "mean", "point_forecast"):
            if key in output:
                return output[key]
    for key in ("mean_predictions", "mean", "point_forecast"):
        value = getattr(output, key, None)
        if value is not None and not callable(value):
            return value
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def _mean_tensor(output: Any, batch: int) -> torch.Tensor:
    output = _mean_value(output)
    if not torch.is_tensor(output):
        raise ValueError("TimesFM mean_predictions must be a torch.Tensor")
    if output.ndim == 1 and batch == 1:
        output = output.unsqueeze(0)
    elif output.ndim == 3 and tuple(output.shape[:2]) == (batch, 1):
        output = output[:, 0, :]
    elif output.ndim == 3 and output.shape[0] == batch and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 2 or output.shape[0] != batch:
        raise ValueError(
            f"TimesFM mean_predictions must have shape [batch, horizon], got {tuple(output.shape)}"
        )
    return output


class TimesFMBackend:
    """TimesFM backend using the differentiable Transformers forward path."""

    model_name = "TimesFM"

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
        model_class = _optional_timesfm_class()
        self.model = model_class.from_pretrained(
            self.model_id,
            revision=self.revision,
            device_map=str(self.device),
        )
        eval_method = getattr(self.model, "eval", None)
        if callable(eval_method):
            eval_method()

    def _past_values(self, history: torch.Tensor):
        return [history[index, :, 0].contiguous() for index in range(history.shape[0])]

    def _forward_mean(self, history: torch.Tensor) -> torch.Tensor:
        output = self.model(
            past_values=self._past_values(history),
            return_dict=True,
        )
        return _mean_tensor(output, batch=int(history.shape[0]))

    def predict(self, history: torch.Tensor, pred_len: int) -> torch.Tensor:
        history = _validate_history(history)
        pred_len = _validate_pred_len(pred_len)
        dtype = _model_dtype(self.model, history.dtype)
        history = history.to(device=self.device, dtype=dtype)
        with torch.inference_mode():
            output = self._forward_mean(history)
        if output.shape[1] < pred_len:
            raise ValueError(
                f"TimesFM mean_predictions horizon {output.shape[1]} is shorter than {pred_len}"
            )
        return ensure_forecast_shape(
            output[:, :pred_len], batch=int(history.shape[0]), pred_len=pred_len
        )

    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        history = _validate_history(history)
        dtype = _model_dtype(self.model, history.dtype)
        history, target, valid = _validate_training_batch(
            history, target, target_mask, self.device, dtype
        )
        output = self._forward_mean(history)
        horizon = int(target.shape[1])
        if output.shape[1] < horizon:
            raise ValueError(
                f"TimesFM mean_predictions horizon {output.shape[1]} is shorter than {horizon}"
            )
        predictions = output[:, :horizon]
        target_values = target[..., 0]
        valid_values = valid[..., 0].to(dtype=predictions.dtype)
        pointwise = (predictions - target_values).square()
        loss = (pointwise * valid_values).sum() / valid_values.sum()
        return _validate_loss(loss)

    def configure_trainable(self, mode: str, lora_settings: Any = None) -> Any:
        validate_model_mode(self.model_name, mode)
        report = configure_trainable(self.model, self.spec, mode, lora_settings)
        if mode == "zero_shot":
            self.model.eval()
        else:
            self.model.train()
        return report


__all__ = ["TimesFMBackend"]
