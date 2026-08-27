"""Chronos-2 foundation-model backend."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from typing import Any, Optional

import torch
from torch import nn

from .base import ensure_forecast_shape
from .common import (
    _extract_loss,
    _model_dtype,
    _positive_config_int,
    _validate_history,
    _validate_loss,
    _validate_pred_len,
    _validate_training_batch,
)
from .registry import FoundationModelSpec, validate_model_mode
from .trainability import configure_trainable


def _optional_chronos_pipeline() -> Any:
    try:
        from chronos import Chronos2Pipeline
    except Exception as exc:
        raise ImportError(
            "Chronos2 requires the optional chronos-forecasting package and "
            "Python 3.10 or newer; install it in a compatible environment"
        ) from exc
    return Chronos2Pipeline


def _point_value(output: Any) -> Any:
    """Extract a point forecast from a Chronos pipeline result."""

    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        for key in ("mean", "point_forecast", "predictions"):
            if key in output:
                return output[key]
    if isinstance(output, tuple) and len(output) == 2:
        # Chronos2Pipeline.predict_quantiles returns (quantiles, mean).
        return output[1]
    for key in ("mean", "mean_predictions"):
        value = getattr(output, key, None)
        if value is not None and not callable(value):
            return value
    return output


def _point_tensor(output: Any, batch: int, pred_len: int) -> torch.Tensor:
    output = _point_value(output)
    if isinstance(output, (list, tuple)):
        if len(output) != batch:
            raise ValueError(
                f"Chronos2 point forecast list must contain {batch} series, got {len(output)}"
            )
        rows = []
        for row in output:
            if not torch.is_tensor(row):
                raise ValueError("Chronos2 point forecast must use torch.Tensor values")
            if row.ndim == 1:
                value = row
            elif row.ndim == 2 and row.shape[0] == 1:
                value = row[0]
            elif row.ndim == 3 and row.shape[0] == 1 and row.shape[-1] == 1:
                value = row[0, :, 0]
            else:
                raise ValueError("Chronos2 point forecast row has an unsupported shape")
            rows.append(value)
        output = torch.stack(rows, dim=0)
    if not torch.is_tensor(output):
        raise ValueError("Chronos2 point forecast must be a torch.Tensor")
    if output.ndim == 1 and batch == 1:
        output = output.unsqueeze(0)
    elif output.ndim == 3 and tuple(output.shape[:2]) == (batch, 1):
        output = output[:, 0, :]
    elif output.ndim == 3 and output.shape[0] == batch and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 2 or output.shape[0] != batch:
        raise ValueError(
            "Chronos2 point forecast must have shape "
            f"[{batch}, horizon] or [{batch}, horizon, 1]"
        )
    if output.shape[1] < pred_len:
        raise ValueError(
            f"Chronos2 point forecast horizon {output.shape[1]} is shorter than {pred_len}"
        )
    output = output[:, :pred_len]
    return ensure_forecast_shape(output, batch=batch, pred_len=pred_len)


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


class Chronos2Backend:
    """Chronos-2 backend using the native pipeline and model loss."""

    model_name = "Chronos2"

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
        pipeline_class = _optional_chronos_pipeline()
        self.pipeline = pipeline_class.from_pretrained(
            self.model_id,
            revision=self.revision,
            device_map=str(self.device),
        )
        model = getattr(self.pipeline, "model", None)
        if not isinstance(model, nn.Module):
            raise ValueError("Chronos2 pipeline must expose its native nn.Module as model")
        self.model = model
        self.model.eval()

    def predict(self, history: torch.Tensor, pred_len: int) -> torch.Tensor:
        history = _validate_history(history)
        pred_len = _validate_pred_len(pred_len)
        if history.device != self.device:
            history = history.to(self.device)
        context = history[..., 0].unsqueeze(1).contiguous()
        with torch.inference_mode():
            predict_quantiles = getattr(self.pipeline, "predict_quantiles", None)
            if callable(predict_quantiles):
                output = predict_quantiles(context, prediction_length=pred_len)
            else:
                predict = getattr(self.pipeline, "predict", None)
                if not callable(predict):
                    raise ValueError("Chronos2 pipeline has no point prediction method")
                output = predict(context, prediction_length=pred_len)
        return _point_tensor(output, batch=int(history.shape[0]), pred_len=pred_len)

    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        history, target, valid = self._training_batch(history, target, target_mask)
        context = history[..., 0].contiguous()
        future_target = target[..., 0].contiguous()
        future_target_mask = valid[..., 0].contiguous()
        config = getattr(self.model, "chronos_config", None)
        if config is None:
            model_config = getattr(self.model, "config", None)
            config = _config_value(model_config, "chronos_config")
            if config is None:
                config = model_config
        if config is None:
            raise ValueError("Chronos2 model is missing chronos_config")
        patch_size = _positive_config_int(
            _config_value(config, "output_patch_size"), "output_patch_size"
        )
        num_output_patches = int(ceil(float(future_target.shape[1]) / patch_size))
        max_patches = _config_value(config, "max_output_patches")
        if max_patches is not None:
            max_patches = _positive_config_int(max_patches, "max_output_patches")
            if num_output_patches > max_patches:
                raise ValueError(
                    "Chronos2 target horizon exceeds the native max_output_patches capacity"
                )
        output = self.model(
            context=context,
            future_target=future_target,
            future_target_mask=future_target_mask,
            num_output_patches=num_output_patches,
        )
        return _validate_loss(_extract_loss(output))

    def _training_batch(self, history: Any, target: Any, target_mask: Any):
        history = _validate_history(history)
        dtype = _model_dtype(self.model, history.dtype)
        return _validate_training_batch(
            history, target, target_mask, self.device, dtype
        )

    def configure_trainable(self, mode: str, lora_settings: Any = None) -> Any:
        validate_model_mode(self.model_name, mode)
        report = configure_trainable(self.model, self.spec, mode, lora_settings)
        if mode == "zero_shot":
            self.model.eval()
        else:
            self.model.train()
        return report


__all__ = ["Chronos2Backend"]
