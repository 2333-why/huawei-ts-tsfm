"""TiRex zero-shot foundation-model backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch
from torch import nn

from .base import ensure_forecast_shape
from .common import _validate_history, _validate_pred_len
from .registry import FoundationModelSpec, validate_model_mode
from .trainability import configure_trainable


def _optional_tirex_loader() -> Any:
    try:
        from tirex import load_model
    except Exception as exc:
        raise ImportError(
            "TiRex requires the optional tirex-ts package and Python 3.10 or newer; "
            "install it in a compatible environment"
        ) from exc
    return load_model


def _mean_value(output: Any) -> Any:
    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        for key in ("mean", "point_forecast", "predictions"):
            if key in output:
                return output[key]
    if isinstance(output, tuple) and len(output) == 2:
        return output[1]
    value = getattr(output, "mean", None)
    if value is not None and not callable(value):
        return value
    return output


def _mean_tensor(output: Any, batch: int, pred_len: int) -> torch.Tensor:
    output = _mean_value(output)
    if isinstance(output, (list, tuple)):
        if len(output) != batch:
            raise ValueError(
                f"TiRex point forecast list must contain {batch} series, got {len(output)}"
            )
        rows = []
        for row in output:
            if not torch.is_tensor(row):
                raise ValueError("TiRex point forecast must use torch.Tensor values")
            if row.ndim == 1:
                rows.append(row)
            elif row.ndim == 2 and row.shape[0] == 1:
                rows.append(row[0])
            else:
                raise ValueError("TiRex point forecast row has an unsupported shape")
        output = torch.stack(rows, dim=0)
    if not torch.is_tensor(output):
        raise ValueError("TiRex point forecast must be a torch.Tensor")
    if output.ndim == 1 and pred_len == 1 and tuple(output.shape) == (batch,):
        output = output.unsqueeze(1)
    elif output.ndim == 1 and batch == 1:
        output = output.unsqueeze(0)
    elif output.ndim == 3 and tuple(output.shape[:2]) == (batch, 1):
        output = output[:, 0, :]
    elif output.ndim == 3 and output.shape[0] == batch and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 2 or output.shape[0] != batch:
        raise ValueError(
            "TiRex point forecast must have shape "
            f"[{batch}, horizon] or [{batch}, horizon, 1]"
        )
    if output.shape[1] < pred_len:
        raise ValueError(
            f"TiRex point forecast horizon {output.shape[1]} is shorter than {pred_len}"
        )
    return ensure_forecast_shape(output[:, :pred_len], batch=batch, pred_len=pred_len)


class TiRexBackend:
    """TiRex backend; the upstream API is intentionally zero-shot only."""

    model_name = "TiRex"

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
        load_model = _optional_tirex_loader()
        model = load_model(
            self.model_id,
            device=str(self.device),
            backend="torch",
            hf_kwargs={"revision": self.revision},
        )
        if not isinstance(model, nn.Module):
            raise ValueError("TiRex loader must return its native torch.nn.Module")
        self.model = model
        self.model.eval()

    def predict(self, history: torch.Tensor, pred_len: int) -> torch.Tensor:
        history = _validate_history(history)
        pred_len = _validate_pred_len(pred_len)
        if history.device != self.device:
            history = history.to(self.device)
        context = history[..., 0].contiguous()
        with torch.inference_mode():
            output = self.model.forecast(
                context=context,
                prediction_length=pred_len,
            )
        forecast = _mean_tensor(output, batch=int(history.shape[0]), pred_len=pred_len)
        return forecast.to(device=history.device)

    def training_loss(self, history: Any, target: Any, target_mask: Any) -> torch.Tensor:
        raise ValueError("TiRex is zero-shot only and does not support training_loss")

    def configure_trainable(self, mode: str, lora_settings: Any = None) -> Any:
        validate_model_mode(self.model_name, mode)
        if mode != "zero_shot":
            raise ValueError("TiRex is zero-shot only and does not support training modes")
        report = configure_trainable(self.model, self.spec, mode, lora_settings)
        self.model.eval()
        return report


__all__ = ["TiRexBackend"]
