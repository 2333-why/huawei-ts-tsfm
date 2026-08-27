"""Hugging Face generate-based backends for the supported models.

The optional Transformers dependency is imported only when a backend is
constructed. Importing this module therefore remains safe for model listing
and command-line parsing.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, List, Optional, Tuple

import torch
from torch import nn

from .base import ensure_forecast_shape
from .registry import FoundationModelSpec
from .trainability import configure_trainable


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

    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized, normalized_target, valid = self._training_inputs(
            history, target, target_mask
        )
        config = getattr(self.model, "config", None)
        if config is None:
            raise ValueError("Sundial model is missing its config")
        patch_len = _positive_config_int(
            getattr(config, "input_token_len", None), "input_token_len"
        )
        output_token_lens = getattr(config, "output_token_lens", None)
        if output_token_lens is None:
            raise ValueError("Sundial model config is missing output_token_lens")
        try:
            output_len = output_token_lens[-1]
        except (IndexError, KeyError, TypeError):
            raise ValueError("Sundial model config output_token_lens is empty")
        output_len = _positive_config_int(output_len, "output_token_lens[-1]")
        horizon = int(normalized_target.shape[1])
        if horizon > output_len:
            raise ValueError(
                f"Sundial target horizon {horizon} exceeds native output horizon {output_len}"
            )

        history_len = int(normalized.shape[1])
        token_count = (history_len + patch_len - 1) // patch_len
        label_offset = (token_count - 1) * patch_len
        losses = []
        for row in range(int(normalized.shape[0])):
            if not bool(valid[row].any()):
                continue
            input_ids = normalized[row : row + 1, :, 0].contiguous()
            labels = torch.zeros(
                (1, label_offset + output_len),
                device=self.device,
                dtype=normalized.dtype,
            )
            labels[:, label_offset : label_offset + horizon] = normalized_target[
                row : row + 1, :, 0
            ]
            loss_masks = torch.zeros(
                (1, token_count), device=self.device, dtype=normalized.dtype
            )
            loss_masks[:, -1] = 1.0
            mask_y = torch.zeros(
                (1, output_len), device=self.device, dtype=normalized.dtype
            )
            mask_y[:, :horizon] = valid[row : row + 1, :, 0].to(normalized.dtype)
            output = self.model(
                input_ids=input_ids,
                labels=labels,
                loss_masks=loss_masks,
                mask_y=mask_y,
                use_cache=False,
                return_dict=True,
                revin=False,
            )
            losses.append(_validate_loss(_extract_loss(output)))
        if not losses:
            raise ValueError("Sundial target_mask contains no valid targets")
        return _validate_loss(torch.stack(losses).mean())


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

    @staticmethod
    def _native_horizons(model: Any, config: Any) -> List[Tuple[int, Any]]:
        """Return ``(horizon, head)`` pairs from the native model metadata."""

        heads = getattr(model, "lm_heads", None)
        if heads is None:
            raise ValueError("TimeMoE model is missing native lm_heads")

        heads_are_mapping = isinstance(heads, Mapping) or isinstance(heads, nn.ModuleDict)
        if heads_are_mapping:
            head_items = list(heads.items())
        else:
            try:
                head_items = list(enumerate(heads))
            except TypeError as exc:
                raise ValueError("TimeMoE lm_heads must be a sequence or mapping") from exc
        if not head_items:
            raise ValueError("TimeMoE model has no native lm_heads")

        configured = getattr(config, "horizon_lengths", None)
        if configured is not None:
            try:
                configured_values = [
                    _positive_config_int(value, "horizon_lengths") for value in configured
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError("TimeMoE config horizon_lengths must be numeric") from exc
            if len(configured_values) != len(head_items):
                raise ValueError(
                    "TimeMoE horizon_lengths and lm_heads must have matching lengths"
                )
            if heads_are_mapping:
                by_key = dict(head_items)
                pairs = []
                for index, horizon in enumerate(configured_values):
                    head = by_key.get(horizon)
                    if head is None:
                        head = by_key.get(str(horizon))
                    if head is None:
                        head = by_key.get(index, by_key.get(str(index)))
                    pairs.append((horizon, head))
                return pairs
            return [
                (horizon, head_items[index][1])
                for index, horizon in enumerate(configured_values)
            ]

        horizon_map = getattr(config, "horizon_length_map", None)
        if horizon_map is None:
            horizon_map = getattr(model, "horizon_length_map", None)
        if isinstance(horizon_map, Mapping) and horizon_map:
            try:
                map_items = [(int(key), int(value)) for key, value in horizon_map.items()]
            except (TypeError, ValueError) as exc:
                raise ValueError("TimeMoE horizon_length_map must be numeric") from exc
            count = len(head_items)
            keys = [key for key, _ in map_items]
            values = [value for _, value in map_items]
            # The native map is normally ``horizon -> head index``.  Accept
            # the inverse ``head index -> horizon`` as well for lightweight
            # model fakes and future checkpoint metadata.
            values_are_indices = all(0 <= value < count for value in values)
            keys_are_indices = all(0 <= key < count for key in keys)
            if values_are_indices and not keys_are_indices:
                orientation = "horizon_to_index"
            elif keys_are_indices and not values_are_indices:
                orientation = "index_to_horizon"
            elif keys_are_indices and set(keys) == set(range(count)):
                orientation = "index_to_horizon"
            else:
                orientation = "horizon_to_index"
            pairs = []
            for key_int, value_int in map_items:
                if orientation == "horizon_to_index":
                    horizon, head_index = key_int, value_int
                else:
                    horizon, head_index = value_int, key_int
                if heads_are_mapping:
                    head = heads.get(head_index)
                    if head is None:
                        head = heads.get(horizon)
                    if head is None:
                        head = heads.get(str(head_index), heads.get(str(horizon)))
                elif 0 <= head_index < len(head_items):
                    head = head_items[head_index][1]
                else:
                    head = None
                if head is not None:
                    pairs.append((horizon, head))
            if pairs:
                return pairs
            raise ValueError("TimeMoE horizon_length_map did not resolve any native lm_heads")

        # A mapping keyed by the native horizon is also a valid source when
        # config metadata is absent.
        if heads_are_mapping:
            try:
                return [(int(key), head) for key, head in head_items]
            except (TypeError, ValueError) as exc:
                raise ValueError("TimeMoE lm_heads mapping keys must be horizons") from exc
        raise ValueError(
            "TimeMoE model config must define horizon_lengths or horizon_length_map"
        )

    @staticmethod
    def _hidden_state(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            hidden = output
        elif isinstance(output, Mapping):
            hidden = output.get("last_hidden_state")
        else:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, (tuple, list)) and output:
                hidden = output[0]
        if not torch.is_tensor(hidden):
            raise ValueError("TimeMoE backbone must return a last_hidden_state tensor")
        if hidden.ndim != 3:
            raise ValueError("TimeMoE backbone hidden state must have shape [B, T, D]")
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError("TimeMoE backbone hidden state must be finite")
        return hidden

    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized, normalized_target, valid = self._training_inputs(
            history, target, target_mask
        )
        config = getattr(self.model, "config", None)
        if config is None:
            raise ValueError("TimeMoE model is missing its config")
        input_size = _positive_config_int(
            getattr(config, "input_size", None), "input_size"
        )
        if input_size != 1:
            raise ValueError("TimeMoE training_loss currently requires input_size == 1")

        horizons = self._native_horizons(self.model, config)
        requested_horizon = int(normalized_target.shape[1])
        candidates = [entry for entry in horizons if entry[0] >= requested_horizon]
        if not candidates:
            max_horizon = max(entry[0] for entry in horizons)
            raise ValueError(
                f"TimeMoE target horizon {requested_horizon} exceeds native maximum {max_horizon}"
            )
        native_horizon, head = min(candidates, key=lambda entry: entry[0])
        if head is None or not callable(head):
            raise ValueError("TimeMoE selected native lm_head is not callable")

        backbone = getattr(self.model, "model", None)
        if backbone is None or not callable(backbone):
            raise ValueError("TimeMoE model is missing its native backbone model")
        input_ids = normalized.contiguous()
        try:
            backbone_output = backbone(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
        except TypeError as first_error:
            # Tiny compatible backbones may expose only a positional input;
            # preserve the native call when the full Transformers signature is
            # available and provide a useful fallback for such fakes.
            try:
                backbone_output = backbone(input_ids)
            except TypeError:
                raise first_error
        hidden = self._hidden_state(backbone_output)
        if hidden.shape[0] != normalized.shape[0] or hidden.shape[1] != normalized.shape[1]:
            raise ValueError(
                "TimeMoE backbone hidden state must preserve batch and history dimensions"
            )
        final_hidden = hidden[:, -1, :].contiguous()
        native_output = head(final_hidden)
        if not torch.is_tensor(native_output):
            raise ValueError("TimeMoE native lm_head must return a tensor")
        if not native_output.is_floating_point():
            raise ValueError("TimeMoE native lm_head output must use a floating-point dtype")
        if not bool(torch.isfinite(native_output).all()):
            raise ValueError("TimeMoE native lm_head output must be finite")
        batch = int(normalized.shape[0])
        flat_output_size = native_horizon * input_size
        if native_output.ndim == 2 and tuple(native_output.shape) == (
            batch,
            flat_output_size,
        ):
            predictions = native_output.reshape(batch, native_horizon, input_size)
        elif native_output.ndim == 3 and tuple(native_output.shape) == (
            batch,
            1,
            flat_output_size,
        ):
            predictions = native_output.reshape(batch, native_horizon, input_size)
        elif native_output.ndim == 3 and tuple(native_output.shape) == (
            batch,
            native_horizon,
            input_size,
        ):
            predictions = native_output
        else:
            raise ValueError(
                "TimeMoE native lm_head output cannot be reshaped to "
                f"[{batch}, {native_horizon}, {input_size}]"
            )
        predictions = predictions[:, :requested_horizon, :]
        if tuple(predictions.shape) != tuple(normalized_target.shape):
            raise ValueError("TimeMoE native prediction shape does not match target")

        loss_function = getattr(self.model, "loss_function", None)
        if loss_function is None or not callable(loss_function):
            raise ValueError("TimeMoE model is missing its native loss_function")
        loss_values = loss_function(predictions, normalized_target)
        if not torch.is_tensor(loss_values):
            raise ValueError("TimeMoE native loss_function must return a tensor")
        if loss_values.ndim == 2 and tuple(loss_values.shape) == (
            batch,
            requested_horizon,
        ):
            loss_values = loss_values.unsqueeze(-1)
        elif loss_values.shape != normalized_target.shape:
            raise ValueError(
                "TimeMoE native loss_function output must match target shape"
            )
        if not bool(torch.isfinite(loss_values).all()):
            raise ValueError("TimeMoE native pointwise loss must be finite")
        valid_float = valid.to(dtype=loss_values.dtype)
        denominator = valid_float.sum()
        if not bool(denominator > 0):
            raise ValueError("target_mask contains no valid targets")
        loss = (loss_values * valid_float).sum() / denominator
        return _validate_loss(loss)


__all__ = ["SundialBackend", "TimeMoEBackend"]
