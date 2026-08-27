"""TimeMoE foundation-model backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List, Tuple

import torch
from torch import nn

from .common import (
    _GenerateBackend,
    _denormalise,
    _model_dtype,
    _positive_config_int,
    _repair_rotary_runtime_buffers,
    _validate_loss,
)


class TimeMoEBackend(_GenerateBackend):
    """TimeMoE backend cropping the final requested generated horizon."""

    model_name = "TimeMoE"

    def __init__(self, spec, device, revision=None):
        super().__init__(spec=spec, device=device, revision=revision)
        _repair_rotary_runtime_buffers(self.model)

    def predict(self, history: torch.Tensor, pred_len: int) -> torch.Tensor:
        with torch.inference_mode():
            history, pred_len, (normalized, means, stdev) = self._inputs(
                history, pred_len
            )
            context = normalized.to(device=self.device, dtype=_model_dtype(self.model, normalized.dtype))
            config = getattr(self.model, "config", None)
            if config is None:
                raise ValueError("TimeMoE model is missing its config")
            input_size = _positive_config_int(
                getattr(config, "input_size", None), "input_size"
            )
            if input_size != 1:
                raise ValueError("TimeMoE prediction currently requires input_size == 1")

            remaining = pred_len
            output_chunks = []
            while remaining:
                native = self.model(
                    input_ids=context,
                    max_horizon_length=remaining,
                    use_cache=False,
                    return_dict=True,
                )
                if isinstance(native, Mapping):
                    logits = native.get("logits")
                else:
                    logits = getattr(native, "logits", None)
                if not torch.is_tensor(logits):
                    raise ValueError("TimeMoE native model must return logits as a tensor")
                if logits.ndim != 3:
                    raise ValueError(
                        "TimeMoE native logits must have shape [batch, time, flat_horizon]"
                    )
                batch, time, flat_horizon = logits.shape
                if batch != context.shape[0] or time != context.shape[1]:
                    raise ValueError(
                        "TimeMoE native logits must preserve batch and context time dimensions"
                    )
                if flat_horizon % input_size:
                    raise ValueError(
                        "TimeMoE native logits horizon must be divisible by input_size"
                    )
                chunk = logits[:, -1, :].reshape(batch, -1, input_size)
                chunk = chunk[:, :remaining, :]
                if chunk.shape[1] == 0:
                    raise ValueError("TimeMoE native logits produced an empty forecast chunk")
                output_chunks.append(chunk)
                context = torch.cat(
                    (context, chunk.to(device=context.device, dtype=context.dtype)), dim=1
                )
                remaining -= int(chunk.shape[1])

            normalized_forecast = torch.cat(output_chunks, dim=1)
            return _denormalise(
                normalized_forecast, means, stdev, history.shape[0], pred_len
            )

    @staticmethod
    def _native_horizons(model: Any, config: Any) -> List[Tuple[int, Any]]:
        """Return ``(horizon, head)`` pairs from native model metadata."""

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


__all__ = ["TimeMoEBackend"]
