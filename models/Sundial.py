"""Sundial foundation-model backend."""

from __future__ import annotations

import torch

from .common import (
    _GenerateBackend,
    _denormalise,
    _extract_loss,
    _generated_tensor,
    _positive_config_int,
    _validate_loss,
)


class SundialBackend(_GenerateBackend):
    """Sundial backend using its native 20-sample generate API."""

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


__all__ = ["SundialBackend"]
