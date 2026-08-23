"""Centralized configuration and construction for the pure TSLib models.

The copied model implementations come from a command-line project and expect
an unusually broad ``configs`` namespace.  This module is the single place
where the power-only contract supplies those fields and handles the few
constructor differences that cannot be expressed by ``Model(configs)``.
"""

from __future__ import annotations

from itertools import islice
from math import gcd
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .tslib_registry import load_model_class


def _int_arg(args: Any, name: str, default: int) -> int:
    return int(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float) -> float:
    return float(getattr(args, name, default))


def _positive_odd(value: Any, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value if value % 2 else value + 1


def _positive(value: Any, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _device(value: Any) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def build_model_config(args: Any, dataset: Any) -> SimpleNamespace:
    """Build the complete common configuration expected by copied models."""

    seq_len = _positive(dataset.seq_len, "dataset.seq_len")
    pred_len = _positive(dataset.pred_len, "dataset.pred_len")
    patch_len = min(_positive(getattr(args, "patch_len", seq_len), "patch_len"), seq_len)
    return SimpleNamespace(
        task_name="long_term_forecast",
        enc_in=1,
        dec_in=1,
        c_out=1,
        seq_len=seq_len,
        label_len=seq_len,
        pred_len=pred_len,
        d_model=_int_arg(args, "d_model", 64),
        n_heads=_int_arg(args, "n_heads", 4),
        e_layers=_int_arg(args, "e_layers", 1),
        d_layers=_int_arg(args, "d_layers", 1),
        d_ff=_int_arg(args, "d_ff", 128),
        factor=_int_arg(args, "factor", 3),
        dropout=_float_arg(args, "dropout", 0.1),
        activation="gelu",
        embed="fixed",
        freq="h",
        moving_avg=_positive_odd(getattr(args, "moving_avg", 3), "moving_avg"),
        top_k=_int_arg(args, "top_k", 2),
        num_kernels=_int_arg(args, "num_kernels", 2),
        num_class=1,
        features="M",
        data="power_only",
        distil=True,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        # SegRNN reshapes both history and forecast into complete segments.
        # A common divisor keeps arbitrary point-count horizons valid while
        # retaining the original maximum segment size of eight.
        seg_len=gcd(gcd(seq_len, pred_len), 8),
        expand=2,
        d_conv=4,
        p_hidden_dims=[16, 16],
        p_hidden_layers=2,
        patch_len=patch_len,
        alpha=0.1,
        top_p=0.5,
        pos=1,
        node_dim=8,
        gcn_depth=2,
        propalpha=0.3,
        conv_channel=8,
        skip_channel=8,
        individual=False,
        ratio=0.5,
        subgraph_size=1,
        batch_size=getattr(args, "batch_size", 1),
        device=_device(getattr(args, "device", "cpu")),
        use_amp=False,
    )


def compute_koopa_mask(
    dataset: Any, alpha: float = 0.2, max_samples: int | None = None
) -> torch.Tensor:
    """Return dominant frequency indices from training power histories.

    The dataset item convention places historical time-series values at item
    zero.  Indexing channel zero before tensor conversion is deliberate: no
    exogenous channel, target window, or other modality is inspected.
    """

    alpha = float(alpha)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1]")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples < 1:
            raise ValueError("max_samples must be positive when provided")

    amplitudes = None
    count = 0
    items = dataset if max_samples is None else islice(dataset, max_samples)
    for item in items:
        if not isinstance(item, (tuple, list)) or not item:
            raise ValueError("dataset items must contain a historical power tensor")
        power_source = item[0]
        try:
            power_source = power_source[..., 0]
        except (IndexError, TypeError) as exc:
            raise ValueError("dataset item zero must contain channel-zero power") from exc
        power = torch.as_tensor(power_source, dtype=torch.float32)
        if power.ndim == 1:
            power = power.unsqueeze(0)
        if power.ndim != 2 or power.shape[1] < 1:
            raise ValueError(
                "historical power must have shape [batch, seq_len] or [seq_len]"
            )
        current = torch.fft.rfft(power, dim=1).abs().mean(dim=0)
        amplitudes = current if amplitudes is None else amplitudes + current
        count += 1

    if count == 0 or amplitudes is None:
        raise ValueError("cannot compute Koopa mask from an empty dataset")
    frequency_count = amplitudes.shape[0]
    top_count = max(1, min(frequency_count, int(frequency_count * alpha)))
    return amplitudes.topk(top_count).indices.to(dtype=torch.long)


def build_power_model(name: str, args: Any, dataset: Any) -> tuple[nn.Module, SimpleNamespace]:
    """Construct one selected model and return it with its effective config."""

    config = build_model_config(args, dataset)
    if name == "MultiPatchFormer":
        config.seq_len = 32
    if name == "Koopa":
        mask_limit = 1 if bool(getattr(args, "smoke", False)) else None
        config.mask_spectrum = compute_koopa_mask(dataset, max_samples=mask_limit)

    model_class = load_model_class(name)
    if name in {"PatchTST", "PAttn"}:
        patch_len = min(config.patch_len, config.seq_len)
        stride = max(1, min(
            _positive(getattr(args, "patch_stride", patch_len), "patch_stride"),
            patch_len,
        ))
        model = model_class(config, patch_len=patch_len, stride=stride)
    else:
        model = model_class(config)
    return model, config


__all__ = [
    "build_model_config",
    "build_power_model",
    "compute_koopa_mask",
]
