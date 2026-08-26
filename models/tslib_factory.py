"""Centralized configuration and construction for the retained TSLib models."""

from __future__ import annotations

from math import gcd
from types import SimpleNamespace
from typing import Any

from torch import nn

from .tslib_registry import load_model_class


def _int_arg(args: Any, name: str, default: int) -> int:
    return int(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float) -> float:
    return float(getattr(args, name, default))


def _positive(value: Any, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def build_model_config(args: Any, dataset: Any) -> SimpleNamespace:
    """Build the common configuration consumed by the retained models."""

    seq_len = _positive(dataset.seq_len, "dataset.seq_len")
    pred_len = _positive(dataset.pred_len, "dataset.pred_len")
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
        num_class=1,
        # SegRNN reshapes both history and forecast into complete segments.
        # A common divisor keeps arbitrary point-count horizons valid while
        # retaining the original maximum segment size of eight.
        seg_len=gcd(gcd(seq_len, pred_len), 8),
        channel_independence=1,
    )


def build_power_model(name: str, args: Any, dataset: Any) -> tuple[nn.Module, SimpleNamespace]:
    """Construct one selected model and return it with its effective config."""

    config = build_model_config(args, dataset)
    model_class = load_model_class(name)
    model = model_class(config)
    return model, config


__all__ = [
    "build_model_config",
    "build_power_model",
]
