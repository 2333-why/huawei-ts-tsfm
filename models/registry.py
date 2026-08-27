"""Lazy catalog of the supported foundation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


RUN_MODES = ("zero_shot", "adapter", "full", "last_layer")
DEFAULT_ADAPTER_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class FoundationModelSpec:
    """Describe one foundation model without importing its backend."""

    name: str
    model_id: str
    revision: str
    entrypoint: str
    last_layer_selector: str
    supported_modes: Tuple[str, ...] = RUN_MODES
    adapter_target_modules: Tuple[str, ...] = DEFAULT_ADAPTER_TARGET_MODULES


MODEL_NAMES = ("Sundial", "TimeMoE")

MODEL_SPECS: Dict[str, FoundationModelSpec] = {
    "Sundial": FoundationModelSpec(
        name="Sundial",
        model_id="thuml/sundial-base-128m",
        revision="3212e42564493f520593e5414af4367fc4b49226",
        entrypoint="models.Sundial:SundialBackend",
        last_layer_selector="flow_loss",
    ),
    "TimeMoE": FoundationModelSpec(
        name="TimeMoE",
        model_id="Maple728/TimeMoE-50M",
        revision="446753ee48ff3726d0606a81d0092d54acee995e",
        entrypoint="models.TimeMoE:TimeMoEBackend",
        last_layer_selector="lm_heads",
    ),
}


def get_model_spec(name: str) -> FoundationModelSpec:
    """Return the immutable specification for ``name``."""

    try:
        return MODEL_SPECS[name]
    except (KeyError, TypeError) as exc:
        available = ", ".join(MODEL_NAMES)
        raise ValueError(
            f"unknown foundation model {name!r}; available models: {available}"
        ) from exc


__all__ = [
    "DEFAULT_ADAPTER_TARGET_MODULES",
    "FoundationModelSpec",
    "MODEL_NAMES",
    "MODEL_SPECS",
    "RUN_MODES",
    "get_model_spec",
]
