"""Lazy catalog of the foundation models used by the experiment matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class FoundationModelSpec:
    """Describe one foundation model without importing its backend."""

    name: str
    model_id: str
    revision: str
    entrypoint: str
    last_layer_selector: str


MODEL_NAMES = ("Sundial", "TimeMoE")
RUN_MODES = ("zero_shot", "adapter", "full", "last_layer")

MODEL_SPECS: Dict[str, FoundationModelSpec] = {
    "Sundial": FoundationModelSpec(
        name="Sundial",
        model_id="thuml/sundial-base-128m",
        revision="3212e42564493f520593e5414af4367fc4b49226",
        entrypoint="foundation_models.backends:SundialBackend",
        last_layer_selector="flow_loss",
    ),
    "TimeMoE": FoundationModelSpec(
        name="TimeMoE",
        model_id="Maple728/TimeMoE-50M",
        revision="446753ee48ff3726d0606a81d0092d54acee995e",
        entrypoint="foundation_models.backends:TimeMoEBackend",
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
    "FoundationModelSpec",
    "MODEL_NAMES",
    "MODEL_SPECS",
    "RUN_MODES",
    "get_model_spec",
]
