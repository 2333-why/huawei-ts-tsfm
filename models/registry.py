"""Lazy catalog of the supported foundation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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
    last_layer_selector: Optional[str]
    supported_modes: Tuple[str, ...] = RUN_MODES
    adapter_target_modules: Tuple[str, ...] = DEFAULT_ADAPTER_TARGET_MODULES


MODEL_NAMES = ("Sundial", "TimeMoE", "Chronos2", "TiRex", "TimesFM")

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
    "Chronos2": FoundationModelSpec(
        name="Chronos2",
        model_id="amazon/chronos-2",
        revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
        entrypoint="models.Chronos2:Chronos2Backend",
        last_layer_selector="output_patch_embedding",
        adapter_target_modules=(
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ),
    ),
    "TiRex": FoundationModelSpec(
        name="TiRex",
        model_id="NX-AI/TiRex",
        revision="63c740922493f5fbe60b277609ec62babfba2762",
        entrypoint="models.TiRex:TiRexBackend",
        last_layer_selector=None,
        supported_modes=("zero_shot",),
    ),
    "TimesFM": FoundationModelSpec(
        name="TimesFM",
        model_id="google/timesfm-2.5-200m-transformers",
        revision="5a9806b9b291fad9233b5249d88263f1846304d3",
        entrypoint="models.TimesFM:TimesFMBackend",
        last_layer_selector="output_projection_point",
        adapter_target_modules=("all-linear",),
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


def get_model_modes(name: str) -> Tuple[str, ...]:
    """Return the immutable run-mode capability tuple for ``name``."""

    return get_model_spec(name).supported_modes


def validate_model_mode(name: str, mode: str) -> None:
    """Reject unsupported model/mode combinations before backend loading."""

    spec = get_model_spec(name)
    if mode not in spec.supported_modes:
        available = ", ".join(spec.supported_modes)
        raise ValueError(
            f"model {name!r} does not support mode {mode!r}; "
            f"valid modes: {available}"
        )


__all__ = [
    "DEFAULT_ADAPTER_TARGET_MODULES",
    "FoundationModelSpec",
    "MODEL_NAMES",
    "MODEL_SPECS",
    "RUN_MODES",
    "get_model_modes",
    "get_model_spec",
    "validate_model_mode",
]
