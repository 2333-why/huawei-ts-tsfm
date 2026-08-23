"""Lazy catalog of forecasting models from Time-Series-Library.

The catalog is intentionally independent from the model implementations.  A
caller can enumerate every supported model without importing optional model
dependencies (notably ``mamba_ssm``).
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelSpec:
    """Describe how to import a model implementation."""

    module: str
    constructor: str = "Model"


SELECTED_MODEL_NAMES: tuple[str, ...] = tuple(sorted(
    (
        "Autoformer",
        "Crossformer",
        "DLinear",
        "ETSformer",
        "FEDformer",
        "FiLM",
        "FreTS",
        "Informer",
        "Koopa",
        "LightTS",
        "MICN",
        "MSGNet",
        "Mamba",
        "MambaSimple",
        "MultiPatchFormer",
        "Nonstationary_Transformer",
        "PAttn",
        "PatchTST",
        "Pyraformer",
        "Reformer",
        "SCINet",
        "SegRNN",
        "TSMixer",
        "TemporalFusionTransformer",
        "TiDE",
        "TimeFilter",
        "TimeMixer",
        "TimeXer",
        "TimesNet",
        "Transformer",
        "WPMixer",
        "iTransformer",
    )
))


EXCLUDED_MODELS: dict[str, str] = {
    "Chronos": "Excluded foundation time-series model.",
    "Chronos2": "Excluded foundation time-series model.",
    "KANAD": "Excluded because the source implements anomaly detection, not forecasting.",
    "Sundial": "Excluded foundation time-series model.",
    "TimeMoE": "Excluded foundation time-series model.",
    "TiRex": "Excluded foundation time-series model.",
    "TimesFM": "Excluded foundation time-series model.",
}


MODEL_SPECS: dict[str, ModelSpec] = {
    name: ModelSpec(module=f"models.{name}")
    for name in SELECTED_MODEL_NAMES
}


def _missing_dependency(exc: ImportError) -> str:
    """Return the most useful package/module name from an import failure."""

    return getattr(exc, "name", None) or str(exc)


def load_model_class(name: str) -> type[torch.nn.Module]:
    """Import and return the configured constructor for ``name``.

    Model modules are imported only when this function is called.  Unknown
    names fail early with the complete valid-name list, while import failures
    retain the original exception as their cause and identify the model and
    missing dependency.
    """

    try:
        spec = MODEL_SPECS[name]
    except KeyError:
        valid_names = ", ".join(SELECTED_MODEL_NAMES)
        raise KeyError(
            f"Unknown model {name!r}; valid names: {valid_names}"
        ) from None

    try:
        module = importlib.import_module(spec.module)
    except ImportError as exc:
        dependency = _missing_dependency(exc)
        raise ImportError(
            f"Unable to load model {name!r}: missing dependency/package "
            f"{dependency!r}. Install it before using this model."
        ) from exc

    try:
        return getattr(module, spec.constructor)
    except AttributeError as exc:
        raise ImportError(
            f"Unable to load model {name!r}: module {spec.module!r} does not "
            f"define constructor {spec.constructor!r}."
        ) from exc


class _LazyModelMapping(Mapping):
    """Read-only compatibility mapping that imports models on lookup."""

    def __getitem__(self, name: str) -> type[torch.nn.Module]:
        return load_model_class(name)

    def __iter__(self) -> Iterator[str]:
        return iter(SELECTED_MODEL_NAMES)

    def __len__(self) -> int:
        return len(SELECTED_MODEL_NAMES)

    def __contains__(self, name: object) -> bool:
        return name in MODEL_SPECS


PURE_TIME_SERIES_MODELS: Mapping[str, type[torch.nn.Module]] = _LazyModelMapping()


__all__ = [
    "EXCLUDED_MODELS",
    "MODEL_SPECS",
    "ModelSpec",
    "PURE_TIME_SERIES_MODELS",
    "SELECTED_MODEL_NAMES",
    "load_model_class",
]
