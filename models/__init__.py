import importlib

from .tslib_registry import (
    EXCLUDED_MODELS,
    MODEL_SPECS,
    PURE_TIME_SERIES_MODELS,
    SELECTED_MODEL_NAMES,
    ModelSpec,
    load_model_class,
)


def __getattr__(name):
    """Load legacy model modules only when a caller requests them."""

    if name in {
        "DLinear", "PatchTST", "iTransformer", "TimesNet", "MTS_31", "MTS_31F",
    }:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DLinear",
    "PatchTST",
    "iTransformer",
    "TimesNet",
    "MTS_31",
    "MTS_31F",
    "EXCLUDED_MODELS",
    "MODEL_SPECS",
    "ModelSpec",
    "PURE_TIME_SERIES_MODELS",
    "SELECTED_MODEL_NAMES",
    "load_model_class",
]
