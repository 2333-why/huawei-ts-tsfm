"""Lightweight public interfaces for the TSFM model catalog.

The registry and task matrix stay importable without importing torch, PEFT, or
any optional model implementation. Runtime helpers are resolved on demand.
"""

import importlib

from .registry import (
    DEFAULT_ADAPTER_TARGET_MODULES,
    MODEL_NAMES,
    MODEL_SPECS,
    RUN_MODES,
    FoundationModelSpec,
    get_model_spec,
)
from .tasks import ExperimentTask, iter_experiment_tasks


_LAZY_EXPORTS = {
    "ADAPTER_TARGET_MODULES": ("models.trainability", "ADAPTER_TARGET_MODULES"),
    "FoundationBackend": ("models.base", "FoundationBackend"),
    "LoraSettings": ("models.trainability", "LoraSettings"),
    "TrainabilityReport": ("models.trainability", "TrainabilityReport"),
    "build_backend": ("models.factory", "build_backend"),
    "configure_trainable": ("models.trainability", "configure_trainable"),
    "ensure_forecast_shape": ("models.base", "ensure_forecast_shape"),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "ADAPTER_TARGET_MODULES",
    "DEFAULT_ADAPTER_TARGET_MODULES",
    "ExperimentTask",
    "FoundationBackend",
    "FoundationModelSpec",
    "LoraSettings",
    "MODEL_NAMES",
    "MODEL_SPECS",
    "RUN_MODES",
    "TrainabilityReport",
    "build_backend",
    "configure_trainable",
    "ensure_forecast_shape",
    "get_model_spec",
    "iter_experiment_tasks",
]
