"""Registry for the classical forecasting baselines."""

from __future__ import annotations

from typing import Any, Type

from .methods import (
    Climatology,
    ForecastBaseline,
    Persistence,
    SeasonalPersistence,
    SmartPersistence,
)


BASELINE_NAMES: tuple[str, ...] = (
    "Persistence",
    "SmartPersistence",
    "SeasonalPersistence",
    "Climatology",
)

_BASELINE_CLASSES: dict[str, Type[ForecastBaseline]] = {
    "Persistence": Persistence,
    "SmartPersistence": SmartPersistence,
    "SeasonalPersistence": SeasonalPersistence,
    "Climatology": Climatology,
}


def load_baseline_class(name: str) -> Type[ForecastBaseline]:
    """Return the registered baseline class or a descriptive error."""

    try:
        return _BASELINE_CLASSES[name]
    except (KeyError, TypeError) as exc:
        available = ", ".join(BASELINE_NAMES)
        raise ValueError(f"unknown baseline {name!r}; available baselines: {available}") from exc


def build_baseline(name: str, train_dataset: Any) -> ForecastBaseline:
    """Construct a baseline fitted to the supplied training dataset."""

    return load_baseline_class(name)(train_dataset)


__all__ = ["BASELINE_NAMES", "build_baseline", "load_baseline_class"]
