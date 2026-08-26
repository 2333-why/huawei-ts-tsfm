"""Classical photovoltaic forecasting baselines."""

from .methods import (
    Climatology,
    ForecastBaseline,
    Persistence,
    SeasonalPersistence,
    SmartPersistence,
    clear_sky_poa,
)
from .registry import BASELINE_NAMES, build_baseline, load_baseline_class

__all__ = [
    "BASELINE_NAMES",
    "ForecastBaseline",
    "Persistence",
    "SmartPersistence",
    "SeasonalPersistence",
    "Climatology",
    "build_baseline",
    "load_baseline_class",
    "clear_sky_poa",
]
