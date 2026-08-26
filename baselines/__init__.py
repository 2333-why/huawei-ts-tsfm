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
from .extended import (
    ClearSkyAR,
    ClearSkyEWMA,
    DriftPersistence,
    MovingMedian,
    PersistenceClimatologyBlend,
    SimilarDay,
)

__all__ = [
    "BASELINE_NAMES",
    "ForecastBaseline",
    "Persistence",
    "SmartPersistence",
    "SeasonalPersistence",
    "Climatology",
    "MovingMedian",
    "DriftPersistence",
    "ClearSkyEWMA",
    "ClearSkyAR",
    "SimilarDay",
    "PersistenceClimatologyBlend",
    "build_baseline",
    "load_baseline_class",
    "clear_sky_poa",
]
