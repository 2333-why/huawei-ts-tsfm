"""Deterministic capability-filtered experiment task matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

from .registry import MODEL_NAMES, get_model_spec


@dataclass(frozen=True)
class ExperimentTask:
    """Identify one model/mode run and its data window."""

    setting: str
    dataset: str
    model: str
    mode: str
    seq_len: int
    pred_len: int


_TASK_SHAPES: Tuple[Tuple[str, str, int, int], ...] = (
    ("seq48_pred1", "skippd_luoyang", 48, 1),
    ("seq48_pred1", "pvod_station00_ylj", 48, 1),
    ("seq96_h4", "skippd_luoyang", 96, 48),
    ("seq96_h4", "pvod_station00_ylj", 96, 16),
)


def iter_experiment_tasks() -> Iterator[ExperimentTask]:
    """Yield fixed windows and only modes supported by each model."""

    for setting, dataset, seq_len, pred_len in _TASK_SHAPES:
        for model in MODEL_NAMES:
            for mode in get_model_spec(model).supported_modes:
                yield ExperimentTask(
                    setting=setting,
                    dataset=dataset,
                    model=model,
                    mode=mode,
                    seq_len=seq_len,
                    pred_len=pred_len,
                )


__all__ = ["ExperimentTask", "iter_experiment_tasks"]
