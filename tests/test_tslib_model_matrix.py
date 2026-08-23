"""Synthetic CUDA forward/backward/step coverage for all 32 selected models."""

from __future__ import annotations

import json
import os

import pytest
import torch

from models.tslib_registry import SELECTED_MODEL_NAMES
from scripts.tslib_model_matrix import run_model


def _matrix_names():
    raw = os.environ.get("TSLIB_MATRIX_NAMES")
    if not raw:
        return SELECTED_MODEL_NAMES
    names = tuple(item.strip() for item in raw.split(",") if item.strip())
    if set(names).difference(SELECTED_MODEL_NAMES):
        raise ValueError(f"TSLIB_MATRIX_NAMES contains unknown names: {names}")
    return names


def _physical_gpu():
    raw = os.environ.get("TSLIB_MATRIX_PHYSICAL_GPU")
    return None if raw is None else int(raw)


def test_matrix_uses_exact_selected_catalog_by_default():
    if os.environ.get("TSLIB_MATRIX_NAMES"):
        pytest.skip("sharded invocation selects an explicit model subset")
    assert len(SELECTED_MODEL_NAMES) == 32
    assert len(set(SELECTED_MODEL_NAMES)) == 32


@pytest.mark.parametrize("model_name", _matrix_names())
def test_synthetic_cuda_model_matrix(model_name):
    if not torch.cuda.is_available():
        pytest.fail("Task7 synthetic model matrix requires an available CUDA GPU")
    row = run_model(
        model_name,
        device=torch.device("cuda:0"),
        physical_gpu=_physical_gpu(),
    )
    assert row["status"] == "PASS", json.dumps(row, sort_keys=True)
    assert row["phase_steps"] == {"train": 1, "val": 1, "test": 1}
    assert row["power_only_pollution_assert"] is True


def test_mamba_entry_is_the_real_mamba_implementation():
    from models.Mamba import Model

    assert Model.__module__ == "models.Mamba"
    assert "MambaSimple" not in Model.__module__
