import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models import (
    EXCLUDED_MODELS,
    MODEL_SPECS,
    PURE_TIME_SERIES_MODELS,
    SELECTED_MODEL_NAMES,
)
from scripts import sync_tslib_models


EXPECTED_MODELS = {
    "Autoformer", "Crossformer", "DLinear", "ETSformer", "FEDformer",
    "FiLM", "FreTS", "Informer", "Koopa", "LightTS", "MICN", "MSGNet",
    "Mamba", "MambaSimple", "MultiPatchFormer", "Nonstationary_Transformer",
    "PAttn", "PatchTST", "Pyraformer", "Reformer", "SCINet", "SegRNN",
    "TSMixer", "TemporalFusionTransformer", "TiDE", "TimeFilter",
    "TimeMixer", "TimeXer", "TimesNet", "Transformer", "WPMixer",
    "iTransformer",
}


def _configs():
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=16,
        label_len=16,
        pred_len=4,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=8,
        n_heads=2,
        e_layers=1,
        d_ff=16,
        factor=2,
        dropout=0.0,
        activation="gelu",
        embed="fixed",
        freq="h",
        moving_avg=3,
        top_k=2,
        num_kernels=1,
        num_class=1,
    )


def test_registry_has_every_forecasting_non_foundation_model():
    assert set(SELECTED_MODEL_NAMES) == EXPECTED_MODELS
    assert set(MODEL_SPECS) == EXPECTED_MODELS


def test_registry_exclusions_are_explicit():
    assert set(EXCLUDED_MODELS) == {
        "Chronos", "Chronos2", "KANAD", "Sundial", "TimeMoE", "TiRex", "TimesFM",
    }
    assert "forecast" in EXCLUDED_MODELS["KANAD"].lower()
    for name in {"Chronos", "Chronos2", "Sundial", "TimeMoE", "TiRex", "TimesFM"}:
        assert "foundation" in EXCLUDED_MODELS[name].lower()


def test_registry_is_lazy_for_optional_mamba_dependency(monkeypatch):
    real_import = importlib.import_module
    calls = []

    def recording_import(name, package=None):
        calls.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", recording_import)
    assert "Mamba" in SELECTED_MODEL_NAMES
    assert "models.Mamba" not in calls


def test_pure_time_series_registry_is_lazy_mapping():
    assert set(PURE_TIME_SERIES_MODELS) == EXPECTED_MODELS
    assert len(PURE_TIME_SERIES_MODELS) == len(EXPECTED_MODELS)


def test_every_selected_model_file_exists():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    missing = sorted(
        name for name in EXPECTED_MODELS
        if not (model_dir / f"{name}.py").is_file()
    )
    assert missing == []


def test_foundation_and_nonforecast_files_are_not_copied():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    for name in EXCLUDED_MODELS:
        assert not (model_dir / f"{name}.py").exists()


@pytest.mark.parametrize("destination_kind", ["equal", "nested"])
def test_sync_rejects_source_destination_overlap_before_writes(tmp_path, destination_kind):
    source_root = tmp_path / "source"
    model_dir = source_root / "models"
    layer_dir = source_root / "layers"
    model_dir.mkdir(parents=True)
    layer_dir.mkdir()
    for name in set(sync_tslib_models.SELECTED_MODEL_NAMES) | sync_tslib_models.EXCLUDED:
        (model_dir / f"{name}.py").write_text(f"# {name}\n", encoding="utf-8")
    (layer_dir / "Core.py").write_text("# core\n", encoding="utf-8")
    destination_root = source_root if destination_kind == "equal" else source_root / "nested"
    before = sorted(path.relative_to(source_root) for path in source_root.rglob("*"))

    with pytest.raises(ValueError, match="overlap"):
        sync_tslib_models.sync_models(source_root, destination_root)

    after = sorted(path.relative_to(source_root) for path in source_root.rglob("*"))
    assert after == before


@pytest.mark.parametrize("name", sorted(EXPECTED_MODELS - {"Mamba"}))
def test_selected_model_module_imports(name):
    module = importlib.import_module(f"models.{name}")
    assert issubclass(module.Model, torch.nn.Module)


def test_existing_model_modules_are_lazy_package_attributes():
    check = """
import importlib
import sys
import models

names = ('DLinear', 'PatchTST', 'iTransformer', 'TimesNet')
assert not any(f'models.{name}' in sys.modules for name in names)
for name in names:
    module = getattr(models, name)
    assert module is importlib.import_module(f'models.{name}')
"""
    result = subprocess.run(
        [sys.executable, "-c", check],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("model_name", ["DLinear", "PatchTST", "iTransformer", "TimesNet"])
def test_pure_time_series_forward_backward(model_name):
    config = _configs()
    model_class = PURE_TIME_SERIES_MODELS[model_name]
    if model_name == "PatchTST":
        model = model_class(config, patch_len=16, stride=8)
    else:
        model = model_class(config)
    inputs = torch.randn(2, config.seq_len, 1, requires_grad=True)
    outputs = model(inputs, None, None, None)
    assert outputs.shape == (2, config.pred_len, 1)
    loss = outputs.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
