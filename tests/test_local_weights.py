from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

from models import local_weights


def _clear_weight_environment(monkeypatch):
    monkeypatch.delenv(local_weights.WEIGHTS_ROOT_ENV, raising=False)
    for name in local_weights.MODEL_WEIGHT_ENVS.values():
        monkeypatch.delenv(name, raising=False)


def _make_model_directory(root: Path, model_name: str) -> Path:
    directory = root / model_name
    directory.mkdir(parents=True)
    for filename in local_weights.REQUIRED_WEIGHT_FILES[model_name]:
        (directory / filename).write_bytes(b"test")
    return directory


def test_unconfigured_source_retains_pinned_hub_identity(monkeypatch):
    _clear_weight_environment(monkeypatch)
    source = local_weights.resolve_model_source("Sundial", "owner/model")
    assert source.location == "owner/model"
    assert source.is_local is False
    assert local_weights.pretrained_kwargs(source, "abc") == {"revision": "abc"}


@pytest.mark.parametrize("model_name", tuple(local_weights.MODEL_WEIGHT_ENVS))
def test_root_resolves_each_flat_browser_download(monkeypatch, tmp_path, model_name):
    _clear_weight_environment(monkeypatch)
    directory = _make_model_directory(tmp_path, model_name)
    monkeypatch.setenv(local_weights.WEIGHTS_ROOT_ENV, str(tmp_path))

    source = local_weights.resolve_model_source(model_name, "owner/model")

    expected = directory / "model.ckpt" if model_name == "TiRex" else directory
    assert source.location == str(expected)
    assert source.is_local is True
    assert local_weights.pretrained_kwargs(source, "abc") == {}


def test_per_model_override_wins_and_incomplete_directory_fails(monkeypatch, tmp_path):
    _clear_weight_environment(monkeypatch)
    monkeypatch.setenv(local_weights.WEIGHTS_ROOT_ENV, str(tmp_path / "unused"))
    override = tmp_path / "override"
    override.mkdir()
    monkeypatch.setenv("SUNDIAL_WEIGHT_DIR", str(override))

    with pytest.raises(FileNotFoundError, match="missing"):
        local_weights.resolve_model_source("Sundial", "owner/model")


def test_inspection_checks_exact_weight_size(monkeypatch, tmp_path):
    _clear_weight_environment(monkeypatch)
    _make_model_directory(tmp_path, "Chronos2")
    monkeypatch.setenv(local_weights.WEIGHTS_ROOT_ENV, str(tmp_path))
    monkeypatch.setitem(
        local_weights.WEIGHT_FILE_METADATA,
        "Chronos2",
        ("model.safetensors", 4, "unused"),
    )

    report = local_weights.inspect_local_weights("Chronos2")

    assert report["complete"] is True
    assert report["size_ok"] is True


def test_generate_backend_passes_local_directory_without_revision(monkeypatch, tmp_path):
    _clear_weight_environment(monkeypatch)
    directory = _make_model_directory(tmp_path, "Sundial")
    monkeypatch.setenv(local_weights.WEIGHTS_ROOT_ENV, str(tmp_path))
    calls = []

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, location, **kwargs):
            calls.append((location, kwargs))
            return object()

    module = types.ModuleType("transformers")
    module.AutoModelForCausalLM = AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", module)
    from models.common import _load_pretrained

    _load_pretrained("owner/model", "revision", "Sundial")

    assert calls == [(str(directory), {"trust_remote_code": True})]


def test_tirex_local_file_bypasses_hub_loader(monkeypatch, tmp_path):
    _clear_weight_environment(monkeypatch)
    directory = _make_model_directory(tmp_path, "TiRex")
    monkeypatch.setenv(local_weights.WEIGHTS_ROOT_ENV, str(tmp_path))
    calls = []

    class FakeModel(nn.Module):
        pass

    class TiRexZero:
        @classmethod
        def from_pretrained(cls, location, **kwargs):
            calls.append((location, kwargs))
            return FakeModel()

    module = types.ModuleType("tirex")
    module.TiRexZero = TiRexZero
    module.load_model = lambda *_args, **_kwargs: pytest.fail("Hub loader was used")
    monkeypatch.setitem(sys.modules, "tirex", module)
    from models.TiRex import TiRexBackend
    from models.registry import get_model_spec

    backend = TiRexBackend(get_model_spec("TiRex"), device="cpu")

    assert isinstance(backend.model, FakeModel)
    assert calls == [
        (
            str(directory / "model.ckpt"),
            {"device": "cpu", "backend": "torch"},
        )
    ]
