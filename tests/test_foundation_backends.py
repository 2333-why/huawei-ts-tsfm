from __future__ import annotations

import math
import subprocess
import sys
import types

import pytest
import torch


def test_ensure_forecast_shape_adds_only_single_channel():
    from foundation_models.base import ensure_forecast_shape

    result = ensure_forecast_shape(torch.tensor([[1.0, 2.0]]), batch=1, pred_len=2)

    assert result.shape == (1, 2, 1)
    assert torch.equal(result, torch.tensor([[[1.0], [2.0]]]))


def test_ensure_forecast_shape_accepts_exact_three_dimensional_tensor():
    from foundation_models.base import ensure_forecast_shape

    value = torch.tensor([[[1.0], [2.0]]])

    assert ensure_forecast_shape(value, batch=1, pred_len=2) is value


@pytest.mark.parametrize(
    "value",
    [
        [[1.0, 2.0]],
        torch.ones(2),
        torch.ones(1, 2, 1, 1),
        torch.ones(2, 2),
        torch.ones(1, 3),
        torch.ones(1, 2, 2),
        torch.ones(1, 2, dtype=torch.int64),
        torch.tensor([[1.0, float("nan")]]),
        torch.tensor([[1.0, float("inf")]]),
    ],
)
def test_ensure_forecast_shape_rejects_invalid_outputs(value):
    from foundation_models.base import ensure_forecast_shape

    with pytest.raises(ValueError):
        ensure_forecast_shape(value, batch=1, pred_len=2)


def test_registry_and_factory_import_without_optional_model_dependencies():
    probe = (
        "import sys\n"
        "import foundation_models.registry\n"
        "import foundation_models.factory\n"
        "import foundation_models.backends\n"
        "assert 'transformers' not in sys.modules\n"
        "assert 'peft' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_factory_resolves_entrypoint_only_when_building(monkeypatch):
    from foundation_models import get_model_spec
    from foundation_models import factory

    calls = []
    constructed = {}

    class FakeBackend:
        def __init__(self, spec, device, revision):
            constructed.update(spec=spec, device=device, revision=revision)

    fake_module = types.SimpleNamespace(SundialBackend=FakeBackend)

    def fake_import(module_name):
        calls.append(module_name)
        return fake_module

    monkeypatch.setattr(factory.importlib, "import_module", fake_import)
    assert calls == []

    backend = factory.build_backend("Sundial", "cpu")

    assert calls == ["foundation_models.backends"]
    assert isinstance(backend, FakeBackend)
    assert constructed["spec"] is get_model_spec("Sundial")
    assert constructed["device"] == "cpu"
    assert constructed["revision"] == get_model_spec("Sundial").revision


def test_factory_unknown_model_lists_all_valid_names():
    from foundation_models.factory import build_backend

    with pytest.raises(ValueError, match="Sundial.*TimeMoE"):
        build_backend("unknown", "cpu")


def test_factory_import_failure_identifies_model_and_entrypoint(monkeypatch):
    from foundation_models import factory

    def fail_import(module_name):
        raise ModuleNotFoundError("No module named 'optional_backend'")

    monkeypatch.setattr(factory.importlib, "import_module", fail_import)

    with pytest.raises(ImportError, match="Sundial.*foundation_models.backends"):
        factory.build_backend("Sundial", "cpu")


class _FakeLoadedModel:
    def __init__(self, generate):
        self._generate = generate
        self.to_devices = []
        self.eval_calls = 0
        self.generate_calls = []

    def to(self, device):
        self.to_devices.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def generate(self, values, **kwargs):
        self.generate_calls.append((values.detach().clone(), kwargs))
        return self._generate(values, kwargs)


def _install_fake_transformers(monkeypatch, model):
    calls = []

    class _AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return model

    module = types.ModuleType("transformers")
    module.AutoModelForCausalLM = _AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", module)
    return calls


def test_sundial_backend_loads_pinned_revision_moves_model_and_denormalizes(monkeypatch):
    history = torch.tensor([[[1.0], [3.0], [5.0]]], requires_grad=True)
    loaded = _FakeLoadedModel(
        lambda values, kwargs: torch.full(
            (values.shape[0], 20, kwargs["max_new_tokens"]), 2.0
        )
    )
    loader_calls = _install_fake_transformers(monkeypatch, loaded)

    from foundation_models.factory import build_backend
    from foundation_models.registry import get_model_spec

    backend = build_backend("Sundial", "cpu")
    output = backend.predict(history, pred_len=2)
    spec = get_model_spec("Sundial")

    assert loader_calls == [
        (
            spec.model_id,
            {"revision": spec.revision, "trust_remote_code": True},
        )
    ]
    assert loaded.to_devices == ["cpu"]
    assert loaded.eval_calls == 1
    generated_values, generated_kwargs = loaded.generate_calls[0]
    expected_scale = math.sqrt(8.0 / 3.0 + 1e-5)
    expected_input = torch.tensor(
        [[-2.0 / expected_scale, 0.0, 2.0 / expected_scale]]
    )
    assert torch.allclose(generated_values, expected_input)
    assert generated_values.shape == (1, 3)
    assert generated_kwargs == {"max_new_tokens": 2, "num_samples": 20}
    assert output.shape == (1, 2, 1)
    assert not output.requires_grad
    assert torch.allclose(
        output,
        torch.full((1, 2, 1), 3.0 + 2.0 * expected_scale),
    )


def test_timemoe_backend_overrides_revision_and_crops_last_horizon(monkeypatch):
    history = torch.tensor([[[2.0], [4.0], [6.0]]])
    loaded = _FakeLoadedModel(
        lambda values, kwargs: torch.tensor([[99.0, 98.0, 2.0, 4.0]])
    )
    loader_calls = _install_fake_transformers(monkeypatch, loaded)

    from foundation_models.factory import build_backend
    from foundation_models.registry import get_model_spec

    backend = build_backend("TimeMoE", "cpu", revision="test-revision")
    output = backend.predict(history, pred_len=2)
    spec = get_model_spec("TimeMoE")

    assert loader_calls == [
        (
            spec.model_id,
            {"revision": "test-revision", "trust_remote_code": True},
        )
    ]
    generated_values, generated_kwargs = loaded.generate_calls[0]
    assert generated_values.shape == (1, 3)
    assert generated_kwargs == {"max_new_tokens": 2}
    expected_scale = math.sqrt(8.0 / 3.0 + 1e-5)
    assert torch.allclose(
        output,
        torch.tensor(
            [[[4.0 + 2.0 * expected_scale], [4.0 + 4.0 * expected_scale]]]
        ),
    )


@pytest.mark.parametrize(
    "history",
    [
        [[1.0], [2.0]],
        torch.ones(1, 2, dtype=torch.int64),
        torch.ones(1, 2, 2),
        torch.tensor([[[float("nan")], [1.0]]]),
        torch.tensor([[[float("inf")], [1.0]]]),
        torch.ones(0, 2, 1),
    ],
)
def test_predict_rejects_nonfinite_nonfloating_or_malformed_history(monkeypatch, history):
    loaded = _FakeLoadedModel(lambda values, kwargs: torch.zeros(values.shape[0], 1))
    _install_fake_transformers(monkeypatch, loaded)

    from foundation_models.factory import build_backend

    backend = build_backend("Sundial", "cpu")

    with pytest.raises(ValueError):
        backend.predict(history, pred_len=1)


@pytest.mark.parametrize("pred_len", [0, -1, True, 1.5])
def test_predict_requires_positive_integer_horizon(monkeypatch, pred_len):
    loaded = _FakeLoadedModel(lambda values, kwargs: torch.zeros(values.shape[0], 1))
    _install_fake_transformers(monkeypatch, loaded)

    from foundation_models.factory import build_backend

    backend = build_backend("TimeMoE", "cpu")

    with pytest.raises(ValueError):
        backend.predict(torch.ones(1, 2, 1), pred_len=pred_len)
