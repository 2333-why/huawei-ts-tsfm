from __future__ import annotations

import math
import subprocess
import sys
import types

import pytest
import torch
from torch import nn


def test_ensure_forecast_shape_adds_only_single_channel():
    from models.base import ensure_forecast_shape

    result = ensure_forecast_shape(torch.tensor([[1.0, 2.0]]), batch=1, pred_len=2)

    assert result.shape == (1, 2, 1)
    assert torch.equal(result, torch.tensor([[[1.0], [2.0]]]))


def test_ensure_forecast_shape_accepts_exact_three_dimensional_tensor():
    from models.base import ensure_forecast_shape

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
    from models.base import ensure_forecast_shape

    with pytest.raises(ValueError):
        ensure_forecast_shape(value, batch=1, pred_len=2)


def test_registry_and_factory_import_without_optional_model_dependencies():
    probe = (
        "import sys\n"
        "import models.registry\n"
        "import models.factory\n"
        "import models.Sundial\n"
        "import models.Chronos2\n"
        "import models.TiRex\n"
        "import models.TimesFM\n"
        "assert 'transformers' not in sys.modules\n"
        "assert 'peft' not in sys.modules\n"
        "assert 'chronos' not in sys.modules\n"
        "assert 'tirex' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_factory_resolves_entrypoint_only_when_building(monkeypatch):
    from models import get_model_spec
    from models import factory

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

    assert calls == ["models.Sundial"]
    assert isinstance(backend, FakeBackend)
    assert constructed["spec"] is get_model_spec("Sundial")
    assert constructed["device"] == "cpu"
    assert constructed["revision"] == get_model_spec("Sundial").revision


def test_factory_unknown_model_lists_all_valid_names():
    from models.factory import build_backend

    with pytest.raises(ValueError, match="Sundial.*TimeMoE"):
        build_backend("unknown", "cpu")


def test_factory_import_failure_identifies_model_and_entrypoint(monkeypatch):
    from models import factory

    def fail_import(module_name):
        raise ModuleNotFoundError("No module named 'optional_backend'")

    monkeypatch.setattr(factory.importlib, "import_module", fail_import)

    with pytest.raises(ImportError, match="Sundial.*models.Sundial"):
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

    from models.factory import build_backend
    from models.registry import get_model_spec

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

    from models.factory import build_backend
    from models.registry import get_model_spec

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

    from models.factory import build_backend

    backend = build_backend("Sundial", "cpu")

    with pytest.raises(ValueError):
        backend.predict(history, pred_len=1)


@pytest.mark.parametrize("pred_len", [0, -1, True, 1.5])
def test_predict_requires_positive_integer_horizon(monkeypatch, pred_len):
    loaded = _FakeLoadedModel(lambda values, kwargs: torch.zeros(values.shape[0], 1))
    _install_fake_transformers(monkeypatch, loaded)

    from models.factory import build_backend

    backend = build_backend("TimeMoE", "cpu")

    with pytest.raises(ValueError):
        backend.predict(torch.ones(1, 2, 1), pred_len=pred_len)


class _RecordingSundialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.config = types.SimpleNamespace(input_token_len=4, output_token_lens=[6])
        self.forward_calls = []

    def forward(self, **kwargs):
        self.forward_calls.append(
            {key: value.detach().clone() if torch.is_tensor(value) else value for key, value in kwargs.items()}
        )
        # Keep the fake loss differentiable while making it depend on native labels/masks.
        output_len = kwargs["mask_y"].shape[-1]
        loss = self.scale * (kwargs["labels"][..., -output_len:] * kwargs["mask_y"]).sum()
        return types.SimpleNamespace(loss=loss)


def _backend_without_loading(backend_type, model):
    backend = backend_type.__new__(backend_type)
    backend.model = model
    backend.device = torch.device("cpu")
    backend.spec = types.SimpleNamespace(last_layer_selector="output")
    backend.model_name = backend_type.model_name
    backend.model_id = "tiny"
    backend.revision = "test"
    return backend


def test_sundial_training_loss_builds_last_patch_labels_and_per_sample_masks():
    from models.Sundial import SundialBackend

    model = _RecordingSundialModel()
    backend = _backend_without_loading(SundialBackend, model)
    history = torch.tensor(
        [[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], [[2.0], [4.0], [6.0], [8.0], [10.0], [12.0]]]
    )
    target = torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]]])
    target_mask = torch.tensor([[True, False], [False, True]])

    loss = backend.training_loss(history, target, target_mask)

    assert len(model.forward_calls) == 2
    assert loss.ndim == 0 and loss.requires_grad and torch.isfinite(loss)
    for index, call in enumerate(model.forward_calls):
        assert call["input_ids"].shape == (1, 6)
        centered = history[index, :, 0] - history[index].mean()
        scale = torch.sqrt(centered.var(unbiased=False) + 1e-5)
        assert torch.allclose(call["input_ids"], (centered / scale).unsqueeze(0))
        assert call["labels"].shape == (1, 10)
        assert torch.all(call["labels"][0, :4] == 0)
        target_centered = target[index, :, 0] - history[index].mean()
        expected_target = target_centered / scale
        expected_target[1 if index == 0 else 0] = 0.0
        assert torch.allclose(call["labels"][0, 4:6], expected_target)
        assert torch.all(call["labels"][0, 6:] == 0)
        assert call["loss_masks"].tolist() == [[0.0, 1.0]]
        assert call["mask_y"].tolist() == [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] if index == 0 else [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        assert call["return_dict"] is True
        assert call["use_cache"] is False
        assert call["revin"] is False


class _RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, input_ids, **kwargs):
        self.calls.append((input_ids.detach().clone(), kwargs))
        hidden = torch.cat((input_ids, input_ids * 2.0, input_ids * 3.0), dim=-1)
        return types.SimpleNamespace(last_hidden_state=hidden * self.weight)


class _RecordingHead(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.weight = nn.Parameter(torch.ones(3, horizon))
        self.calls = []

    def forward(self, hidden):
        self.calls.append(hidden.detach().clone())
        return hidden.matmul(self.weight)


class _RecordingTimeMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(input_size=1, horizon_lengths=[1, 8, 32, 64])
        self.model = _RecordingBackbone()
        self.lm_heads = nn.ModuleList([_RecordingHead(horizon) for horizon in [1, 8, 32, 64]])
        self.loss_function = nn.HuberLoss(reduction="none", delta=1.0)


@pytest.mark.parametrize(
    ("horizon", "head_index"), [(1, 0), (16, 2), (48, 3)]
)
def test_timemoe_training_loss_is_history_only_and_selects_smallest_covering_head(horizon, head_index):
    from models.TimeMoE import TimeMoEBackend

    model = _RecordingTimeMoEModel()
    backend = _backend_without_loading(TimeMoEBackend, model)
    history = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    target = torch.arange(1.0, horizon + 1.0).reshape(1, horizon, 1)
    target_mask = torch.ones(1, horizon, dtype=torch.bool)

    loss = backend.training_loss(history, target, target_mask)

    assert loss.ndim == 0 and loss.requires_grad and torch.isfinite(loss)
    assert len(model.model.calls) == 1
    input_ids, kwargs = model.model.calls[0]
    assert input_ids.shape == (1, 4, 1)
    history_centered = history[..., 0] - history.mean()
    history_scale = torch.sqrt(history_centered.var(unbiased=False) + 1e-5)
    assert torch.allclose(input_ids[..., 0], history_centered / history_scale)
    assert kwargs["return_dict"] is True
    assert kwargs["use_cache"] is False
    assert all(not head.calls for index, head in enumerate(model.lm_heads) if index != head_index)
    assert len(model.lm_heads[head_index].calls) == 1


def test_timemoe_training_loss_applies_native_huber_point_mask():
    from models.TimeMoE import TimeMoEBackend

    model = _RecordingTimeMoEModel()
    backend = _backend_without_loading(TimeMoEBackend, model)
    history = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    target = torch.tensor([[[0.0], [4.0], [100.0]]])
    target_mask = torch.tensor([[True, False, True]])

    loss = backend.training_loss(history, target, target_mask)
    history_mean = history.mean()
    history_scale = torch.sqrt(history[..., 0].var(unbiased=False) + 1e-5)
    normalized_target = (target[..., 0] - history_mean) / history_scale
    # The selected horizon-8 head emits three times the final hidden value for this fake.
    normalized_history = (history[..., 0] - history_mean) / history_scale
    prediction_value = normalized_history[0, -1] * 6.0
    expected = torch.nn.functional.huber_loss(
        torch.tensor([prediction_value, prediction_value]),
        torch.tensor([normalized_target[0, 0], normalized_target[0, 2]]),
        reduction="mean",
    )
    assert torch.allclose(loss.detach(), expected, atol=1e-5)


@pytest.mark.parametrize(
    "history,target,target_mask",
    [
        (torch.ones(1, 2, 1), torch.ones(1, 1, 1), torch.zeros(1, 1, dtype=torch.bool)),
        (torch.ones(1, 2, 1), torch.ones(1, 1, 1), torch.tensor([[0.5]])),
        (torch.ones(1, 2, 1), torch.ones(1, 1, 1), torch.tensor([[float("nan")]])),
        (torch.ones(1, 2, 1), torch.ones(1, 1, 2), torch.ones(1, 1, 2)),
        (torch.ones(1, 2, 1), torch.ones(2, 1, 1), torch.ones(2, 1, dtype=torch.bool)),
        (torch.ones(1, 2, 1), torch.tensor([[[float("nan")]]]), torch.ones(1, 1, dtype=torch.bool)),
    ],
)
def test_training_loss_rejects_malformed_or_zero_valid_targets_before_forward(history, target, target_mask):
    from models import Sundial

    model = _RecordingSundialModel()
    backend = _backend_without_loading(Sundial.SundialBackend, model)

    with pytest.raises(ValueError):
        backend.training_loss(history, target, target_mask)
    assert model.forward_calls == []


@pytest.mark.parametrize(
    "loss_value",
    [torch.tensor(1.0), torch.tensor(float("nan")), torch.ones(1)],
)
def test_training_loss_rejects_non_scalar_nonfinite_or_nograd_native_losses(loss_value):
    from models import Sundial

    class BadSundial(_RecordingSundialModel):
        def forward(self, **kwargs):
            return types.SimpleNamespace(loss=loss_value)

    model = BadSundial()
    backend = _backend_without_loading(Sundial.SundialBackend, model)
    with pytest.raises(ValueError, match="loss"):
        backend.training_loss(
            torch.ones(1, 2, 1), torch.ones(1, 1, 1), torch.ones(1, 1, dtype=torch.bool)
        )


def test_backend_configure_trainable_sets_eval_or_train(monkeypatch):
    from models import Sundial
    from models import common

    model = _RecordingSundialModel()
    backend = _backend_without_loading(Sundial.SundialBackend, model)
    seen = []

    def fake_configure(model_arg, spec_arg, mode, lora_settings):
        seen.append((model_arg, spec_arg, mode, lora_settings))
        return types.SimpleNamespace(mode=mode)

    monkeypatch.setattr(common, "configure_trainable", fake_configure)
    backend.configure_trainable("zero_shot", None)
    assert not model.training
    backend.configure_trainable("full", None)
    assert model.training
    assert [entry[2] for entry in seen] == ["zero_shot", "full"]


class _FakeChronosModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.self_attention = nn.Module()
        self.self_attention.q = nn.Linear(1, 1)
        self.output_patch_embedding = nn.Module()
        self.output_patch_embedding.output_layer = nn.Linear(1, 1)
        self.chronos_config = types.SimpleNamespace(
            output_patch_size=4,
            max_output_patches=8,
        )
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        loss = self.weight * (
            kwargs["future_target"] * kwargs["future_target_mask"]
        ).sum()
        return types.SimpleNamespace(loss=loss)


class _FakeChronosPipeline:
    loader_calls = []
    predict_calls = []

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.loader_calls.append((model_id, kwargs))
        return cls(_FakeChronosModel())

    def predict_quantiles(self, inputs, prediction_length):
        self.predict_calls.append((inputs.detach().clone(), prediction_length))
        batch = inputs.shape[0]
        means = [torch.tensor([[10.0, 11.0, 12.0, 13.0][:prediction_length]]) for _ in range(batch)]
        quantiles = [torch.zeros(1, prediction_length, 9) for _ in range(batch)]
        return quantiles, means


def _install_fake_chronos(monkeypatch):
    module = types.ModuleType("chronos")
    module.Chronos2Pipeline = _FakeChronosPipeline
    monkeypatch.setitem(sys.modules, "chronos", module)
    _FakeChronosPipeline.loader_calls = []
    _FakeChronosPipeline.predict_calls = []


def test_chronos2_loader_prediction_and_native_training_contract(monkeypatch):
    _install_fake_chronos(monkeypatch)
    from models.Chronos2 import Chronos2Backend
    from models.registry import get_model_spec

    spec = get_model_spec("Chronos2")
    backend = Chronos2Backend(spec=spec, device="cpu")
    history = torch.tensor([[[1.0], [2.0], [3.0]]])
    forecast = backend.predict(history, 2)

    assert _FakeChronosPipeline.loader_calls == [
        (
            spec.model_id,
            {"revision": spec.revision, "device_map": "cpu"},
        )
    ]
    values, horizon = _FakeChronosPipeline.predict_calls[0]
    assert values.shape == (1, 1, 3)
    assert torch.equal(values[0, 0], history[0, :, 0])
    assert horizon == 2
    assert torch.equal(forecast, torch.tensor([[[10.0], [11.0]]]))
    assert backend.model is backend.pipeline.model

    loss = backend.training_loss(
        history,
        torch.tensor([[[4.0], [5.0], [6.0], [7.0], [8.0]]]),
        torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.bool),
    )
    call = backend.model.calls[0]
    assert call["context"].shape == (1, 3)
    assert call["future_target"].shape == (1, 5)
    assert call["future_target_mask"].tolist() == [[True, False, True, False, True]]
    assert call["num_output_patches"] == 2
    assert loss.ndim == 0 and loss.requires_grad and torch.isfinite(loss)


class _FakeTiRexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forecast(self, **kwargs):
        self.calls.append(kwargs)
        context = kwargs["context"]
        prediction_length = kwargs["prediction_length"]
        quantiles = torch.zeros(context.shape[0], 9, prediction_length)
        mean = torch.arange(prediction_length, dtype=context.dtype).repeat(context.shape[0], 1)
        return quantiles, mean


def _install_fake_tirex(monkeypatch):
    module = types.ModuleType("tirex")
    loaded = _FakeTiRexModel()
    calls = []

    def load_model(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return loaded

    module.load_model = load_model
    monkeypatch.setitem(sys.modules, "tirex", module)
    return loaded, calls


def test_tirex_loader_uses_hf_kwargs_and_rejects_training(monkeypatch):
    loaded, loader_calls = _install_fake_tirex(monkeypatch)
    from models.TiRex import TiRexBackend
    from models.registry import get_model_spec

    spec = get_model_spec("TiRex")
    backend = TiRexBackend(spec=spec, device="cpu")
    history = torch.tensor([[[3.0], [4.0], [5.0]]])
    forecast = backend.predict(history, 2)

    assert loader_calls == [
        (
            spec.model_id,
            {
                "device": "cpu",
                "backend": "torch",
                "hf_kwargs": {"revision": spec.revision},
            },
        )
    ]
    assert loaded.calls[0]["context"].shape == (1, 3)
    assert loaded.calls[0]["prediction_length"] == 2
    assert torch.equal(forecast, torch.tensor([[[0.0], [1.0]]]))
    report = backend.configure_trainable("zero_shot")
    assert report.mode == "zero_shot"
    assert report.trainable_parameters == 0
    assert all(not parameter.requires_grad for parameter in backend.model.parameters())
    with pytest.raises(ValueError, match="TiRex.*zero_shot|unsupported"):
        backend.configure_trainable("full")
    with pytest.raises(ValueError, match="TiRex.*training|unsupported"):
        backend.training_loss(history, torch.ones(1, 1, 1), torch.ones(1, 1, dtype=torch.bool))


class _FakeTimesFMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        past_values = kwargs["past_values"]
        horizon = 5
        point = torch.stack([values[-1].expand(horizon) for values in past_values]) * self.scale
        return types.SimpleNamespace(mean_predictions=point)


class _FakeTimesFMClass:
    model = _FakeTimesFMModel()
    loader_calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.loader_calls.append((model_id, kwargs))
        return cls.model


def _install_fake_timesfm(monkeypatch):
    module = types.ModuleType("transformers")
    module.TimesFm2_5ModelForPrediction = _FakeTimesFMClass
    monkeypatch.setitem(sys.modules, "transformers", module)
    _FakeTimesFMClass.model = _FakeTimesFMModel()
    _FakeTimesFMClass.loader_calls = []


def test_timesfm_loader_crops_mean_predictions_and_computes_masked_mse(monkeypatch):
    _install_fake_timesfm(monkeypatch)
    from models.TimesFM import TimesFMBackend
    from models.registry import get_model_spec

    spec = get_model_spec("TimesFM")
    backend = TimesFMBackend(spec=spec, device="cpu")
    history = torch.tensor([[[1.0], [2.0], [3.0]]])
    forecast = backend.predict(history, 3)

    assert _FakeTimesFMClass.loader_calls == [
        (
            spec.model_id,
            {"revision": spec.revision, "device_map": "cpu"},
        )
    ]
    assert _FakeTimesFMClass.model.calls[0]["past_values"][0].shape == (3,)
    assert torch.equal(forecast, torch.tensor([[[3.0], [3.0], [3.0]]]))

    target = torch.tensor([[[1.0], [5.0], [9.0]]])
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    loss = backend.training_loss(history, target, mask)
    expected = ((torch.tensor([3.0, 3.0]) - torch.tensor([1.0, 9.0])) ** 2).mean()
    assert torch.allclose(loss.detach(), expected)
    assert loss.ndim == 0 and loss.requires_grad and torch.isfinite(loss)
