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
        self.native_calls = []

    def to(self, device):
        self.to_devices.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def generate(self, values, **kwargs):
        self.generate_calls.append((values.detach().clone(), kwargs))
        return self._generate(values, kwargs)

    def __call__(self, **kwargs):
        self.native_calls.append(
            {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in kwargs.items()
            }
        )
        return types.SimpleNamespace(logits=self._generate(kwargs["input_ids"], kwargs))


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
            (values.shape[0], 20, kwargs["max_output_length"]), 2.0
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
    native_call = loaded.native_calls[0]
    expected_scale = math.sqrt(8.0 / 3.0 + 1e-5)
    expected_input = torch.tensor(
        [[-2.0 / expected_scale, 0.0, 2.0 / expected_scale]]
    )
    assert torch.allclose(native_call["input_ids"], expected_input)
    assert native_call["input_ids"].shape == (1, 3)
    assert {
        key: value for key, value in native_call.items() if key != "input_ids"
    } == {
        "max_output_length": 2,
        "num_samples": 20,
        "use_cache": False,
        "return_dict": True,
        "revin": False,
    }
    assert output.shape == (1, 2, 1)
    assert not output.requires_grad
    assert torch.allclose(
        output,
        torch.full((1, 2, 1), 3.0 + 2.0 * expected_scale),
    )


def test_sundial_backend_uses_native_forward_when_generate_cache_api_is_incompatible(monkeypatch):
    class _SundialCompatLoadedModel:
        def __init__(self):
            self.to_devices = []
            self.eval_calls = 0
            self.generate_calls = []
            self.native_calls = []

        def to(self, device):
            self.to_devices.append(device)
            return self

        def eval(self):
            self.eval_calls += 1
            return self

        def generate(self, *args, **kwargs):
            self.generate_calls.append((args, kwargs))
            raise AttributeError("'DynamicCache' object has no attribute 'seen_tokens'")

        def __call__(self, **kwargs):
            self.native_calls.append(
                {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in kwargs.items()
                }
            )
            input_ids = kwargs["input_ids"]
            horizon = kwargs["max_output_length"]
            return types.SimpleNamespace(
                logits=torch.full((input_ids.shape[0], 20, horizon), 2.0)
            )

    history = torch.tensor(
        [[[1.0], [3.0], [5.0]], [[2.0], [4.0], [6.0]]],
        requires_grad=True,
    )
    loaded = _SundialCompatLoadedModel()
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
    assert loaded.generate_calls == []
    assert len(loaded.native_calls) == 1
    native_call = loaded.native_calls[0]
    expected_scale = math.sqrt(8.0 / 3.0 + 1e-5)
    expected_input = torch.tensor(
        [
            [-2.0 / expected_scale, 0.0, 2.0 / expected_scale],
            [-2.0 / expected_scale, 0.0, 2.0 / expected_scale],
        ]
    )
    assert torch.allclose(native_call["input_ids"], expected_input)
    assert native_call["input_ids"].shape == (2, 3)
    assert {
        key: value
        for key, value in native_call.items()
        if key != "input_ids"
    } == {
        "max_output_length": 2,
        "num_samples": 20,
        "use_cache": False,
        "return_dict": True,
        "revin": False,
    }
    assert output.shape == (2, 2, 1)
    assert torch.isfinite(output).all()
    assert torch.allclose(
        output,
        torch.tensor(
            [
                [[3.0 + 2.0 * expected_scale], [3.0 + 2.0 * expected_scale]],
                [[4.0 + 2.0 * expected_scale], [4.0 + 2.0 * expected_scale]],
            ]
        ),
    )


class _CorruptRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 4
        self.base = 10000
        self.max_position_embeddings = 8
        self.register_buffer(
            "inv_freq", torch.tensor([float("nan"), float("inf")]), persistent=False
        )
        self.register_buffer(
            "cos_cached",
            torch.full((1, 1, self.max_position_embeddings, self.dim), float("nan")),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            torch.full((1, 1, self.max_position_embeddings, self.dim), float("nan")),
            persistent=False,
        )
        self.cache_calls = []

    def _set_cos_sin_cache(self, seq_len, device=None, dtype=None):
        self.cache_calls.append((seq_len, device, dtype))
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        frequencies = torch.outer(positions, self.inv_freq.to(device=device, dtype=dtype))
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.cos_cached = embedding.cos()[None, None, :, :]
        self.sin_cached = embedding.sin()[None, None, :, :]


def test_sundial_backend_rebuilds_corrupt_rotary_runtime_buffers(monkeypatch):

    class _SundialCorruptLoadedModel:
        def __init__(self):
            self.rotary = _CorruptRotary()
            self.to_devices = []
            self.eval_calls = 0
            self.native_calls = []

        def to(self, device):
            self.to_devices.append(device)
            return self

        def eval(self):
            self.eval_calls += 1
            return self

        def modules(self):
            return (self.rotary,)

        def __call__(self, **kwargs):
            self.native_calls.append(
                {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in kwargs.items()
                }
            )
            expected_positions = torch.arange(0, self.rotary.dim, 2, dtype=torch.float32)
            expected_inv_freq = 1.0 / (
                self.rotary.base ** (expected_positions / self.rotary.dim)
            )
            expected_frequencies = torch.outer(
                torch.arange(
                    self.rotary.max_position_embeddings,
                    dtype=self.rotary.inv_freq.dtype,
                ),
                expected_inv_freq,
            )
            expected_embedding = torch.cat((expected_frequencies, expected_frequencies), dim=-1)
            buffers_match = (
                torch.isfinite(self.rotary.inv_freq).all()
                and torch.allclose(self.rotary.inv_freq, expected_inv_freq)
                and torch.isfinite(self.rotary.cos_cached).all()
                and torch.isfinite(self.rotary.sin_cached).all()
                and torch.allclose(
                    self.rotary.cos_cached,
                    expected_embedding.cos()[None, None, :, :],
                )
                and torch.allclose(
                    self.rotary.sin_cached,
                    expected_embedding.sin()[None, None, :, :],
                )
            )
            input_ids = kwargs["input_ids"]
            horizon = kwargs["max_output_length"]
            logits = torch.full(
                (input_ids.shape[0], 20, horizon), 2.0 if buffers_match else float("nan")
            )
            return types.SimpleNamespace(logits=logits)

    history = torch.tensor([[[1.0], [3.0], [5.0]]], requires_grad=True)
    loaded = _SundialCorruptLoadedModel()
    _install_fake_transformers(monkeypatch, loaded)

    from models.factory import build_backend

    backend = build_backend("Sundial", "cpu")
    output = backend.predict(history, pred_len=1)

    assert loaded.to_devices == ["cpu"]
    assert loaded.eval_calls == 1
    assert len(loaded.native_calls) == 1
    assert torch.isfinite(output).all()
    assert output.shape == (1, 1, 1)
    assert torch.allclose(loaded.rotary.inv_freq, torch.tensor([1.0, 0.01]), atol=1e-6)
    assert torch.isfinite(loaded.rotary.cos_cached).all()
    assert torch.isfinite(loaded.rotary.sin_cached).all()
    assert torch.allclose(
        loaded.rotary.cos_cached[0, 0, 0],
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )
    assert torch.allclose(
        loaded.rotary.sin_cached[0, 0, 0],
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        atol=1e-6,
    )


class _TimeMoECompatLoadedModel:
    def __init__(self):
        self.rotary = _CorruptRotary()
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.bfloat16))
        self.config = types.SimpleNamespace(input_size=1, horizon_lengths=[1, 2, 4])
        self.to_devices = []
        self.eval_calls = 0
        self.generate_calls = []
        self.native_calls = []
        self._normalized_chunks = {
            4: torch.tensor([0.5, 1.0, 1.5, 2.0]),
            2: torch.tensor([2.5, 3.0]),
            1: torch.tensor([3.5]),
        }

    def parameters(self):
        yield self.weight

    def to(self, device):
        self.to_devices.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def modules(self):
        return (self.rotary,)

    def generate(self, *args, **kwargs):
        self.generate_calls.append((args, kwargs))
        raise AttributeError("'DynamicCache' object has no attribute 'seen_tokens'")

    def __call__(self, **kwargs):
        self.native_calls.append(
            {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in kwargs.items()
            }
        )
        input_ids = kwargs["input_ids"]
        if input_ids.ndim != 3 or input_ids.dtype != torch.bfloat16:
            raise AssertionError("TimeMoE native context must be bfloat16 [batch, time, 1]")
        expected_positions = torch.arange(0, self.rotary.dim, 2, dtype=torch.float32)
        expected_inv_freq = 1.0 / (
            self.rotary.base ** (expected_positions / self.rotary.dim)
        )
        expected_frequencies = torch.outer(
            torch.arange(
                self.rotary.max_position_embeddings,
                dtype=self.rotary.inv_freq.dtype,
            ),
            expected_inv_freq,
        )
        expected_embedding = torch.cat((expected_frequencies, expected_frequencies), dim=-1)
        buffers_match = (
            torch.isfinite(self.rotary.inv_freq).all()
            and torch.allclose(self.rotary.inv_freq, expected_inv_freq)
            and torch.isfinite(self.rotary.cos_cached).all()
            and torch.isfinite(self.rotary.sin_cached).all()
            and torch.allclose(
                self.rotary.cos_cached,
                expected_embedding.cos()[None, None, :, :],
            )
            and torch.allclose(
                self.rotary.sin_cached,
                expected_embedding.sin()[None, None, :, :],
            )
        )
        max_horizon = kwargs["max_horizon_length"]
        selected_horizon = max(
            horizon for horizon in self.config.horizon_lengths if horizon <= max_horizon
        )
        normalized_chunk = self._normalized_chunks[selected_horizon]
        logits = torch.full(
            (
                input_ids.shape[0],
                input_ids.shape[1],
                selected_horizon * self.config.input_size,
            ),
            float("nan") if not buffers_match else 0.0,
        )
        if buffers_match:
            logits[:, -1, :] = normalized_chunk
        return types.SimpleNamespace(logits=logits)


def test_timemoe_backend_uses_native_no_cache_chunks_for_transformers_compatibility(
    monkeypatch,
):
    history = torch.tensor([[[1.0], [3.0], [5.0]]])
    loaded = _TimeMoECompatLoadedModel()
    _install_fake_transformers(monkeypatch, loaded)

    from models.factory import build_backend

    backend = build_backend("TimeMoE", "cpu")
    output = backend.predict(history, pred_len=6)

    assert loaded.to_devices == ["cpu"]
    assert loaded.eval_calls == 1
    assert loaded.generate_calls == []
    assert len(loaded.native_calls) == 2
    assert [call["max_horizon_length"] for call in loaded.native_calls] == [6, 2]
    assert [call["input_ids"].shape for call in loaded.native_calls] == [
        (1, 3, 1),
        (1, 7, 1),
    ]
    assert all(call["input_ids"].dtype == torch.bfloat16 for call in loaded.native_calls)
    assert all(call["input_ids"].ndim == 3 for call in loaded.native_calls)
    assert all(call["use_cache"] is False for call in loaded.native_calls)
    assert all(call["return_dict"] is True for call in loaded.native_calls)
    assert torch.equal(
        loaded.native_calls[1]["input_ids"].to(dtype=torch.float32)[:, -4:, 0],
        torch.tensor([[0.5, 1.0, 1.5, 2.0]]),
    )
    assert torch.allclose(loaded.rotary.inv_freq, torch.tensor([1.0, 0.01]), atol=1e-6)
    assert torch.isfinite(loaded.rotary.cos_cached).all()
    assert torch.isfinite(loaded.rotary.sin_cached).all()
    assert torch.allclose(
        loaded.rotary.cos_cached[0, 0, 0],
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )
    assert torch.allclose(
        loaded.rotary.sin_cached[0, 0, 0],
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        atol=1e-6,
    )
    expected_scale = math.sqrt(8.0 / 3.0 + 1e-5)
    expected = torch.tensor(
        [
            [
                [3.0 + 0.5 * expected_scale],
                [3.0 + 1.0 * expected_scale],
                [3.0 + 1.5 * expected_scale],
                [3.0 + 2.0 * expected_scale],
                [3.0 + 2.5 * expected_scale],
                [3.0 + 3.0 * expected_scale],
            ]
        ]
    )
    assert output.shape == (1, 6, 1)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, expected)


def test_timemoe_backend_overrides_revision_and_crops_last_horizon(monkeypatch):
    history = torch.tensor([[[2.0], [4.0], [6.0]]])
    loaded = _TimeMoECompatLoadedModel()
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
    assert loaded.generate_calls == []
    assert len(loaded.native_calls) == 1
    native_call = loaded.native_calls[0]
    assert native_call["input_ids"].shape == (1, 3, 1)
    assert native_call["input_ids"].dtype == torch.bfloat16
    assert native_call["max_horizon_length"] == 2
    assert native_call["use_cache"] is False
    assert native_call["return_dict"] is True
    expected_scale = math.sqrt(8.0 / 3.0 + 1e-5)
    assert torch.allclose(
        output,
        torch.tensor(
            [[[4.0 + 2.5 * expected_scale], [4.0 + 3.0 * expected_scale]]]
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
        if inputs.device.type != "cpu":
            raise RuntimeError(
                "cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned"
            )
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_chronos2_moves_cpu_pipeline_forecast_to_backend_cuda_device(monkeypatch):
    _install_fake_chronos(monkeypatch)
    from models.Chronos2 import Chronos2Backend
    from models.registry import get_model_spec

    spec = get_model_spec("Chronos2")
    backend = Chronos2Backend(spec=spec, device="cuda:0")
    forecast = backend.predict(torch.tensor([[[1.0], [2.0], [3.0]]]), 2)

    values, horizon = _FakeChronosPipeline.predict_calls[0]
    assert values.device == torch.device("cpu")
    assert values.is_contiguous()
    assert values.shape == (1, 1, 3)
    assert horizon == 2
    assert forecast.device == torch.device("cuda:0")
    assert forecast.shape == (1, 2, 1)
    assert torch.isfinite(forecast).all()
    assert torch.equal(forecast.cpu(), torch.tensor([[[10.0], [11.0]]]))


def test_tirex_moves_cpu_forecast_to_backend_cuda_device(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    loaded, _ = _install_fake_tirex(monkeypatch)
    from models.TiRex import TiRexBackend
    from models.registry import get_model_spec

    spec = get_model_spec("TiRex")
    backend = TiRexBackend(spec=spec, device="cuda:0")
    forecast = backend.predict(torch.tensor([[[3.0], [4.0], [5.0]]]), 2)

    assert loaded.calls[0]["context"].device == torch.device("cuda:0")
    assert forecast.device == torch.device("cuda:0")
    assert forecast.shape == (1, 2, 1)
    assert torch.isfinite(forecast).all()
    assert torch.equal(forecast.cpu(), torch.tensor([[[0.0], [1.0]]]))


def _assert_backend_optimizer_step_is_audited(backend, model, mode, loss_inputs):
    from models.trainability import LoraSettings

    report = backend.configure_trainable(
        mode,
        LoraSettings(r=2, lora_alpha=4, lora_dropout=0.0)
        if mode == "adapter"
        else None,
    )
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    allowed = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert report.trainable_parameters == sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name in allowed
    )
    if mode == "adapter":
        assert allowed
        assert all("lora_" in name for name in allowed)
    elif mode == "last_layer":
        assert allowed
        assert all(
            name.startswith("output_patch_embedding.")
            or name.startswith("output_projection_point.")
            for name in allowed
        )
    else:
        assert allowed == set(name for name, _ in model.named_parameters())

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.2,
    )
    loss = backend.training_loss(*loss_inputs)
    loss.backward()
    if mode == "adapter":
        assert any(
            parameter.grad is not None and bool(torch.any(parameter.grad != 0))
            for name, parameter in model.named_parameters()
            if "lora_" in name
        )
    optimizer.step()

    changed = {
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, before[name])
    }
    assert changed
    assert changed <= allowed
    assert all(
        name in allowed or torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )


class _BackendOptimizerChronosModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.Module()
        self.self_attention.q = nn.Linear(1, 1)
        self.self_attention.v = nn.Linear(1, 1)
        self.self_attention.k = nn.Linear(1, 1)
        self.self_attention.o = nn.Linear(1, 1)
        self.output_patch_embedding = nn.Module()
        self.output_patch_embedding.output_layer = nn.Linear(1, 1)
        self.unrelated = nn.Linear(1, 1)
        self.chronos_config = types.SimpleNamespace(
            output_patch_size=1,
            max_output_patches=16,
        )

    def forward(self, *, context, future_target, future_target_mask, num_output_patches):
        hidden = context.unsqueeze(-1)
        hidden = torch.tanh(self.self_attention.q(hidden))
        hidden = torch.tanh(self.self_attention.v(hidden))
        hidden = torch.tanh(self.self_attention.k(hidden))
        hidden = torch.tanh(self.self_attention.o(hidden))
        point = self.output_patch_embedding.output_layer(hidden[:, -1, :])
        point = point.expand(-1, future_target.shape[1])
        residual = (point - future_target) * future_target_mask.to(point.dtype)
        return types.SimpleNamespace(loss=residual.square().sum())


class _BackendOptimizerChronosPipeline:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()

    def __init__(self):
        self.model = _BackendOptimizerChronosModel()

    def predict_quantiles(self, context, prediction_length):
        point = torch.zeros(context.shape[0], prediction_length)
        quantiles = torch.zeros(context.shape[0], prediction_length, 9)
        return quantiles, point


def _install_backend_optimizer_chronos(monkeypatch):
    module = types.ModuleType("chronos")
    module.Chronos2Pipeline = _BackendOptimizerChronosPipeline
    monkeypatch.setitem(sys.modules, "chronos", module)


@pytest.mark.parametrize("mode", ["adapter", "full", "last_layer"])
def test_chronos2_backend_optimizer_step_changes_only_permitted_state(monkeypatch, mode):
    _install_backend_optimizer_chronos(monkeypatch)
    from models.Chronos2 import Chronos2Backend
    from models.registry import get_model_spec

    backend = Chronos2Backend(get_model_spec("Chronos2"), device="cpu")
    history = torch.ones(1, 3, 1)
    target = torch.zeros(1, 2, 1)
    target_mask = torch.ones(1, 2, dtype=torch.bool)
    _assert_backend_optimizer_step_is_audited(
        backend,
        backend.model,
        mode,
        (history, target, target_mask),
    )


class _BackendOptimizerTimesFMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(1, 1)
        self.output_projection_point = nn.Linear(1, 1)

    def forward(self, *, past_values, return_dict):
        values = torch.stack(past_values, dim=0)
        hidden = self.decoder(values[:, -1:].unsqueeze(-1))
        point = self.output_projection_point(hidden).squeeze(-1)
        return types.SimpleNamespace(mean_predictions=point.expand(-1, 5))


class _BackendOptimizerTimesFMClass:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return _BackendOptimizerTimesFMModel()


def _install_backend_optimizer_timesfm(monkeypatch):
    module = types.ModuleType("transformers")
    module.TimesFm2_5ModelForPrediction = _BackendOptimizerTimesFMClass
    monkeypatch.setitem(sys.modules, "transformers", module)


@pytest.mark.parametrize("mode", ["adapter", "full", "last_layer"])
def test_timesfm_backend_optimizer_step_changes_only_permitted_state(monkeypatch, mode):
    _install_backend_optimizer_timesfm(monkeypatch)
    from models.TimesFM import TimesFMBackend
    from models.registry import get_model_spec

    backend = TimesFMBackend(get_model_spec("TimesFM"), device="cpu")
    history = torch.ones(1, 3, 1)
    target = torch.zeros(1, 2, 1)
    target_mask = torch.ones(1, 2, dtype=torch.bool)
    _assert_backend_optimizer_step_is_audited(
        backend,
        backend.model,
        mode,
        (history, target, target_mask),
    )


class _BoundaryChronosModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.chronos_config = types.SimpleNamespace(output_patch_size=1)

    def forward(self, **kwargs):
        return types.SimpleNamespace(loss=self.weight * kwargs["future_target"].sum())


class _BoundaryChronosPipeline:
    output_kind = "valid"

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()

    def __init__(self):
        self.model = _BoundaryChronosModel()

    def predict_quantiles(self, context, prediction_length):
        batch = context.shape[0]
        if self.output_kind == "malformed":
            point = torch.ones(batch, prediction_length, 2)
        elif self.output_kind == "nonfinite":
            point = torch.tensor([[1.0, float("nan")]]).expand(batch, -1)
        elif self.output_kind == "short":
            point = torch.ones(batch, max(1, prediction_length - 1))
        else:
            point = torch.arange(prediction_length, dtype=context.dtype).expand(batch, -1)
        return torch.zeros(batch, prediction_length, 9), point


def _install_boundary_chronos(monkeypatch, output_kind):
    module = types.ModuleType("chronos")
    _BoundaryChronosPipeline.output_kind = output_kind
    module.Chronos2Pipeline = _BoundaryChronosPipeline
    monkeypatch.setitem(sys.modules, "chronos", module)


@pytest.mark.parametrize("output_kind", ["malformed", "nonfinite", "short"])
def test_chronos2_rejects_malformed_nonfinite_and_short_point_outputs(monkeypatch, output_kind):
    _install_boundary_chronos(monkeypatch, output_kind)
    from models.Chronos2 import Chronos2Backend
    from models.registry import get_model_spec

    backend = Chronos2Backend(get_model_spec("Chronos2"), device="cpu")
    with pytest.raises(ValueError):
        backend.predict(torch.ones(1, 3, 1), 2)


def test_chronos2_reports_directed_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "chronos", None)
    from models.Chronos2 import Chronos2Backend
    from models.registry import get_model_spec

    with pytest.raises(ImportError, match="chronos-forecasting.*Python 3.10"):
        Chronos2Backend(get_model_spec("Chronos2"), device="cpu")


class _BoundaryTiRexModel(nn.Module):
    output_kind = "valid"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forecast(self, *, context, prediction_length):
        batch = context.shape[0]
        if self.output_kind == "malformed":
            point = torch.ones(batch, prediction_length, 2)
        elif self.output_kind == "nonfinite":
            point = torch.tensor([[1.0, float("nan")]]).expand(batch, -1)
        elif self.output_kind == "short":
            point = torch.ones(batch, max(1, prediction_length - 1))
        else:
            point = torch.arange(prediction_length, dtype=context.dtype).expand(batch, -1)
        return torch.zeros(batch, 9, prediction_length), point


def _install_boundary_tirex(monkeypatch, output_kind):
    module = types.ModuleType("tirex")
    model = _BoundaryTiRexModel()
    model.output_kind = output_kind

    def load_model(model_id, **kwargs):
        return model

    module.load_model = load_model
    monkeypatch.setitem(sys.modules, "tirex", module)


@pytest.mark.parametrize("output_kind", ["malformed", "nonfinite", "short"])
def test_tirex_rejects_malformed_nonfinite_and_short_point_outputs(monkeypatch, output_kind):
    _install_boundary_tirex(monkeypatch, output_kind)
    from models.TiRex import TiRexBackend
    from models.registry import get_model_spec

    backend = TiRexBackend(get_model_spec("TiRex"), device="cpu")
    with pytest.raises(ValueError):
        backend.predict(torch.ones(1, 3, 1), 2)


def test_tirex_reports_directed_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "tirex", None)
    from models.TiRex import TiRexBackend
    from models.registry import get_model_spec

    with pytest.raises(ImportError, match="tirex-ts.*Python 3.10"):
        TiRexBackend(get_model_spec("TiRex"), device="cpu")


class _BoundaryTimesFMModel(nn.Module):
    output_kind = "valid"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, *, past_values, return_dict):
        batch = len(past_values)
        horizon = 2
        if self.output_kind == "malformed":
            point = torch.ones(batch, horizon, 2)
        elif self.output_kind == "nonfinite":
            point = torch.tensor([[1.0, float("nan")]]).expand(batch, -1)
        elif self.output_kind == "short":
            point = torch.ones(batch, 1)
        else:
            point = torch.ones(batch, horizon)
        return types.SimpleNamespace(mean_predictions=point)


class _BoundaryTimesFMClass:
    output_kind = "valid"

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        model = _BoundaryTimesFMModel()
        model.output_kind = cls.output_kind
        return model


def _install_boundary_timesfm(monkeypatch, output_kind):
    module = types.ModuleType("transformers")
    _BoundaryTimesFMClass.output_kind = output_kind
    module.TimesFm2_5ModelForPrediction = _BoundaryTimesFMClass
    monkeypatch.setitem(sys.modules, "transformers", module)


@pytest.mark.parametrize("output_kind", ["malformed", "nonfinite", "short"])
def test_timesfm_rejects_malformed_nonfinite_and_short_point_outputs(monkeypatch, output_kind):
    _install_boundary_timesfm(monkeypatch, output_kind)
    from models.TimesFM import TimesFMBackend
    from models.registry import get_model_spec

    backend = TimesFMBackend(get_model_spec("TimesFM"), device="cpu")
    with pytest.raises(ValueError):
        backend.predict(torch.ones(1, 3, 1), 2)


def test_timesfm_reports_directed_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    from models.TimesFM import TimesFMBackend
    from models.registry import get_model_spec

    with pytest.raises(ImportError, match="TimesFm2_5ModelForPrediction.*Python 3.10"):
        TimesFMBackend(get_model_spec("TimesFM"), device="cpu")
