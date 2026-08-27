from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest
import torch
from torch import nn

from models.registry import FoundationModelSpec, get_model_spec


class TinyTrainableModel(nn.Module):
    def __init__(self, *, output_name: str = "output"):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.q_proj = nn.Linear(3, 3)
        self.backbone.k_proj = nn.Linear(3, 3)
        self.backbone.v_proj = nn.Linear(3, 3)
        self.backbone.o_proj = nn.Linear(3, 3)
        self.backbone.gate_proj = nn.Linear(3, 3)
        self.backbone.up_proj = nn.Linear(3, 3)
        self.backbone.down_proj = nn.Linear(3, 3)
        setattr(self, output_name, nn.Linear(3, 1))


def _spec(selector: str = "output") -> FoundationModelSpec:
    return FoundationModelSpec(
        name="Tiny",
        model_id="tiny",
        revision="test",
        entrypoint="tests.test_foundation_trainability:TinyTrainableModel",
        last_layer_selector=selector,
    )


def test_zero_shot_freezes_all_and_report_is_deterministic():
    from models.trainability import configure_trainable

    model = TinyTrainableModel()
    report = configure_trainable(model, _spec(), "zero_shot")

    assert report.mode == "zero_shot"
    assert report.total_parameters == sum(p.numel() for p in model.parameters())
    assert report.trainable_parameters == 0
    assert report.trainable_ratio == 0.0
    assert report.trainable_names == ()
    assert report.trainable_names_sha256 == hashlib.sha256(b"").hexdigest()
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_full_unfreezes_every_parameter_and_reports_sorted_names():
    from models.trainability import configure_trainable

    model = TinyTrainableModel()
    report = configure_trainable(model, _spec(), "full")
    expected_names = tuple(sorted(name for name, _ in model.named_parameters()))

    assert report.trainable_parameters == report.total_parameters
    assert report.trainable_names == expected_names[:50]
    assert report.trainable_names_sha256 == hashlib.sha256(
        "\n".join(expected_names).encode("utf-8")
    ).hexdigest()
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_last_layer_matches_only_the_exact_output_subtree():
    from models.trainability import configure_trainable

    model = TinyTrainableModel()
    report = configure_trainable(model, _spec("output"), "last_layer")

    assert report.trainable_parameters == sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name == "output.weight" or name.startswith("output.")
    )
    assert report.trainable_names == ("output.bias", "output.weight")
    assert all(
        parameter.requires_grad
        == (name == "output" or name.startswith("output."))
        for name, parameter in model.named_parameters()
    )


def test_adapter_injects_real_lora_parameters_and_freezes_base():
    from models.trainability import LoraSettings, configure_trainable

    model = TinyTrainableModel()
    report = configure_trainable(
        model,
        _spec(),
        "adapter",
        LoraSettings(r=2, lora_alpha=4, lora_dropout=0.0),
    )

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert report.trainable_parameters > 0
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    assert all(
        parameter.requires_grad == ("lora_" in name)
        for name, parameter in model.named_parameters()
    )


def test_adapter_does_not_import_peft_until_configuration():
    probe = (
        "import sys\n"
        "import models.trainability\n"
        "assert 'peft' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["unknown", "", "FULL"])
def test_unknown_trainability_mode_is_rejected(mode):
    from models.trainability import configure_trainable

    with pytest.raises(ValueError, match="mode"):
        configure_trainable(TinyTrainableModel(), _spec(), mode)


def test_last_layer_rejects_empty_selector_and_selector_covering_every_parameter():
    from models.trainability import configure_trainable

    with pytest.raises(ValueError, match="selector"):
        configure_trainable(TinyTrainableModel(), _spec(""), "last_layer")

    model = TinyTrainableModel()
    del model.output
    with pytest.raises(ValueError, match="all|total|subtree"):
        configure_trainable(model, _spec("backbone"), "last_layer")


def test_adapter_rejects_models_without_matching_target_modules():
    from models.trainability import configure_trainable

    class NoTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="target"):
        configure_trainable(NoTarget(), _spec(), "adapter")


def test_adapter_rejects_injection_that_adds_no_lora_parameters(monkeypatch):
    import peft

    from models import trainability

    monkeypatch.setattr(peft, "inject_adapter_in_model", lambda config, model: model)
    with pytest.raises(ValueError, match="no trainable|LoRA"):
        trainability.configure_trainable(
            TinyTrainableModel(), _spec(), "adapter", trainability.LoraSettings()
        )


def test_adapter_rejects_base_parameter_leakage(monkeypatch):
    import peft

    from models import trainability

    def leak_base_parameter(config, model):
        model.backbone.q_proj.bias.requires_grad_(True)
        return model

    monkeypatch.setattr(peft, "inject_adapter_in_model", leak_base_parameter)
    with pytest.raises(ValueError, match="base parameters|leak"):
        trainability.configure_trainable(
            TinyTrainableModel(), _spec(), "adapter", trainability.LoraSettings()
        )


def test_full_rejects_an_audit_when_unfreeze_fails(monkeypatch):
    from models import trainability

    model = TinyTrainableModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    monkeypatch.setattr(trainability, "_unfreeze", lambda parameters: None)
    with pytest.raises(ValueError, match="full.*audit|every parameter"):
        trainability.configure_trainable(model, _spec(), "full")


def test_zero_parameter_models_are_rejected():
    from models.trainability import configure_trainable

    with pytest.raises(ValueError, match="parameter"):
        configure_trainable(nn.Identity(), _spec(), "zero_shot")


def test_report_is_frozen():
    from dataclasses import FrozenInstanceError

    from models.trainability import configure_trainable

    report = configure_trainable(TinyTrainableModel(), _spec(), "full")
    with pytest.raises(FrozenInstanceError):
        report.mode = "changed"


@pytest.mark.parametrize("mode", ["zero_shot", "adapter", "full", "last_layer"])
def test_optimizer_step_changes_only_parameters_allowed_by_mode(mode):
    from models.trainability import LoraSettings, configure_trainable

    model = TinyTrainableModel()
    configure_trainable(
        model,
        _spec(),
        mode,
        LoraSettings(r=2, lora_alpha=4) if mode == "adapter" else None,
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if mode == "zero_shot":
        assert trainable == []
        return

    optimizer = torch.optim.SGD(trainable, lr=0.1)
    values = torch.ones(2, 3)
    hidden = model.backbone.q_proj(values)
    loss = model.output(hidden).sum()
    loss.backward()
    optimizer.step()

    changed = {
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, before[name])
    }
    intended = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert changed.intersection(intended)
    assert all(
        name in intended or torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )


class ModernChronosTrainableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.Module()
        self.self_attention.q = nn.Linear(2, 2)
        self.self_attention.v = nn.Linear(2, 2)
        self.self_attention.k = nn.Linear(2, 2)
        self.self_attention.o = nn.Linear(2, 2)
        self.output_patch_embedding = nn.Module()
        self.output_patch_embedding.output_layer = nn.Linear(2, 1)
        self.unrelated = nn.Linear(2, 2)


class ModernTimesFMTrainableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_projection_point = nn.Linear(2, 1)
        self.decoder = nn.Linear(2, 2)


@pytest.mark.parametrize(
    ("model_name", "model_type", "last_prefix"),
    [
        ("Chronos2", ModernChronosTrainableModel, "output_patch_embedding"),
        ("TimesFM", ModernTimesFMTrainableModel, "output_projection_point"),
    ],
)
def test_new_model_specs_support_real_adapter_full_and_strict_last_layer(
    model_name, model_type, last_prefix
):
    from models.trainability import LoraSettings, configure_trainable

    spec = get_model_spec(model_name)
    for mode in ("adapter", "full", "last_layer"):
        model = model_type()
        report = configure_trainable(
            model,
            spec,
            mode,
            LoraSettings(r=2, lora_alpha=4) if mode == "adapter" else None,
        )
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if mode == "adapter":
            assert names and all("lora_" in name for name in names)
        elif mode == "full":
            assert report.trainable_parameters == report.total_parameters
            assert names == tuple(name for name, _ in model.named_parameters())
        else:
            assert names and all(name == last_prefix or name.startswith(last_prefix + ".") for name in names)
