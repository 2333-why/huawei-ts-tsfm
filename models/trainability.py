"""Deterministic trainability policies and parameter audits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from torch import nn

from .registry import DEFAULT_ADAPTER_TARGET_MODULES


ADAPTER_TARGET_MODULES = DEFAULT_ADAPTER_TARGET_MODULES
_RUN_MODES = ("zero_shot", "adapter", "full", "last_layer")


@dataclass(frozen=True)
class LoraSettings:
    """Configuration for the explicit transformer-linear LoRA adapter."""

    r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: Optional[str] = None
    fan_in_fan_out: bool = False


@dataclass(frozen=True)
class TrainabilityReport:
    """Immutable, deterministic audit data for one trainability policy."""

    mode: str
    total_parameters: int
    trainable_parameters: int
    trainable_ratio: float
    trainable_names: Tuple[str, ...]
    trainable_names_sha256: str

    @property
    def total_params(self) -> int:
        return self.total_parameters

    @property
    def trainable_params(self) -> int:
        return self.trainable_parameters

    @property
    def trainable_parameter_names(self) -> Tuple[str, ...]:
        return self.trainable_names

    @property
    def trainable_name_digest(self) -> str:
        return self.trainable_names_sha256


def _freeze_all(parameters: Sequence[Tuple[str, nn.Parameter]]) -> None:
    for _, parameter in parameters:
        parameter.requires_grad_(False)


def _unfreeze(parameters: Sequence[Tuple[str, nn.Parameter]]) -> None:
    for _, parameter in parameters:
        parameter.requires_grad_(True)


def _matching_prefix(name: str, selector: str) -> bool:
    return name == selector or name.startswith(selector + ".")


def _build_report(
    mode: str,
    parameters: Sequence[Tuple[str, nn.Parameter]],
) -> TrainabilityReport:
    total_parameters = sum(parameter.numel() for _, parameter in parameters)
    if total_parameters <= 0:
        raise ValueError("cannot audit a model with zero parameters")

    trainable = [
        (name, parameter)
        for name, parameter in parameters
        if bool(parameter.requires_grad)
    ]
    trainable_names_complete = tuple(sorted(name for name, _ in trainable))
    trainable_parameters = sum(parameter.numel() for _, parameter in trainable)
    digest = hashlib.sha256(
        "\n".join(trainable_names_complete).encode("utf-8")
    ).hexdigest()
    return TrainabilityReport(
        mode=mode,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        trainable_ratio=float(trainable_parameters) / float(total_parameters),
        trainable_names=trainable_names_complete[:50],
        trainable_names_sha256=digest,
    )


def _audit_report(
    mode: str,
    parameters: Sequence[Tuple[str, nn.Parameter]],
    expected_names: Optional[Sequence[str]] = None,
) -> TrainabilityReport:
    report = _build_report(mode, parameters)
    actual_names = tuple(sorted(name for name, parameter in parameters if parameter.requires_grad))

    if mode == "zero_shot" and actual_names:
        raise ValueError(
            "zero_shot trainability audit failed: expected zero trainable parameters, "
            f"got {len(actual_names)}"
        )
    if mode == "full" and report.trainable_parameters != report.total_parameters:
        raise ValueError(
            "full trainability audit failed: expected every parameter to be trainable, "
            f"got {report.trainable_parameters}/{report.total_parameters}"
        )
    if expected_names is not None and actual_names != tuple(sorted(expected_names)):
        raise ValueError(
            f"{mode} trainability audit failed: trainable parameter names leaked or "
            "were not enabled as requested"
        )
    if mode == "adapter":
        if not actual_names:
            raise ValueError(
                "adapter trainability audit failed: LoRA injection added no trainable parameters"
            )
        leaked = [name for name in actual_names if "lora_" not in name]
        if leaked:
            raise ValueError(
                "adapter trainability audit failed: base parameters are trainable: "
                + ", ".join(leaked[:5])
            )
    return report


def _adapter_module_matches(name: str, target_modules: Sequence[str]) -> bool:
    return any(name == suffix or name.endswith("." + suffix) for suffix in target_modules)


def _configure_adapter(
    model: nn.Module,
    lora_settings: LoraSettings,
    target_modules: Sequence[str],
) -> TrainabilityReport:
    if lora_settings.bias != "none":
        raise ValueError(
            "adapter trainability requires bias='none' so no base bias parameters leak"
        )
    matched_modules = [
        name
        for name, module in model.named_modules()
        if name and _adapter_module_matches(name, target_modules) and isinstance(module, nn.Linear)
    ]
    if not matched_modules:
        raise ValueError(
            "adapter configuration found no matching nn.Linear target modules; "
            "expected configured transformer linear target modules"
        )

    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:
        raise ImportError(
            "adapter trainability requires PEFT 0.13.2; install the optional dependency"
        ) from exc

    try:
        config = LoraConfig(
            r=lora_settings.r,
            lora_alpha=lora_settings.lora_alpha,
            lora_dropout=lora_settings.lora_dropout,
            bias=lora_settings.bias,
            task_type=lora_settings.task_type,
            target_modules=list(target_modules),
            fan_in_fan_out=lora_settings.fan_in_fan_out,
        )
        inject_adapter_in_model(config, model)
    except ImportError:
        raise
    except Exception as exc:
        raise ValueError(f"adapter LoRA injection failed: {exc}") from exc

    updated_parameters = list(model.named_parameters())
    if not updated_parameters:
        raise ValueError("adapter LoRA injection left the model with no parameters")
    return _audit_report("adapter", updated_parameters)


def configure_trainable(
    model: nn.Module,
    spec: Any,
    mode: str,
    lora_settings: Optional[LoraSettings] = None,
) -> TrainabilityReport:
    """Apply ``mode`` to ``model`` in place and return a parameter audit."""

    if mode not in _RUN_MODES:
        raise ValueError(
            f"unknown trainability mode {mode!r}; available modes: {', '.join(_RUN_MODES)}"
        )
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module")

    parameters = list(model.named_parameters())
    total_parameters = sum(parameter.numel() for _, parameter in parameters)
    if total_parameters <= 0:
        raise ValueError("cannot configure a model with zero parameters")

    if mode == "zero_shot":
        _freeze_all(parameters)
        return _audit_report(mode, parameters, expected_names=())

    if mode == "full":
        _unfreeze(parameters)
        return _audit_report(mode, parameters, expected_names=[name for name, _ in parameters])

    if mode == "last_layer":
        selector = getattr(spec, "last_layer_selector", None)
        if not isinstance(selector, str) or not selector.strip():
            raise ValueError("last_layer requires a nonempty last_layer_selector")
        selector = selector.strip()
        selected_names = [
            name for name, _ in parameters if _matching_prefix(name, selector)
        ]
        if not selected_names:
            raise ValueError(
                f"last_layer selector {selector!r} matched no parameters"
            )
        selected_parameters = [
            (name, parameter)
            for name, parameter in parameters
            if name in selected_names
        ]
        selected_count = sum(parameter.numel() for _, parameter in selected_parameters)
        if selected_count <= 0:
            raise ValueError(
                f"last_layer selector {selector!r} matched no parameters"
            )
        if selected_count >= total_parameters:
            raise ValueError(
                f"last_layer selector {selector!r} matches all model parameters; "
                "a strict last-layer subtree must be a proper subset"
            )
        _freeze_all(parameters)
        _unfreeze(selected_parameters)
        return _audit_report(mode, parameters, expected_names=selected_names)

    if lora_settings is None:
        lora_settings = LoraSettings()
    if not isinstance(lora_settings, LoraSettings):
        raise ValueError("adapter lora_settings must be a LoraSettings instance")
    _freeze_all(parameters)
    target_modules = getattr(spec, "adapter_target_modules", ADAPTER_TARGET_MODULES)
    if not isinstance(target_modules, (tuple, list)) or not target_modules:
        raise ValueError("adapter requires nonempty adapter_target_modules")
    return _configure_adapter(model, lora_settings, tuple(target_modules))


__all__ = [
    "ADAPTER_TARGET_MODULES",
    "LoraSettings",
    "TrainabilityReport",
    "configure_trainable",
]
