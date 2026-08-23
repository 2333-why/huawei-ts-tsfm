"""Minimal synthetic CUDA compatibility matrix for the selected TSLib models.

The matrix deliberately uses the real registry, factory, and power-only
adapter.  It is bounded to one train step and one forward for each validation
and test phase; it is not a dataset or epoch runner.
"""

from __future__ import annotations

import gc
import importlib
import math
import traceback
from types import SimpleNamespace
from typing import Iterable, Optional

import torch

from models.tslib_adapter import forward_power_model, prepare_model_inputs
from models.tslib_factory import build_power_model
from models.tslib_registry import SELECTED_MODEL_NAMES


BATCH_SIZE = 2
SEQ_LEN = 16
PRED_LEN = 16
D_MODEL = 8
N_HEADS = 2
E_LAYERS = 1
D_LAYERS = 1
D_FF = 16


class SyntheticPowerDataset:
    """Tiny iterable used only to derive Koopa's power-frequency mask."""

    seq_len = SEQ_LEN
    pred_len = PRED_LEN

    def __iter__(self):
        for offset in range(BATCH_SIZE):
            yield (
                torch.arange(SEQ_LEN, dtype=torch.float32).reshape(SEQ_LEN, 1)
                + offset,
            )


def matrix_args(device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        e_layers=E_LAYERS,
        d_layers=D_LAYERS,
        d_ff=D_FF,
        factor=2,
        dropout=0.0,
        moving_avg=3,
        top_k=2,
        num_kernels=1,
        patch_len=SEQ_LEN,
        patch_stride=8,
        batch_size=BATCH_SIZE,
        device=str(device),
    )


def _model_class_name(model) -> str:
    cls = type(model)
    return f"{cls.__module__}.{cls.__name__}"


def _assert_power_only_inputs(model_name: str, x: torch.Tensor) -> None:
    """Ensure a large extra channel cannot affect any model-facing tensor."""

    polluted = torch.cat((x, torch.full_like(x, 987654.0)), dim=-1)
    clean_inputs = prepare_model_inputs(model_name, x, SEQ_LEN, PRED_LEN)
    polluted_inputs = prepare_model_inputs(
        model_name, polluted, SEQ_LEN, PRED_LEN
    )
    for clean, dirty in zip(clean_inputs, polluted_inputs):
        if torch.is_tensor(clean):
            assert torch.equal(clean, dirty), (
                f"{model_name} model input changed when a non-power channel "
                "was polluted"
            )
        else:
            assert clean is dirty or clean == dirty


def _base_row(model_name: str, physical_gpu: Optional[int], device: torch.device):
    process_gpu = torch.cuda.current_device() if device.type == "cuda" else None
    gpu_name = (
        torch.cuda.get_device_name(process_gpu) if process_gpu is not None else None
    )
    return {
        "model": model_name,
        "status": "FAIL",
        "physical_gpu": physical_gpu,
        "process_gpu": process_gpu,
        "gpu": gpu_name,
        "model_class": None,
        "phase_steps": {"train": 0, "val": 0, "test": 0},
        "train_forward": False,
        "loss": False,
        "backward": False,
        "optimizer_step": False,
        "val_forward": False,
        "test_forward": False,
        "power_only_pollution_assert": False,
        "exception": None,
    }


def run_model(
    model_name: str,
    *,
    device: torch.device,
    physical_gpu: Optional[int] = None,
) -> dict:
    """Run exactly one train/val/test step for one model and return its row."""

    if device.type != "cuda":
        raise RuntimeError(
            "Task7 synthetic model matrix requires CUDA; use a CUDA device"
        )
    row = _base_row(model_name, physical_gpu, device)
    model = None
    try:
        args = matrix_args(device)
        dataset = SyntheticPowerDataset()
        model, config = build_power_model(model_name, args, dataset)
        row["model_class"] = _model_class_name(model)
        model = model.to(device)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, 1, device=device)
        target = torch.randn(BATCH_SIZE, PRED_LEN, 1, device=device)

        _assert_power_only_inputs(model_name, x)
        row["power_only_pollution_assert"] = True

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_power_model(
            model_name, model, x, config.label_len, config.pred_len
        )
        row["train_forward"] = True
        row["phase_steps"]["train"] = 1
        assert tuple(prediction.shape) == (BATCH_SIZE, PRED_LEN, 1), tuple(
            prediction.shape
        )
        loss = torch.nn.functional.mse_loss(prediction, target)
        assert torch.isfinite(loss).item(), f"non-finite loss: {loss!r}"
        row["loss"] = True
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert gradients, "no parameter received a gradient"
        assert all(torch.isfinite(gradient).all().item() for gradient in gradients)
        row["backward"] = True
        optimizer.step()
        row["optimizer_step"] = True

        model.eval()
        with torch.no_grad():
            validation = forward_power_model(
                model_name, model, x, config.label_len, config.pred_len
            )
            assert tuple(validation.shape) == (BATCH_SIZE, PRED_LEN, 1), tuple(
                validation.shape
            )
            row["val_forward"] = True
            row["phase_steps"]["val"] = 1
            test = forward_power_model(
                model_name, model, x, config.label_len, config.pred_len
            )
            assert tuple(test.shape) == (BATCH_SIZE, PRED_LEN, 1), tuple(test.shape)
            row["test_forward"] = True
            row["phase_steps"]["test"] = 1
        row["status"] = "PASS"
    except Exception as exc:  # record and continue to the next matrix model
        row["exception"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def mamba_verification() -> dict:
    """Return evidence that the selected Mamba entry is not MambaSimple."""

    module = importlib.import_module("models.Mamba")
    package = importlib.import_module("mamba_ssm")
    return {
        "model_constructor": f"{module.Model.__module__}.{module.Model.__name__}",
        "mamba_ssm_version": getattr(package, "__version__", None),
        "mamba_ssm_constructor": (
            f"{package.Mamba.__module__}.{package.Mamba.__name__}"
        ),
        "is_mamba_simple": module.Model.__module__ == "models.MambaSimple",
    }


def run_matrix(
    model_names: Iterable[str],
    *,
    device: torch.device,
    physical_gpu: Optional[int] = None,
) -> dict:
    """Run a deterministic serial shard and return a machine-readable report."""

    model_names = tuple(model_names)
    unknown = set(model_names).difference(SELECTED_MODEL_NAMES)
    if unknown:
        raise ValueError(f"unknown selected model names: {sorted(unknown)}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Task7 matrix requires an available CUDA device")
    rows = [
        run_model(name, device=device, physical_gpu=physical_gpu)
        for name in model_names
    ]
    passed = sum(row["status"] == "PASS" for row in rows)
    return {
        "schema_version": 1,
        "configuration": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "e_layers": E_LAYERS,
            "d_layers": D_LAYERS,
            "d_ff": D_FF,
            "model_count": len(model_names),
            "serial_per_gpu": True,
        },
        "physical_gpu": physical_gpu,
        "process_gpu": torch.cuda.current_device(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "mamba": mamba_verification(),
        "summary": {
            "total": len(rows),
            "passed": passed,
            "failed": len(rows) - passed,
        },
        "models": rows,
    }


__all__ = [
    "BATCH_SIZE",
    "D_MODEL",
    "D_FF",
    "D_LAYERS",
    "E_LAYERS",
    "N_HEADS",
    "PRED_LEN",
    "SEQ_LEN",
    "run_matrix",
    "run_model",
    "mamba_verification",
]
