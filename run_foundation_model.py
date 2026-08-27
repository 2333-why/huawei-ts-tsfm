"""Run one foundation-model power forecasting experiment.

The foundation runner deliberately has its own lifecycle.  It shares only the
stable prediction/metric/hash primitives with ``run_time_series`` and keeps
optional model dependencies behind backend construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.power_only import PowerOnlyParquetDataset
from foundation_models.factory import build_backend
from foundation_models.registry import MODEL_NAMES, RUN_MODES, get_model_spec
from foundation_models.trainability import LoraSettings
from models.tslib_adapter import power_only_batch
from run_time_series import _json_safe, _sha256, _write_predictions


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = {
    "skippd_luoyang": REPO_ROOT / "configs/datasets/skippd_luoyang.json",
    "pvod_station00_ylj": REPO_ROOT / "configs/datasets/pvod_station00_ylj.yaml",
}
DATASET_LOADERS = {
    "skippd_luoyang": PowerOnlyParquetDataset,
    "pvod_station00_ylj": PowerOnlyParquetDataset,
}
_ORIGINAL_DATASET_CLASS = PowerOnlyParquetDataset

FOUNDATION_SCHEMA_VERSION = "2"
CHECKPOINT_SCHEMA_VERSION = "foundation-checkpoint-v1"
COMPLETION_HEADER = (
    "schema_version",
    "dataset",
    "model",
    "seq_len",
    "pred_len",
    "output_dir",
    "run_mode",
    "epochs",
    "max_train_steps",
    "max_eval_steps",
    "max_test_steps",
    "train_steps",
    "val_steps",
    "test_steps",
    "best_sha256",
    "predictions_sha256",
    "metrics_sha256",
    "mode",
    "model_id",
    "model_revision",
)
FOUNDATION_COMPLETION_HEADER = COMPLETION_HEADER


def _read_config(path: Path) -> dict:
    """Read a JSON or YAML dataset configuration when a default output is used."""

    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _device_from_arg(value: Any) -> torch.device:
    """Resolve a CPU/CUDA device and reject unavailable CUDA spellings."""

    if isinstance(value, str) and value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(value, torch.device):
        device = value
    else:
        try:
            device = torch.device(str(value).lower())
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid device {value!r}") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {device} was requested but CUDA is unavailable"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is unavailable; "
                f"device_count={torch.cuda.device_count()}"
            )
    elif device.type != "cpu":
        raise ValueError(f"unsupported device type {device.type!r}; use cpu or cuda")
    return device


def _step_limit(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_steps must be a non-negative integer") from exc
    if value < 0:
        raise ValueError("max_steps must be non-negative")
    return None if value == 0 else value


def _validate_runtime_args(args: Any) -> None:
    """Validate the runtime contract before constructing any dataset/backend."""

    dataset = getattr(args, "dataset", None)
    model = getattr(args, "model", None)
    mode = getattr(args, "mode", None)
    if dataset not in DATASET_LOADERS:
        raise ValueError(
            f"dataset is required and must be one of: {', '.join(sorted(DATASET_LOADERS))}"
        )
    if model not in MODEL_NAMES:
        raise ValueError(
            f"model is required and must be one of: {', '.join(MODEL_NAMES)}"
        )
    if mode not in RUN_MODES:
        raise ValueError(
            f"mode is required and must be one of: {', '.join(RUN_MODES)}"
        )

    for name in ("seq_len", "pred_len", "batch_size"):
        value = getattr(args, name, None)
        if value is None or isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must be positive")

    for name in ("max_train_steps", "max_eval_steps", "max_test_steps"):
        value = getattr(args, name, 0)
        if value is None:
            setattr(args, name, 0)
            value = 0
        elif int(value) < 0:
            raise ValueError(f"{name} must be non-negative")
        elif not hasattr(args, name):
            setattr(args, name, int(value))

    learning_rate = float(getattr(args, "learning_rate", 1e-3))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    workers = int(getattr(args, "num_workers", 0))
    if workers < 0:
        raise ValueError("num_workers must be non-negative")
    prefetch = int(getattr(args, "prefetch_factor", 2))
    if workers > 0 and prefetch <= 0:
        raise ValueError("prefetch_factor must be positive when num_workers is positive")

    if bool(getattr(args, "smoke", False)):
        args.epochs = 0 if mode == "zero_shot" else 1
        args.max_train_steps = 1
        args.max_eval_steps = 1
        args.max_test_steps = 1
        args.batch_size = 2

    # zero_shot always reports a truthful zero-epoch run, even when callers
    # construct a Namespace directly instead of going through parse_args.
    if mode == "zero_shot":
        args.epochs = 0
    else:
        epochs = getattr(args, "epochs", 1)
        if epochs is None or int(epochs) < 1:
            raise ValueError("epochs must be at least 1 for training modes")
        args.epochs = int(epochs)

    if getattr(args, "config", None) is None:
        args.config = DEFAULT_CONFIGS[dataset]
    args.config = Path(args.config).resolve()


def _dataset_and_loader(args: Any, flag: str):
    """Construct one split and its loader; this is the zero-shot test seam."""

    # Keep the concrete class as a direct seam: tests and offline callers can
    # replace it without having to rebuild the catalog mapping used by argparse.
    dataset_class = DATASET_LOADERS[args.dataset]
    if dataset_class is _ORIGINAL_DATASET_CLASS:
        dataset_class = PowerOnlyParquetDataset
    device_value = getattr(args, "device", "auto")
    device = device_value if isinstance(device_value, torch.device) else _device_from_arg(device_value)
    dataset = dataset_class(
        config_path=args.config,
        flag=flag,
        history_points=int(args.seq_len),
        forecast_steps=int(args.pred_len),
    )
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": flag == "train",
        "drop_last": False,
        "num_workers": int(getattr(args, "num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=int(getattr(args, "prefetch_factor", 2)),
        )
    return dataset, DataLoader(dataset, **loader_kwargs)


def _prepare_batch(batch: Any, device: torch.device, seq_len: int, pred_len: int):
    """Validate the fixed dataset mapping and move its tensors to ``device``."""

    if not isinstance(batch, Mapping):
        raise ValueError("dataset batch must be a mapping")
    if "target_mask" not in batch:
        raise ValueError("dataset batch mapping must contain target_mask")

    try:
        raw_mask = torch.as_tensor(batch["target_mask"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("target_mask must be convertible to a tensor") from exc
    if raw_mask.ndim == 3 and raw_mask.shape[-1] == 1:
        mask_for_check = raw_mask[..., 0]
    else:
        mask_for_check = raw_mask
    if mask_for_check.ndim != 2:
        raise ValueError("target_mask must have shape [batch, horizon] or [batch, horizon, 1]")
    if mask_for_check.is_complex():
        raise ValueError("target_mask must be boolean or real numeric")
    if mask_for_check.is_floating_point() and not bool(torch.isfinite(mask_for_check).all()):
        raise ValueError("target_mask must contain only finite values")
    if not bool(((mask_for_check == 0) | (mask_for_check == 1)).all()):
        raise ValueError("target_mask must contain only binary 0/1 values")

    power_batch = power_only_batch(batch, device)
    history, target, target_mask = power_batch.x, power_batch.y, power_batch.mask
    if tuple(history.shape[1:]) != (int(seq_len), 1):
        raise ValueError(
            f"history must have shape [batch, {seq_len}, 1], got {tuple(history.shape)}"
        )
    if tuple(target.shape[1:]) != (int(pred_len), 1):
        raise ValueError(
            f"target must have shape [batch, {pred_len}, 1], got {tuple(target.shape)}"
        )
    if not bool(torch.isfinite(history).all()):
        raise ValueError("history must contain only finite values")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target must contain only finite values")
    if target_mask.shape != target.shape[:2]:
        raise ValueError("target_mask shape does not match target")
    return history, target, target_mask


def _issue_times(batch: Mapping, batch_size: int) -> np.ndarray:
    value = batch.get("issue_time_ns")
    if value is None:
        return np.full((batch_size,), np.nan, dtype=np.float64)
    try:
        result = torch.as_tensor(value).detach().cpu().numpy().reshape(-1)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("issue_time_ns must be convertible to a one-dimensional array") from exc
    if result.size != batch_size:
        raise ValueError(
            f"issue_time_ns must contain one value per batch row, got {result.size} for {batch_size}"
        )
    return result


def _validate_prediction(prediction: Any, batch_size: int, pred_len: int) -> torch.Tensor:
    """Validate the backend trust boundary before any clipping or serialization."""

    if not torch.is_tensor(prediction):
        raise ValueError("backend prediction must be a torch.Tensor")
    if not prediction.is_floating_point():
        raise ValueError("backend prediction must use a floating-point dtype")
    expected = (int(batch_size), int(pred_len), 1)
    if tuple(prediction.shape) != expected:
        raise ValueError(f"backend prediction must have shape {expected}, got {tuple(prediction.shape)}")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("backend prediction must contain only finite values")
    return prediction


def _limited_batches(loader: Iterable, max_steps: Optional[int]):
    limit = _step_limit(max_steps)
    return loader if limit is None else islice(loader, limit)


def _validate_loss(loss: Any) -> torch.Tensor:
    if not torch.is_tensor(loss):
        raise ValueError("native training loss must be a torch.Tensor")
    if loss.ndim != 0:
        raise ValueError("native training loss must be a scalar tensor")
    if not bool(torch.isfinite(loss).all()):
        raise ValueError("native training loss must be finite")
    if not loss.requires_grad:
        raise ValueError("native training loss must require gradients")
    return loss


def _train_epoch(
    backend: Any,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seq_len: int,
    pred_len: int,
    max_steps: Optional[int],
) -> Tuple[float, int]:
    model = getattr(backend, "model", None)
    if model is None or not callable(getattr(model, "train", None)):
        raise ValueError("backend must expose a trainable model")
    model.train()
    total_loss = 0.0
    total_valid = 0
    steps = 0
    for batch in _limited_batches(loader, max_steps):
        history, target, target_mask = _prepare_batch(batch, device, seq_len, pred_len)
        steps += 1
        valid = int(target_mask.sum().item())
        if valid == 0:
            # The batch counts toward the requested limit, but contributes no
            # optimizer or native-loss work.
            continue
        optimizer.zero_grad(set_to_none=True)
        loss = _validate_loss(backend.training_loss(history, target, target_mask))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item()) * valid
        total_valid += valid

    if steps == 0:
        limit_label = "unlimited" if _step_limit(max_steps) is None else str(max_steps)
        raise RuntimeError(f"train loader is empty; max_steps={limit_label} requires at least one batch")
    if total_valid == 0:
        raise RuntimeError("train phase has zero valid targets")
    return total_loss / total_valid, steps


def _collect_predictions(
    backend: Any,
    loader: Iterable,
    device: torch.device,
    seq_len: int,
    pred_len: int,
    max_steps: Optional[int],
    phase: str,
) -> Dict[str, Any]:
    model = getattr(backend, "model", None)
    if model is None or not callable(getattr(model, "eval", None)):
        raise ValueError("backend must expose an evaluable model")
    model.eval()
    total_sse = 0.0
    total_valid = 0
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    issue_times: List[np.ndarray] = []
    steps = 0

    for batch in _limited_batches(loader, max_steps):
        history, target, target_mask = _prepare_batch(batch, device, seq_len, pred_len)
        steps += 1
        with torch.inference_mode():
            prediction = _validate_prediction(
                backend.predict(history, pred_len), history.shape[0], pred_len
            )
        batch_mask = target_mask.unsqueeze(-1).to(dtype=prediction.dtype)
        squared = (prediction - target).square()
        if not bool(torch.isfinite(squared).all()):
            raise ValueError(f"{phase} prediction loss is non-finite")
        total_sse += float((squared * batch_mask).sum().detach().cpu().item())
        total_valid += int(target_mask.sum().item())
        predictions.append(prediction.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())
        issue_times.append(_issue_times(batch, history.shape[0]))

    if steps == 0:
        limit_label = "unlimited" if _step_limit(max_steps) is None else str(max_steps)
        raise RuntimeError(f"{phase} loader is empty; max_steps={limit_label} requires at least one batch")
    if total_valid == 0:
        raise RuntimeError(f"{phase} phase has zero valid targets")
    loss = total_sse / total_valid
    if not math.isfinite(loss):
        raise ValueError(f"{phase} prediction loss is non-finite")
    return {
        "loss": loss,
        "steps": steps,
        "prediction": np.concatenate(predictions, axis=0),
        "target": np.concatenate(targets, axis=0),
        "mask": np.concatenate(masks, axis=0),
        "issue_time_ns": np.concatenate(issue_times, axis=0),
    }


def _clip_result(result: Mapping[str, Any], dataset: Any) -> Dict[str, Any]:
    """Clip finite normalized predictions once to the dataset's allowed range."""

    try:
        scale = float(dataset.power_scale)
        rated_power = float(dataset.rated_power)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("dataset power_scale and rated_power must be numeric") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("dataset power_scale must be finite and positive")
    if not math.isfinite(rated_power) or rated_power <= 0:
        raise ValueError("dataset rated_power must be finite and positive")
    floor = getattr(dataset, "target_floor", None)
    lower = 0.0 if floor is None else float(floor) / scale
    upper = rated_power / scale
    if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0 or lower > upper:
        raise ValueError("dataset prediction clipping bounds must be finite and ordered")

    prediction = np.asarray(result["prediction"])
    if prediction.ndim != 3 or prediction.shape[-1] != 1:
        raise ValueError("prediction must have shape [batch, horizon, 1]")
    if not np.isfinite(prediction).all():
        raise ValueError("backend prediction must be finite before clipping")
    clipped = np.clip(prediction, lower, upper)
    if not np.isfinite(clipped).all():
        raise ValueError("clipped prediction must be finite")
    copied = dict(result)
    copied["prediction"] = clipped
    mask = np.asarray(copied["mask"], dtype=bool)
    target = np.asarray(copied["target"])
    valid = int(mask.sum())
    if valid <= 0:
        raise RuntimeError("test phase has zero valid targets")
    squared = (clipped - target) ** 2
    if not np.isfinite(squared).all():
        raise ValueError("test prediction loss is non-finite")
    copied["loss"] = float((squared[..., 0][mask]).sum()) / valid
    if not math.isfinite(copied["loss"]):
        raise ValueError("test prediction loss is non-finite")
    return copied


def _report_value(report: Any, name: str, default: Any = None) -> Any:
    if isinstance(report, Mapping):
        return report.get(name, default)
    return getattr(report, name, default)


def _trainability_payload(report: Any) -> Dict[str, Any]:
    if report is None:
        raise ValueError("backend trainability configuration returned no report")
    if is_dataclass(report):
        data = asdict(report)
    elif isinstance(report, Mapping):
        data = dict(report)
    else:
        names = _report_value(report, "trainable_names", None)
        data = {
            "mode": _report_value(report, "mode"),
            "total_parameters": _report_value(report, "total_parameters", _report_value(report, "total_params")),
            "trainable_parameters": _report_value(report, "trainable_parameters", _report_value(report, "trainable_params")),
            "trainable_ratio": _report_value(report, "trainable_ratio"),
        }
        if names is None:
            names = _report_value(report, "trainable_parameter_names", None)
        if names is not None:
            data["trainable_names"] = tuple(names)
        digest = _report_value(report, "trainable_names_sha256", None)
        if digest is None:
            digest = _report_value(report, "trainable_name_digest", None)
        if digest is not None:
            data["trainable_names_sha256"] = digest
    if "trainable_names" in data and isinstance(data["trainable_names"], tuple):
        data["trainable_names"] = list(data["trainable_names"])
    return _json_safe(data)


def _audit_trainability(backend: Any, mode: str, report: Any) -> Dict[str, Any]:
    payload = _trainability_payload(report)
    if payload.get("mode") != mode:
        raise ValueError("backend trainability report mode does not match requested mode")
    model = getattr(backend, "model", None)
    if model is None or not callable(getattr(model, "named_parameters", None)):
        raise ValueError("backend must expose model parameters for trainability audit")
    parameters = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    if total <= 0:
        raise ValueError("cannot audit a backend model with zero parameters")
    trainable = [(name, parameter) for name, parameter in parameters if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    reported_total = payload.get("total_parameters")
    reported_trainable = payload.get("trainable_parameters")
    if reported_total is not None and int(reported_total) != total:
        raise ValueError("backend trainability report total parameter count is stale")
    if reported_trainable is not None and int(reported_trainable) != trainable_count:
        raise ValueError("backend trainability report trainable parameter count is stale")
    actual_names = tuple(sorted(name for name, _ in trainable))
    reported_names = payload.get("trainable_names")
    if reported_names is not None and list(reported_names) != list(actual_names[:50]):
        raise ValueError("backend trainability report names are stale")
    if reported_names is None:
        payload["trainable_names"] = list(actual_names[:50])
    expected_digest = hashlib.sha256("\n".join(actual_names).encode("utf-8")).hexdigest()
    reported_digest = payload.get("trainable_names_sha256")
    if reported_digest is not None and str(reported_digest) != expected_digest:
        raise ValueError("backend trainability report digest is stale")
    if reported_digest is None:
        payload["trainable_names_sha256"] = expected_digest
    if mode == "zero_shot" and trainable:
        raise ValueError("zero_shot trainability audit found trainable parameters")
    if mode != "zero_shot" and not trainable:
        raise ValueError(f"{mode} trainability audit found no trainable parameters")
    payload["total_parameters"] = total
    payload["trainable_parameters"] = trainable_count
    payload["trainable_ratio"] = float(trainable_count) / float(total) if total else 0.0
    return _json_safe(payload)


def _lora_settings(args: Any) -> LoraSettings:
    return LoraSettings(
        r=int(getattr(args, "lora_r", 8)),
        lora_alpha=int(getattr(args, "lora_alpha", 8)),
        lora_dropout=float(getattr(args, "lora_dropout", 0.0)),
        bias=str(getattr(args, "lora_bias", "none")),
        task_type=getattr(args, "lora_task_type", None),
        fan_in_fan_out=bool(getattr(args, "lora_fan_in_fan_out", False)),
    )


def _lora_payload(settings: LoraSettings) -> Dict[str, Any]:
    return _json_safe(asdict(settings))


def _identity(backend: Any, model_name: str) -> Tuple[str, str]:
    spec = get_model_spec(model_name)
    model_id_value = getattr(backend, "model_id", None)
    model_id = str(spec.model_id if model_id_value is None else model_id_value)
    revision_value = getattr(backend, "revision", None)
    if revision_value is None:
        revision_value = getattr(backend, "effective_revision", None)
    if revision_value is None:
        revision_value = getattr(backend, "model_revision", spec.revision)
    revision = str(revision_value)
    return model_id, revision


def _state_dict_cpu(model: Any) -> Dict[str, Any]:
    state = model.state_dict()
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def _temporary_path(path: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".tmp.", dir=str(path.parent))
    os.close(fd)
    return Path(name)


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temp_path = _temporary_path(path)
    try:
        with temp_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    temp_path = _temporary_path(path)
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_write_predictions(path: Path, result: Mapping[str, Any], dataset: Any, step_minutes: int) -> Dict[str, Any]:
    temp_path = _temporary_path(path)
    try:
        metrics = _write_predictions(temp_path, result, dataset, step_minutes)
        _fsync_file(temp_path)
        os.replace(temp_path, path)
        return metrics
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _checkpoint_payload(
    args: Any,
    backend: Any,
    dataset: Any,
    mode: str,
    report: Mapping[str, Any],
    lora_settings: Mapping[str, Any],
    epoch: Optional[int],
    val_loss: Optional[float],
    model_id: str,
    revision: str,
) -> Dict[str, Any]:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset": str(args.dataset),
        "model": str(args.model),
        "model_name": str(args.model),
        "model_id": model_id,
        "model_revision": revision,
        "effective_revision": revision,
        "revision": revision,
        "mode": mode,
        "seq_len": int(dataset.seq_len),
        "pred_len": int(dataset.pred_len),
        "lora_settings": dict(lora_settings),
        "trainability": dict(report),
        "trainability_report": dict(report),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    if mode != "zero_shot":
        payload["state_dict"] = _state_dict_cpu(backend.model)
    return _json_safe(payload)


def _validate_checkpoint_identity(
    checkpoint: Mapping[str, Any],
    args: Any,
    dataset: Any,
    mode: str,
    model_id: str,
    revision: str,
    report: Mapping[str, Any],
    lora_settings: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset": str(args.dataset),
        "model": str(args.model),
        "model_id": model_id,
        "model_revision": revision,
        "effective_revision": revision,
        "revision": revision,
        "mode": mode,
        "seq_len": int(dataset.seq_len),
        "pred_len": int(dataset.pred_len),
        "lora_settings": dict(lora_settings),
        "trainability": dict(report),
        "trainability_report": dict(report),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"checkpoint identity mismatch for {key}")
    if "state_dict" not in checkpoint or not isinstance(checkpoint["state_dict"], Mapping):
        raise RuntimeError("training checkpoint has no restorable state_dict")


def _restore_checkpoint(
    path: Path,
    backend: Any,
    args: Any,
    dataset: Any,
    mode: str,
    model_id: str,
    revision: str,
    report: Mapping[str, Any],
    lora_settings: Mapping[str, Any],
    device: torch.device,
    expected_epoch: Optional[int] = None,
    expected_val_loss: Optional[float] = None,
) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as exc:
        raise RuntimeError(f"could not load checkpoint {path}") from exc
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("checkpoint must contain a mapping")
    _validate_checkpoint_identity(
        checkpoint, args, dataset, mode, model_id, revision, report, lora_settings
    )
    if expected_epoch is not None and checkpoint.get("epoch") != int(expected_epoch):
        raise RuntimeError("checkpoint identity mismatch for epoch")
    if expected_val_loss is not None:
        try:
            observed_val_loss = float(checkpoint.get("val_loss"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("checkpoint identity mismatch for val_loss") from exc
        if not math.isfinite(observed_val_loss) or observed_val_loss != float(expected_val_loss):
            raise RuntimeError("checkpoint identity mismatch for val_loss")
    try:
        backend.model.load_state_dict(checkpoint["state_dict"], strict=True)
    except Exception as exc:
        raise RuntimeError("strict checkpoint state restoration failed") from exc
    return checkpoint


def _remove_completion(output_dir: Path) -> None:
    completion = output_dir / "completion.tsv"
    if completion.exists():
        completion.unlink()


def _resolve_output_dir(args: Any) -> Path:
    if getattr(args, "output_dir", None) is not None:
        output_dir = Path(args.output_dir).resolve()
    else:
        config = _read_config(Path(args.config))
        try:
            root = Path(config["paths"]["results_root"])
        except (KeyError, TypeError) as exc:
            raise ValueError("dataset config must define paths.results_root") from exc
        if not root.is_absolute():
            root = root.resolve()
        output_dir = (root / "foundation_models" / args.dataset / args.model / args.mode).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_completion(output_dir)
    return output_dir


def _write_completion_manifest(
    output_dir: Path,
    args: Any,
    dataset: Any,
    metrics_payload: Mapping[str, Any],
    checkpoint_path: Path,
    prediction_path: Path,
    metrics_path: Path,
    model_id: str,
    revision: str,
) -> Path:
    phase_steps = metrics_payload["phase_steps"]
    values = (
        FOUNDATION_SCHEMA_VERSION,
        str(args.dataset),
        str(args.model),
        str(int(dataset.seq_len)),
        str(int(dataset.pred_len)),
        str(output_dir.resolve()),
        str(metrics_payload["run_mode"]),
        str(int(metrics_payload["epochs"])),
        str(int(getattr(args, "max_train_steps", 0))),
        str(int(getattr(args, "max_eval_steps", 0))),
        str(int(getattr(args, "max_test_steps", 0))),
        str(int(phase_steps["train"])),
        str(int(phase_steps["val"])),
        str(int(phase_steps["test"])),
        _sha256(checkpoint_path),
        _sha256(prediction_path),
        _sha256(metrics_path),
        str(args.mode),
        str(model_id),
        str(revision),
    )
    manifest_path = output_dir / "completion.tsv"
    temp_path = _temporary_path(manifest_path)
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(COMPLETION_HEADER)
            writer.writerow(values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, manifest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return manifest_path


def run(args: Any) -> Dict[str, Path]:
    """Run one experiment and return only its four final artifact paths."""

    output_dir: Optional[Path] = None
    try:
        # When callers provide an explicit output, remove an old completion
        # before validating or constructing anything else.  This keeps a stale
        # success marker from surviving even an early runtime-argument failure.
        if getattr(args, "output_dir", None) is not None:
            output_dir = Path(args.output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            _remove_completion(output_dir)
        elif (
            getattr(args, "dataset", None) in DATASET_LOADERS
            and getattr(args, "model", None) in MODEL_NAMES
            and getattr(args, "mode", None) in RUN_MODES
        ):
            # Identity is sufficient to resolve the default location.  Doing
            # this before full validation also removes a stale completion when
            # a later argument check fails.
            if getattr(args, "config", None) is None:
                args.config = DEFAULT_CONFIGS[args.dataset]
            output_dir = _resolve_output_dir(args)
        _validate_runtime_args(args)
        if output_dir is None:
            output_dir = _resolve_output_dir(args)
        args.device = _device_from_arg(getattr(args, "device", "auto"))
        settings = _lora_settings(args)
        settings_payload = _lora_payload(settings)
        checkpoint_path = output_dir / "best.pt"
        prediction_path = output_dir / "predictions.csv"
        metrics_path = output_dir / "metrics.json"

        if args.mode == "zero_shot":
            test_dataset, test_loader = _dataset_and_loader(args, "test")
            backend = build_backend(args.model, args.device)
            report = _audit_trainability(
                backend,
                args.mode,
                backend.configure_trainable(args.mode, settings),
            )
            model_id, revision = _identity(backend, args.model)
            _atomic_torch_save(
                _checkpoint_payload(
                    args,
                    backend,
                    test_dataset,
                    args.mode,
                    report,
                    settings_payload,
                    None,
                    None,
                    model_id,
                    revision,
                ),
                checkpoint_path,
            )
            test_result = _collect_predictions(
                backend,
                test_loader,
                args.device,
                int(args.seq_len),
                int(args.pred_len),
                args.max_test_steps,
                "test",
            )
            history: List[Dict[str, Any]] = []
            best_epoch = None
            best_val_loss = None
            phase_steps = {"train": 0, "val": 0, "test": int(test_result["steps"])}
        else:
            train_dataset, train_loader = _dataset_and_loader(args, "train")
            val_dataset, val_loader = _dataset_and_loader(args, "val")
            test_dataset, test_loader = _dataset_and_loader(args, "test")
            backend = build_backend(args.model, args.device)
            report = _audit_trainability(
                backend,
                args.mode,
                backend.configure_trainable(args.mode, settings),
            )
            trainable_parameters = [
                parameter for parameter in backend.model.parameters() if parameter.requires_grad
            ]
            if not trainable_parameters:
                raise ValueError("training mode has no requires_grad parameters")
            optimizer = torch.optim.Adam(
                trainable_parameters, lr=float(getattr(args, "learning_rate", 1e-3))
            )
            model_id, revision = _identity(backend, args.model)
            best_epoch = None
            best_val_loss = float("inf")
            history = []
            total_train_steps = 0
            total_val_steps = 0
            for epoch in range(int(args.epochs)):
                train_loss, train_steps = _train_epoch(
                    backend,
                    train_loader,
                    optimizer,
                    args.device,
                    int(args.seq_len),
                    int(args.pred_len),
                    args.max_train_steps,
                )
                val_result = _collect_predictions(
                    backend,
                    val_loader,
                    args.device,
                    int(args.seq_len),
                    int(args.pred_len),
                    args.max_eval_steps,
                    "val",
                )
                val_loss = float(val_result["loss"])
                if best_epoch is None or val_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = val_loss
                    _atomic_torch_save(
                        _checkpoint_payload(
                            args,
                            backend,
                            train_dataset,
                            args.mode,
                            report,
                            settings_payload,
                            epoch,
                            val_loss,
                            model_id,
                            revision,
                        ),
                        checkpoint_path,
                    )
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_steps": train_steps,
                        "val_loss": val_loss,
                        "val_steps": int(val_result["steps"]),
                    }
                )
                total_train_steps += int(train_steps)
                total_val_steps += int(val_result["steps"])
            if best_epoch is None or not checkpoint_path.is_file():
                raise RuntimeError("training did not produce a checkpoint")
            _restore_checkpoint(
                checkpoint_path,
                backend,
                args,
                train_dataset,
                args.mode,
                model_id,
                revision,
                report,
                settings_payload,
                args.device,
                expected_epoch=best_epoch,
                expected_val_loss=best_val_loss,
            )
            test_result = _collect_predictions(
                backend,
                test_loader,
                args.device,
                int(args.seq_len),
                int(args.pred_len),
                args.max_test_steps,
                "test",
            )
            phase_steps = {
                "train": total_train_steps,
                "val": total_val_steps,
                "test": int(test_result["steps"]),
            }

        test_result = _clip_result(test_result, test_dataset)
        step_value = getattr(test_dataset, "forecast_step", None)
        if isinstance(step_value, timedelta):
            step_minutes = int(step_value.total_seconds() // 60)
        elif hasattr(step_value, "total_seconds"):
            step_minutes = int(step_value.total_seconds() // 60)
        else:
            step_minutes = int(getattr(args, "step_minutes", 1))
        original_metrics = _atomic_write_predictions(
            prediction_path, test_result, test_dataset, step_minutes
        )
        metrics_payload = _json_safe(
            {
                "dataset": args.dataset,
                "model": args.model,
                "model_id": model_id,
                "model_revision": revision,
                "effective_revision": revision,
                "revision": revision,
                "mode": args.mode,
                "seq_len": int(test_dataset.seq_len),
                "pred_len": int(test_dataset.pred_len),
                "epochs": int(args.epochs),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_loss": best_val_loss,
                "test_loss_normalized": float(test_result["loss"]),
                "train_steps": int(phase_steps["train"]),
                "val_steps": int(phase_steps["val"]),
                "test_steps": int(phase_steps["test"]),
                "phase_steps": phase_steps,
                "limits": {
                    "train": int(getattr(args, "max_train_steps", 0)),
                    "val": int(getattr(args, "max_eval_steps", 0)),
                    "test": int(getattr(args, "max_test_steps", 0)),
                },
                "trainability": report,
                "trainability_report": report,
                "lora_settings": settings_payload,
                "metrics_original_power_units": original_metrics,
                "rated_power": float(test_dataset.rated_power),
                "run_mode": "smoke" if getattr(args, "smoke", False) else "full",
                "history": history,
            }
        )
        _atomic_json_dump(metrics_payload, metrics_path)
        completion_path = _write_completion_manifest(
            output_dir,
            args,
            test_dataset,
            metrics_payload,
            checkpoint_path,
            prediction_path,
            metrics_path,
            model_id,
            revision,
        )
        return {
            "checkpoint": checkpoint_path,
            "predictions": prediction_path,
            "metrics": metrics_path,
            "completion": completion_path,
        }
    except Exception:
        if output_dir is not None:
            _remove_completion(output_dir)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_LOADERS))
    parser.add_argument("--model", choices=list(MODEL_NAMES))
    parser.add_argument("--mode", choices=list(RUN_MODES))
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-modes", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_eval_steps", type=int, default=0)
    parser.add_argument("--max_test_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_bias", default="none")
    parser.add_argument("--lora_task_type")
    parser.add_argument("--lora_fan_in_fan_out", action="store_true")
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models or args.list_modes:
        return args
    if args.dataset is None:
        parser.error("--dataset is required unless --list-models or --list-modes is used")
    if args.model is None:
        parser.error("--model is required unless --list-models or --list-modes is used")
    if args.mode is None:
        parser.error("--mode is required unless --list-models or --list-modes is used")
    if args.seq_len is None or args.seq_len <= 0:
        parser.error("--seq_len must be positive")
    if args.pred_len is None or args.pred_len <= 0:
        parser.error("--pred_len must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    for name in ("max_train_steps", "max_eval_steps", "max_test_steps"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative")
    if args.smoke:
        args.epochs = 0 if args.mode == "zero_shot" else 1
        args.max_train_steps = 1
        args.max_eval_steps = 1
        args.max_test_steps = 1
        args.batch_size = 2
    if args.mode == "zero_shot":
        args.epochs = 0
    elif args.epochs < 1:
        parser.error("--epochs must be at least 1 for training modes")
    if args.config is None:
        args.config = DEFAULT_CONFIGS[args.dataset]
    args.config = Path(args.config).resolve()
    try:
        _device_from_arg(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_models:
        print("\n".join(MODEL_NAMES))
        return 0
    if args.list_modes:
        print("\n".join(RUN_MODES))
        return 0
    result = run(args)
    print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
