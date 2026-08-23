"""Train and evaluate power-only time-series forecasting models.

The existing Parquet adapters return several modalities for the multimodal
pipeline. This entry point deliberately consumes only the first time-series
channel (normalized power) and never forwards time marks or privileged data to
the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader_luoyang import LuoyangParquetDataset
from data_provider.data_loader_ylj import YLJParquetDataset
from models.tslib_adapter import forward_power_model, power_only_batch
from models.tslib_factory import build_model_config, build_power_model
from models.tslib_registry import SELECTED_MODEL_NAMES


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = {
    "skippd_luoyang": REPO_ROOT / "configs/datasets/skippd_luoyang.json",
    "pvod_station00_ylj": REPO_ROOT / "configs/datasets/pvod_station00_ylj.yaml",
}
DATASET_LOADERS = {
    "skippd_luoyang": LuoyangParquetDataset,
    "pvod_station00_ylj": YLJParquetDataset,
}


def _read_config(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_arg(value: str) -> torch.device:
    value = value.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device(value)


def _dataset_and_loader(args, flag: str):
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset_class = DATASET_LOADERS[args.dataset]
    kwargs = {"config_path": args.config, "flag": flag}
    seq_len = getattr(args, "seq_len", None)
    pred_len = getattr(args, "pred_len", None)
    if seq_len is not None:
        kwargs["history_points"] = seq_len
    if pred_len is not None:
        kwargs["forecast_steps"] = pred_len
    if args.dataset == "skippd_luoyang":
        kwargs.update(load_images=False, privileged_teacher=False)
    else:
        kwargs.update(privileged_teacher=False, power_only=True)
    dataset = dataset_class(**kwargs)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": flag == "train",
        "drop_last": False,
        "num_workers": args.num_workers,
        "pin_memory": args.device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    return dataset, DataLoader(dataset, **loader_kwargs)


def _model_config(args, dataset) -> SimpleNamespace:
    """Compatibility wrapper around the shared TSLib factory."""

    return build_model_config(args, dataset)


def _build_model(args, dataset):
    """Compatibility wrapper around the shared power-model factory."""

    return build_power_model(args.model, args, dataset)


def _as_target_mask(metadata, target: torch.Tensor) -> torch.Tensor:
    mask = metadata.get("target_mask") if isinstance(metadata, dict) else None
    if mask is None:
        return torch.ones(target.shape[:-1], dtype=torch.bool, device=target.device)
    mask = torch.as_tensor(mask, device=target.device, dtype=torch.bool)
    if mask.ndim == target.ndim:
        mask = mask[..., 0]
    return mask


def _power_only_batch(batch, device: torch.device):
    """Compatibility wrapper returning the historical tuple shape."""

    result = power_only_batch(batch, device)
    return result.x, result.y, result.mask


def _masked_mse(prediction, target, mask):
    mask = mask.unsqueeze(-1).to(dtype=prediction.dtype)
    denominator = mask.sum()
    if denominator.item() == 0:
        return prediction.sum() * 0.0
    return ((prediction - target).square() * mask).sum() / denominator


def _step_limit(limit: Optional[int]) -> Optional[int]:
    if limit is None:
        return None
    limit = int(limit)
    if limit < 0:
        raise ValueError("max_steps must be non-negative")
    return None if limit == 0 else limit


def _validate_runtime_args(args) -> None:
    """Reject invalid limits before constructing datasets or models."""

    for name in ("max_train_steps", "max_eval_steps", "max_test_steps"):
        raw_value = getattr(args, name)
        if raw_value is not None and int(raw_value) < 0:
            raise ValueError(f"{name} must be non-negative")
    batch_size = int(getattr(args, "batch_size"))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if int(getattr(args, "epochs")) < 1:
        raise ValueError("epochs must be at least 1")
    for name in ("seq_len", "pred_len"):
        value = getattr(args, name, None)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive")


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    max_steps: Optional[int],
    model_name: str = "DLinear",
    label_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    phase: str = "train",
) -> Tuple[float, int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    steps = 0
    limit = _step_limit(max_steps)
    batches = loader if limit is None else islice(loader, limit)
    for batch in batches:
        batch_x, batch_y, target_mask = _power_only_batch(batch, device)
        effective_label_len = batch_x.shape[1] if label_len is None else int(label_len)
        effective_pred_len = batch_y.shape[1] if pred_len is None else int(pred_len)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = forward_power_model(
                model_name, model, batch_x, effective_label_len, effective_pred_len
            )
            loss = _masked_mse(prediction, batch_y, target_mask)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
    if steps == 0:
        limit_label = "unlimited" if limit is None else str(limit)
        raise RuntimeError(
            f"{phase} loader is empty; max_steps={limit_label} requires at least one batch"
        )
    return (total_loss / steps if steps else float("nan")), steps


@torch.no_grad()
def _collect_predictions(
    model,
    loader,
    device,
    max_steps: Optional[int],
    model_name: str = "DLinear",
    label_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    phase: str = "eval",
):
    model.eval()
    losses: List[float] = []
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    issue_times: List[np.ndarray] = []
    limit = _step_limit(max_steps)
    batches = loader if limit is None else islice(loader, limit)
    for batch in batches:
        batch_x, batch_y, target_mask = _power_only_batch(batch, device)
        effective_label_len = batch_x.shape[1] if label_len is None else int(label_len)
        effective_pred_len = batch_y.shape[1] if pred_len is None else int(pred_len)
        prediction = forward_power_model(
            model_name, model, batch_x, effective_label_len, effective_pred_len
        )
        losses.append(float(_masked_mse(prediction, batch_y, target_mask).cpu()))
        predictions.append(prediction.detach().cpu().numpy())
        targets.append(batch_y.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())
        metadata = batch[9] if len(batch) > 9 and isinstance(batch[9], dict) else {}
        issue = metadata.get("issue_time_ns")
        if issue is None:
            issue = np.full(batch_y.shape[0], np.nan)
        issue_times.append(torch.as_tensor(issue).detach().cpu().numpy().reshape(-1))
    if not predictions:
        limit_label = "unlimited" if limit is None else str(limit)
        raise RuntimeError(
            f"{phase} loader is empty; max_steps={limit_label} requires at least one batch"
        )
        return {
            "loss": float("nan"), "steps": 0,
            "prediction": np.empty((0, 0, 1), np.float32),
            "target": np.empty((0, 0, 1), np.float32),
            "mask": np.empty((0, 0), bool), "issue_time_ns": np.empty(0),
        }
    return {
        "loss": float(np.mean(losses)), "steps": len(predictions),
        "prediction": np.concatenate(predictions),
        "target": np.concatenate(targets), "mask": np.concatenate(masks),
        "issue_time_ns": np.concatenate(issue_times),
    }


def _safe_metric(value: float):
    return None if not math.isfinite(float(value)) else float(value)


def _metrics(prediction, target, mask) -> dict:
    valid = np.asarray(mask, dtype=bool)
    pred = np.asarray(prediction)[..., 0][valid]
    truth = np.asarray(target)[..., 0][valid]
    if truth.size == 0:
        return {"count": 0, "mae": None, "rmse": None, "mape_percent": None}
    error = pred - truth
    nonzero = np.abs(truth) > 1e-8
    return {
        "count": int(truth.size),
        "mae": _safe_metric(np.mean(np.abs(error))),
        "rmse": _safe_metric(np.sqrt(np.mean(error ** 2))),
        "mape_percent": _safe_metric(np.mean(np.abs(error[nonzero] / truth[nonzero])) * 100)
        if nonzero.any() else None,
    }


def _write_predictions(path: Path, result, dataset, step_minutes: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = float(dataset.power_scale)
    prediction = result["prediction"] * scale
    target = result["target"] * scale
    mask = result["mask"]
    aggregate = _metrics(prediction, target, mask)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "issue_time", "target_time", "horizon_minutes", "y_true", "y_pred",
        ])
        writer.writeheader()
        for row_index, issue_ns in enumerate(result["issue_time_ns"]):
            if not np.isfinite(issue_ns):
                continue
            issue_seconds = float(issue_ns) / 1_000_000_000.0
            for horizon_index in range(prediction.shape[1]):
                if not mask[row_index, horizon_index]:
                    continue
                from datetime import datetime, timedelta, timezone

                issue = datetime.fromtimestamp(issue_seconds, tz=timezone.utc)
                target_time = issue + timedelta(minutes=step_minutes * (horizon_index + 1))
                writer.writerow({
                    "issue_time": issue.isoformat(),
                    "target_time": target_time.isoformat(),
                    "horizon_minutes": step_minutes * (horizon_index + 1),
                    "y_true": float(target[row_index, horizon_index, 0]),
                    "y_pred": float(prediction[row_index, horizon_index, 0]),
                })
    return aggregate


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(args) -> dict:
    _validate_runtime_args(args)
    _seed_everything(args.seed)
    args.device = _device_from_arg(args.device)
    config = _read_config(args.config)
    train_dataset, train_loader = _dataset_and_loader(args, "train")
    val_dataset, val_loader = _dataset_and_loader(args, "val")
    test_dataset, test_loader = _dataset_and_loader(args, "test")
    model, model_config = _build_model(args, train_dataset)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.output_dir is None:
        root = Path(config["paths"]["results_root"])
        output_dir = root / "pure_time_series" / args.dataset / args.model
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"

    best_val = float("inf")
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        train_loss, train_steps = _run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            args.max_train_steps,
            model_name=args.model,
            label_len=model_config.label_len,
            pred_len=model_config.pred_len,
            phase="train",
        )
        val_result = _collect_predictions(
            model,
            val_loader,
            args.device,
            args.max_eval_steps,
            model_name=args.model,
            label_len=model_config.label_len,
            pred_len=model_config.pred_len,
            phase="val",
        )
        val_loss = val_result["loss"]
        criterion = val_loss if math.isfinite(val_loss) else train_loss
        if best_epoch < 0 or criterion < best_val:
            best_val = criterion
            best_epoch = epoch
            torch.save({
                "model_name": args.model,
                "dataset": args.dataset,
                "config": vars(model_config),
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, checkpoint_path)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_steps": train_steps,
            "val_loss": val_loss, "val_steps": val_result["steps"],
        })
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.6f} ({train_steps} steps) "
            f"val_loss={val_loss:.6f} ({val_result['steps']} steps)",
            flush=True,
        )
        if args.max_train_steps and train_steps >= args.max_train_steps:
            # A smoke run intentionally performs only the requested small
            # number of iterations per epoch; epoch count remains explicit.
            pass

    if not checkpoint_path.is_file():
        raise RuntimeError("training did not produce a checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["state_dict"])
    test_limit = args.max_test_steps
    test_result = _collect_predictions(
        model,
        test_loader,
        args.device,
        test_limit,
        model_name=args.model,
        label_len=model_config.label_len,
        pred_len=model_config.pred_len,
        phase="test",
    )
    step_minutes = int(test_dataset.forecast_step.total_seconds() // 60)
    prediction_path = output_dir / "predictions.csv"
    metrics = _write_predictions(prediction_path, test_result, test_dataset, step_minutes)
    metrics_payload = _json_safe({
        "dataset": args.dataset,
        "model": args.model,
        "seq_len": int(test_dataset.seq_len),
        "pred_len": int(test_dataset.pred_len),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss_normalized": test_result["loss"],
        "train_steps": int(sum(item["train_steps"] for item in history)),
        "val_steps": int(sum(item["val_steps"] for item in history)),
        "test_steps": test_result["steps"],
        "phase_steps": {
            "train": int(sum(item["train_steps"] for item in history)),
            "val": int(sum(item["val_steps"] for item in history)),
            "test": int(test_result["steps"]),
        },
        "metrics_original_power_units": metrics,
        "history": history,
    })
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=True)
    return {
        "checkpoint": checkpoint_path,
        "predictions": prediction_path,
        "metrics": metrics_path,
        "payload": metrics_payload,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_LOADERS))
    parser.add_argument("--model", choices=list(SELECTED_MODEL_NAMES))
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print the complete selected model catalog and exit",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="bound training, validation, and test to one batch each",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_eval_steps", type=int, default=0)
    parser.add_argument("--max_test_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--moving_avg", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--num_kernels", type=int, default=2)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse runner arguments and apply the smoke-run contract."""

    parser = build_parser()
    args = parser.parse_args(argv)

    for name in ("max_train_steps", "max_eval_steps", "max_test_steps"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    for name in ("seq_len", "pred_len"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive")
    if args.list_models:
        return args
    if args.dataset is None:
        parser.error("--dataset is required unless --list-models is used")
    if args.model is None:
        parser.error("--model is required unless --list-models is used")
    if args.config is None:
        args.config = DEFAULT_CONFIGS[args.dataset]
    args.config = Path(args.config).resolve()
    if args.smoke:
        args.epochs = 1
        args.max_train_steps = 1
        args.max_eval_steps = 1
        args.max_test_steps = 1
        args.batch_size = 2
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_models:
        print("\n".join(SELECTED_MODEL_NAMES))
        return 0
    result = run(args)
    print(json.dumps({
        "checkpoint": str(result["checkpoint"]),
        "predictions": str(result["predictions"]),
        "metrics": str(result["metrics"]),
    }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
