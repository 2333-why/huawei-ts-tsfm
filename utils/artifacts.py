"""Shared prediction, metric, JSON, and digest artifact helpers."""

from __future__ import annotations

import csv
import hashlib
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _safe_metric(value: float):
    return None if not math.isfinite(float(value)) else float(value)


def _metrics(prediction, target, mask, rated_power: float) -> dict:
    try:
        rated_power = float(rated_power)
    except (TypeError, ValueError) as exc:
        raise ValueError("rated_power must be finite and positive") from exc
    if not math.isfinite(rated_power) or rated_power <= 0:
        raise ValueError("rated_power must be finite and positive")
    valid = np.asarray(mask, dtype=bool)
    pred = np.asarray(prediction)[..., 0][valid]
    truth = np.asarray(target)[..., 0][valid]
    if truth.size == 0:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "nmae": None,
            "nrmse": None,
            "mape_percent": None,
            "nmae_percent": None,
            "nrmse_percent": None,
        }
    error = pred - truth
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    nonzero = np.abs(truth) > 1e-8
    return {
        "count": int(truth.size),
        "mae": _safe_metric(mae),
        "rmse": _safe_metric(rmse),
        "nmae": _safe_metric(mae / rated_power),
        "nrmse": _safe_metric(rmse / rated_power),
        "mape_percent": _safe_metric(np.mean(np.abs(error[nonzero] / truth[nonzero])) * 100)
        if nonzero.any()
        else None,
        "nmae_percent": _safe_metric(mae / rated_power * 100),
        "nrmse_percent": _safe_metric(rmse / rated_power * 100),
    }


def write_predictions(path: Path, result: Mapping[str, Any], dataset: Any, step_minutes: int) -> dict:
    """Write valid prediction rows in raw power units and return metrics."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = float(dataset.power_scale)
    prediction = result["prediction"] * scale
    target = result["target"] * scale
    mask = result["mask"]
    aggregate = _metrics(prediction, target, mask, dataset.rated_power)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "issue_time",
                "target_time",
                "horizon_minutes",
                "y_true",
                "y_pred",
            ],
        )
        writer.writeheader()
        for row_index, issue_ns in enumerate(result["issue_time_ns"]):
            if not np.isfinite(issue_ns):
                continue
            issue_seconds = float(issue_ns) / 1_000_000_000.0
            for horizon_index in range(prediction.shape[1]):
                if not mask[row_index, horizon_index]:
                    continue
                issue = datetime.utcfromtimestamp(issue_seconds)
                target_time = issue + timedelta(minutes=step_minutes * (horizon_index + 1))
                writer.writerow(
                    {
                        "issue_time": issue.isoformat(),
                        "target_time": target_time.isoformat(),
                        "horizon_minutes": step_minutes * (horizon_index + 1),
                        "y_true": float(target[row_index, horizon_index, 0]),
                        "y_pred": float(prediction[row_index, horizon_index, 0]),
                    }
                )
    return aggregate


def json_safe(value: Any):
    """Convert common NumPy containers and non-finite floats for JSON."""

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Private aliases preserve the names used by the legacy foundation runner
# while its imports are migrated to this module.
_json_safe = json_safe
_sha256 = sha256
_write_predictions = write_predictions


__all__ = [
    "json_safe",
    "sha256",
    "write_predictions",
]
