"""Official Luoyang long-form predictions and raw-unit metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from data_provider.data_loader_luoyang import load_luoyang_config


def _metric_block(errors: np.ndarray, rated_power: float) -> Dict[str, Any]:
    if errors.size == 0:
        return {"count": 0, "MSE": None, "RMSE": None, "MAE": None, "NRMSE": None, "NMAE": None}
    mse = float(np.mean(np.square(errors)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    return {
        "count": int(errors.size), "MSE": mse, "RMSE": rmse, "MAE": mae,
        "NRMSE": rmse / rated_power, "NMAE": mae / rated_power,
    }


def _direction_accuracy(frame: pd.DataFrame, minimum_delta: float) -> Optional[float]:
    truth_delta = frame["y_true"].to_numpy() - frame["current_power"].to_numpy()
    pred_delta = frame["y_pred"].to_numpy() - frame["current_power"].to_numpy()
    selected = np.abs(truth_delta) >= minimum_delta
    if not selected.any():
        return None
    return float(np.mean(np.sign(truth_delta[selected]) == np.sign(pred_delta[selected])))


def build_official_outputs(
    preds: np.ndarray,
    trues: np.ndarray,
    dataset: Any,
    output_dir: Union[str, Path],
    config_path: Union[str, Path],
) -> tuple[Path, Path]:
    config = load_luoyang_config(config_path)
    steps = int(config["dimensions"]["forecast_steps"])
    step_minutes = int(config["time"]["forecast_step_minutes"])
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, steps)
    trues = np.asarray(trues, dtype=np.float64).reshape(-1, steps)
    issue_times = pd.to_datetime(np.asarray(dataset.sample_times))
    current = np.asarray(dataset.current_powers, dtype=np.float64)
    if preds.shape != trues.shape or preds.shape[0] != len(issue_times):
        raise ValueError("predictions, truths, and issue times are not aligned")
    if len(pd.Index(issue_times).unique()) != len(issue_times):
        raise ValueError("distributed inference produced duplicate issue_time values")
    if not np.isfinite(preds).all() or not np.isfinite(trues).all():
        raise ValueError("predictions and truths must be finite")
    preds = np.maximum(preds, float(config["output"]["prediction_floor"]))

    records = []
    for sample, issue in enumerate(issue_times):
        for horizon_index in range(steps):
            horizon = (horizon_index + 1) * step_minutes
            records.append({
                "issue_time": issue,
                "target_time": issue + pd.Timedelta(minutes=horizon),
                "horizon_minutes": horizon,
                "y_true": trues[sample, horizon_index],
                "y_pred": preds[sample, horizon_index],
                "current_power": current[sample],
            })
    frame = pd.DataFrame.from_records(records).sort_values(["issue_time", "horizon_minutes"])
    expected = list(range(step_minutes, steps * step_minutes + 1, step_minutes))
    grouped = frame.groupby("issue_time", sort=False)["horizon_minutes"].apply(list)
    if not grouped.map(lambda values: values == expected).all():
        raise ValueError("each issue_time must contain exactly the configured horizons")
    if frame.duplicated(["issue_time", "horizon_minutes"]).any():
        raise ValueError("duplicate issue_time/horizon pair")
    expected_targets = frame["issue_time"] + pd.to_timedelta(frame["horizon_minutes"], unit="m")
    if not frame["target_time"].equals(expected_targets):
        raise ValueError("target_time does not match issue_time plus horizon")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / config["output"]["predictions_file"]
    frame[["issue_time", "target_time", "horizon_minutes", "y_true", "y_pred"]].to_csv(
        predictions_path, index=False, date_format="%Y-%m-%d %H:%M:%S"
    )

    rated = float(config["power"]["rated_power"])
    hard_threshold = float(config["power"]["hard_delta_fraction"]) * rated
    direction_threshold = float(config["power"]["direction_min_delta_fraction"]) * rated
    horizons: Dict[str, Any] = {}
    for horizon in config["time"]["evaluation_horizons_minutes"]:
        subset = frame[frame["horizon_minutes"] == int(horizon)].copy()
        errors = subset["y_pred"].to_numpy() - subset["y_true"].to_numpy()
        hard = np.abs(subset["y_true"].to_numpy() - subset["current_power"].to_numpy()) >= hard_threshold
        overall = _metric_block(errors, rated)
        overall["direction_accuracy"] = _direction_accuracy(subset, direction_threshold)
        hard_block = _metric_block(errors[hard], rated)
        hard_block["direction_accuracy"] = _direction_accuracy(subset.loc[hard], direction_threshold)
        horizons[f"{int(horizon)}_minutes"] = {"overall": overall, "hard_delta": hard_block}
    metrics = {
        "units": config["power"]["units"],
        "rated_power": rated,
        "forecast_step_minutes": step_minutes,
        "forecast_steps": steps,
        "hard_delta_threshold": hard_threshold,
        "direction_min_delta": direction_threshold,
        "horizons": horizons,
    }
    metrics_path = output_dir / config["output"]["metrics_file"]
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return predictions_path, metrics_path
