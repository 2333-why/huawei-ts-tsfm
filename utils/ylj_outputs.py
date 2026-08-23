"""Official YLJ predictions and mask-aware metrics in MW."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_provider.data_loader_ylj import load_ylj_config


def _metrics(errors, rated_power):
    if not errors.size:
        return {"count": 0, "RMSE": None, "MAE": None, "NRMSE": None, "NMAE": None}
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mae = float(np.mean(np.abs(errors)))
    return {
        "count": int(errors.size), "RMSE": rmse, "MAE": mae,
        "NRMSE": rmse / rated_power, "NMAE": mae / rated_power,
    }


def build_ylj_outputs(preds, trues, dataset, output_dir, config_path):
    config = load_ylj_config(config_path)
    steps = int(config["dimensions"]["forecast_steps"])
    step_minutes = int(config["time"]["forecast_step_minutes"])
    preds = np.asarray(preds, np.float64).reshape(-1, steps)
    trues = np.asarray(trues, np.float64).reshape(-1, steps)
    masks = np.asarray(dataset.target_masks, bool).reshape(-1, steps)
    issue_times = pd.to_datetime(np.asarray(dataset.sample_times))
    current = np.asarray(dataset.current_powers, np.float64)
    current_valid = np.asarray(dataset.current_power_valid, bool)
    if preds.shape != trues.shape or preds.shape != masks.shape or preds.shape[0] != len(issue_times):
        raise ValueError("YLJ predictions, targets, masks and issue times are not aligned")
    if len(pd.Index(issue_times).unique()) != len(issue_times):
        raise ValueError("YLJ inference contains duplicate issue_time values")
    if not np.isfinite(preds).all() or not np.isfinite(trues[masks]).all():
        raise ValueError("YLJ valid predictions and targets must be finite")
    preds = np.maximum(preds, float(config["power"]["target_floor"]))

    records = []
    for sample, issue in enumerate(issue_times):
        for index in range(steps):
            horizon = (index + 1) * step_minutes
            valid = bool(masks[sample, index])
            records.append({
                "issue_time": issue,
                "target_time": issue + pd.Timedelta(minutes=horizon),
                "horizon_minutes": horizon,
                "y_true": trues[sample, index] if valid else np.nan,
                "y_pred": preds[sample, index],
                "target_valid": valid,
                "current_power": current[sample],
                "current_power_valid": current_valid[sample],
            })
    frame = pd.DataFrame(records).sort_values(["issue_time", "horizon_minutes"])
    expected = list(range(step_minutes, step_minutes * steps + 1, step_minutes))
    if not frame.groupby("issue_time")["horizon_minutes"].apply(list).map(lambda row: row == expected).all():
        raise ValueError("each YLJ issue_time must contain every configured horizon")
    if frame.duplicated(["issue_time", "horizon_minutes"]).any():
        raise ValueError("duplicate YLJ issue_time/horizon pair")
    expected_targets = frame["issue_time"] + pd.to_timedelta(frame["horizon_minutes"], unit="m")
    if not frame["target_time"].equals(expected_targets):
        raise ValueError("YLJ target_time does not match issue_time plus horizon")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / config["output"]["predictions_file"]
    frame[["issue_time", "target_time", "horizon_minutes", "y_true", "y_pred", "target_valid"]].to_csv(
        predictions_path, index=False, date_format="%Y-%m-%d %H:%M:%S")

    rated = float(config["power"]["rated_power"])
    hard_threshold = rated * float(config["power"]["hard_delta_fraction"])
    horizons = {}
    for horizon in config["time"]["evaluation_horizons_minutes"]:
        subset = frame[(frame["horizon_minutes"] == int(horizon)) & frame["target_valid"]].copy()
        errors = (subset["y_pred"] - subset["y_true"]).to_numpy(np.float64)
        hard = (
            subset["current_power_valid"].to_numpy(bool)
            & (np.abs(subset["y_true"].to_numpy() - subset["current_power"].to_numpy()) >= hard_threshold))
        horizons[f"{int(horizon)}_minutes"] = {
            "overall": _metrics(errors, rated),
            "hard_delta": _metrics(errors[hard], rated),
        }
    metrics = {
        "units": config["power"]["units"], "rated_power": rated,
        "forecast_step_minutes": step_minutes, "forecast_steps": steps,
        "horizons": horizons,
    }
    metrics_path = output_dir / config["output"]["metrics_file"]
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return predictions_path, metrics_path
