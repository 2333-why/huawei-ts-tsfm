import csv
import os

import numpy as np

from utils.metrics import STANFORD_SUNNY_DATES


def _as_1d(values):
    return np.asarray(values, dtype=np.float32).reshape(len(values), -1)[:, -1]


def _current_pv_from_dataset(dataset):
    count = len(dataset.sample_times)
    current_pv = np.full(count, np.nan, dtype=np.float32)

    pv_values = getattr(dataset, "pv_values", None)
    sample_indices = getattr(dataset, "sample_indices", None)
    if pv_values is None or sample_indices is None:
        return current_pv

    sample_indices = np.asarray(sample_indices)
    if sample_indices.ndim != 2 or sample_indices.shape[1] < getattr(dataset, "seq_len", 1):
        return current_pv

    if getattr(dataset, "history_order", "current_first") == "current_first":
        history_position = 0
    else:
        history_position = int(getattr(dataset, "seq_len", 1)) - 1

    history_indices = sample_indices[:, history_position].astype(np.int64)
    return np.asarray(pv_values[history_indices], dtype=np.float32).reshape(-1)


def save_stanford_sample_predictions(preds, trues, dataset, output_dir, capacity_kw):
    """Persist per-sample Stanford predictions for residual analysis."""
    os.makedirs(output_dir, exist_ok=True)

    pred = _as_1d(preds)
    true = _as_1d(trues)
    if len(pred) != len(dataset.sample_times):
        raise ValueError(
            f"prediction count {len(pred)} does not match sample_times {len(dataset.sample_times)}"
        )

    sample_times = np.asarray(dataset.sample_times, dtype="datetime64[s]")
    dates = sample_times.astype("datetime64[D]")
    sunny_mask = np.isin(dates, STANFORD_SUNNY_DATES)
    split = np.where(sunny_mask, "Sunny", "Cloudy")
    current_pv = _current_pv_from_dataset(dataset)
    signed_ramp = true - current_pv
    abs_ramp = np.abs(signed_ramp)
    residual = pred - true
    abs_error = np.abs(residual)
    squared_error = residual ** 2

    horizon = getattr(dataset, "forecast_horizon", np.timedelta64(15 * 60, "s"))
    target_times = sample_times + horizon
    capacity_kw = float(capacity_kw)

    npz_path = os.path.join(output_dir, "sample_predictions.npz")
    np.savez_compressed(
        npz_path,
        pred=pred,
        true=true,
        residual=residual,
        abs_error=abs_error,
        squared_error=squared_error,
        sample_times=sample_times.astype("datetime64[s]").astype(str),
        target_times=target_times.astype("datetime64[s]").astype(str),
        dates=dates.astype(str),
        split=split,
        current_pv=current_pv,
        signed_ramp_kw=signed_ramp,
        abs_ramp_kw=abs_ramp,
        target_pv_kw=true,
        pred_pv_kw=pred,
        capacity_kw=np.asarray([capacity_kw], dtype=np.float32),
    )

    csv_path = os.path.join(output_dir, "sample_predictions.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_time",
            "target_time",
            "date",
            "split",
            "current_pv_kw",
            "target_pv_kw",
            "pred_pv_kw",
            "residual_kw",
            "abs_error_kw",
            "squared_error_kw2",
            "signed_ramp_kw",
            "abs_ramp_kw",
            "residual_norm",
            "abs_error_norm",
        ])
        for i in range(len(pred)):
            writer.writerow([
                str(sample_times[i]),
                str(target_times[i]),
                str(dates[i]),
                split[i],
                float(current_pv[i]),
                float(true[i]),
                float(pred[i]),
                float(residual[i]),
                float(abs_error[i]),
                float(squared_error[i]),
                float(signed_ramp[i]),
                float(abs_ramp[i]),
                float(residual[i] / capacity_kw),
                float(abs_error[i] / capacity_kw),
            ])

    return npz_path, csv_path
