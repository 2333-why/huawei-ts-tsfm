from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import run_time_series
from baselines import BASELINE_NAMES, build_baseline
from data_provider.power_only import PowerOnlyParquetDataset


SITE = {
    "latitude": 37.427,
    "longitude": -122.174,
    "timezone": "America/Los_Angeles",
    "surface_tilt": 37,
    "surface_azimuth": 195,
}


def _write_runner_fixture(root: Path) -> Path:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=14, freq="1h")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "power": [float(value) for value in range(1, 15)],
    })
    parquet_path = root / "power.parquet"
    frame.to_parquet(parquet_path, index=False)
    config = {
        "dataset": "PowerOnlyParquet",
        "paths": {
            "parquet_file": str(parquet_path),
            "results_root": str(root / "results"),
        },
        "fields": {"timestamp": "timestamp", "target": "power"},
        "time": {
            "sampling_interval_minutes": 60,
            "forecast_step_minutes": 60,
            "train_start_timestamp": "2025-01-01 00:00:00",
            "test_start_timestamp": "2025-01-01 09:00:00",
            "test_end_exclusive_timestamp": "2025-01-01 12:00:00",
            "validation_fraction": 0.5,
        },
        "power": {"power_scale": 100.0, "rated_power": 100.0, "units": "kW"},
        "site": SITE,
    }
    config_path = root / "power.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _runner_args(config_path: Path, output_dir: Path):
    return run_time_series.parse_args([
        "--dataset", "skippd_luoyang", "--model", "Persistence",
        "--config", str(config_path), "--seq_len", "2", "--pred_len", "1",
        "--smoke", "--device", "cpu", "--output_dir", str(output_dir),
    ])


def test_baseline_collector_matches_neural_result_contract(tmp_path: Path):
    config_path = _write_runner_fixture(tmp_path)
    dataset = PowerOnlyParquetDataset(
        config_path, "test", history_points=2, forecast_steps=1,
    )
    baseline = build_baseline("Persistence", dataset)
    loader = [
        {
            "history": torch.tensor([[[0.01], [0.02]]]),
            "target": torch.tensor([[[0.03]]]),
            "target_mask": torch.tensor([[True]]),
            "issue_time_ns": torch.tensor([0], dtype=torch.int64),
        },
    ]

    result = run_time_series._collect_baseline_predictions(
        baseline, loader, dataset, torch.device("cpu"), max_steps=1,
    )

    assert set(result) == {
        "loss", "steps", "prediction", "target", "mask", "issue_time_ns",
    }
    assert result["steps"] == 1
    np.testing.assert_allclose(result["prediction"], [[[0.02]]])
    np.testing.assert_allclose(result["issue_time_ns"], [0])


def test_persistence_run_skips_neural_construction_and_writes_uniform_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    config_path = _write_runner_fixture(tmp_path)
    output_dir = tmp_path / "output"

    def fail_neural_construction(*unused):
        raise AssertionError("baseline run attempted neural model construction")

    monkeypatch.setattr(run_time_series, "_build_model", fail_neural_construction)
    result = run_time_series.run(_runner_args(config_path, output_dir))

    assert all(path.is_file() for path in (
        result["checkpoint"], result["predictions"],
        result["metrics"], result["completion"],
    ))
    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    assert checkpoint["method_name"] == "Persistence"
    assert checkpoint["method_type"] == "baseline"
    assert checkpoint["dataset"] == "skippd_luoyang"
    assert checkpoint["config"]["seq_len"] == 2
    assert checkpoint["config"]["pred_len"] == 1
    assert checkpoint["checkpoint_payload"]["method_type"] == "baseline"

    with result["predictions"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["y_pred"]) for row in rows] == [10.0, 11.0]
    assert [float(row["y_true"]) for row in rows] == [11.0, 12.0]

    metrics = json.loads(result["metrics"].read_text(encoding="utf-8"))
    assert metrics["method_type"] == "baseline"
    assert metrics["training_skipped"] is True
    assert metrics["best_epoch"] is None
    assert metrics["phase_steps"] == {"train": 0, "val": 0, "test": 1}
    assert metrics["epochs"] == 0

    completion_rows = result["completion"].read_text(encoding="utf-8").splitlines()
    fields = dict(zip(completion_rows[0].split("\t"), completion_rows[1].split("\t")))
    assert fields["epochs"] == "0"
    assert fields["train_steps"] == "0"
    assert fields["val_steps"] == "0"
    assert fields["test_steps"] == "1"


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_catalog_names_are_distinct_from_neural_catalog(name):
    assert name not in run_time_series.SELECTED_MODEL_NAMES
