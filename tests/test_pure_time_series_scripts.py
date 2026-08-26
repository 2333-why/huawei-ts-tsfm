"""Executable contracts for the pure time-series batch scripts."""

from __future__ import annotations

import json
import hashlib
import os
from collections import Counter
from pathlib import Path
import stat
import subprocess
import time
from typing import Optional

import pytest


EXPECTED_TASKS = {
    (24, 1, "skippd_luoyang"),
    (24, 1, "pvod_station00_ylj"),
    (48, 1, "skippd_luoyang"),
    (48, 1, "pvod_station00_ylj"),
    (48, 48, "skippd_luoyang"),
    (48, 16, "pvod_station00_ylj"),
    (96, 48, "skippd_luoyang"),
    (96, 16, "pvod_station00_ylj"),
}
EXPECTED_LABEL_BY_TASK = {
    (24, 1, "skippd_luoyang"): "seq24_pred1",
    (24, 1, "pvod_station00_ylj"): "seq24_pred1",
    (48, 1, "skippd_luoyang"): "seq48_pred1",
    (48, 1, "pvod_station00_ylj"): "seq48_pred1",
    (48, 48, "skippd_luoyang"): "seq48_h4",
    (48, 16, "pvod_station00_ylj"): "seq48_h4",
    (96, 48, "skippd_luoyang"): "seq96_h4",
    (96, 16, "pvod_station00_ylj"): "seq96_h4",
}
EXPECTED_SETTING_LABELS = {"seq24_pred1", "seq48_pred1", "seq48_h4", "seq96_h4"}
EXPECTED_MODELS = (
    "TSMixer", "Pyraformer", "SegRNN", "Transformer",
    "LightTS", "Crossformer", "FreTS", "MICN",
)
EXPECTED_BASELINES = (
    "Persistence", "SmartPersistence", "SeasonalPersistence", "Climatology",
)


def _fake_python(tmp_path: Path) -> Path:
    fake = tmp_path / "fake_python"
    fake.write_text(
        """#!/usr/bin/env python3
import json
import hashlib
import os
from pathlib import Path
import signal
import sys
import time

models = %r
baselines = %r
args = sys.argv[1:]
record_path = Path(os.environ["FAKE_RECORD"])
event = {
    "args": args,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
if "--list-models" in args:
    event["kind"] = "list"
    if os.environ.get("FAKE_DUPLICATE_MODELS") == "1":
        models = models[:-1] + (models[0],)
    with record_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\\n")
    print("\\n".join(models))
    raise SystemExit(0)

if "--list-baselines" in args:
    event["kind"] = "list-baselines"
    if os.environ.get("FAKE_DUPLICATE_BASELINES") == "1":
        baselines = baselines[:-1] + (baselines[0],)
    with record_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\\n")
    print("\\n".join(baselines))
    raise SystemExit(0)

event["kind"] = "run"
dataset = args[args.index("--dataset") + 1]
model = args[args.index("--model") + 1]
seq_len = int(args[args.index("--seq_len") + 1])
pred_len = int(args[args.index("--pred_len") + 1])
event["dataset"] = dataset
event["model"] = model
event["seq_len"] = seq_len
event["pred_len"] = pred_len
gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
event["gpu"] = gpu
is_baseline = model in baselines
if (
    os.environ.get("FAKE_KILL_WORKER_MODEL") == model
    and os.environ.get("FAKE_FAIL_DATASET") == dataset
):
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(0.05)
lock_root = Path(os.environ["FAKE_LOCK_ROOT"])
lock_root.mkdir(parents=True, exist_ok=True)
lock_path = lock_root / ("gpu-" + str(gpu) + ".lock")
acquired_lock = False
try:
    lock_path.mkdir()
    acquired_lock = True
except FileExistsError:
    event["overlap"] = True
event["started_at"] = time.monotonic_ns()

output_dir = Path(args[args.index("--output_dir") + 1])
try:
    time.sleep(0.02)
    if (
        os.environ.get("FAKE_FAIL_MODEL") == model
        and (
            not os.environ.get("FAKE_FAIL_DATASET")
            or os.environ.get("FAKE_FAIL_DATASET") == dataset
        )
    ):
        event["exit_code"] = 17
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "best.pt").write_bytes(b"checkpoint")
        (output_dir / "predictions.csv").write_text(
            "issue_time\\ttarget_time\\thorizon_minutes\\ty_true\\ty_pred\\n"
            "2025-01-01T00:00:00\\t2025-01-01T00:05:00\\t5\\t1.0\\t1.0\\n",
            encoding="utf-8",
        )
        smoke = "--smoke" in args
        epochs = 0 if is_baseline else (1 if smoke else (
            int(args[args.index("--epochs") + 1]) if "--epochs" in args else 1
        ))
        phase_steps = (
            {"train": 0, "val": 0, "test": 1}
            if is_baseline
            else {"train": 1, "val": 1, "test": 1}
        )
        max_train_steps = 1 if smoke else 0
        max_eval_steps = 1 if smoke else 0
        max_test_steps = 1 if smoke else 0
        metrics = {
            "dataset": dataset,
            "model": model,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "method_type": "baseline" if is_baseline else "neural",
            "training_skipped": is_baseline,
            "best_epoch": 0,
            "best_val_loss": 0.0,
            "test_loss_normalized": 0.0,
            "phase_steps": phase_steps,
            "metrics_original_power_units": {
                "count": 1,
                "mae": 0.0,
                "rmse": 0.0,
                "nmae": 0.0,
                "nrmse": 0.0,
                "mape_percent": 0.0,
                "nmae_percent": 0.0,
                "nrmse_percent": 0.0,
            },
            "run_mode": "smoke" if smoke else "full",
            "epochs": epochs,
            "limits": {"train": max_train_steps, "val": max_eval_steps, "test": max_test_steps},
            "history": [],
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, sort_keys=True, indent=2), encoding="utf-8"
        )
        def sha256(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_header = [
            "schema_version", "dataset", "model", "seq_len", "pred_len",
            "output_dir", "run_mode", "epochs", "max_train_steps",
            "max_eval_steps", "max_test_steps", "train_steps", "val_steps",
            "test_steps", "best_sha256", "predictions_sha256", "metrics_sha256",
        ]
        manifest_values = [
            "1", dataset, model, str(seq_len), str(pred_len), str(output_dir.resolve()),
            "smoke" if smoke else "full", str(epochs),
            str(max_train_steps), str(max_eval_steps), str(max_test_steps),
            str(phase_steps["train"]), str(phase_steps["val"]), str(phase_steps["test"]),
            sha256(output_dir / "best.pt"),
            sha256(output_dir / "predictions.csv"), sha256(output_dir / "metrics.json"),
        ]
        manifest = output_dir / "completion.tsv"
        temp_manifest = output_dir / ".completion.tsv.tmp.fake"
        temp_manifest.write_text(
            "\\t".join(manifest_header) + "\\n" + "\\t".join(manifest_values) + "\\n",
            encoding="utf-8",
        )
        temp_manifest.replace(manifest)
        event["exit_code"] = 0
finally:
    event["finished_at"] = time.monotonic_ns()
    if acquired_lock:
        lock_path.rmdir()
    with record_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\\n")
raise SystemExit(event["exit_code"])
""" % (EXPECTED_MODELS, EXPECTED_BASELINES),
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake


def _run_smoke(
    tmp_path: Path,
    *,
    fail_model: Optional[str] = None,
    resume: bool = False,
    script_name: str = "scripts/smoke_all_pure_time_series.sh",
    extra_environment: Optional[dict[str, str]] = None,
):
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "smoke-results"
    record_path = tmp_path / "invocations.jsonl"
    fake_python = _fake_python(tmp_path)
    environment = os.environ.copy()
    environment.update(
        PYTHON=str(fake_python),
        OUTPUT_ROOT=str(output_root),
        FAKE_RECORD=str(record_path),
        FAKE_LOCK_ROOT=str(tmp_path / "gpu-locks"),
        RESUME="1" if resume else "0",
        FAKE_FAIL_DATASET="skippd_luoyang",
    )
    environment.pop("GPUS", None)
    environment.pop("SMOKE", None)
    if fail_model is not None:
        environment["FAKE_FAIL_MODEL"] = fail_model
    else:
        environment.pop("FAKE_FAIL_MODEL", None)
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        ["bash", str(repo_root / script_name)],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
    ), record_path, output_root


def _events(record_path: Path):
    return [
        json.loads(line)
        for line in record_path.read_text(encoding="utf-8").splitlines()
    ]


def _task_output_dir(output_root: Path, task=(24, 1, "skippd_luoyang", "TSMixer")) -> Path:
    seq_len, pred_len, dataset, model = task
    setting = EXPECTED_LABEL_BY_TASK[(seq_len, pred_len, dataset)]
    return output_root / setting / dataset / model


def test_smoke_expands_exactly_96_runs_with_gpu_neural_and_cpu_baseline_queues(tmp_path):
    result, record_path, output_root = _run_smoke(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    events = _events(record_path)
    assert [event["kind"] for event in events].count("list") == 1
    assert [event["kind"] for event in events].count("list-baselines") == 1
    runs = [event for event in events if event["kind"] == "run"]
    assert len(runs) == 96
    neural_runs = [event for event in runs if event["model"] in EXPECTED_MODELS]
    baseline_runs = [event for event in runs if event["model"] in EXPECTED_BASELINES]
    assert len(neural_runs) == 64
    assert len(baseline_runs) == 32
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in neural_runs
    } == {(seq_len, pred_len, dataset, model)
          for seq_len, pred_len, dataset in EXPECTED_TASKS
          for model in EXPECTED_MODELS}
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in baseline_runs
    } == {(seq_len, pred_len, dataset, baseline)
          for seq_len, pred_len, dataset in EXPECTED_TASKS
          for baseline in EXPECTED_BASELINES}
    assert {event["gpu"] for event in neural_runs} == {"0", "1"}
    assert sum(event["gpu"] == "0" for event in neural_runs) == 32
    assert sum(event["gpu"] == "1" for event in neural_runs) == 32
    assert all(event["gpu"] is None for event in baseline_runs)
    assert all(not event.get("overlap", False) for event in runs)
    gpu_zero = [event for event in neural_runs if event["gpu"] == "0"]
    gpu_one = [event for event in neural_runs if event["gpu"] == "1"]
    assert any(
        left["started_at"] < right["finished_at"]
        and right["started_at"] < left["finished_at"]
        for left in gpu_zero
        for right in gpu_one
    )
    assert all("--smoke" in event["args"] for event in runs)
    assert all(
        event["args"][event["args"].index("--device") + 1] == "cuda:0"
        for event in neural_runs
    )
    assert all(
        event["args"][event["args"].index("--device") + 1] == "cpu"
        and "cuda" not in event["args"]
        for event in baseline_runs
    )
    output_dirs = [
        event["args"][event["args"].index("--output_dir") + 1]
        for event in runs
    ]
    assert len(output_dirs) == 96
    assert len(set(output_dirs)) == 96
    assert {
        Path(path).resolve().relative_to(output_root.resolve()).parts[0]
        for path in output_dirs
    } == EXPECTED_SETTING_LABELS
    for event, output_dir in zip(runs, output_dirs):
        task = (event["seq_len"], event["pred_len"], event["dataset"])
        assert Path(output_dir).resolve().relative_to(output_root.resolve()).parts[0] \
            == EXPECTED_LABEL_BY_TASK[task]
    for path in output_dirs:
        try:
            Path(path).resolve().relative_to(output_root.resolve())
        except ValueError:
            pytest.fail("run output escaped the configured output root")

    for path in output_dirs:
        task_dir = Path(path)
        metrics = json.loads((task_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["metrics_original_power_units"]["nmae"] is not None
        assert metrics["metrics_original_power_units"]["nrmse"] is not None
        if metrics["model"] in EXPECTED_BASELINES:
            assert metrics["method_type"] == "baseline"
            assert metrics["training_skipped"] is True
            assert metrics["epochs"] == 0
            assert metrics["phase_steps"] == {"train": 0, "val": 0, "test": 1}
        else:
            assert metrics["phase_steps"] == {"train": 1, "val": 1, "test": 1}
        manifest_rows = (task_dir / "completion.tsv").read_text(encoding="utf-8").splitlines()
        assert len(manifest_rows) == 2
        assert manifest_rows[0].split("\t")[0:5] == [
            "schema_version", "dataset", "model", "seq_len", "pred_len",
        ]

    summary = output_root / "smoke_summary.tsv"
    rows = summary.read_text(encoding="utf-8").splitlines()
    assert rows[0].split("\t") == [
        "seq_len", "pred_len", "dataset", "model", "status", "output_dir",
        "exit_code", "launch_gpu",
    ]
    assert len(rows[1:]) == 96
    assert all(row.split("\t")[4] == "PASS" for row in rows[1:])
    assert len({tuple(row.split("\t")[:4]) for row in rows[1:]}) == 96
    assert Counter(row.split("\t")[7] for row in rows[1:]) == Counter(
        {"0": 32, "1": 32, "cpu": 32}
    )


def test_summary_persists_launch_gpu_provenance_matching_cuda_assignment(tmp_path):
    result, record_path, output_root = _run_smoke(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    runs = {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"]): event
        for event in _events(record_path)
        if event["kind"] == "run"
    }
    rows = (output_root / "smoke_summary.tsv").read_text(encoding="utf-8").splitlines()
    assert rows[0].split("\t") == [
        "seq_len", "pred_len", "dataset", "model", "status", "output_dir",
        "exit_code", "launch_gpu",
    ]
    assert len(rows[1:]) == 96

    summary_launch_gpu_counts = Counter()
    for row in rows[1:]:
        fields = row.split("\t")
        task = (int(fields[0]), int(fields[1]), fields[2], fields[3])
        launch_gpu = fields[7]
        if task[3] in EXPECTED_BASELINES:
            assert launch_gpu == "cpu"
            assert runs[task]["cuda_visible_devices"] is None
        else:
            assert launch_gpu in {"0", "1"}
            assert launch_gpu == runs[task]["cuda_visible_devices"]
        summary_launch_gpu_counts[launch_gpu] += 1
    assert summary_launch_gpu_counts == Counter({"0": 32, "1": 32, "cpu": 32})


def test_legacy_summary_without_launch_gpu_is_incompatible_with_resume(tmp_path):
    first, record_path, output_root = _run_smoke(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr

    summary_path = output_root / "smoke_summary.tsv"
    rows = [line.split("\t") for line in summary_path.read_text().splitlines()]
    if rows[0][-1] == "launch_gpu":
        rows = [row[:-1] for row in rows]
    summary_path.write_text(
        "\n".join("\t".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    second, record_path, _ = _run_smoke(tmp_path, resume=True)
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 192


def test_legacy_eight_column_gpu_header_is_incompatible_with_resume(tmp_path):
    first, record_path, output_root = _run_smoke(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr

    summary_path = output_root / "smoke_summary.tsv"
    rows = [line.split("\t") for line in summary_path.read_text().splitlines()]
    assert rows[0][-1] == "launch_gpu"
    rows[0][-1] = "gpu"
    summary_path.write_text(
        "\n".join("\t".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    second, record_path, _ = _run_smoke(tmp_path, resume=True)
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 192


def test_resume_preserves_prior_launch_gpu_when_assignment_changes(tmp_path):
    first, record_path, output_root = _run_smoke(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr

    second, record_path, _ = _run_smoke(
        tmp_path,
        resume=True,
        extra_environment={"GPUS": "2 3"},
    )
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 96

    rows = (output_root / "smoke_summary.tsv").read_text().splitlines()[1:]
    assert Counter(row.split("\t")[7] for row in rows) == Counter(
        {"0": 32, "1": 32, "cpu": 32}
    )


@pytest.mark.parametrize("mutation", [
    "metrics",
    "predictions",
    "checkpoint",
    "identity",
    "summary_identity",
    "manifest_output_dir",
    "truncated_manifest",
])
def test_resume_rejects_incomplete_or_mismatched_completion_artifacts(tmp_path, mutation):
    first, record_path, output_root = _run_smoke(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    task_dir = _task_output_dir(output_root)

    if mutation == "metrics":
        (task_dir / "metrics.json").write_text("not json", encoding="utf-8")
    elif mutation == "predictions":
        with (task_dir / "predictions.csv").open("ab") as handle:
            handle.write(b"tampered\\n")
    elif mutation == "checkpoint":
        (task_dir / "best.pt").write_bytes(b"tampered checkpoint")
    elif mutation == "summary_identity":
        summary = output_root / "smoke_summary.tsv"
        rows = [line.split("\t") for line in summary.read_text(encoding="utf-8").splitlines()]
        for row in rows[1:]:
            if tuple(row[:4]) == ("24", "1", "skippd_luoyang", "TSMixer"):
                row[2] = "pvod_station00_ylj"
                break
        summary.write_text(
            "\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8"
        )
    elif mutation in {"identity", "manifest_output_dir"}:
        manifest = task_dir / "completion.tsv"
        rows = [line.split("\t") for line in manifest.read_text(encoding="utf-8").splitlines()]
        if mutation == "identity":
            rows[1][1] = "pvod_station00_ylj"
        else:
            rows[1][5] = str(task_dir) + "-wrong"
        manifest.write_text(
            "\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8"
        )
    else:
        manifest = task_dir / "completion.tsv"
        rows = manifest.read_text(encoding="utf-8").splitlines()
        manifest.write_text(rows[0] + "\n", encoding="utf-8")

    second, record_path, _ = _run_smoke(
        tmp_path,
        resume=True,
        extra_environment={"GPUS": "2 3"},
    )
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    expected_retries = 2 if mutation == "summary_identity" else 1
    assert len(runs) == 96 + expected_retries
    retried_tasks = {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in runs[96:]
    }
    assert (24, 1, "skippd_luoyang", "TSMixer") in retried_tasks
    if mutation == "summary_identity":
        assert (24, 1, "pvod_station00_ylj", "TSMixer") in retried_tasks


def test_resume_skips_valid_baseline_completion_artifacts(tmp_path):
    first, record_path, output_root = _run_smoke(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr

    second, record_path, _ = _run_smoke(
        tmp_path,
        resume=True,
        extra_environment={"GPUS": "2 3"},
    )
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 96
    assert sum(event["model"] in EXPECTED_BASELINES for event in runs) == 32

    rows = (output_root / "smoke_summary.tsv").read_text(encoding="utf-8").splitlines()[1:]
    baseline_rows = [row.split("\t") for row in rows if row.split("\t")[3] in EXPECTED_BASELINES]
    assert len(baseline_rows) == 32
    assert {row[7] for row in baseline_rows} == {"cpu"}


def test_resume_rejects_baseline_completion_with_zero_test_steps(tmp_path):
    first, record_path, output_root = _run_smoke(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    task_dir = _task_output_dir(output_root, (24, 1, "skippd_luoyang", "Persistence"))

    manifest = task_dir / "completion.tsv"
    rows = [line.split("\t") for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[1][13] = "0"
    manifest.write_text(
        "\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8"
    )

    second, record_path, _ = _run_smoke(
        tmp_path,
        resume=True,
        extra_environment={"GPUS": "2 3"},
    )
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 97
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in runs[96:]
    } == {(24, 1, "skippd_luoyang", "Persistence")}


def test_full_resume_skips_valid_baseline_completion_artifacts(tmp_path):
    first, record_path, output_root = _run_smoke(
        tmp_path,
        script_name="scripts/run_all_pure_time_series.sh",
        extra_environment={"EPOCHS": "40"},
    )
    assert first.returncode == 0, first.stdout + first.stderr
    task_dir = _task_output_dir(output_root, (24, 1, "skippd_luoyang", "Persistence"))
    manifest_rows = (task_dir / "completion.tsv").read_text(encoding="utf-8").splitlines()
    fields = manifest_rows[1].split("\t")
    assert fields[6:14] == ["full", "0", "0", "0", "0", "0", "0", "1"]

    second, record_path, _ = _run_smoke(
        tmp_path,
        resume=True,
        script_name="scripts/run_all_pure_time_series.sh",
        extra_environment={"EPOCHS": "40", "GPUS": "2 3"},
    )
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 96
    assert sum(event["model"] in EXPECTED_BASELINES for event in runs) == 32
    rows = (output_root / "run_summary.tsv").read_text(encoding="utf-8").splitlines()[1:]
    assert len(rows) == 96
    assert {row.split("\t")[7] for row in rows if row.split("\t")[3] in EXPECTED_BASELINES} == {"cpu"}


def test_full_resume_retries_invalid_baseline_completion_artifacts(tmp_path):
    first, record_path, output_root = _run_smoke(
        tmp_path,
        script_name="scripts/run_all_pure_time_series.sh",
        extra_environment={"EPOCHS": "40"},
    )
    assert first.returncode == 0, first.stdout + first.stderr
    task_dir = _task_output_dir(output_root, (24, 1, "skippd_luoyang", "Persistence"))

    manifest = task_dir / "completion.tsv"
    rows = [line.split("\t") for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[1][13] = "0"
    manifest.write_text(
        "\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8"
    )

    second, record_path, _ = _run_smoke(
        tmp_path,
        resume=True,
        script_name="scripts/run_all_pure_time_series.sh",
        extra_environment={"EPOCHS": "40", "GPUS": "2 3"},
    )
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 97
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in runs[96:]
    } == {(24, 1, "skippd_luoyang", "Persistence")}


@pytest.mark.parametrize("mutation", [
    "metrics",
    "predictions",
    "checkpoint",
    "steps",
    "identity",
])
def test_resume_rejects_corrupt_baseline_completion_artifacts(tmp_path, mutation):
    first, record_path, output_root = _run_smoke(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    task_dir = _task_output_dir(output_root, (24, 1, "skippd_luoyang", "Persistence"))

    if mutation == "metrics":
        (task_dir / "metrics.json").write_text("not json", encoding="utf-8")
    elif mutation == "predictions":
        with (task_dir / "predictions.csv").open("ab") as handle:
            handle.write(b"tampered\\n")
    elif mutation == "checkpoint":
        (task_dir / "best.pt").write_bytes(b"tampered checkpoint")
    else:
        manifest = task_dir / "completion.tsv"
        rows = [line.split("\t") for line in manifest.read_text(encoding="utf-8").splitlines()]
        if mutation == "steps":
            rows[1][11] = "1"
        else:
            rows[1][2] = "SmartPersistence"
        manifest.write_text(
            "\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8"
        )

    second, record_path, _ = _run_smoke(
        tmp_path,
        resume=True,
        extra_environment={"GPUS": "2 3"},
    )
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 97
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in runs[96:]
    } == {(24, 1, "skippd_luoyang", "Persistence")}


def test_resume_rejects_duplicate_summary_identity(tmp_path):
    first, record_path, output_root = _run_smoke(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    summary = output_root / "smoke_summary.tsv"
    rows = summary.read_text(encoding="utf-8").splitlines()
    rows.append(rows[1])
    summary.write_text("\n".join(rows) + "\n", encoding="utf-8")

    second, record_path, _ = _run_smoke(tmp_path, resume=True)
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 97


def test_resume_rejects_completion_from_different_full_epoch_setting(tmp_path):
    first, record_path, output_root = _run_smoke(
        tmp_path,
        script_name="scripts/run_all_pure_time_series.sh",
        extra_environment={"EPOCHS": "40"},
    )
    assert first.returncode == 0, first.stdout + first.stderr

    second, record_path, _ = _run_smoke(
        tmp_path,
        resume=True,
        script_name="scripts/run_all_pure_time_series.sh",
        extra_environment={"EPOCHS": "41"},
    )
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 160


def test_smoke_resume_retries_only_failed_combination_and_rewrites_summary(tmp_path):
    first, record_path, output_root = _run_smoke(tmp_path, fail_model="TSMixer")
    assert first.returncode != 0
    first_runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(first_runs) == 96
    failed_event = next(
        event for event in first_runs
        if (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        == (24, 1, "skippd_luoyang", "TSMixer")
    )
    failed_output = Path(
        failed_event["args"][failed_event["args"].index("--output_dir") + 1]
    )
    failed_output.mkdir(parents=True, exist_ok=True)
    (failed_output / "best.pt").write_bytes(b"left by failed run")
    (failed_output / "metrics.json").write_text("{}", encoding="utf-8")
    (failed_output / "predictions.csv").write_text("prediction\n", encoding="utf-8")

    second, record_path, _ = _run_smoke(
        tmp_path,
        resume=True,
        extra_environment={"GPUS": "2 3"},
    )
    assert second.returncode == 0, second.stdout + second.stderr
    all_runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(all_runs) == 100
    retried = all_runs[96:]
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in retried
    } == {
        (24, 1, "skippd_luoyang", "TSMixer"),
        (48, 1, "skippd_luoyang", "TSMixer"),
        (48, 48, "skippd_luoyang", "TSMixer"),
        (96, 48, "skippd_luoyang", "TSMixer"),
    }

    rows = (output_root / "smoke_summary.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows[1:]) == 96
    assert len({tuple(row.split("\t")[:4]) for row in rows[1:]}) == 96
    assert all(row.split("\t")[4] == "PASS" for row in rows[1:])

    summary_gpu = {}
    for row in rows[1:]:
        fields = row.split("\t")
        summary_gpu[(int(fields[0]), int(fields[1]), fields[2], fields[3])] = fields[7]
    first_gpu = {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"]): event["gpu"]
        for event in first_runs
    }
    retried_gpu = {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"]): event["gpu"]
        for event in retried
    }
    retried_tasks = set(retried_gpu)
    assert retried_tasks
    for task, prior_gpu in first_gpu.items():
        expected_gpu = retried_gpu[task] if task in retried_tasks else (
            "cpu" if prior_gpu is None else prior_gpu
        )
        assert summary_gpu[task] == expected_gpu


def test_full_runner_uses_same_96_tasks_without_smoke_flag(tmp_path):
    result, record_path, output_root = _run_smoke(
        tmp_path,
        script_name="scripts/run_all_pure_time_series.sh",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 96
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in runs
    } == ({(seq_len, pred_len, dataset, model)
           for seq_len, pred_len, dataset in EXPECTED_TASKS
           for model in EXPECTED_MODELS}
          | {(seq_len, pred_len, dataset, baseline)
             for seq_len, pred_len, dataset in EXPECTED_TASKS
             for baseline in EXPECTED_BASELINES})
    assert all("--smoke" not in event["args"] for event in runs)
    assert all("--epochs" in event["args"] for event in runs if event["model"] in EXPECTED_MODELS)
    assert all(
        event["args"][event["args"].index("--device") + 1]
        == ("cpu" if event["model"] in EXPECTED_BASELINES else "cuda:0")
        for event in runs
    )
    rows = (output_root / "run_summary.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows[1:]) == 96
    assert all(row.split("\t")[4] == "PASS" for row in rows[1:])


def test_full_runner_rejects_duplicate_model_names(tmp_path):
    result, record_path, _ = _run_smoke(
        tmp_path,
        script_name="scripts/run_all_pure_time_series.sh",
        extra_environment={"FAKE_DUPLICATE_MODELS": "1"},
    )

    assert result.returncode == 2
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert runs == []


def test_full_runner_rejects_duplicate_baseline_names(tmp_path):
    result, record_path, _ = _run_smoke(
        tmp_path,
        script_name="scripts/run_all_pure_time_series.sh",
        extra_environment={"FAKE_DUPLICATE_BASELINES": "1"},
    )

    assert result.returncode == 2
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert runs == []


def test_full_runner_fails_if_a_gpu_worker_stops_early(tmp_path):
    result, _, output_root = _run_smoke(
        tmp_path,
        script_name="scripts/run_all_pure_time_series.sh",
        extra_environment={"FAKE_KILL_WORKER_MODEL": "TSMixer"},
    )

    assert result.returncode != 0
    rows = (output_root / "run_summary.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows[1:]) < 96


@pytest.mark.parametrize("script_name", [
    "run_all_pure_time_series.sh",
    "smoke_all_pure_time_series.sh",
])
def test_orchestration_scripts_are_executable(script_name):
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name
    assert script.is_file()
    assert script.stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize("script_name", [
    "run_time_series_models.sh",
    "smoke_time_series_models.sh",
])
def test_legacy_wrappers_are_removed(script_name):
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name
    assert not script.exists()
