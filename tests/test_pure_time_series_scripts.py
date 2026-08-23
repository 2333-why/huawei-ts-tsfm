"""Executable contracts for the pure time-series batch scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import time
from typing import Optional

import pytest


EXPECTED_DATASETS = ("skippd_luoyang", "pvod_station00_ylj")
EXPECTED_SETTINGS = ((24, 1), (48, 12))
EXPECTED_MODELS = (
    "Autoformer", "Crossformer", "DLinear", "ETSformer", "FEDformer",
    "FiLM", "FreTS", "Informer", "Koopa", "LightTS", "MICN", "MSGNet",
    "Mamba", "MambaSimple", "MultiPatchFormer", "Nonstationary_Transformer",
    "PAttn", "PatchTST", "Pyraformer", "Reformer", "SCINet", "SegRNN",
    "TSMixer", "TemporalFusionTransformer", "TiDE", "TimeFilter",
    "TimeMixer", "TimeXer", "TimesNet", "Transformer", "WPMixer",
    "iTransformer",
)


def _fake_python(tmp_path: Path) -> Path:
    fake = tmp_path / "fake_python"
    fake.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import signal
import sys
import time

models = %r
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
        (output_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (output_dir / "predictions.csv").write_text("prediction\\n", encoding="utf-8")
        event["exit_code"] = 0
finally:
    event["finished_at"] = time.monotonic_ns()
    if acquired_lock:
        lock_path.rmdir()
    with record_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\\n")
raise SystemExit(event["exit_code"])
""" % (EXPECTED_MODELS,),
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


def test_smoke_expands_exactly_128_power_runs_with_required_arguments(tmp_path):
    result, record_path, output_root = _run_smoke(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    events = _events(record_path)
    assert [event["kind"] for event in events].count("list") == 1
    runs = [event for event in events if event["kind"] == "run"]
    assert len(runs) == 128
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in runs
    } == {
        (seq_len, pred_len, dataset, model)
        for seq_len, pred_len in EXPECTED_SETTINGS
        for dataset in EXPECTED_DATASETS
        for model in EXPECTED_MODELS
    }
    assert {event["gpu"] for event in runs} == {"0", "1"}
    assert sum(event["gpu"] == "0" for event in runs) == 64
    assert sum(event["gpu"] == "1" for event in runs) == 64
    assert all(not event.get("overlap", False) for event in runs)
    gpu_zero = [event for event in runs if event["gpu"] == "0"]
    gpu_one = [event for event in runs if event["gpu"] == "1"]
    assert any(
        left["started_at"] < right["finished_at"]
        and right["started_at"] < left["finished_at"]
        for left in gpu_zero
        for right in gpu_one
    )
    assert all("--smoke" in event["args"] for event in runs)
    assert all(
        event["args"][event["args"].index("--device") + 1] == "cuda:0"
        for event in runs
    )
    output_dirs = [
        event["args"][event["args"].index("--output_dir") + 1]
        for event in runs
    ]
    assert len(set(output_dirs)) == 128
    for path in output_dirs:
        try:
            Path(path).resolve().relative_to(output_root.resolve())
        except ValueError:
            pytest.fail("run output escaped the configured output root")

    summary = output_root / "smoke_summary.tsv"
    rows = summary.read_text(encoding="utf-8").splitlines()
    assert rows[0].split("\t") == [
        "seq_len", "pred_len", "dataset", "model", "status", "output_dir",
        "exit_code",
    ]
    assert len(rows[1:]) == 128
    assert all(row.split("\t")[4] == "PASS" for row in rows[1:])


def test_smoke_resume_retries_only_failed_combination_and_rewrites_summary(tmp_path):
    first, record_path, output_root = _run_smoke(tmp_path, fail_model="DLinear")
    assert first.returncode != 0
    first_runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(first_runs) == 128

    second, record_path, _ = _run_smoke(tmp_path, resume=True)
    assert second.returncode == 0, second.stdout + second.stderr
    all_runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(all_runs) == 130
    retried = all_runs[128:]
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in retried
    } == {
        (24, 1, "skippd_luoyang", "DLinear"),
        (48, 12, "skippd_luoyang", "DLinear"),
    }

    rows = (output_root / "smoke_summary.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows[1:]) == 128
    assert len({tuple(row.split("\t")[:4]) for row in rows[1:]}) == 128
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


def test_full_runner_fails_if_a_gpu_worker_stops_early(tmp_path):
    result, _, output_root = _run_smoke(
        tmp_path,
        script_name="scripts/run_all_pure_time_series.sh",
        extra_environment={"FAKE_KILL_WORKER_MODEL": "Autoformer"},
    )

    assert result.returncode != 0
    rows = (output_root / "run_summary.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows[1:]) < 128


@pytest.mark.parametrize(
    ("script_name", "smoke"),
    [
        ("scripts/smoke_time_series_models.sh", True),
        ("scripts/run_time_series_models.sh", False),
    ],
)
def test_legacy_wrappers_preserve_128_run_dual_gpu_matrix(tmp_path, script_name, smoke):
    result, record_path, output_root = _run_smoke(tmp_path, script_name=script_name)

    assert result.returncode == 0, result.stdout + result.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 128
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in runs
    } == {
        (seq_len, pred_len, dataset, model)
        for seq_len, pred_len in EXPECTED_SETTINGS
        for dataset in EXPECTED_DATASETS
        for model in EXPECTED_MODELS
    }
    assert {event["gpu"] for event in runs} == {"0", "1"}
    assert sum(event["gpu"] == "0" for event in runs) == 64
    assert sum(event["gpu"] == "1" for event in runs) == 64
    assert all(not event.get("overlap", False) for event in runs)
    assert len({
        event["args"][event["args"].index("--output_dir") + 1]
        for event in runs
    }) == 128
    if smoke:
        assert all("--smoke" in event["args"] for event in runs)
    else:
        assert all("--smoke" not in event["args"] for event in runs)
        assert all("--epochs" in event["args"] for event in runs)

    summary_name = "smoke_summary.tsv" if smoke else "run_summary.tsv"
    rows = (output_root / summary_name).read_text(encoding="utf-8").splitlines()
    assert len(rows[1:]) == 128
    assert all(row.split("\t")[4] == "PASS" for row in rows[1:])


@pytest.mark.parametrize("script_name", [
    "run_all_pure_time_series.sh",
    "smoke_all_pure_time_series.sh",
    "run_time_series_models.sh",
    "smoke_time_series_models.sh",
])
def test_orchestration_scripts_are_executable(script_name):
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name
    assert script.is_file()
    assert script.stat().st_mode & stat.S_IXUSR
