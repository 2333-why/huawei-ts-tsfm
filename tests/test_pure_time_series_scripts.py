"""Executable contracts for the pure time-series batch scripts."""

from __future__ import annotations

import json
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


def test_smoke_expands_exactly_64_power_runs_with_required_arguments(tmp_path):
    result, record_path, output_root = _run_smoke(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    events = _events(record_path)
    assert [event["kind"] for event in events].count("list") == 1
    runs = [event for event in events if event["kind"] == "run"]
    assert len(runs) == 64
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in runs
    } == {(seq_len, pred_len, dataset, model)
          for seq_len, pred_len, dataset in EXPECTED_TASKS
          for model in EXPECTED_MODELS}
    assert {event["gpu"] for event in runs} == {"0", "1"}
    assert sum(event["gpu"] == "0" for event in runs) == 32
    assert sum(event["gpu"] == "1" for event in runs) == 32
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
    assert len(set(output_dirs)) == 64
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

    summary = output_root / "smoke_summary.tsv"
    rows = summary.read_text(encoding="utf-8").splitlines()
    assert rows[0].split("\t") == [
        "seq_len", "pred_len", "dataset", "model", "status", "output_dir",
        "exit_code", "gpu",
    ]
    assert len(rows[1:]) == 64
    assert all(row.split("\t")[4] == "PASS" for row in rows[1:])


def test_summary_persists_gpu_provenance_matching_cuda_assignment(tmp_path):
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
        "exit_code", "gpu",
    ]
    assert len(rows[1:]) == 64

    summary_gpu_counts = Counter()
    for row in rows[1:]:
        fields = row.split("\t")
        task = (int(fields[0]), int(fields[1]), fields[2], fields[3])
        gpu = fields[7]
        assert gpu in {"0", "1"}
        assert gpu == runs[task]["cuda_visible_devices"]
        summary_gpu_counts[gpu] += 1
    assert summary_gpu_counts == Counter({"0": 32, "1": 32})


def test_legacy_summary_without_gpu_is_incompatible_with_resume(tmp_path):
    first, record_path, output_root = _run_smoke(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr

    summary_path = output_root / "smoke_summary.tsv"
    rows = [line.split("\t") for line in summary_path.read_text().splitlines()]
    if rows[0][-1] == "gpu":
        rows = [row[:-1] for row in rows]
    summary_path.write_text(
        "\n".join("\t".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    second, record_path, _ = _run_smoke(tmp_path, resume=True)
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 128


def test_resume_key_ignores_gpu_assignment(tmp_path):
    first, record_path, output_root = _run_smoke(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr

    second, record_path, _ = _run_smoke(
        tmp_path,
        resume=True,
        extra_environment={"GPUS": "2 3"},
    )
    assert second.returncode == 0, second.stdout + second.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 64

    rows = (output_root / "smoke_summary.tsv").read_text().splitlines()[1:]
    assert Counter(row.split("\t")[7] for row in rows) == Counter({"2": 32, "3": 32})


def test_smoke_resume_retries_only_failed_combination_and_rewrites_summary(tmp_path):
    first, record_path, output_root = _run_smoke(tmp_path, fail_model="TSMixer")
    assert first.returncode != 0
    first_runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(first_runs) == 64
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

    second, record_path, _ = _run_smoke(tmp_path, resume=True)
    assert second.returncode == 0, second.stdout + second.stderr
    all_runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(all_runs) == 68
    retried = all_runs[64:]
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
    assert len(rows[1:]) == 64
    assert len({tuple(row.split("\t")[:4]) for row in rows[1:]}) == 64
    assert all(row.split("\t")[4] == "PASS" for row in rows[1:])


def test_full_runner_uses_same_64_tasks_without_smoke_flag(tmp_path):
    result, record_path, output_root = _run_smoke(
        tmp_path,
        script_name="scripts/run_all_pure_time_series.sh",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    runs = [event for event in _events(record_path) if event["kind"] == "run"]
    assert len(runs) == 64
    assert {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"])
        for event in runs
    } == {(seq_len, pred_len, dataset, model)
          for seq_len, pred_len, dataset in EXPECTED_TASKS
          for model in EXPECTED_MODELS}
    assert all("--smoke" not in event["args"] for event in runs)
    assert all("--epochs" in event["args"] for event in runs)
    rows = (output_root / "run_summary.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows[1:]) == 64
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
        extra_environment={"FAKE_KILL_WORKER_MODEL": "TSMixer"},
    )

    assert result.returncode != 0
    rows = (output_root / "run_summary.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows[1:]) < 64


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
