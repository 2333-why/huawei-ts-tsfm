"""Executable contracts for the two-GPU foundation-model scripts."""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

import pytest


MODELS = ("Sundial", "TimeMoE")
MODES = ("zero_shot", "adapter", "full", "last_layer")
SHAPES = (
    ("seq48_pred1", "skippd_luoyang", 48, 1),
    ("seq48_pred1", "pvod_station00_ylj", 48, 1),
    ("seq96_h4", "skippd_luoyang", 96, 48),
    ("seq96_h4", "pvod_station00_ylj", 96, 16),
)
SUMMARY_HEADER = (
    "seq_len\tpred_len\tdataset\tmodel\tmode\tstatus\toutput_dir\t"
    "exit_code\tlaunch_gpu\tdata_fingerprint"
)
COMPLETION_HEADER = (
    "schema_version\tdataset\tmodel\tseq_len\tpred_len\toutput_dir\t"
    "run_mode\tepochs\tmax_train_steps\tmax_eval_steps\tmax_test_steps\t"
    "train_steps\tval_steps\ttest_steps\tbest_sha256\tpredictions_sha256\t"
    "metrics_sha256\tmode\tmodel_id\tmodel_revision"
)


def _fake_python(tmp_path: Path, repo: Path) -> Path:
    fake = tmp_path / "fake_foundation_python"
    fake.write_text(
        textwrap.dedent(
            r'''
            #!/usr/bin/env python3
            import hashlib
            import json
            import os
            from pathlib import Path
            import subprocess
            import sys
            import time

            sys.path.insert(0, os.environ["FAKE_REPO"])
            from models.registry import get_model_spec

            COMPLETION_HEADER = (
                "schema_version\tdataset\tmodel\tseq_len\tpred_len\toutput_dir\t"
                "run_mode\tepochs\tmax_train_steps\tmax_eval_steps\tmax_test_steps\t"
                "train_steps\tval_steps\ttest_steps\tbest_sha256\tpredictions_sha256\t"
                "metrics_sha256\tmode\tmodel_id\tmodel_revision"
            )

            args = sys.argv[1:]
            record_path = Path(os.environ["FAKE_RECORD"])

            def event_write(event):
                with record_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")

            if args and args[-1] == "--list-models":
                event_write({"kind": "list", "args": args,
                             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
                print("Sundial\nTimeMoE")
                raise SystemExit(0)
            if args and args[-1] == "--list-modes":
                event_write({"kind": "list-modes", "args": args,
                             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
                print("zero_shot\nadapter\nfull\nlast_layer")
                raise SystemExit(0)

            def value(name):
                return args[args.index(name) + 1]

            dataset = value("--dataset")
            model = value("--model")
            mode = value("--mode")
            seq_len = int(value("--seq_len"))
            pred_len = int(value("--pred_len"))
            output = Path(value("--output_dir"))
            config = Path(value("--config"))
            gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            key = f"{seq_len}|{pred_len}|{dataset}|{model}|{mode}"
            event = {
                "kind": "run", "args": args, "dataset": dataset, "model": model,
                "mode": mode, "seq_len": seq_len, "pred_len": pred_len, "gpu": gpu,
                "cuda_visible_devices": gpu, "key": key,
                "parquet_file": json.loads(config.read_text(encoding="utf-8"))["paths"]["parquet_file"],
                "started_at": time.monotonic_ns(),
            }
            lock_root = Path(os.environ["FAKE_LOCK_ROOT"])
            lock_root.mkdir(parents=True, exist_ok=True)
            lock_path = lock_root / ("gpu-" + gpu + ".lock")
            acquired = False
            try:
                lock_path.mkdir()
                acquired = True
            except FileExistsError:
                event["overlap"] = True
            exit_code = 0
            child = None
            try:
                pid_dir_value = os.environ.get("FAKE_PID_DIR")
                if pid_dir_value:
                    pid_dir = Path(pid_dir_value)
                    pid_dir.mkdir(parents=True, exist_ok=True)
                    (pid_dir / ("runner-" + str(os.getpid()))).write_text(str(os.getpid()), encoding="utf-8")
                if os.environ.get("FAKE_LONG_SLEEP") == "1":
                    child_code = (
                        "import subprocess,sys,time; "
                        "grandchild=subprocess.Popen(['sleep','30']); "
                        "open(sys.argv[1],'w').write(str(grandchild.pid)); "
                        "time.sleep(30)"
                    )
                    child = subprocess.Popen([
                        sys.executable, "-c", child_code,
                        str(Path(pid_dir_value) / ("grandchild-" + str(os.getpid()))),
                    ])
                    (Path(pid_dir_value) / ("child-" + str(os.getpid()))).write_text(
                        str(child.pid), encoding="utf-8"
                    )
                time.sleep(float(os.environ.get("FAKE_RUN_SLEEP", "0.03")))
                if os.environ.get("FAKE_FAIL_KEY") == key:
                    exit_code = 17
                else:
                    output.mkdir(parents=True, exist_ok=True)
                    spec = get_model_spec(model)
                    smoke = "--smoke" in args
                    run_mode = "smoke" if smoke else "full"
                    limits = {"train": 1, "val": 1, "test": 1} if smoke else {
                        "train": 0, "val": 0, "test": 0,
                    }
                    phase = {"train": 0, "val": 0, "test": 1} if mode == "zero_shot" else {
                        "train": 1, "val": 1, "test": 1,
                    }
                    epochs = 0 if mode == "zero_shot" else 1
                    (output / "best.pt").write_bytes((key + "\n").encode("utf-8"))
                    (output / "predictions.csv").write_text(
                        "issue_time,target_time,horizon_minutes,y_true,y_pred\n"
                        "2025-01-01T00:00:00,2025-01-01T00:05:00,5,1.0,1.0\n",
                        encoding="utf-8",
                    )
                    metrics = {
                        "dataset": dataset, "model": model, "model_id": spec.model_id,
                        "model_revision": spec.revision, "effective_revision": spec.revision,
                        "revision": spec.revision, "mode": mode, "seq_len": seq_len,
                        "pred_len": pred_len, "epochs": epochs, "run_mode": run_mode,
                        "limits": limits, "train_steps": phase["train"],
                        "val_steps": phase["val"], "test_steps": phase["test"],
                        "phase_steps": phase,
                        "trainability": {"mode": mode},
                        "trainability_report": {"mode": mode},
                    }
                    (output / "metrics.json").write_text(
                        json.dumps(metrics, sort_keys=True, indent=2) + "\n", encoding="utf-8"
                    )
                    def digest(name):
                        return hashlib.sha256((output / name).read_bytes()).hexdigest()
                    values = [
                        "2", dataset, model, str(seq_len), str(pred_len), str(output.resolve()),
                        run_mode, str(epochs), str(limits["train"]), str(limits["val"]),
                        str(limits["test"]), str(phase["train"]), str(phase["val"]),
                        str(phase["test"]), digest("best.pt"), digest("predictions.csv"),
                        digest("metrics.json"), mode, spec.model_id, spec.revision,
                    ]
                    manifest = output / "completion.tsv"
                    temporary = output / (".completion." + str(os.getpid()) + ".tmp")
                    temporary.write_text(COMPLETION_HEADER + "\n" + "\t".join(values) + "\n", encoding="utf-8")
                    temporary.replace(manifest)
            finally:
                if child is not None:
                    child.terminate()
                    child.wait(timeout=2)
                if acquired:
                    lock_path.rmdir()
                event["finished_at"] = time.monotonic_ns()
                event["exit_code"] = exit_code
                event_write(event)
            raise SystemExit(exit_code)
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake


def _harness(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    fake = _fake_python(tmp_path, repo)
    skippd = tmp_path / "skippd.parquet"
    pvod = tmp_path / "pvod.parquet"
    skippd.write_bytes(b"skippd")
    pvod.write_bytes(b"pvod")
    output_root = tmp_path / "results"
    record = tmp_path / "invocations.jsonl"
    env = os.environ.copy()
    env.update(
        PYTHON=str(fake), CONFIG_PYTHON=sys.executable, FAKE_REPO=str(repo),
        FAKE_RECORD=str(record), FAKE_LOCK_ROOT=str(tmp_path / "locks"),
        SKIPPD_PARQUET=str(skippd), PVOD_PARQUET=str(pvod),
        OUTPUT_ROOT=str(output_root), SMOKE="1", RESUME="0",
    )
    return repo, env, output_root, record


def _copied_default_harness(tmp_path: Path):
    source_repo = Path(__file__).resolve().parents[1]
    copied_repo = tmp_path / "copied repo"
    inherited_ignore = shutil.ignore_patterns(".git", "__pycache__", "*.pyc")

    def ignore_checkout_entry(directory, names):
        ignored = set(inherited_ignore(directory, names))
        if Path(directory).resolve() == source_repo.resolve():
            ignored.add("codex")
        return ignored

    shutil.copytree(source_repo, copied_repo, ignore=ignore_checkout_entry)
    fake = _fake_python(tmp_path, copied_repo)
    skippd = tmp_path / "default-skippd.parquet"
    pvod = tmp_path / "default-pvod.parquet"
    skippd.write_bytes(b"skippd")
    pvod.write_bytes(b"pvod")
    record = tmp_path / "default-invocations.jsonl"
    env = os.environ.copy()
    env.update(
        PYTHON=str(fake), CONFIG_PYTHON=sys.executable, FAKE_REPO=str(copied_repo),
        FAKE_RECORD=str(record), FAKE_LOCK_ROOT=str(tmp_path / "default-locks"),
        SKIPPD_PARQUET=str(skippd), PVOD_PARQUET=str(pvod), SMOKE="1", RESUME="0",
    )
    env.pop("OUTPUT_ROOT", None)
    env.pop("SUMMARY_PATH", None)
    return copied_repo, env


def _events(record: Path):
    return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]


def test_two_gpu_smoke_expands_32_tasks_with_serial_queues_and_contained_outputs(tmp_path):
    repo, env, output_root, record = _harness(tmp_path)
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = [event for event in _events(record) if event["kind"] == "run"]
    expected = {
        (seq, pred, dataset, model, mode)
        for _setting, dataset, seq, pred in SHAPES
        for model in MODELS for mode in MODES
    }
    observed = {
        (event["seq_len"], event["pred_len"], event["dataset"], event["model"], event["mode"])
        for event in events
    }
    assert len(events) == 32 and observed == expected
    assert Counter(event["gpu"] for event in events) == Counter({"0": 16, "1": 16})
    assert all(not event.get("overlap", False) for event in events)
    assert any(
        left["started_at"] < right["finished_at"] and right["started_at"] < left["finished_at"]
        for left in events if left["gpu"] == "0"
        for right in events if right["gpu"] == "1"
    )
    assert all(event["args"][event["args"].index("--device") + 1] == "cuda:0" for event in events)
    for event in events:
        path = Path(event["args"][event["args"].index("--output_dir") + 1]).resolve()
        assert path != output_root.resolve()
        path.relative_to(output_root.resolve())
        assert path.parts[-1] == event["mode"]
        completion = (path / "completion.tsv").read_text(encoding="utf-8").splitlines()
        assert len(completion) == 2 and completion[0] == COMPLETION_HEADER
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        expected_phase = {"train": 0, "val": 0, "test": 1} if event["mode"] == "zero_shot" else {
            "train": 1, "val": 1, "test": 1,
        }
        assert metrics["phase_steps"] == expected_phase
        assert metrics["limits"] == {"train": 1, "val": 1, "test": 1}
    rows = (output_root / "smoke_summary.tsv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == SUMMARY_HEADER and len(rows) == 33
    assert all(row.split("\t")[5] == "PASS" for row in rows[1:])


@pytest.mark.parametrize("gpus", ["0,1 0", "00 0", "0 00", "-1 0", "0 gpu", "0 GPU-12345678"])
def test_invalid_gpu_tokens_fail_before_any_fake_worker_run(tmp_path, gpus):
    repo, env, output_root, record = _harness(tmp_path)
    env["GPUS"] = gpus
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "canonical distinct physical GPU ordinals" in result.stderr
    assert not record.exists() or not [event for event in _events(record) if event["kind"] == "run"]
    assert not (output_root / "smoke_summary.tsv").exists()


def test_in_root_output_symlink_aliases_are_rejected_before_workers_start(tmp_path):
    repo, env, output_root, record = _harness(tmp_path)
    alias_parent = output_root / "seq48_pred1" / "skippd_luoyang" / "Sundial"
    alias_parent.mkdir(parents=True)
    (alias_parent / "zero_shot").mkdir()
    (alias_parent / "adapter").symlink_to(alias_parent / "zero_shot", target_is_directory=True)
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode != 0
    assert [event for event in _events(record) if event["kind"] == "run"] == []


def test_one_runner_failure_is_recorded_and_propagates_after_independent_tasks(tmp_path):
    repo, env, output_root, record = _harness(tmp_path)
    env["FAKE_FAIL_KEY"] = "48|1|skippd_luoyang|Sundial|adapter"
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    events = [event for event in _events(record) if event["kind"] == "run"]
    assert len(events) == 32
    rows = (output_root / "smoke_summary.tsv").read_text(encoding="utf-8").splitlines()
    failures = [row.split("\t") for row in rows[1:] if row.split("\t")[5] == "FAIL"]
    assert len(rows) == 33 and len(failures) == 1
    assert failures[0][:5] == ["48", "1", "skippd_luoyang", "Sundial", "adapter"]
    assert failures[0][7] == "17"


def test_valid_resume_skips_every_task_and_keeps_original_gpu_provenance(tmp_path):
    repo, env, output_root, record = _harness(tmp_path)
    first = subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    before = len([event for event in _events(record) if event["kind"] == "run"])

    env.update(RESUME="1", GPUS="2 3")
    second = subprocess.run(
        ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert len([event for event in _events(record) if event["kind"] == "run"]) == before
    rows = (output_root / "smoke_summary.tsv").read_text(encoding="utf-8").splitlines()
    assert Counter(row.split("\t")[8] for row in rows[1:]) == Counter({"0": 16, "1": 16})


def _mutate_completion(output: Path, mutation: str):
    manifest = output / "completion.tsv"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    fields = rows[1].split("\t")
    if mutation == "mode":
        fields[17] = "adapter"
    elif mutation == "steps":
        fields[13] = "0"  # smoke zero-shot test_steps must be exactly one
    elif mutation == "identity":
        fields[1] = "pvod_station00_ylj"
    manifest.write_text(rows[0] + "\n" + "\t".join(fields) + "\n", encoding="utf-8")


@pytest.mark.parametrize("mutation", ["metrics", "predictions", "checkpoint", "fingerprint", "mode", "steps", "identity"])
def test_stale_or_mutated_completion_evidence_reruns_only_one_task(tmp_path, mutation):
    repo, env, _output_root, record = _harness(tmp_path)
    first = subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    output = tmp_path / "results" / "seq48_pred1" / "skippd_luoyang" / "Sundial" / "zero_shot"
    if mutation == "metrics":
        (output / "metrics.json").write_text("not json\n", encoding="utf-8")
    elif mutation == "predictions":
        with (output / "predictions.csv").open("a", encoding="utf-8") as handle:
            handle.write("tampered\n")
    elif mutation == "checkpoint":
        (output / "best.pt").write_bytes(b"tampered")
    elif mutation == "fingerprint":
        (output / "data_fingerprint.txt").write_text("wrong\n", encoding="utf-8")
    else:
        _mutate_completion(output, mutation)
    before = len([event for event in _events(record) if event["kind"] == "run"])

    env.update(RESUME="1", GPUS="2 3")
    second = subprocess.run(
        ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    rerun = [event for event in _events(record) if event["kind"] == "run"][before:]
    assert len(rerun) == 1
    assert (rerun[0]["seq_len"], rerun[0]["pred_len"], rerun[0]["dataset"], rerun[0]["model"], rerun[0]["mode"]) == (
        48, 1, "skippd_luoyang", "Sundial", "zero_shot"
    )


def test_duplicate_summary_identity_invalidates_only_that_task(tmp_path):
    repo, env, output_root, record = _harness(tmp_path)
    first = subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    summary = output_root / "smoke_summary.tsv"
    rows = summary.read_text(encoding="utf-8").splitlines()
    summary.write_text("\n".join(rows + [rows[1]]) + "\n", encoding="utf-8")
    before = len([event for event in _events(record) if event["kind"] == "run"])
    env.update(RESUME="1")
    second = subprocess.run(
        ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert len([event for event in _events(record) if event["kind"] == "run"]) == before + 1


def test_resume_rejects_empty_surplus_summary_column_for_only_that_task(tmp_path):
    repo, env, output_root, record = _harness(tmp_path)
    first = subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    summary = output_root / "smoke_summary.tsv"
    rows = summary.read_text(encoding="utf-8").splitlines()
    summary.write_text("\n".join([rows[0], rows[1] + "\t"] + rows[2:]) + "\n", encoding="utf-8")
    before = len([event for event in _events(record) if event["kind"] == "run"])
    env.update(RESUME="1", GPUS="2 3")
    second = subprocess.run(
        ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    rerun = [event for event in _events(record) if event["kind"] == "run"][before:]
    assert len(rerun) == 1
    assert (rerun[0]["seq_len"], rerun[0]["pred_len"], rerun[0]["dataset"], rerun[0]["model"], rerun[0]["mode"]) == (
        48, 1, "skippd_luoyang", "Sundial", "zero_shot"
    )


def test_dataset_content_change_retries_exactly_16_tasks(tmp_path):
    repo, env, _output_root, record = _harness(tmp_path)
    assert subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    ).returncode == 0
    before = len([event for event in _events(record) if event["kind"] == "run"])
    (tmp_path / "skippd.parquet").write_bytes(b"skippd-v2")
    env.update(RESUME="1")
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rerun = [event for event in _events(record) if event["kind"] == "run"][before:]
    assert len(rerun) == 16 and {event["dataset"] for event in rerun} == {"skippd_luoyang"}


def test_effective_config_change_retries_exactly_16_tasks(tmp_path):
    repo, env, _output_root, record = _harness(tmp_path)
    assert subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    ).returncode == 0
    before = len([event for event in _events(record) if event["kind"] == "run"])
    config = repo / "configs" / "datasets" / "pvod_station00_ylj.yaml"
    original = config.read_text(encoding="utf-8")
    config.write_text(original + "fingerprint_only: changed\n", encoding="utf-8")
    try:
        env.update(RESUME="1")
        result = subprocess.run(
            ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
            cwd=repo, env=env, text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        rerun = [event for event in _events(record) if event["kind"] == "run"][before:]
        assert len(rerun) == 16 and {event["dataset"] for event in rerun} == {"pvod_station00_ylj"}
    finally:
        config.write_text(original, encoding="utf-8")


def test_registry_revision_change_retries_exactly_one_model(tmp_path):
    repo, env, _output_root, record = _harness(tmp_path)
    assert subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    ).returncode == 0
    before = len([event for event in _events(record) if event["kind"] == "run"])
    registry = repo / "models" / "registry.py"
    original = registry.read_text(encoding="utf-8")
    changed = original.replace(
        "3212e42564493f520593e5414af4367fc4b49226",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    registry.write_text(changed, encoding="utf-8")
    try:
        env.update(RESUME="1")
        result = subprocess.run(
            ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
            cwd=repo, env=env, text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        rerun = [event for event in _events(record) if event["kind"] == "run"][before:]
        assert len(rerun) == 16 and {event["model"] for event in rerun} == {"Sundial"}
    finally:
        registry.write_text(original, encoding="utf-8")


def test_unset_output_and_summary_paths_use_smoke_defaults_in_isolated_repo(tmp_path):
    repo, env = _copied_default_harness(tmp_path)
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = repo / "results_foundation_models_smoke" / "smoke_summary.tsv"
    assert summary.is_file()
    rows = summary.read_text(encoding="utf-8").splitlines()
    assert rows[0] == SUMMARY_HEADER and len(rows) == 33


def test_unrelated_file_change_keeps_all_valid_resume_tasks_skipped(tmp_path):
    repo, env, _output_root, record = _harness(tmp_path)
    first = subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    before = len([event for event in _events(record) if event["kind"] == "run"])
    (tmp_path / "unrelated.txt").write_text("unrelated change\n", encoding="utf-8")
    env.update(RESUME="1", GPUS="2 3")
    second = subprocess.run(
        ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert len([event for event in _events(record) if event["kind"] == "run"]) == before


def test_paths_with_spaces_are_supported_and_summary_is_exact(tmp_path):
    repo, env, output_root, record = _harness(tmp_path)
    spaced = tmp_path / "results with spaces"
    env["OUTPUT_ROOT"] = str(spaced)
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh"), "ignored-arg"],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = spaced / "smoke_summary.tsv"
    rows = summary.read_text(encoding="utf-8").splitlines()
    assert rows[0] == SUMMARY_HEADER and len(rows) == 33
    assert all(str(spaced.resolve()) in row for row in rows[1:])


def test_full_run_branch_omits_smoke_and_publishes_full_summary(tmp_path):
    repo, env, output_root, record = _harness(tmp_path)
    env.update(SMOKE="0")
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = [event for event in _events(record) if event["kind"] == "run"]
    assert len(events) == 32
    assert all("--smoke" not in event["args"] for event in events)
    summary = output_root / "run_summary.tsv"
    rows = summary.read_text(encoding="utf-8").splitlines()
    assert rows[0] == SUMMARY_HEADER and len(rows) == 33
    for event in events:
        output = Path(event["args"][event["args"].index("--output_dir") + 1])
        manifest = (output / "completion.tsv").read_text(encoding="utf-8").splitlines()[1].split("\t")
        assert manifest[6] == "full"
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["run_mode"] == "full"
        assert metrics["limits"] == {"train": 0, "val": 0, "test": 0}
        if event["mode"] == "zero_shot":
            assert metrics["phase_steps"] == {"train": 0, "val": 0, "test": 1}
        else:
            assert metrics["phase_steps"] == {"train": 1, "val": 1, "test": 1}


@pytest.mark.parametrize(("signal", "expected_return"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_signal_kills_fake_descendants_and_preserves_prior_summary(tmp_path, signal, expected_return):
    repo, env, output_root, _record = _harness(tmp_path)
    first = subprocess.run(
        ["bash", str(repo / "scripts" / "smoke_all_foundation_models_2gpu.sh")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    summary = output_root / "smoke_summary.tsv"
    prior = summary.read_bytes()
    pid_dir = tmp_path / "signal-pids"
    signal_env = dict(env, SMOKE="1", RESUME="0", FAKE_LONG_SLEEP="1", FAKE_RUN_SLEEP="30", FAKE_PID_DIR=str(pid_dir))
    process = subprocess.Popen(
        ["bash", str(repo / "scripts" / "run_all_foundation_models_2gpu.sh")],
        cwd=repo, env=signal_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    pids = []
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            pids = [int(path.read_text()) for path in pid_dir.glob("runner-*")]
            pids += [int(path.read_text()) for path in pid_dir.glob("child-*")]
            pids += [int(path.read_text()) for path in pid_dir.glob("grandchild-*")]
            if len(pids) >= 6:
                break
            time.sleep(0.05)
        assert len(pids) >= 6
        process.send_signal(signal)
        assert process.wait(timeout=10) == expected_return
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(Path(f"/proc/{pid}").exists() for pid in pids):
            time.sleep(0.05)
        assert not any(Path(f"/proc/{pid}").exists() for pid in pids)
        assert summary.read_bytes() == prior
        assert not list(summary.parent.glob(".smoke_summary.tsv.tmp.*"))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
