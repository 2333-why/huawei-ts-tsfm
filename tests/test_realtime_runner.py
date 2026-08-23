from pathlib import Path
import subprocess
import sys

from scripts import run_all_datasets


ROOT = Path(__file__).parents[1]


def test_subprocess_environment_forces_unbuffered_output(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr(run_all_datasets.subprocess, "run", fake_run)
    run_all_datasets.run(["python", "child.py"])
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert captured["check"] is True


def test_preflight_resolves_project_imports_outside_repository(tmp_path):
    script = ROOT / "scripts" / "dual_dataset_preflight.py"
    command = (
        "import runpy; "
        f"runpy.run_path({str(script)!r}, run_name='preflight_import_test'); "
        "import data_provider"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_continue_after_luoyang_teacher_skips_only_completed_teacher(monkeypatch):
    commands = []

    def fake_run(command):
        commands.append(command)

    monkeypatch.setattr(run_all_datasets, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_datasets.py",
            "--skip-preflight",
            "--after-luoyang-teacher",
        ],
    )
    run_all_datasets.main()

    stages = [command[2:4] for command in commands if len(command) >= 4]
    assert ["scripts/luoyang_pipeline.py", "teacher"] not in stages
    assert ["scripts/luoyang_pipeline.py", "student"] in stages
    assert ["scripts/luoyang_pipeline.py", "test"] in stages
    assert ["scripts/ylj_pipeline.py", "teacher"] in stages
    assert ["scripts/ylj_pipeline.py", "student"] in stages
    assert ["scripts/ylj_pipeline.py", "test"] in stages


def test_ylj_only_runner_never_starts_luoyang(monkeypatch):
    commands = []

    monkeypatch.setattr(run_all_datasets, "run", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all_datasets.py", "--datasets", "ylj"],
    )
    run_all_datasets.main()

    flattened = [" ".join(command) for command in commands]
    assert "--datasets ylj" in flattened[0]
    assert "test_luoyang_parquet.py" not in flattened[0]
    assert not any("luoyang_pipeline.py" in command for command in flattened)
    assert sum("ylj_pipeline.py teacher" in command for command in flattened) == 1
    assert sum("ylj_pipeline.py student" in command for command in flattened) == 1
    assert sum("ylj_pipeline.py test" in command for command in flattened) == 1
    assert sum("verify_official_outputs.py ylj" in command for command in flattened) == 1
