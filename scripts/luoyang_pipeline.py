#!/usr/bin/env python3
"""Configuration-driven FACTS Luoyang teacher/KD runner and inference entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: Path):
    from data_provider.data_loader_luoyang import load_luoyang_config
    return load_luoyang_config(path)


def _configured_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_teacher_checkpoint(config, config_path, prefer_latest=False):
    """Resolve the configured Teacher weight without guessing across model types."""
    from utils.checkpoint_contract import validate_checkpoint_contract

    configured = _configured_path(config["paths"]["teacher_checkpoint"])
    if configured.is_file() and not prefer_latest:
        validate_checkpoint_contract(configured, config_path)
        return configured

    checkpoint_root = _configured_path(config["paths"]["checkpoints_root"])
    candidates = []
    if configured.is_file():
        try:
            validate_checkpoint_contract(configured, config_path)
            candidates.append(configured)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
    if checkpoint_root.is_dir():
        for candidate in checkpoint_root.rglob("checkpoint.pth"):
            relative_parts = [part.lower() for part in candidate.relative_to(checkpoint_root).parts[:-1]]
            if not any("teacher" in part for part in relative_parts):
                continue
            try:
                validate_checkpoint_contract(candidate, config_path)
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                continue
            if candidate not in candidates:
                candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            "Teacher checkpoint was not found at the configured path and no "
            f"contract-matching Teacher checkpoint exists under {checkpoint_root}. "
            "Finish the Teacher run first or update paths.teacher_checkpoint."
        )
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return candidates[0]


def resolve_student_checkpoint(config, config_path, prefer_latest=False):
    """Resolve the newest contract-matching Student/KD weight for inference."""
    from utils.checkpoint_contract import validate_checkpoint_contract

    configured = _configured_path(config["paths"]["student_checkpoint"])
    if configured.is_file() and not prefer_latest:
        validate_checkpoint_contract(configured, config_path)
        return configured
    checkpoint_root = _configured_path(config["paths"]["checkpoints_root"])
    candidates = []
    if configured.is_file():
        try:
            validate_checkpoint_contract(configured, config_path)
            candidates.append(configured)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
    if checkpoint_root.is_dir():
        for candidate in checkpoint_root.rglob("checkpoint.pth"):
            parts = [part.lower() for part in candidate.relative_to(checkpoint_root).parts[:-1]]
            if not any("student" in part or "_kd_" in part for part in parts):
                continue
            try:
                validate_checkpoint_contract(candidate, config_path)
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                continue
            if candidate not in candidates:
                candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            "Student checkpoint was not found at the configured path and no "
            f"contract-matching Student checkpoint exists under {checkpoint_root}. "
            "Finish the Student/KD run first or update paths.student_checkpoint."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def common_args(config, config_path):
    dims, runtime, model, paths = (
        config["dimensions"], config["runtime"], config["model"], config["paths"]
    )
    values = [
        "--task_name", "long_term_forecast", "--data", "LuoyangParquet",
        "--luoyang_config", str(config_path), "--root_path", ".", "--data_path", ".",
        "--features", "MS", "--target", config["fields"]["target"],
        "--freq", f'{config["time"]["sampling_interval_minutes"]}t',
        "--seq_len", str(dims["history_points"]), "--label_len", "0",
        "--pred_len", str(dims["forecast_steps"]), "--enc_in", str(dims["timeseries_input_dim"]),
        "--seq_len_hours", str(
            (int(dims["history_points"]) - 1)
            * int(config["time"]["sampling_interval_minutes"]) / 60.0),
        "--pred_len_hours", str(int(config["time"]["forecast_horizon_minutes"]) / 60.0),
        "--dec_in", str(dims["timeseries_input_dim"]), "--c_out", str(dims["target_dim"]),
        "--weather_feature_dim", str(dims["weather_placeholder_dim"]),
        "--privileged_teacher",
        "--forecast_horizon_minutes", str(config["time"]["forecast_horizon_minutes"]),
        "--sample_interval_minutes", str(config["time"]["sampling_interval_minutes"]),
        "--history_order", config["time"]["history_order"], "--image_height", str(dims["image_height"]),
        "--image_width", str(dims["image_width"]), "--stanford_image_mode", "rgb",
        "--image_encoder_type", model["image_encoder_type"],
        "--image_cnn_dim", str(model["image_cnn_dim"]),
        "--image_temporal_hidden_dim", str(model["image_temporal_hidden_dim"]),
        "--d_model", str(model["d_model"]), "--n_heads", str(model["n_heads"]),
        "--e_layers", str(model["e_layers"]), "--d_layers", str(model["d_layers"]),
        "--d_ff", str(model["d_ff"]), "--factor", str(model["factor"]),
        "--num_workers", str(runtime["num_workers"]),
        "--prefetch_factor", str(runtime["prefetch_factor"]),
        "--patience", str(runtime["patience"]),
        "--checkpoints", paths["checkpoints_root"], "--results_root", paths["results_root"],
        "--use_multi_gpu", "--devices", runtime["gpu_devices"],
    ]
    if runtime["use_amp"]:
        values.append("--use_amp")
    monitoring = config.get("monitoring", {})
    if monitoring.get("enabled", False):
        values.extend([
            "--monitor",
            "--monitor_level", str(monitoring.get("level", "standard")),
            "--monitor_epoch_interval", str(monitoring.get("epoch_interval", 1)),
            "--monitor_gradient_interval", str(
                monitoring.get("gradient_interval_steps", 50)),
            "--monitor_min_slice_count", str(monitoring.get("min_slice_count", 100)),
            "--monitor_min_slice_days", str(monitoring.get("min_slice_days", 5)),
        ])
        if monitoring.get("include_test_during_training", False):
            values.append("--monitor_include_test_during_training")
        if monitoring.get("save_raw_predictions", False):
            values.append("--monitor_save_raw_predictions")
    return values


def command(config, config_path, action, teacher_checkpoint=None, student_checkpoint=None):
    py = sys.executable
    student_only = action == "student-only"
    base = common_args(config, config_path)
    runtime, model, paths = config["runtime"], config["model"], config["paths"]
    if action == "teacher":
        return [
            py, "-u", "run_teacher.py", "--is_training", "1", "--model", "MTS_31F",
            "--model_id", f'Luoyang_teacher_{config["dimensions"]["history_points"]}_{config["dimensions"]["forecast_steps"]}',
            "--batch_size", str(runtime["teacher_batch_size"]),
            "--learning_rate", str(runtime.get(
                "teacher_learning_rate", runtime.get("learning_rate", 0.0001))),
            "--train_epochs", str(runtime["teacher_epochs"]), "--fuse_strategy", model["teacher_fuse_strategy"],
            "--des", "Luoyang_teacher", *base,
        ]
    student = [py, "-u", "run_kd.py", "--model", "MTS_31"]
    if not student_only:
        student.extend([
            "--teacher_model", "MTS_31F",
            "--teacher_path", str(teacher_checkpoint or paths["teacher_checkpoint"]),
            "--teacher_image_encoder_type", model["image_encoder_type"],
            "--teacher_fuse_strategy", model["teacher_fuse_strategy"],
        ])
    student.extend([
        "--model_id", f'Luoyang_student_{config["dimensions"]["history_points"]}_{config["dimensions"]["forecast_steps"]}',
        "--batch_size", str(runtime["student_batch_size"]), "--train_epochs", str(runtime["student_epochs"]),
        "--learning_rate", str(runtime.get(
            "student_learning_rate", runtime.get("learning_rate", 0.0001))),
        "--fuse_strategy", model["student_fuse_strategy"],
        "--student_fuse_strategy", model["student_fuse_strategy"],
        "--strict_causal_student", "--kd_type", model["kd_type"],
        "--kd_loss_weight_sim", str(0.0 if student_only else model["kd_similarity_weight"]),
        "--kd_loss_weight_inter", str(0.0 if student_only else model["kd_intervention_weight"]),
        "--kd_intervention_min_feature_delta", str(model["kd_intervention_min_feature_delta"]),
        "--modality_dropout_rate", str(model["modality_dropout_rate"]),
        "--val_selection_metric", model.get("student_checkpoint_metric", "loss"),
        "--extra_tag", "luoyang_48step", "--des", "Luoyang_student", "--trail",
        "student_only_luoyang" if student_only else "kd_luoyang",
        *base,
    ])
    if student_only:
        student.remove("--privileged_teacher")
        return [*student, "--student_only", "--is_training", "1"]
    if action == "student":
        return [*student, "--is_training", "1"]
    if action == "test":
        student.remove("--privileged_teacher")
        return [*student, "--is_training", "0", "--test_checkpoint", str(student_checkpoint or paths["student_checkpoint"])]
    raise ValueError(action)


def run_logged(config, config_path, action, cmd):
    log_root = Path(config["paths"]["logs_root"])
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log_path = log_root / f"{stamp}_{action}_{os.getpid()}.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = config["runtime"]["gpu_devices"]
    environment["PYTHONUNBUFFERED"] = "1"
    header = {
        "command": shlex.join(cmd), "resolved_config": config,
        "python": sys.version, "platform": platform.platform(),
        "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "rank": environment.get("RANK", "single-process-DataParallel"),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps(header, ensure_ascii=False, indent=2) + "\n")
        log.flush()
        process = subprocess.Popen(
            cmd, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        log.write(f"end_time={time.strftime('%Y-%m-%d %H:%M:%S')} return_code={return_code}\n")
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
    return log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["teacher", "student", "student-only", "test", "dry-run", "validate-config"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "luoyang_parquet.json")
    parser.add_argument(
        "--prefer-latest-checkpoint", action="store_true",
        help="select the newest matching checkpoint, including over configured paths",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    actions = ["teacher", "student", "test"] if args.action in {"dry-run", "validate-config"} else [args.action]
    for action in actions:
        teacher_checkpoint = None
        student_checkpoint = None
        if args.action not in {"dry-run", "validate-config"} and action in {"student", "test"}:
            teacher_checkpoint = resolve_teacher_checkpoint(
                config, config_path, args.prefer_latest_checkpoint)
            print(f"Resolved Teacher checkpoint: {teacher_checkpoint}")
        if args.action not in {"dry-run", "validate-config"} and action == "test":
            student_checkpoint = resolve_student_checkpoint(
                config, config_path, args.prefer_latest_checkpoint)
            print(f"Resolved Student checkpoint: {student_checkpoint}")
        cmd = command(
            config, config_path, action, teacher_checkpoint=teacher_checkpoint,
            student_checkpoint=student_checkpoint)
        if args.action == "dry-run":
            print(f"{action}: {shlex.join(cmd)}")
        elif args.action == "validate-config":
            validation = [item for item in cmd if item not in ["--test_checkpoint", config["paths"]["student_checkpoint"]]]
            validation.extend(["--config_only"])
            completed = subprocess.run(validation, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if completed.returncode or "Configuration validation completed" not in completed.stdout:
                print(completed.stdout)
                raise subprocess.CalledProcessError(completed.returncode, validation)
            print(f"[valid] {action}")
        else:
            print(f"log: {run_logged(config, config_path, action, cmd)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
