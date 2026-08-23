#!/usr/bin/env python3
"""Configuration-driven FACTS YLJ teacher, Student/KD and inference runner."""

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader_ylj import load_ylj_config
from utils.ylj_checkpoint_contract import validate_ylj_checkpoint_contract


def _path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def resolve_teacher(config, config_path, prefer_latest=False):
    configured = _path(config["paths"]["teacher_checkpoint"])
    if configured.is_file() and not prefer_latest:
        validate_ylj_checkpoint_contract(configured, config_path)
        return configured
    root = _path(config["paths"]["checkpoints_root"])
    candidates = []
    if configured.is_file():
        try:
            validate_ylj_checkpoint_contract(configured, config_path)
            candidates.append(configured)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
    if root.is_dir():
        for candidate in root.rglob("checkpoint.pth"):
            if "teacher" not in str(candidate.relative_to(root).parent).lower():
                continue
            try:
                validate_ylj_checkpoint_contract(candidate, config_path)
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                continue
            if candidate not in candidates:
                candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"No contract-matching YLJ Teacher checkpoint under {root}; run teacher first")
    return max(candidates, key=lambda item: item.stat().st_mtime_ns)


def resolve_student(config, config_path, prefer_latest=False):
    """Resolve the newest contract-matching Student/KD weight for inference."""
    configured = _path(config["paths"]["student_checkpoint"])
    if configured.is_file() and not prefer_latest:
        validate_ylj_checkpoint_contract(configured, config_path)
        return configured
    root = _path(config["paths"]["checkpoints_root"])
    candidates = []
    if configured.is_file():
        try:
            validate_ylj_checkpoint_contract(configured, config_path)
            candidates.append(configured)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
    if root.is_dir():
        for candidate in root.rglob("checkpoint.pth"):
            parent = str(candidate.relative_to(root).parent).lower()
            if "student" not in parent and "_kd_" not in parent:
                continue
            try:
                validate_ylj_checkpoint_contract(candidate, config_path)
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                continue
            if candidate not in candidates:
                candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"No contract-matching YLJ Student checkpoint under {root}; run student first")
    return max(candidates, key=lambda item: item.stat().st_mtime_ns)


def common_args(config, config_path, multi_gpu=True):
    dims, timing, runtime, model, paths = (
        config["dimensions"], config["time"], config["runtime"],
        config["model"], config["paths"])
    values = [
        "--task_name", "long_term_forecast", "--data", "YLJParquet",
        "--ylj_config", str(config_path), "--root_path", ".", "--data_path", ".",
        "--features", "MS", "--target", config["fields"]["target"],
        "--freq", f'{timing["sampling_interval_minutes"]}t',
        "--seq_len", str(dims["history_points"]), "--label_len", "0",
        "--pred_len", str(dims["forecast_steps"]),
        "--enc_in", str(dims["time_series_input_dim"]),
        "--dec_in", str(dims["time_series_input_dim"]), "--c_out", str(dims["target_dim"]),
        "--seq_len_hours", str(dims["history_points"] * timing["sampling_interval_minutes"] / 60),
        "--pred_len_hours", str(timing["forecast_horizon_minutes"] / 60),
        "--forecast_horizon_minutes", str(timing["forecast_horizon_minutes"]),
        "--sample_interval_minutes", str(timing["sampling_interval_minutes"]),
        "--history_order", timing["history_order"],
        "--weather_feature_dim", str(dims["weather_placeholder_dim"]),
        "--privileged_teacher",
        "--image_height", str(dims["image_height"]), "--image_width", str(dims["image_width"]),
        "--stanford_image_mode", "rgb", "--image_encoder_type", model["image_encoder_type"],
        "--image_cnn_dim", str(model["image_cnn_dim"]),
        "--image_temporal_hidden_dim", str(model["image_temporal_hidden_dim"]),
        "--d_model", str(model["d_model"]), "--n_heads", str(model["n_heads"]),
        "--e_layers", str(model["e_layers"]), "--d_layers", str(model["d_layers"]),
        "--d_ff", str(model["d_ff"]), "--factor", str(model["factor"]),
        "--num_workers", str(runtime["num_workers"]),
        "--prefetch_factor", str(runtime["prefetch_factor"]),
        "--patience", str(runtime["patience"]), "--learning_rate", str(runtime["learning_rate"]),
        "--checkpoints", paths["checkpoints_root"], "--results_root", paths["results_root"],
    ]
    if multi_gpu:
        values += ["--use_multi_gpu", "--devices", str(runtime["gpu_devices"])]
    if runtime["precision"] == "fp16":
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


def command(config, config_path, action, teacher_checkpoint=None, resume_checkpoint=None,
            student_checkpoint=None):
    py = sys.executable
    smoke = action == "teacher-smoke"
    student_only = action == "student-only"
    base = common_args(config, config_path, multi_gpu=not smoke)
    runtime, model = config["runtime"], config["model"]
    if action in {"teacher", "teacher-smoke"}:
        return [
            py, "-u", "run_teacher.py", "--is_training", "1", "--model", "MTS_31F",
            "--model_id", f'YLJ_teacher_{config["dimensions"]["history_points"]}_{config["dimensions"]["forecast_steps"]}',
            "--batch_size", "1" if smoke else str(runtime["teacher_batch_size"]),
            "--train_epochs", "1" if smoke else str(runtime["teacher_epochs"]),
            "--fuse_strategy", model["teacher_fuse_strategy"], "--des", "YLJ_teacher", *base]
    student = [py, "-u", "run_kd.py", "--model", "MTS_31"]
    if not student_only:
        student.extend([
            "--teacher_model", "MTS_31F",
            "--teacher_path", str(teacher_checkpoint or config["paths"]["teacher_checkpoint"]),
            "--teacher_image_encoder_type", model["image_encoder_type"],
            "--teacher_fuse_strategy", model["teacher_fuse_strategy"],
        ])
    student.extend([
        "--model_id", f'YLJ_student_{config["dimensions"]["history_points"]}_{config["dimensions"]["forecast_steps"]}',
        "--batch_size", str(runtime["student_batch_size"]), "--train_epochs", str(runtime["student_epochs"]),
        "--fuse_strategy", model["student_fuse_strategy"],
        "--student_fuse_strategy", model["student_fuse_strategy"],
        "--strict_causal_student", "--kd_type", model["kd_type"],
        "--kd_loss_weight_sim", str(0.0 if student_only else model["kd_similarity_weight"]),
        "--kd_loss_weight_inter", str(0.0 if student_only else model["kd_intervention_weight"]),
        "--kd_intervention_min_feature_delta", str(model["kd_intervention_min_feature_delta"]),
        "--modality_dropout_rate", str(model["modality_dropout_rate"]),
        "--extra_tag", "ylj_16step", "--des", "YLJ_student", "--trail",
        "student_only_ylj" if student_only else "kd_ylj", *base])
    if student_only:
        student.remove("--privileged_teacher")
        return [*student, "--student_only", "--is_training", "1"]
    if action == "student":
        return [*student, "--is_training", "1"]
    if action == "student-resume":
        if not resume_checkpoint:
            raise ValueError("student-resume requires --resume-checkpoint")
        return [*student, "--student_init_path", str(resume_checkpoint), "--is_training", "1"]
    if action == "test":
        student.remove("--privileged_teacher")
        return [*student, "--is_training", "0", "--test_checkpoint", str(student_checkpoint or config["paths"]["student_checkpoint"])]
    raise ValueError(action)


def run_logged(config, action, cmd):
    root = _path(config["paths"]["logs_root"])
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / f'{time.strftime("%Y%m%dT%H%M%S")}_{action}_{os.getpid()}.log'
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(config["runtime"]["gpu_devices"])
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({
            "command": shlex.join(cmd), "resolved_config": config,
            "python": sys.version, "platform": platform.platform(),
            "pytorch": torch.__version__, "cuda": torch.version.cuda,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2) + "\n")
        log.flush()
        process = subprocess.Popen(cmd, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
        log.write(f"end_time={time.strftime('%Y-%m-%d %H:%M:%S')} return_code={code}\n")
    if code:
        raise subprocess.CalledProcessError(code, cmd)
    return log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["teacher-smoke", "teacher", "student", "student-only", "student-resume", "test", "dry-run", "validate-config"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "datasets" / "ylj.yaml")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--prefer-latest-checkpoint", action="store_true",
        help="select the newest matching checkpoint, including over configured paths",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_ylj_config(config_path)
    actions = ["teacher-smoke", "teacher", "student", "test"] if args.action in {"dry-run", "validate-config"} else [args.action]
    for action in actions:
        teacher = None
        student = None
        if args.action not in {"dry-run", "validate-config"} and action in {"student", "student-resume", "test"}:
            teacher = resolve_teacher(
                config, config_path, args.prefer_latest_checkpoint)
            print(f"Resolved YLJ Teacher checkpoint: {teacher}")
        if args.action not in {"dry-run", "validate-config"} and action == "test":
            student = resolve_student(
                config, config_path, args.prefer_latest_checkpoint)
            print(f"Resolved YLJ Student checkpoint: {student}")
        cmd = command(config, config_path, action, teacher, args.resume_checkpoint, student)
        if args.action == "dry-run":
            print(f"{action}: {shlex.join(cmd)}")
        elif args.action == "validate-config":
            validation = [item for item in cmd if item not in ["--test_checkpoint", config["paths"]["student_checkpoint"]]]
            validation.append("--config_only")
            completed = subprocess.run(validation, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if completed.returncode or "Configuration validation completed" not in completed.stdout:
                print(completed.stdout)
                raise subprocess.CalledProcessError(completed.returncode, validation)
            print(f"[valid] {action}")
        else:
            print(f"log: {run_logged(config, action, cmd)}")


if __name__ == "__main__":
    main()
