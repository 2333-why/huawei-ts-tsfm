#!/usr/bin/env python3
"""Reproducible, resumable FACTS t11 -> S14 -> physics-path pipeline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "pipeline.json"
STAGE_ORDER = [
    "teacher",
    "s14",
    "appearance",
    "solar",
    "phase_pretrain",
    "physics",
]


def load_config() -> Dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    override = os.environ.get("FACTS_DATA_ROOT")
    if override:
        config["data"]["root"] = override
    if os.environ.get("FACTS_NUM_WORKERS"):
        config["runtime"]["num_workers"] = int(os.environ["FACTS_NUM_WORKERS"])
    return config


def config_digest() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def stage_tag(config: Dict, stage: str) -> str:
    return str(config["stages"][stage]["tag"])


def artifact_dirs(config: Dict, stage: str) -> List[Path]:
    tag = stage_tag(config, stage)
    if stage == "teacher":
        pattern = f"long_term_forecast_Stanford_{tag}_*"
    else:
        pattern = f"long_term_forecast_Stanford_student_{tag}_*"
    return sorted(path for path in (ROOT / "checkpoints").glob(pattern) if path.is_dir())


def artifact_dir(config: Dict, stage: str, required: bool = True) -> Optional[Path]:
    matches = artifact_dirs(config, stage)
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous {stage} checkpoint directories: {matches}")
    if not matches:
        if required:
            raise FileNotFoundError(f"{stage} checkpoint directory is missing")
        return None
    return matches[0]


def accepted_checkpoint(config: Dict, stage: str) -> Path:
    directory = artifact_dir(config, stage)
    checkpoint = directory / "checkpoint.pth"
    if stage in {"appearance", "solar", "physics"}:
        manifest_path = directory / "checkpoint_selection.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{stage} manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("accepted", False):
            raise RuntimeError(f"{stage} was not validation-accepted: {manifest_path}")
        checkpoint = Path(manifest["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.is_file():
            # Manifests keep an absolute audit path. Fall back to the local
            # artifact when the complete FACTS_NEW directory has been moved.
            checkpoint = directory / "checkpoint.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{stage} checkpoint is missing: {checkpoint}")
    return checkpoint.resolve()


def phase_magnitude_checkpoint(config: Dict) -> Path:
    checkpoint = artifact_dir(config, "phase_pretrain") / "phase_magnitude_checkpoint.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"phase magnitude checkpoint is missing: {checkpoint}")
    return checkpoint.resolve()


def data_paths(config: Dict) -> List[Path]:
    data = config["data"]
    root = Path(data["root"])
    return [root / data[name] for name in ("hdf5", "times_trainval", "times_test")]


def require_data(config: Dict) -> None:
    missing = [path for path in data_paths(config) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Stanford data artifacts: " + ", ".join(map(str, missing)))


def common_model_args(config: Dict) -> List[str]:
    model = config["model"]
    return [
        "--e_layers", str(model["e_layers"]),
        "--d_layers", str(model["d_layers"]),
        "--d_model", str(model["d_model"]),
        "--n_heads", str(model["n_heads"]),
        "--d_ff", str(model["d_ff"]),
        "--factor", str(model["factor"]),
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
        "--weather_feature_dim", "1",
    ]


def common_data_args(config: Dict) -> List[str]:
    data = config["data"]
    return [
        "--root_path", str(data["root"]),
        "--data_path", str(data["hdf5"]),
        "--times_trainval_path", str(data["times_trainval"]),
        "--times_test_path", str(data["times_test"]),
        "--data", "Stanford", "--features", "S", "--target", "pv_log",
        "--freq", "t", "--seq_len", str(data["history_length"]),
        "--label_len", "0", "--pred_len", "1",
        "--forecast_horizon_minutes", str(data["forecast_horizon_minutes"]),
        "--sample_interval_minutes", "1", "--history_order", "current_first",
        "--stanford_capacity_kw", str(data["capacity_kw"]),
        "--split_strategy", "day_block", "--num_fold", "10", "--fold_index", "0",
    ]


def runtime_args(config: Dict, stage: Dict) -> List[str]:
    runtime = config["runtime"]
    args = [
        "--batch_size", str(stage["batch_size"]),
        "--num_workers", str(runtime["num_workers"]),
        "--prefetch_factor", str(runtime["prefetch_factor"]),
        "--train_epochs", str(stage["epochs"]),
        "--patience", str(stage["patience"]),
        "--learning_rate", str(stage.get("learning_rate", 0.0001)),
        "--lradj", str(stage.get("lradj", "type3")),
        "--gpu", str(runtime["gpu"]), "--itr", "1",
    ]
    if runtime.get("use_amp", True):
        args.append("--use_amp")
    return args


def teacher_command(config: Dict, is_training: int = 1) -> List[str]:
    stage = config["stages"]["teacher"]
    tag = stage["tag"]
    return [
        sys.executable, "-u", "run_teacher.py",
        "--task_name", "long_term_forecast", "--is_training", str(is_training),
        "--model_id", f"Stanford_{tag}_16_1", "--model", "MTS_31F",
        *common_data_args(config), *common_model_args(config),
        "--fuse_strategy", "no_weather",
        "--image_height", "64", "--image_width", "64",
        "--image_proj_h", "8", "--image_proj_w", "8",
        "--stanford_image_mode", stage["image_mode"],
        "--image_encoder_type", stage["image_encoder"],
        "--image_cnn_dim", str(config["model"]["image_cnn_dim"]),
        "--image_temporal_hidden_dim", str(config["model"]["image_temporal_hidden_dim"]),
        "--stanford_sample_weight_mode", "none",
        "--stanford_sample_weight_scale", "1.0",
        "--stanford_sample_weight_clip", "3.0",
        "--ramp_aux_weight", "0.0", "--ramp_direction_aux_weight", "0.0",
        *runtime_args(config, stage),
        "--des", f"Stanford_{tag}", "--tors", "teacher",
    ]


def student_base_command(
    config: Dict,
    stage_name: str,
    teacher_checkpoint: Path,
    init_checkpoint: Optional[Path],
    is_training: int = 1,
) -> List[str]:
    stage = config["stages"][stage_name]
    tag = stage["tag"]
    command = [
        sys.executable, "-u", "run_kd.py",
        "--task_name", "long_term_forecast", "--is_training", str(is_training),
        "--model_id", f"Stanford_student_{tag}_16_1",
        "--model", "MTS_31", "--teacher_model", "MTS_31F",
        "--teacher_path", str(teacher_checkpoint),
        *common_data_args(config), *common_model_args(config),
        "--image_height", "64", "--image_width", "64",
        "--image_proj_h", "8", "--image_proj_w", "8",
        "--image_cnn_dim", str(config["model"]["image_cnn_dim"]),
        "--image_temporal_hidden_dim", str(config["model"]["image_temporal_hidden_dim"]),
        "--teacher_image_encoder_type", "cnn",
        "--stanford_sample_weight_mode", "none",
        "--stanford_sample_weight_scale", "1.0",
        "--stanford_sample_weight_clip", "3.0",
        "--stanford_ramp_weight_5", "1.5", "--stanford_ramp_weight_8", "2.0",
        "--ramp_aux_weight", "0.0", "--ramp_direction_aux_weight", "0.0",
        "--residual_aux_weight", "0.0", "--residual_correction_clip_kw", "15.0",
        "--stable_ramp_threshold_kw", "5.0",
        "--expert_gate_bias", "-1.0", "--expert_gate_aux_weight", "0.0",
        "--expert_gate_threshold_kw", "5.0", "--expert_gate_target", "extreme",
        "--expert_gate_hidden_dim", "32",
        "--event_state_aux_weight", "0.0", "--event_residual_aux_weight", "0.0",
        "--event_routing_mode", "soft",
        "--tail_routing_mode", "confidence", "--tail_confidence_threshold", "0.5",
        "--tail_error_threshold_kw", "3.0", "--tail_gate_pretrain_epochs", "0",
        "--clear_sky_aux_weight", "0.0", "--clear_sky_quantile", "0.95",
        "--phase_state_aux_weight", "0.0", "--phase_path_aux_weight", "0.0",
        "--phase_motion_consistency_weight", "0.0",
        "--phase_correction_mode", "learned_residual",
        "--phase_benefit_aux_weight", "0.0", "--phase_benefit_mode", "winner",
        "--phase_constraint_dual_lr", "0.05", "--phase_constraint_penalty", "1.0",
        "--phase_validation_calibration", "0", "--phase_risk_scale_kw", "3.0",
        "--phase_transport_mode", "mean_flow", "--phase_transport_hypotheses", "3",
        "--phase_stable_expert_all_samples", "0", "--phase_specialist_abstention", "0",
        "--phase_state_pretrain_epochs", "0", "--phase_magnitude_pretrain_epochs", "0",
        "--phase_benefit_pretrain_epochs", "0", "--phase_resume_from", "none",
        "--phase_route_classes", "active_down,active_up",
        "--kd_extreme_response_weight", "0.0",
        "--kd_extreme_response_threshold_kw", "5.0",
        "--val_selection_metric", "loss", "--val_extreme_metric_weight", "0.25",
        "--val_min_improvement", "0.0",
        "--val_ordinary_relative_tolerance", "0.0",
        "--val_no_path_relative_tolerance", "0.0",
        "--val_cloud_ordinary_relative_tolerance", "0.0",
        "--val_cloud_no_path_relative_tolerance", "0.0",
        "--val_tail_min_improvement", "0.0",
        "--val_extreme_relative_tolerance", "0.0",
        "--max_train_batches", "0", "--max_val_batches", "0",
        *runtime_args(config, stage),
        "--seed", str(stage.get("seed", 2021)),
        "--extra_tag", tag, "--des", f"Stanford_{tag}",
        "--fusion_type", "attention", "--trail", f"kd_{tag}", "--tors", "student",
        "--strict_causal_student",
    ]
    if init_checkpoint is not None:
        command.extend(["--student_init_path", str(init_checkpoint)])
    return command


def replace_arg(command: List[str], name: str, value: str) -> None:
    index = command.index(name)
    command[index + 1] = str(value)


def add_flags(command: List[str], flags: Iterable[str]) -> None:
    command.extend(flags)


def student_command(
    config: Dict,
    stage_name: str,
    teacher_checkpoint: Path,
    init_checkpoint: Optional[Path],
    is_training: int = 1,
) -> List[str]:
    stage = config["stages"][stage_name]
    command = student_base_command(
        config, stage_name, teacher_checkpoint, init_checkpoint, is_training
    )

    if stage_name == "s14":
        command.extend([
            "--fuse_strategy", "no_weather", "--student_fuse_strategy", "no_weather",
            "--teacher_fuse_strategy", "no_weather", "--stanford_image_mode", "gray",
            "--image_encoder_type", "cnn", "--kd_type", stage["kd_type"],
            "--kd_loss_weight_sim", str(stage["kd_similarity_weight"]),
            "--kd_loss_weight_inter", str(stage["kd_intervention_weight"]),
            "--modality_dropout_rate", str(stage["modality_dropout"]),
            "--stanford_privileged_teacher",
        ])
        return command

    strategy = stage["strategy"] if "strategy" in stage else "causal_phase_field_tail"
    parent_strategy = stage.get("parent_strategy", "causal_solar_event")
    command.extend([
        "--freeze_ts_epochs", "0", "--freeze_ts_permanently",
        "--train_image_residual_only", "--conservative_checkpointing",
        "--parent_fuse_strategy", parent_strategy,
        "--fuse_strategy", "no_weather", "--student_fuse_strategy", strategy,
        "--teacher_fuse_strategy", "no_weather",
        "--stanford_image_mode", stage.get("image_mode", "rgb"),
        "--image_encoder_type", stage.get("image_encoder", "cnn_solar_advection"),
        "--modality_dropout_rate", "0.0", "--val_ordinary_guard",
        "--val_min_improvement", "0.000001",
    ])

    if stage_name == "appearance":
        replace_arg(command, "--val_selection_metric", stage["selection_metric"])
        replace_arg(command, "--val_ordinary_relative_tolerance", stage["ordinary_tolerance"])
        command.extend([
            "--stable_correction_aux_weight", "0.0",
            "--kd_type", "inter_causal", "--kd_loss_weight_sim", "0.001",
            "--kd_loss_weight_inter", "0.1", "--stanford_privileged_teacher",
        ])
    elif stage_name == "solar":
        replace_arg(command, "--event_state_aux_weight", stage["event_state_weight"])
        replace_arg(command, "--val_selection_metric", stage["selection_metric"])
        command.extend([
            "--stable_correction_aux_weight", str(stage["stable_correction_weight"]),
            "--kd_type", "spatial_causal", "--kd_loss_weight_sim", "0.0",
            "--kd_loss_weight_inter", str(stage["spatial_kd_weight"]),
            "--stanford_privileged_teacher",
        ])
    else:
        add_flags(command, [
            "--val_no_path_guard", "--stanford_return_phase_path",
            "--skip_test_after_train", "--test_only_if_accepted",
        ])
        replace_arg(command, "--event_residual_aux_weight", "1.0")
        replace_arg(command, "--phase_state_aux_weight", "1.0")
        replace_arg(command, "--phase_path_aux_weight", "1.0")
        replace_arg(command, "--phase_benefit_aux_weight", "1.0")
        replace_arg(command, "--phase_transport_mode", "multi_hypothesis")
        replace_arg(command, "--phase_transport_hypotheses", "3")
        replace_arg(command, "--phase_route_classes", "all")
        replace_arg(command, "--val_selection_metric", "cloud_event_rmse")
        command.extend([
            "--stable_correction_aux_weight", "0.05",
            "--kd_type", "response", "--kd_loss_weight_sim", "0.0",
            "--kd_loss_weight_inter", "0.0",
        ])

        if stage_name == "phase_pretrain":
            replace_arg(command, "--phase_state_pretrain_epochs", stage["state_epochs"])
            replace_arg(command, "--phase_magnitude_pretrain_epochs", stage["magnitude_epochs"])
            replace_arg(command, "--phase_benefit_pretrain_epochs", stage["benefit_epochs"])
            replace_arg(command, "--phase_stable_expert_all_samples", stage["stable_expert_all_samples"])
            replace_arg(command, "--phase_specialist_abstention", stage["specialist_abstention"])
        elif stage_name == "physics":
            replace_arg(command, "--phase_correction_mode", stage["correction_mode"])
            replace_arg(command, "--phase_benefit_mode", stage["benefit_mode"])
            replace_arg(command, "--phase_validation_calibration", stage["validation_calibration"])
            replace_arg(command, "--phase_benefit_pretrain_epochs", stage["benefit_epochs"])
            replace_arg(command, "--phase_resume_from", "magnitude")
            # Physics-path is constrained directly; learned residual stability is disabled.
            command[command.index("--stable_correction_aux_weight") + 1] = "0.0"
    return command


def symbolic_checkpoint(stage: str, suffix: str = "checkpoint.pth") -> Path:
    return Path(f"<checkpoint-from-{stage}>/{suffix}")


def command_for_stage(config: Dict, stage: str, dry_run: bool = False) -> List[str]:
    if stage == "teacher":
        return teacher_command(config)
    teacher = symbolic_checkpoint("teacher") if dry_run else accepted_checkpoint(config, "teacher")
    if stage == "s14":
        init = None
    elif stage == "appearance":
        init = symbolic_checkpoint("s14") if dry_run else accepted_checkpoint(config, "s14")
    elif stage == "solar":
        init = symbolic_checkpoint("appearance") if dry_run else accepted_checkpoint(config, "appearance")
    elif stage == "phase_pretrain":
        init = symbolic_checkpoint("solar") if dry_run else accepted_checkpoint(config, "solar")
    elif stage == "physics":
        init = symbolic_checkpoint("phase_pretrain", "phase_magnitude_checkpoint.pth") if dry_run else phase_magnitude_checkpoint(config)
    else:
        raise ValueError(f"unknown stage: {stage}")
    return student_command(config, stage, teacher, init)


def stage_complete(config: Dict, stage: str) -> bool:
    try:
        if stage == "phase_pretrain":
            phase_magnitude_checkpoint(config)
        else:
            accepted_checkpoint(config, stage)
        return True
    except (FileNotFoundError, RuntimeError):
        return False


def run_command(command: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + shlex.join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def write_lineage(config: Dict) -> None:
    stages = {}
    for name in STAGE_ORDER:
        directory = artifact_dir(config, name, required=False)
        record = {"tag": stage_tag(config, name), "complete": stage_complete(config, name)}
        if directory:
            record["directory"] = str(directory.resolve())
            manifest = directory / "checkpoint_selection.json"
            if manifest.is_file():
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                record["accepted"] = payload.get("accepted")
                record["config_fingerprint"] = payload.get("config_fingerprint")
        stages[name] = record
    output = {
        "schema_version": 1,
        "pipeline_config_sha256": config_digest(),
        "stages": stages,
    }
    path = ROOT / "reports" / "pipeline_lineage.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def print_status(config: Dict) -> None:
    print("stage\ttag\tstatus")
    for stage in STAGE_ORDER:
        print(f"{stage}\t{stage_tag(config, stage)}\t{'complete' if stage_complete(config, stage) else 'missing'}")


def run_stage(config: Dict, stage: str) -> None:
    require_data(config)
    if stage_complete(config, stage):
        print(f"[skip] {stage}: verified artifact already exists")
        return
    command = command_for_stage(config, stage)
    run_command(command, ROOT / "logs" / f"{stage_tag(config, stage)}.log")
    if not stage_complete(config, stage):
        raise RuntimeError(f"{stage} finished without a verified artifact")
    write_lineage(config)


def run_test(config: Dict) -> None:
    require_data(config)
    accepted_checkpoint(config, "physics")
    teacher = accepted_checkpoint(config, "teacher")
    init = phase_magnitude_checkpoint(config)
    command = student_command(config, "physics", teacher, init, is_training=0)
    run_command(command, ROOT / "logs" / "s14physics_full1_test.log")


def validate_commands(config: Dict) -> None:
    for stage in STAGE_ORDER:
        command = command_for_stage(config, stage, dry_run=True)
        command.append("--config_only")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode:
            print(completed.stdout)
            raise subprocess.CalledProcessError(completed.returncode, command)
        marker = "Configuration validation completed; training was not started."
        if marker not in completed.stdout:
            raise RuntimeError(f"{stage} did not reach configuration validation")
        print(f"[valid] {stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["all", "status", "dry-run", "validate-config", "test", *STAGE_ORDER],
        help="pipeline action or one explicit stage",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    os.chdir(ROOT)
    if args.action == "status":
        print_status(config)
    elif args.action == "dry-run":
        for stage in STAGE_ORDER:
            print(f"\n# {stage}\n{shlex.join(command_for_stage(config, stage, dry_run=True))}")
    elif args.action == "validate-config":
        validate_commands(config)
    elif args.action == "all":
        for stage in STAGE_ORDER:
            run_stage(config, stage)
        print_status(config)
    elif args.action == "test":
        run_test(config)
    else:
        run_stage(config, args.action)
    write_lineage(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
