"""Sidecar contract preventing Luoyang runs from loading legacy one-step weights."""

import hashlib
import json
from pathlib import Path

from data_provider.data_loader_luoyang import load_luoyang_config


def expected_contract(config_path):
    config = load_luoyang_config(config_path)
    model_keys = (
        "d_model", "n_heads", "e_layers", "d_layers", "d_ff", "factor",
        "image_encoder_type", "image_cnn_dim", "image_temporal_hidden_dim",
        "teacher_fuse_strategy", "student_fuse_strategy",
    )
    semantic_config = {
        key: config[key]
        for key in (
            "dataset", "fields", "dimensions", "time", "power", "images",
            "missing", "privileged_teacher")
    }
    semantic_config["model"] = {
        key: config["model"][key] for key in model_keys
    }
    semantic_digest = hashlib.sha256(
        json.dumps(
            semantic_config, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 3,
        "dataset": "LuoyangParquet",
        "timeseries_input_dim": int(config["dimensions"]["timeseries_input_dim"]),
        "history_points": int(config["dimensions"]["history_points"]),
        "image_history_frames": int(config["dimensions"]["image_history_frames"]),
        "pred_len": int(config["dimensions"]["forecast_steps"]),
        "rated_power": float(config["power"]["rated_power"]),
        "image_sampling_strategy": config["images"]["sampling_strategy"],
        "privileged_teacher": config["privileged_teacher"],
        "data_config_sha256": semantic_digest,
    }


def write_checkpoint_contract(checkpoint_path, config_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Luoyang checkpoint is missing: {checkpoint_path}")
    sidecar = checkpoint_path.with_name(checkpoint_path.name + ".metadata.json")
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(
        json.dumps(expected_contract(config_path), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(sidecar)
    return sidecar


def validate_checkpoint_contract(checkpoint_path, config_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Luoyang checkpoint is missing: {checkpoint_path}")
    sidecar = checkpoint_path.with_name(checkpoint_path.name + ".metadata.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Luoyang checkpoint metadata is missing: {sidecar}")
    actual = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = expected_contract(config_path)
    if actual != expected:
        raise ValueError(f"Luoyang checkpoint metadata mismatch: {sidecar}")
    return actual
