"""Resolve pinned foundation-model weights from browser-downloaded folders."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


WEIGHTS_ROOT_ENV = "TSFM_WEIGHTS_ROOT"
MODEL_WEIGHT_ENVS = {
    "Sundial": "SUNDIAL_WEIGHT_DIR",
    "TimeMoE": "TIMEMOE_WEIGHT_DIR",
    "TimeMoE200M": "TIMEMOE_200M_WEIGHT_DIR",
    "Chronos2": "CHRONOS2_WEIGHT_DIR",
    "TiRex": "TIREX_WEIGHT_DIR",
    "TimesFM": "TIMESFM_WEIGHT_DIR",
}

REQUIRED_WEIGHT_FILES = {
    "Sundial": (
        "config.json",
        "generation_config.json",
        "configuration_sundial.py",
        "modeling_sundial.py",
        "flow_loss.py",
        "ts_generation_mixin.py",
        "model.safetensors",
    ),
    "TimeMoE": (
        "config.json",
        "generation_config.json",
        "configuration_time_moe.py",
        "modeling_time_moe.py",
        "ts_generation_mixin.py",
        "model.safetensors",
    ),
    "TimeMoE200M": (
        "config.json",
        "generation_config.json",
        "configuration_time_moe.py",
        "modeling_time_moe.py",
        "ts_generation_mixin.py",
        "model.safetensors",
    ),
    "Chronos2": ("config.json", "model.safetensors"),
    "TiRex": ("model.ckpt",),
    "TimesFM": ("config.json", "model.safetensors"),
}

# Exact LFS objects at the revisions pinned in models.registry.
WEIGHT_FILE_METADATA = {
    "Sundial": (
        "model.safetensors",
        513_341_448,
        "414435b508391f92afadd2aaeec418c806776aeccbce12e638d73a139ca5ca78",
    ),
    "TimeMoE": (
        "model.safetensors",
        226_760_264,
        "0127209833663df6f5ae3cf1c3316f739a8dd1dae27e59036268bbfdb48f91a4",
    ),
    "TimeMoE200M": (
        "model.safetensors",
        906_450_104,
        "28d1cbf480c1f63d4ad56d8b7649d9860b9afa9f31abeedff9decea48d9aee57",
    ),
    "Chronos2": (
        "model.safetensors",
        477_930_472,
        "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42",
    ),
    "TiRex": (
        "model.ckpt",
        141_230_262,
        "b8c3f5a036c63272ce4b91c00187e26922a394cb6cb49d4e16db070ad0422314",
    ),
    "TimesFM": (
        "model.safetensors",
        925_187_448,
        "b53f6d52114e2ad786890f3c4637ce05f580b7800d6e24401f88b398b76035ef",
    ),
}


@dataclass(frozen=True)
class ModelSource:
    """Loader location while preserving the registry ID/revision as identity."""

    location: str
    is_local: bool
    directory: Optional[Path] = None


def _environment_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError(f"{name} is set but empty")
    return Path(value).expanduser().resolve()


def configured_weight_directory(model_name: str) -> Optional[Path]:
    """Return a configured flat model directory, or None for Hub/cache mode."""

    if model_name not in MODEL_WEIGHT_ENVS:
        raise ValueError(f"unknown foundation model {model_name!r}")
    override = _environment_path(MODEL_WEIGHT_ENVS[model_name])
    if override is not None:
        return override.parent if override.is_file() else override
    root = _environment_path(WEIGHTS_ROOT_ENV)
    return None if root is None else root / model_name


def inspect_local_weights(
    model_name: str, *, verify_sha256: bool = False
) -> Dict[str, object]:
    """Validate one configured browser-download directory without loading it."""

    directory = configured_weight_directory(model_name)
    required = REQUIRED_WEIGHT_FILES[model_name]
    missing = []
    if directory is None or not directory.is_dir():
        missing = list(required)
    else:
        for filename in required:
            path = directory / filename
            try:
                info = path.stat()
                if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                    missing.append(filename)
            except OSError:
                missing.append(filename)

    weight_filename, expected_size, expected_sha256 = WEIGHT_FILE_METADATA[model_name]
    actual_size = 0
    sha256 = None
    size_ok = False
    checksum_ok = None
    if directory is not None and weight_filename not in missing:
        weight_path = directory / weight_filename
        actual_size = int(weight_path.stat().st_size)
        size_ok = actual_size == expected_size
        if verify_sha256 and size_ok:
            digest = hashlib.sha256()
            with weight_path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            sha256 = digest.hexdigest()
            checksum_ok = sha256 == expected_sha256

    complete = not missing and size_ok and (checksum_ok is not False)
    return {
        "name": model_name,
        "environment": MODEL_WEIGHT_ENVS[model_name],
        "directory": "" if directory is None else str(directory),
        "required_files": list(required),
        "missing_files": missing,
        "weight_file": weight_filename,
        "expected_size": expected_size,
        "actual_size": actual_size,
        "size_ok": size_ok,
        "expected_sha256": expected_sha256,
        "sha256": sha256,
        "checksum_ok": checksum_ok,
        "complete": complete,
    }


def resolve_model_source(model_name: str, model_id: str) -> ModelSource:
    """Choose a validated local source when configured, otherwise the pinned ID."""

    directory = configured_weight_directory(model_name)
    if directory is None:
        return ModelSource(location=model_id, is_local=False)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"local weights for {model_name} do not exist: {directory}"
        )
    missing = [
        filename
        for filename in REQUIRED_WEIGHT_FILES[model_name]
        if not (directory / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"local weights for {model_name} are incomplete in {directory}; "
            f"missing: {', '.join(missing)}"
        )
    location = directory / "model.ckpt" if model_name == "TiRex" else directory
    return ModelSource(location=str(location), is_local=True, directory=directory)


def pretrained_kwargs(source: ModelSource, revision: str) -> Dict[str, str]:
    """Do not pass a Hub revision to a direct filesystem loader."""

    return {} if source.is_local else {"revision": revision}


__all__ = [
    "MODEL_WEIGHT_ENVS",
    "ModelSource",
    "REQUIRED_WEIGHT_FILES",
    "WEIGHTS_ROOT_ENV",
    "WEIGHT_FILE_METADATA",
    "configured_weight_directory",
    "inspect_local_weights",
    "pretrained_kwargs",
    "resolve_model_source",
]
