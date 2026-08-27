#!/usr/bin/env python3
"""Check the pinned runtime needed by the foundation-model runner.

The checker intentionally reports availability only.  It never loads a model,
downloads weights, reads Parquet contents, or sends credentials to the network.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.registry import MODEL_NAMES, get_model_spec  # noqa: E402


SCHEMA_VERSION = "foundation-environment-v1"
EXPECTED_INTERPRETER = "/opt/data/private/penv/time/bin/python"
EXPECTED_PYTHON = "3.8.18"
MAX_NETWORK_TIMEOUT = 10.0

LEGACY_PACKAGE_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("torch", "torch", ">=2.3,<2.4"),
    ("transformers", "transformers", ">=4.46.2,<4.47"),
    ("peft", "peft", ">=0.13.2,<0.14"),
)
# The Python 3.10+ profile is intentionally separate: TimesFM 2.5's
# Transformers class is not exposed by the legacy 4.46 runtime.  Keep
# ``PACKAGE_SPECS`` as the historical alias for callers that inspect it.
MODERN_PACKAGE_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("torch", "torch", ">=2.4,<3"),
    ("transformers", "transformers", ">=5.3,<6"),
    ("peft", "peft", ">=0.13.2,<1"),
)
PACKAGE_SPECS = LEGACY_PACKAGE_SPECS

_DATASET_CONFIGS: Tuple[Tuple[str, Path], ...] = (
    ("skippd_luoyang", REPOSITORY_ROOT / "configs/datasets/skippd_luoyang.json"),
    ("pvod_station00_ylj", REPOSITORY_ROOT / "configs/datasets/pvod_station00_ylj.yaml"),
)
_DATASET_OVERRIDES = {
    "skippd_luoyang": "SKIPPD_PARQUET",
    "pvod_station00_ylj": "PVOD_PARQUET",
}
_REQUIRED_CACHE_FILES = {
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
    # The modern checkpoints intentionally keep a smaller, model-specific
    # local-cache contract.  Presence is reported only; no loader is imported.
    "Chronos2": (
        "config.json",
        "model.safetensors",
    ),
    "TiRex": (
        "model.ckpt",
    ),
    "TimesFM": (
        "config.json",
        "model.safetensors",
    ),
}

_MODEL_PYTHON_FLOORS = {
    "Sundial": "3.8",
    "TimeMoE": "3.8",
    "Chronos2": "3.10",
    "TiRex": "3.10",
    "TimesFM": "3.10",
}
_MODERN_MODEL_PACKAGES = {
    "Chronos2": "chronos-forecasting (chronos/Chronos2Pipeline)",
    "TiRex": "tirex-ts (tirex/load_model)",
    "TimesFM": "transformers (TimesFm2_5ModelForPrediction)",
}
_MODERN_MODEL_PACKAGE_SPECS = {
    "Chronos2": ("chronos-forecasting", ">=2,<3"),
    "TiRex": ("tirex-ts", ">=1,<2"),
    "TimesFM": ("transformers", ">=5.3,<6"),
}


def _runtime_python_version() -> str:
    """Return the running interpreter version without exposing build metadata."""

    info = sys.version_info
    return "{}.{}.{}".format(int(info[0]), int(info[1]), int(info[2]))


def _package_version(distribution_name: str) -> str:
    """Read one distribution version through importlib metadata."""

    return str(importlib.metadata.version(distribution_name))


def _import_package(module_name: str) -> Any:
    """Import one package as a separately monkeypatchable preflight seam."""

    return importlib.import_module(module_name)


def _version_satisfies(value: Any, specifier: str) -> bool:
    """Evaluate a released package version against a PEP 440 specifier."""

    try:
        version_module = importlib.import_module("packaging.version")
        specifier_module = importlib.import_module("packaging.specifiers")
        version_type = getattr(version_module, "Version")
        specifier_type = getattr(specifier_module, "SpecifierSet")
        parsed = version_type(str(value))
        if parsed.is_prerelease or parsed.is_devrelease:
            return False
        return bool(specifier_type(specifier).contains(parsed, prereleases=False))
    except Exception:
        # A missing or unusable packaging installation is a failed check, not
        # an import-time failure or an exception disclosure in the report.
        return False


def _package_report(
    display_name: str,
    distribution_name: str,
    expected: str,
) -> Tuple[Dict[str, Any], Optional[Any]]:
    item: Dict[str, Any] = {
        "name": display_name,
        "status": "pass",
        "expected": expected,
        "version": None,
        "metadata_status": "pass",
        "import_status": "pass",
        "version_status": "pass",
        "cuda_build": None,
    }
    version: Optional[str]
    try:
        version = _package_version(distribution_name)
        item["version"] = version
    except Exception:
        version = None
        item["metadata_status"] = "missing"
        item["status"] = "fail"

    imported: Optional[Any]
    try:
        imported = _import_package(distribution_name)
        if imported is None:
            raise ImportError
    except Exception:
        imported = None
        item["import_status"] = "fail"
        item["status"] = "fail"

    if version is None or not _version_satisfies(version, expected):
        item["version_status"] = "fail"
        item["status"] = "fail"

    if display_name == "torch" and imported is not None:
        try:
            build = getattr(getattr(imported, "version"), "cuda")
            item["cuda_build"] = str(build) if build is not None else None
        except Exception:
            item["cuda_build"] = None
        if item["cuda_build"] != "11.8":
            item["status"] = "fail"
    return item, imported


def _torch_cuda_report(torch_module: Any) -> Dict[str, Any]:
    """Inspect CUDA through Torch while retaining only fixed safe fields."""

    result: Dict[str, Any] = {
        "status": "fail",
        "available": False,
        "device_count": 0,
        "devices": [],
    }
    if torch_module is None:
        return result
    try:
        cuda = getattr(torch_module, "cuda")
        available = bool(cuda.is_available())
        result["available"] = available
    except Exception:
        return result
    if not result["available"]:
        return result
    try:
        count = int(cuda.device_count())
    except Exception:
        return result
    result["device_count"] = count
    if count < 0:
        return result
    devices: List[Dict[str, Any]] = []
    valid = count >= 2
    for index in range(count):
        try:
            properties = cuda.get_device_properties(index)
            name = getattr(properties, "name")
            total_memory = getattr(properties, "total_memory")
            if isinstance(total_memory, bool):
                raise ValueError
            total_bytes = int(total_memory)
            if not isinstance(name, str) or not name.strip() or total_bytes <= 0:
                valid = False
            devices.append(
                {
                    "index": index,
                    "name": name if isinstance(name, str) else "",
                    "total_bytes": total_bytes if total_bytes > 0 else 0,
                }
            )
        except Exception:
            valid = False
            devices.append({"index": index, "name": "", "total_bytes": 0})
    result["devices"] = devices
    if valid and len(devices) == count:
        result["status"] = "pass"
    return result


def _load_dataset_config(path: Path) -> Mapping[str, Any]:
    """Read only the small configuration mapping; Parquet data stays unopened."""

    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    else:
        import yaml

        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError
    return value


def _effective_dataset_paths() -> List[Dict[str, Any]]:
    """Resolve the two configured Parquet paths and their allowed overrides."""

    result: List[Dict[str, Any]] = []
    for name, config_path in _DATASET_CONFIGS:
        item: Dict[str, Any] = {
            "name": name,
            "config_path": str(config_path),
            "path": "",
            "source": "config",
            "config_status": "pass",
        }
        override_name = _DATASET_OVERRIDES[name]
        override = os.environ.get(override_name)
        if isinstance(override, str) and override.strip():
            item["path"] = override.strip()
            item["source"] = "environment_override"
            result.append(item)
            continue
        try:
            if not config_path.is_file():
                raise ValueError
            config = _load_dataset_config(config_path)
            paths = config.get("paths")
            if not isinstance(paths, Mapping):
                raise ValueError
            configured = paths.get("parquet_file")
            if not isinstance(configured, str) or not configured.strip():
                raise ValueError
            configured_path = Path(configured.strip())
            if not configured_path.is_absolute():
                configured_path = config_path.parent / configured_path
            item["path"] = str(configured_path)
        except Exception:
            item["config_status"] = "fail"
        result.append(item)
    return result


def _dataset_report(paths: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Validate effective paths without opening or parsing Parquet content."""

    by_name = {
        str(item.get("name")): item
        for item in paths
        if isinstance(item, Mapping) and item.get("name") is not None
    }
    result: List[Dict[str, Any]] = []
    for name, _config_path in _DATASET_CONFIGS:
        raw = by_name.get(name, {})
        config_path = raw.get("config_path", "")
        parquet_path = raw.get("path", "")
        source = raw.get("source", "config")
        config_status = raw.get("config_status", "pass")
        item: Dict[str, Any] = {
            "name": name,
            "status": "fail",
            "config_path": str(config_path),
            "path": str(parquet_path) if parquet_path else "",
            "source": str(source) if source in {"config", "environment_override"} else "config",
            "size_bytes": 0,
        }
        if config_status != "pass" or not isinstance(parquet_path, str) or not parquet_path.strip():
            result.append(item)
            continue
        try:
            path = Path(parquet_path)
            if not path.is_file():
                result.append(item)
                continue
            file_stat = path.stat()
            if not stat.S_ISREG(file_stat.st_mode) or int(file_stat.st_size) <= 0:
                result.append(item)
                continue
            item["status"] = "pass"
            item["size_bytes"] = int(file_stat.st_size)
        except Exception:
            pass
        result.append(item)
    return result


def _effective_cache_root() -> Path:
    """Return the effective Hub cache directory without inspecting token files."""

    # Transformers resolves legacy variables and HF_HOME into this effective
    # value.  Consult that installed value first so this checker does not
    # accidentally choose a raw HF_HUB_CACHE over a legacy override.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from transformers.utils.hub import TRANSFORMERS_CACHE

        if isinstance(TRANSFORMERS_CACHE, str) and TRANSFORMERS_CACHE.strip():
            return Path(TRANSFORMERS_CACHE).expanduser()
    except Exception:
        pass
    # If Transformers itself is unavailable, retain its relevant precedence as
    # a credential-free fallback before considering HF_HOME.
    for name in (
        "TRANSFORMERS_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "PYTORCH_PRETRAINED_BERT_CACHE",
        "HF_HUB_CACHE",
    ):
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return Path(value.strip()).expanduser()
    home = os.environ.get("HF_HOME")
    if isinstance(home, str) and home.strip():
        return (Path(home.strip()).expanduser() / "hub").resolve()
    try:
        from huggingface_hub import constants

        value = getattr(constants, "HF_HUB_CACHE", None)
        if isinstance(value, str) and value.strip():
            return Path(value).expanduser()
    except Exception:
        pass
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _try_to_load_from_cache(
    model_id: str,
    filename: str,
    revision: str,
    cache_dir: Path,
    token: bool = False,
) -> Optional[str]:
    """Call the Hub local-cache primitive, never authorizing a request."""

    try:
        from huggingface_hub import try_to_load_from_cache

        try:
            result = try_to_load_from_cache(
                model_id,
                filename,
                cache_dir=str(cache_dir),
                revision=revision,
                token=False,
            )
        except TypeError:
            # huggingface-hub versions before token became an accepted
            # parameter are still local-only for this primitive; retain the
            # explicit no-token attempt above and use the compatible call.
            result = try_to_load_from_cache(
                model_id,
                filename,
                cache_dir=str(cache_dir),
                revision=revision,
            )
    except Exception:
        return None
    if result is None:
        return None
    return str(result)


def _cached_model_files(
    model_id: str,
    revision: str,
    required_files: Sequence[str],
    cache_root: Path,
) -> Dict[str, Any]:
    """Check every required file at one exact revision using local cache only."""

    missing: List[str] = []
    for filename in required_files:
        try:
            candidate = _try_to_load_from_cache(
                model_id,
                filename,
                revision,
                cache_root,
                token=False,
            )
            if not candidate:
                raise ValueError
            path = Path(candidate)
            if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
                raise ValueError
        except Exception:
            missing.append(str(filename))
    return {"complete": not missing, "missing_files": missing}


def _probe_pinned_metadata(
    model_id: str,
    revision: str,
    timeout: float,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Issue a credential-free HEAD request for one pinned cache file."""

    if filename is None:
        filename = "model.ckpt" if model_id == "NX-AI/TiRex" else "config.json"
    if not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename:
        filename = "config.json"
    url = "https://huggingface.co/{}/resolve/{}/{}".format(model_id, revision, filename)
    curl_binary = shutil.which("curl")
    if not curl_binary:
        return {"status": "fail", "network_status": "missing_curl"}
    curl_binary = str(Path(curl_binary).resolve())
    command = [
        curl_binary,
        "--disable",
        "--head",
        "--location",
        "--max-redirs",
        "3",
        "--max-time",
        str(float(timeout)),
        "--connect-timeout",
        str(float(timeout)),
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--silent",
        "--show-error",
        "--output",
        "/dev/null",
        "--write-out",
        "%{http_code}",
        url,
    ]
    # Do not inherit proxy, token, or other credential-bearing environment
    # variables.  --disable also prevents a user's curl configuration file
    # from adding headers, proxies, or a request body.
    clean_env = {
        "PATH": str(Path(curl_binary).parent),
        "LANG": "C",
        "LC_ALL": "C",
    }
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout),
            env=clean_env,
        )
    except subprocess.TimeoutExpired:
        return {"status": "fail", "network_status": "timeout"}
    except FileNotFoundError:
        return {"status": "fail", "network_status": "missing_curl"}
    except OSError:
        return {"status": "fail", "network_status": "connection_error"}
    except Exception:
        return {"status": "fail", "network_status": "connection_error"}

    try:
        return_code = int(getattr(completed, "returncode", 1))
    except Exception:
        return_code = 1
    if return_code == 28:
        return {"status": "fail", "network_status": "timeout"}
    if return_code == 6:
        return {"status": "fail", "network_status": "dns"}
    if return_code != 0:
        return {"status": "fail", "network_status": "connection_error"}
    raw_status = getattr(completed, "stdout", "")
    if not isinstance(raw_status, str):
        return {"status": "fail", "network_status": "connection_error"}
    status_text = raw_status.strip()
    if not re.fullmatch(r"\d{3}", status_text):
        return {"status": "fail", "network_status": "connection_error"}
    status = int(status_text)
    if 200 <= status < 300:
        return {"status": "pass", "http_status": status}
    return {"status": "fail", "network_status": "http_{}".format(status), "http_status": status}


def _interpreter_report() -> Dict[str, Any]:
    executable = str(getattr(sys, "executable", ""))
    python_version = _runtime_python_version()
    supported_python = python_version == EXPECTED_PYTHON or _python_version_at_least(
        python_version, "3.10"
    )
    return {
        "status": "pass"
        if executable == EXPECTED_INTERPRETER and supported_python
        else "fail",
        "expected_executable": EXPECTED_INTERPRETER,
        "expected_python": EXPECTED_PYTHON,
        "executable": executable,
        "python": python_version,
    }


def _normalise_cuda_report(value: Any) -> Dict[str, Any]:
    """Copy only the CUDA fields that are safe and stable to render."""

    source = value if isinstance(value, Mapping) else {}
    devices: List[Dict[str, Any]] = []
    raw_devices = source.get("devices", [])
    if isinstance(raw_devices, Sequence) and not isinstance(raw_devices, (str, bytes)):
        for raw in raw_devices:
            if not isinstance(raw, Mapping):
                continue
            try:
                index = int(raw.get("index", len(devices)))
            except Exception:
                index = len(devices)
            name = raw.get("name", "")
            if not isinstance(name, str):
                name = ""
            try:
                total_bytes = int(raw.get("total_bytes", 0))
            except Exception:
                total_bytes = 0
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "total_bytes": total_bytes if total_bytes > 0 else 0,
                }
            )
    try:
        device_count = int(source.get("device_count", len(devices)))
    except Exception:
        device_count = len(devices)
    available = bool(source.get("available", False))
    status = "pass" if source.get("status") == "pass" else "fail"
    if status == "pass":
        if not available or device_count < 2 or len(devices) != device_count:
            status = "fail"
        elif any(not item["name"].strip() or item["total_bytes"] <= 0 for item in devices):
            status = "fail"
    return {
        "status": status,
        "available": available,
        "device_count": device_count,
        "devices": devices,
    }


def _normalise_dataset_paths(value: Any) -> List[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        nested = value.get("datasets")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            value = nested
        else:
            value = [dict(item, name=name) if isinstance(item, Mapping) else {"name": name, "path": item}
                     for name, item in value.items()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _normalise_cache_result(value: Any, required_files: Sequence[str]) -> Dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    raw_missing = source.get("missing_files", source.get("missing", []))
    if isinstance(raw_missing, str):
        raw_missing = [raw_missing]
    if not isinstance(raw_missing, Sequence):
        raw_missing = []
    missing = [str(name) for name in raw_missing if str(name) in set(required_files)]
    complete = bool(source.get("complete", False)) and not missing
    if complete:
        missing = []
    else:
        # A malformed fake or cache helper result is an incomplete cache; do
        # not silently treat it as a usable exact snapshot.
        if not missing:
            missing = [str(name) for name in required_files]
    return {"complete": complete, "missing_files": missing}


def _normalise_probe_result(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    raw_status = source.get("status")
    if raw_status in {"pass", "ok", "reachable"}:
        return {"status": "pass", "network_status": "pass"}
    label = source.get("network_status", source.get("error", raw_status))
    if isinstance(label, str) and label in {
        "missing_curl",
        "timeout",
        "dns",
        "connection_error",
    }:
        return {"status": "fail", "network_status": label}
    http_status = source.get("http_status")
    try:
        code = int(http_status)
    except Exception:
        code = None
    if code is not None and code >= 400:
        return {"status": "fail", "network_status": "http_{}".format(code)}
    if isinstance(label, str) and label.startswith("http_") and label[5:].isdigit():
        return {"status": "fail", "network_status": label}
    return {"status": "fail", "network_status": "connection_error"}


def _failed_dataset_paths() -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "config_path": str(config_path),
            "path": "",
            "source": "config",
            "config_status": "fail",
        }
        for name, config_path in _DATASET_CONFIGS
    ]


def _python_version_at_least(value: Any, minimum: str) -> bool:
    """Compare interpreter versions without importing model packages."""

    def parts(raw: Any) -> Tuple[int, int, int]:
        match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(raw))
        if not match:
            return (0, 0, 0)
        return tuple(int(item or 0) for item in match.groups())  # type: ignore[return-value]

    return parts(value) >= parts(minimum)


def _package_specs_for_python(python_version: str) -> Tuple[Tuple[str, str, str], ...]:
    """Select the complete dependency profile without importing model code."""

    if _python_version_at_least(python_version, "3.10"):
        return MODERN_PACKAGE_SPECS
    return PACKAGE_SPECS


def _model_package_report(model_name: str, package_items: Sequence[Mapping[str, Any]]) -> Tuple[str, str, Optional[str]]:
    """Check optional distributions through metadata only, never importing them."""

    package_spec = _MODERN_MODEL_PACKAGE_SPECS.get(model_name)
    if package_spec is None:
        return "pass", "-", None
    distribution_name, version_specifier = package_spec
    if distribution_name == "transformers":
        version = next(
            (
                item.get("version")
                for item in package_items
                if item.get("name") == "transformers"
            ),
            None,
        )
    else:
        try:
            version = _package_version(distribution_name)
        except Exception:
            version = None
    if version is None:
        return "missing", distribution_name, None
    version = str(version)
    if not _version_satisfies(version, version_specifier):
        return "incompatible", distribution_name, version
    return "pass", distribution_name, version


def collect_environment(*, offline: bool = False, network_timeout: float = 3.0) -> Dict[str, Any]:
    """Collect a deterministic, credential-free report of local readiness."""

    try:
        timeout = float(network_timeout)
    except (TypeError, ValueError):
        raise ValueError("network_timeout must be positive and at most 10 seconds")
    if not math.isfinite(timeout) or timeout <= 0 or timeout > MAX_NETWORK_TIMEOUT:
        raise ValueError("network_timeout must be positive and at most 10 seconds")

    interpreter = _interpreter_report()
    runtime_python = _runtime_python_version()
    package_specs = _package_specs_for_python(runtime_python)
    packages: List[Dict[str, Any]] = []
    imported: Dict[str, Any] = {}
    for display_name, distribution_name, expected in package_specs:
        item, package = _package_report(display_name, distribution_name, expected)
        packages.append(item)
        imported[display_name] = package

    try:
        cuda = _normalise_cuda_report(_torch_cuda_report(imported.get("torch")))
    except Exception:
        cuda = _normalise_cuda_report({})

    try:
        raw_paths = _effective_dataset_paths()
    except Exception:
        raw_paths = _failed_dataset_paths()
    datasets = _dataset_report(_normalise_dataset_paths(raw_paths))

    cache_root_status = "pass"
    try:
        cache_root = _effective_cache_root()
        cache_root_path = Path(cache_root)
        cache_root_value = str(cache_root_path)
    except Exception:
        cache_root_status = "fail"
        cache_root_path = Path(".")
        cache_root_value = ""
    cache_root_report = {"status": cache_root_status, "path": cache_root_value}

    models: List[Dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        spec = get_model_spec(model_name)
        required_files = _REQUIRED_CACHE_FILES.get(model_name, ("config.json",))
        python_floor = _MODEL_PYTHON_FLOORS.get(model_name, EXPECTED_PYTHON)
        python_blocked = not _python_version_at_least(runtime_python, python_floor)
        package_status, package_name, package_version = _model_package_report(model_name, packages)
        try:
            raw_cache = _cached_model_files(
                spec.model_id, spec.revision, required_files, cache_root_path
            )
        except Exception:
            raw_cache = {"complete": False, "missing_files": list(required_files)}
        cache = _normalise_cache_result(raw_cache, required_files)
        item: Dict[str, Any] = {
            "name": model_name,
            "model_id": spec.model_id,
            "revision": spec.revision,
            "required_files": list(required_files),
            "status": "pass" if cache["complete"] else "download_required",
            "cache_complete": cache["complete"],
            "missing_files": cache["missing_files"],
            "network_attempted": False,
            "network_status": "not_checked" if cache["complete"] else "offline",
            "download_required": not cache["complete"],
            "python_floor": python_floor,
            "python_status": "blocked" if python_blocked else "pass",
            "package_status": package_status,
            "package": package_name,
            "package_version": package_version,
            "block_reason": None,
        }
        if python_blocked:
            item["status"] = "blocked"
            item["network_status"] = "python_blocked"
            item["block_reason"] = (
                "requires Python >={} and package {} ({}); current Python {}"
                .format(
                    python_floor,
                    _MODERN_MODEL_PACKAGES.get(model_name, package_name),
                    package_status,
                    runtime_python,
                )
            )
        elif package_status == "pass" and not cache["complete"] and not offline:
            item["network_attempted"] = True
            try:
                probe = _normalise_probe_result(
                    _probe_pinned_metadata(spec.model_id, spec.revision, timeout)
                )
            except Exception:
                probe = {"status": "fail", "network_status": "connection_error"}
            item["network_status"] = probe["network_status"]
        if not python_blocked and package_status != "pass":
            item["status"] = "missing" if package_status == "missing" else "incompatible"
            if item["block_reason"] is None:
                item["block_reason"] = "required package {} is {}".format(
                    package_name, package_status
                )
        # A reachable HEAD does not make an absent local checkpoint usable.
        if not cache["complete"] and item["status"] == "pass":
            item["status"] = "download_required"
        models.append(item)

    checks = [interpreter, cuda, cache_root_report]
    checks.extend(packages)
    checks.extend(datasets)
    checks.extend(models)
    ok = all(item.get("status") == "pass" for item in checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": bool(ok),
        "interpreter": interpreter,
        "packages": packages,
        "cuda": cuda,
        "datasets": datasets,
        "cache_root": cache_root_report,
        "models": models,
    }


def _text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def render_text(report: Dict[str, Any]) -> str:
    """Render only the fixed report fields in deterministic section order."""

    lines = [
        "foundation environment preflight",
        "status: {}".format("PASS" if report.get("ok") else "FAIL"),
    ]
    interpreter = report.get("interpreter", {})
    if isinstance(interpreter, Mapping):
        lines.append(
            "interpreter: {} executable={} python={}".format(
                _text(interpreter.get("status")),
                _text(interpreter.get("executable")),
                _text(interpreter.get("python")),
            )
        )
    packages = report.get("packages", [])
    if isinstance(packages, Sequence) and not isinstance(packages, (str, bytes)):
        for item in packages:
            if isinstance(item, Mapping):
                lines.append(
                    "package {}: {} version={} cuda_build={}".format(
                        _text(item.get("name")),
                        _text(item.get("status")),
                        _text(item.get("version")),
                        _text(item.get("cuda_build")),
                    )
                )
    cuda = report.get("cuda", {})
    if isinstance(cuda, Mapping):
        lines.append(
            "cuda: {} available={} devices={}".format(
                _text(cuda.get("status")),
                _text(cuda.get("available")),
                _text(cuda.get("device_count")),
            )
        )
        devices = cuda.get("devices", [])
        if isinstance(devices, Sequence) and not isinstance(devices, (str, bytes)):
            for device in devices:
                if isinstance(device, Mapping):
                    lines.append(
                        "cuda_device {}: name={} total_bytes={}".format(
                            _text(device.get("index")),
                            _text(device.get("name")),
                            _text(device.get("total_bytes")),
                        )
                    )
    datasets = report.get("datasets", [])
    if isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)):
        for item in datasets:
            if isinstance(item, Mapping):
                lines.append(
                    "dataset {}: {} source={} path={} size_bytes={}".format(
                        _text(item.get("name")),
                        _text(item.get("status")),
                        _text(item.get("source")),
                        _text(item.get("path")),
                        _text(item.get("size_bytes")),
                    )
                )
    cache_root = report.get("cache_root", {})
    if isinstance(cache_root, Mapping):
        lines.append(
            "cache_root: {} path={}".format(
                _text(cache_root.get("status")), _text(cache_root.get("path"))
            )
        )
    models = report.get("models", [])
    if isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
        for item in models:
            if isinstance(item, Mapping):
                missing = item.get("missing_files", [])
                if not isinstance(missing, Sequence) or isinstance(missing, (str, bytes)):
                    missing = []
                lines.append(
                    "model {}: {} cache_complete={} network={} download_required={} python={} floor={} package={} reason={} missing={}".format(
                        _text(item.get("name")),
                        _text(item.get("status")),
                        _text(item.get("cache_complete")),
                        _text(item.get("network_status")),
                        _text(item.get("download_required")),
                        _text(item.get("python_status")),
                        _text(item.get("python_floor")),
                        _text(item.get("package_status")),
                        _text(item.get("block_reason")),
                        ",".join(_text(value) for value in missing),
                    )
                )
    return "\n".join(lines) + "\n"


def render_json(report: Dict[str, Any]) -> str:
    """Render the machine-readable report with stable key ordering."""

    return json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def _network_timeout_argument(value: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a positive number no greater than 10")
    if not math.isfinite(timeout) or timeout <= 0 or timeout > MAX_NETWORK_TIMEOUT:
        raise argparse.ArgumentTypeError("must be a positive number no greater than 10")
    return timeout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--network-timeout", type=_network_timeout_argument, default=3.0)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Run the preflight CLI and return 0 for pass, 1 for failed checks."""

    args = _parser().parse_args(argv)
    report = collect_environment(offline=bool(args.offline), network_timeout=args.network_timeout)
    output = render_json(report) if args.json_output else render_text(report)
    sys.stdout.write(output)
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
