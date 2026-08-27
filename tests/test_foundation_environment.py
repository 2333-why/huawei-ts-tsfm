"""Focused, offline tests for the foundation-model environment preflight."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "check_foundation_environment.py"
SENTINEL = "credential-sentinel-should-never-appear"


def _load_module():
    spec = importlib.util.spec_from_file_location("foundation_environment_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def environment_module():
    return _load_module()


class FakeCuda:
    def __init__(self, available=True, names=None, memories=None):
        self.available = available
        self.names = list(names or ["Fake GPU 0", "Fake GPU 1"])
        self.memories = list(memories or [25_393_692_672, 25_393_692_672])

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.names)

    def get_device_properties(self, index):
        return SimpleNamespace(name=self.names[index], total_memory=self.memories[index])


class FakeTorch:
    version = SimpleNamespace(cuda="11.8")

    def __init__(self, cuda=None):
        self.cuda = cuda or FakeCuda()


def _fake_paths(tmp_path):
    skippd = tmp_path / "skippd.parquet"
    pvod = tmp_path / "pvod.parquet"
    skippd.write_bytes(b"skippd parquet")
    pvod.write_bytes(b"pvod parquet")
    return [
        {
            "name": "skippd_luoyang",
            "config_path": str(tmp_path / "skippd.json"),
            "path": str(skippd),
            "source": "config",
            "config_status": "pass",
        },
        {
            "name": "pvod_station00_ylj",
            "config_path": str(tmp_path / "pvod.yaml"),
            "path": str(pvod),
            "source": "config",
            "config_status": "pass",
        },
    ]


def _install_happy_fakes(monkeypatch, module, tmp_path, *, cache_complete=True):
    versions = {
        "torch": "2.3.1+cu118",
        "transformers": "4.46.2",
        "peft": "0.13.2",
    }
    imports = {
        "torch": FakeTorch(),
        "transformers": object(),
        "peft": object(),
    }
    monkeypatch.setattr(module, "_package_version", lambda name: versions[name])
    monkeypatch.setattr(module, "_import_package", lambda name: imports[name])
    monkeypatch.setattr(module, "_effective_dataset_paths", lambda: _fake_paths(tmp_path))
    monkeypatch.setattr(module, "_effective_cache_root", lambda: tmp_path / "hf-cache")
    monkeypatch.setattr(
        module,
        "_cached_model_files",
        lambda model_id, revision, required_files, cache_root: {
            "complete": cache_complete,
            "missing_files": [] if cache_complete else list(required_files),
        },
    )
    return imports["torch"]


def _install_transformers_hub_fake(monkeypatch, cache_value):
    transformers = types.ModuleType("transformers")
    utils = types.ModuleType("transformers.utils")
    hub = types.ModuleType("transformers.utils.hub")
    hub.TRANSFORMERS_CACHE = cache_value
    transformers.utils = utils
    utils.hub = hub
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", hub)
    return hub


def test_complete_reference_environment_is_ok_and_skips_network(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    network_calls = []
    monkeypatch.setattr(module, "_probe_pinned_metadata", lambda *args: network_calls.append(args))

    report = module.collect_environment()

    assert report["ok"] is False
    assert report["interpreter"]["status"] == "pass"
    assert [item["name"] for item in report["packages"]] == [
        "torch",
        "transformers",
        "peft",
    ]
    assert report["cuda"]["status"] == "pass"
    assert len(report["cuda"]["devices"]) == 2
    assert all(item["network_attempted"] is False for item in report["models"])
    assert all(item["download_required"] is False for item in report["models"])
    assert [item["status"] for item in report["models"]] == [
        "pass", "pass", "blocked", "blocked", "blocked"
    ]
    assert all(item["network_status"] == "python_blocked" for item in report["models"][2:])
    assert network_calls == []


def test_report_rendering_and_section_order_are_deterministic(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")

    first = module.collect_environment()
    second = module.collect_environment()

    assert first == second
    assert module.render_text(first) == module.render_text(second)
    assert module.render_json(first) == module.render_json(second)
    assert list(first) == [
        "schema_version",
        "ok",
        "interpreter",
        "packages",
        "cuda",
        "datasets",
        "cache_root",
        "models",
    ]
    assert [item["name"] for item in first["datasets"]] == [
        "skippd_luoyang",
        "pvod_station00_ylj",
    ]
    assert [item["name"] for item in first["models"]] == [
        "Sundial", "TimeMoE", "Chronos2", "TiRex", "TimesFM"
    ]
    assert list(first["interpreter"]) == [
        "status",
        "expected_executable",
        "expected_python",
        "executable",
        "python",
    ]
    assert list(first["packages"][0]) == [
        "name",
        "status",
        "expected",
        "version",
        "metadata_status",
        "import_status",
        "version_status",
        "cuda_build",
    ]
    assert list(first["cuda"]) == ["status", "available", "device_count", "devices"]
    assert list(first["cuda"]["devices"][0]) == ["index", "name", "total_bytes"]
    assert list(first["datasets"][0]) == [
        "name",
        "status",
        "config_path",
        "path",
        "source",
        "size_bytes",
    ]
    assert list(first["cache_root"]) == ["status", "path"]
    assert list(first["models"][0]) == [
        "name",
        "model_id",
        "revision",
        "required_files",
        "status",
        "cache_complete",
        "missing_files",
        "network_attempted",
        "network_status",
        "download_required",
        "python_floor",
        "python_status",
        "package_status",
        "package",
        "package_version",
        "block_reason",
    ]
    json_lines = module.render_json(first).splitlines()
    assert json_lines[1].strip() == '"cache_root": {'
    assert json_lines[-1] == "}"
    text_lines = module.render_text(first).splitlines()
    assert text_lines[:3] == [
        "foundation environment preflight",
        "status: FAIL",
        "interpreter: pass executable=/opt/data/private/penv/time/bin/python python=3.8.18",
    ]
    assert text_lines.index(next(line for line in text_lines if line.startswith("cuda: "))) < text_lines.index(
        next(line for line in text_lines if line.startswith("dataset "))
    )


@pytest.mark.parametrize(
    ("change", "expected_fragment"),
    [
        ("executable", "interpreter"),
        ("python", "python"),
        ("torch_version", "torch"),
        ("transformers_version", "transformers"),
        ("peft_version", "peft"),
    ],
)
def test_interpreter_and_package_mismatches_fail_cleanly(
    environment_module, monkeypatch, tmp_path, change, expected_fragment
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    if change == "executable":
        monkeypatch.setattr(module.sys, "executable", "/wrong/python")
    elif change == "python":
        monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.9.0")
    elif change == "torch_version":
        monkeypatch.setattr(module, "_package_version", lambda name: "2.2.0" if name == "torch" else {
            "transformers": "4.46.2",
            "peft": "0.13.2",
        }[name])
    elif change == "transformers_version":
        monkeypatch.setattr(module, "_package_version", lambda name: "4.47.0" if name == "transformers" else {
            "torch": "2.3.1+cu118",
            "peft": "0.13.2",
        }[name])
    else:
        monkeypatch.setattr(module, "_package_version", lambda name: "0.12.0" if name == "peft" else {
            "torch": "2.3.1+cu118",
            "transformers": "4.46.2",
        }[name])

    report = module.collect_environment()

    assert report["ok"] is False
    assert any(
        item["status"] == "fail" and expected_fragment in item.get("name", "")
        for item in report["packages"]
    ) or report["interpreter"]["status"] == "fail"


def test_missing_distribution_and_import_failure_are_named_without_exception_text(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")

    def missing(name):
        if name == "transformers":
            raise RuntimeError(SENTINEL)
        return {"torch": "2.3.1+cu118", "peft": "0.13.2"}[name]

    def imports(name):
        if name == "peft":
            raise ImportError(SENTINEL)
        return FakeTorch() if name == "torch" else object()

    monkeypatch.setattr(module, "_package_version", missing)
    monkeypatch.setattr(module, "_import_package", imports)

    report = module.collect_environment()
    text = module.render_text(report)
    rendered = module.render_json(report)

    assert report["ok"] is False
    assert report["packages"][1]["status"] == "fail"
    assert report["packages"][2]["status"] == "fail"
    assert SENTINEL not in text and SENTINEL not in rendered


@pytest.mark.parametrize(
    ("package_name", "version"),
    [
        ("torch", "2.3.1.dev0"),
        ("transformers", "4.46.2rc1"),
        ("peft", "0.13.2.dev1"),
    ],
)
def test_prerelease_and_dev_versions_do_not_satisfy_pinned_ranges(
    environment_module, monkeypatch, tmp_path, package_name, version
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    versions = {
        "torch": "2.3.1+cu118",
        "transformers": "4.46.2",
        "peft": "0.13.2",
    }
    versions[package_name] = version
    monkeypatch.setattr(module, "_package_version", lambda name: versions[name])

    report = module.collect_environment()

    item = next(item for item in report["packages"] if item["name"] == package_name)
    assert item["version"] == version
    assert item["version_status"] == "fail"
    assert item["status"] == "fail"
    assert report["ok"] is False


@pytest.mark.parametrize(
    ("package_name", "version", "expected_status"),
    [
        ("transformers", "4.46.1", "fail"),
        ("transformers", "4.46.2", "pass"),
        ("transformers", "4.47.0", "fail"),
        ("peft", "0.13.1", "fail"),
        ("peft", "0.13.2", "pass"),
        ("peft", "0.14.0", "fail"),
    ],
)
def test_dependency_patch_boundaries_match_the_supported_contract(
    environment_module, monkeypatch, tmp_path, package_name, version, expected_status
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    versions = {
        "torch": "2.3.1+cu118",
        "transformers": "4.46.2",
        "peft": "0.13.2",
    }
    versions[package_name] = version
    monkeypatch.setattr(module, "_package_version", lambda name: versions[name])

    report = module.collect_environment()

    item = next(item for item in report["packages"] if item["name"] == package_name)
    assert item["version"] == version
    assert item["status"] == expected_status


@pytest.mark.parametrize(
    ("package_name", "version", "expected_status"),
    [
        ("torch", "0!2.3.1", "pass"),
        ("torch", "2.3.1.0", "pass"),
        ("torch", "2.3.1+cu118", "pass"),
        ("torch", "2.3.not-a-version", "fail"),
    ],
)
def test_dependency_versions_use_pep440_release_comparison(
    environment_module, monkeypatch, tmp_path, package_name, version, expected_status
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    versions = {
        "torch": "2.3.1+cu118",
        "transformers": "4.46.2",
        "peft": "0.13.2",
    }
    versions[package_name] = version
    monkeypatch.setattr(module, "_package_version", lambda name: versions[name])

    report = module.collect_environment()

    item = next(item for item in report["packages"] if item["name"] == package_name)
    assert item["version"] == version
    assert item["status"] == expected_status


def test_missing_packaging_dependency_is_a_failed_version_check(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    original_import = module.importlib.import_module

    def missing_packaging(name):
        if name.startswith("packaging."):
            raise ImportError(SENTINEL)
        return original_import(name)

    monkeypatch.setattr(module.importlib, "import_module", missing_packaging)

    report = module.collect_environment()

    torch_item = next(item for item in report["packages"] if item["name"] == "torch")
    assert torch_item["version_status"] == "fail"
    assert torch_item["status"] == "fail"
    assert SENTINEL not in module.render_text(report)


def test_torch_cuda_build_mismatch_is_a_package_failure(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    wrong_torch = FakeTorch()
    wrong_torch.version = SimpleNamespace(cuda="12.1")
    monkeypatch.setattr(
        module,
        "_import_package",
        lambda name: wrong_torch if name == "torch" else object(),
    )
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")

    report = module.collect_environment()

    torch_item = next(item for item in report["packages"] if item["name"] == "torch")
    assert torch_item["cuda_build"] == "12.1"
    assert torch_item["status"] == "fail"
    assert report["ok"] is False


def test_required_cache_files_and_registry_pins_are_exact(environment_module):
    module = environment_module

    assert module.MODEL_NAMES == ("Sundial", "TimeMoE", "Chronos2", "TiRex", "TimesFM")
    assert module._REQUIRED_CACHE_FILES == {
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
        "Chronos2": ("config.json", "model.safetensors"),
        "TiRex": ("model.ckpt",),
        "TimesFM": ("config.json", "model.safetensors"),
    }
    assert (
        module.get_model_spec("Sundial").model_id,
        module.get_model_spec("Sundial").revision,
    ) == (
        "thuml/sundial-base-128m",
        "3212e42564493f520593e5414af4367fc4b49226",
    )
    assert (
        module.get_model_spec("TimeMoE").model_id,
        module.get_model_spec("TimeMoE").revision,
    ) == (
        "Maple728/TimeMoE-50M",
        "446753ee48ff3726d0606a81d0092d54acee995e",
    )
    expected = {
        "Sundial": (
            "thuml/sundial-base-128m",
            "3212e42564493f520593e5414af4367fc4b49226",
        ),
        "TimeMoE": (
            "Maple728/TimeMoE-50M",
            "446753ee48ff3726d0606a81d0092d54acee995e",
        ),
        "Chronos2": (
            "amazon/chronos-2",
            "29ec3766d36d6f73f0696f85560a422f50e8498c",
        ),
        "TiRex": (
            "NX-AI/TiRex",
            "63c740922493f5fbe60b277609ec62babfba2762",
        ),
        "TimesFM": (
            "google/timesfm-2.5-200m-transformers",
            "5a9806b9b291fad9233b5249d88263f1846304d3",
        ),
    }
    assert {
        name: (module.get_model_spec(name).model_id, module.get_model_spec(name).revision)
        for name in module.MODEL_NAMES
    } == expected


@pytest.mark.parametrize(
    "mapping_name",
    ["_REQUIRED_CACHE_FILES", "_MODEL_PYTHON_FLOORS", "_MODEL_PACKAGE_DESCRIPTORS", "_MODEL_DESCRIPTORS"],
)
@pytest.mark.parametrize("drift", ["missing", "extra"])
def test_descriptor_mapping_drift_fails_closed(environment_module, monkeypatch, mapping_name, drift):
    module = environment_module
    original = getattr(module, mapping_name)
    broken = dict(original)
    if drift == "missing":
        broken.pop(module.MODEL_NAMES[-1])
    else:
        broken["UnexpectedModel"] = next(iter(original.values()))
    monkeypatch.setattr(module, mapping_name, broken)

    with pytest.raises(RuntimeError, match="descriptor"):
        module.collect_environment(offline=True)


@pytest.mark.parametrize(
    "cuda_report",
    [
        {"status": "fail", "available": False, "device_count": 0, "devices": []},
        {
            "status": "fail",
            "available": True,
            "device_count": 1,
            "devices": [{"index": 0, "name": "Fake GPU", "total_bytes": 1}],
        },
        {
            "status": "fail",
            "available": True,
            "device_count": 2,
            "devices": [
                {"index": 0, "name": "", "total_bytes": 1},
                {"index": 1, "name": "Fake GPU 1", "total_bytes": 1},
            ],
        },
        {
            "status": "fail",
            "available": True,
            "device_count": 2,
            "devices": [
                {"index": 0, "name": "Fake GPU 0", "total_bytes": 0},
                {"index": 1, "name": "Fake GPU 1", "total_bytes": -1},
            ],
        },
    ],
)
def test_cuda_failures_propagate_as_a_failed_report(
    environment_module, monkeypatch, tmp_path, cuda_report
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    monkeypatch.setattr(module, "_torch_cuda_report", lambda _torch: cuda_report)

    report = module.collect_environment()

    assert report["ok"] is False
    assert report["cuda"]["status"] == "fail"


@pytest.mark.parametrize(
    "fake_cuda",
    [
        FakeCuda(available=False),
        FakeCuda(names=["Fake GPU 0"], memories=[1]),
        FakeCuda(names=["", "Fake GPU 1"]),
        FakeCuda(memories=[0, -1]),
    ],
)
def test_cuda_leaf_rejects_unavailable_or_invalid_device_properties(environment_module, fake_cuda):
    report = environment_module._torch_cuda_report(FakeTorch(fake_cuda))
    assert report["status"] == "fail"


def test_effective_dataset_override_takes_precedence_and_file_validation_is_strict(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    skippd_override = tmp_path / "override.parquet"
    pvod_override = tmp_path / "directory.parquet"
    skippd_override.write_bytes(b"override")
    pvod_override.mkdir()
    monkeypatch.setenv("SKIPPD_PARQUET", str(skippd_override))
    monkeypatch.setenv("PVOD_PARQUET", str(pvod_override))
    monkeypatch.setattr(
        module,
        "_DATASET_CONFIGS",
        (
            ("skippd_luoyang", tmp_path / "missing.json"),
            ("pvod_station00_ylj", tmp_path / "missing.yaml"),
        ),
    )

    paths = module._effective_dataset_paths()

    assert [item["source"] for item in paths] == [
        "environment_override",
        "environment_override",
    ]
    assert paths[0]["path"] == str(skippd_override)
    assert paths[1]["path"] == str(pvod_override)
    report = module._dataset_report(paths)
    assert report[0]["status"] == "pass"
    assert report[1]["status"] == "fail"

    pvod_override.rmdir()
    pvod_override.write_bytes(b"")
    report = module._dataset_report(paths)
    assert report[1]["status"] == "fail"


def test_cache_root_matches_transformers_precedence(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    transformers_cache = tmp_path / "transformers-cache"
    hub_cache = tmp_path / "hub-cache"
    hf_home = tmp_path / "hf-home"
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(transformers_cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    _install_transformers_hub_fake(monkeypatch, str(transformers_cache))

    assert module._effective_cache_root() == transformers_cache


def test_cache_root_uses_transformers_effective_legacy_precedence(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    legacy_transformers_cache = tmp_path / "legacy-transformers-cache"
    legacy_pretrained_cache = tmp_path / "legacy-pretrained-cache"
    hub_cache = tmp_path / "hub-cache"
    for name in (
        "TRANSFORMERS_CACHE",
        "HF_HUB_CACHE",
        "HF_HOME",
        "PYTORCH_TRANSFORMERS_CACHE",
        "PYTORCH_PRETRAINED_BERT_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTORCH_TRANSFORMERS_CACHE", str(legacy_transformers_cache))
    monkeypatch.setenv("PYTORCH_PRETRAINED_BERT_CACHE", str(legacy_pretrained_cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    _install_transformers_hub_fake(monkeypatch, str(legacy_transformers_cache))

    assert module._effective_cache_root() == legacy_transformers_cache


def test_cache_root_fallback_keeps_legacy_precedence_when_transformers_import_fails(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    legacy_transformers_cache = tmp_path / "legacy-transformers-cache"
    legacy_pretrained_cache = tmp_path / "legacy-pretrained-cache"
    hub_cache = tmp_path / "hub-cache"
    for name in ("TRANSFORMERS_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTORCH_TRANSFORMERS_CACHE", str(legacy_transformers_cache))
    monkeypatch.setenv("PYTORCH_PRETRAINED_BERT_CACHE", str(legacy_pretrained_cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    for name in ("transformers", "transformers.utils", "transformers.utils.hub"):
        monkeypatch.setitem(sys.modules, name, None)

    assert module._effective_cache_root() == legacy_transformers_cache


def test_invalid_config_and_missing_parquet_are_reported_as_failures(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    missing = tmp_path / "missing.parquet"
    monkeypatch.setattr(
        module,
        "_effective_dataset_paths",
        lambda: [
            {
                "name": "skippd_luoyang",
                "config_path": str(invalid),
                "path": str(missing),
                "source": "config",
                "config_status": "fail",
            },
            {
                "name": "pvod_station00_ylj",
                "config_path": str(invalid),
                "path": str(missing),
                "source": "config",
                "config_status": "fail",
            },
        ],
    )

    report = module._dataset_report(module._effective_dataset_paths())

    assert all(item["status"] == "fail" for item in report)
    assert all("error" not in item for item in report)


def test_dataset_leaf_rejects_missing_invalid_and_bad_override_files(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    missing_config = tmp_path / "missing.json"
    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text("not json", encoding="utf-8")
    monkeypatch.delenv("SKIPPD_PARQUET", raising=False)
    monkeypatch.delenv("PVOD_PARQUET", raising=False)
    monkeypatch.setattr(
        module,
        "_DATASET_CONFIGS",
        (
            ("skippd_luoyang", missing_config),
            ("pvod_station00_ylj", invalid_config),
        ),
    )

    paths = module._effective_dataset_paths()
    report = module._dataset_report(paths)

    assert [item["config_status"] for item in paths] == ["fail", "fail"]
    assert [item["status"] for item in report] == ["fail", "fail"]

    valid_config = tmp_path / "valid.json"
    valid_config.write_text(
        json.dumps({"paths": {"parquet_file": str(tmp_path / "unused.parquet")}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_DATASET_CONFIGS",
        (
            ("skippd_luoyang", valid_config),
            ("pvod_station00_ylj", valid_config),
        ),
    )
    bad_paths = [
        tmp_path / "absent.parquet",
        tmp_path / "directory.parquet",
        tmp_path / "empty.parquet",
    ]
    bad_paths[1].mkdir()
    bad_paths[2].write_bytes(b"")
    for bad_path in bad_paths:
        monkeypatch.setenv("SKIPPD_PARQUET", str(bad_path))
        override_report = module._dataset_report(module._effective_dataset_paths())
        assert override_report[0]["source"] == "environment_override"
        assert override_report[0]["status"] == "fail"


def test_exact_revision_cache_is_complete_only_when_every_pinned_file_exists(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    requested = []
    (tmp_path / "config.json").write_bytes(b"{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    def fake_cache(model_id, filename, revision, cache_dir, token=False):
        requested.append((model_id, filename, revision, cache_dir, token))
        if filename == "generation_config.json":
            return None
        return str(tmp_path / filename)

    monkeypatch.setattr(module, "_try_to_load_from_cache", fake_cache)
    result = module._cached_model_files(
        "thuml/sundial-base-128m",
        "3212e42564493f520593e5414af4367fc4b49226",
        ("config.json", "generation_config.json", "model.safetensors"),
        tmp_path,
    )

    assert result["complete"] is False
    assert result["missing_files"] == ["generation_config.json"]
    assert all(item[2] != "main" for item in requested)
    assert all(item[4] is False for item in requested)


def test_cache_miss_reachable_head_passes_but_marks_download_required(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path, cache_complete=False)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    calls = []

    def reachable(model_id, revision, timeout):
        calls.append((model_id, revision, timeout))
        return {"status": "pass", "http_status": 200}

    monkeypatch.setattr(module, "_probe_pinned_metadata", reachable)
    report = module.collect_environment(network_timeout=2.5)

    assert report["ok"] is False
    assert [item["status"] for item in report["models"]] == [
        "download_required", "download_required", "blocked", "blocked", "blocked"
    ]
    assert [item["network_attempted"] for item in report["models"]] == [True, True, False, False, False]
    assert all(item["download_required"] is True for item in report["models"])
    assert [call[2] for call in calls] == [2.5, 2.5]
    assert calls[0][:2] == (
        "thuml/sundial-base-128m",
        "3212e42564493f520593e5414af4367fc4b49226",
    )


def test_probe_uses_credential_free_head_and_pinned_revision(
    environment_module, monkeypatch
):
    module = environment_module
    observed = {}

    class Completed:
        returncode = 0
        stdout = "200"
        stderr = ""

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module._probe_pinned_metadata(
        "Maple728/TimeMoE-50M",
        "446753ee48ff3726d0606a81d0092d54acee995e",
        3.0,
    )

    assert result == {"status": "pass", "http_status": 200}
    command = observed["command"]
    assert command[0] == "/usr/bin/curl"
    assert any(
        "/Maple728/TimeMoE-50M/resolve/446753ee48ff3726d0606a81d0092d54acee995e/config.json"
        in argument
        for argument in command
    )
    assert observed["kwargs"]["timeout"] == 3.0


def test_tirex_probe_uses_model_ckpt_at_the_pinned_revision(
    environment_module, monkeypatch
):
    module = environment_module
    observed = {}

    class Completed:
        returncode = 0
        stdout = "200"
        stderr = ""

    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: (observed.update(command=command, kwargs=kwargs) or Completed()),
    )
    result = module._probe_pinned_metadata(
        "NX-AI/TiRex",
        "63c740922493f5fbe60b277609ec62babfba2762",
        3.0,
    )

    assert result == {"status": "pass", "http_status": 200}
    assert any(
        "/NX-AI/TiRex/resolve/63c740922493f5fbe60b277609ec62babfba2762/model.ckpt"
        in argument
        for argument in observed["command"]
    )


def test_probe_uses_bounded_curl_head_with_clean_environment(
    environment_module, monkeypatch
):
    module = environment_module
    observed = {}

    class Completed:
        returncode = 0
        stdout = "200"
        stderr = ""

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return Completed()

    monkeypatch.setenv("HTTPS_PROXY", "https://user:" + SENTINEL + "@proxy.invalid")
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module._probe_pinned_metadata(
        "Maple728/TimeMoE-50M",
        "446753ee48ff3726d0606a81d0092d54acee995e",
        3.0,
    )

    command = observed["command"]
    kwargs = observed["kwargs"]
    assert result == {"status": "pass", "http_status": 200}
    assert command[0] == "/usr/bin/curl"
    assert "--head" in command
    assert "--location" in command
    assert "--max-time" in command
    assert command[command.index("--max-time") + 1] == "3.0"
    assert "--max-redirs" in command
    assert "--proto" in command and "=https" in command
    assert "--proto-redir" in command and "=https" in command
    assert "--output" in command and "/dev/null" in command
    assert "--write-out" in command and "%{http_code}" in command
    assert not any(flag in command for flag in ("--data", "--request", "GET"))
    assert kwargs["timeout"] == 3.0
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    clean_env = kwargs["env"]
    assert SENTINEL not in repr(clean_env)
    assert not {"HTTP_PROXY", "HTTPS_PROXY", "HF_TOKEN"}.intersection(clean_env)


def test_probe_requires_final_2xx_instead_of_accepting_a_bare_redirect(
    environment_module, monkeypatch
):
    module = environment_module

    class Completed:
        returncode = 0
        stdout = "302"
        stderr = ""

    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Completed())

    result = module._probe_pinned_metadata(
        "Maple728/TimeMoE-50M",
        "446753ee48ff3726d0606a81d0092d54acee995e",
        3.0,
    )

    assert result == {"status": "fail", "network_status": "http_302", "http_status": 302}


@pytest.mark.parametrize(
    ("probe_case", "expected"),
    [
        ("missing_curl", "missing_curl"),
        ("timeout", "timeout"),
        ("dns", "dns"),
        ("http_404", "http_404"),
    ],
)
def test_probe_curl_failures_are_missing_or_sanitized(
    environment_module, monkeypatch, probe_case, expected
):
    module = environment_module
    if probe_case == "missing_curl":
        monkeypatch.setattr(module.shutil, "which", lambda name: None)
    else:
        monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/curl")

        class Completed:
            returncode = 28 if probe_case == "timeout" else 6 if probe_case == "dns" else 0
            stdout = "404" if probe_case == "http_404" else ""
            stderr = SENTINEL

        monkeypatch.setattr(
            module.subprocess, "run", lambda *args, **kwargs: Completed()
        )

    result = module._probe_pinned_metadata(
        "Maple728/TimeMoE-50M",
        "446753ee48ff3726d0606a81d0092d54acee995e",
        3.0,
    )

    assert result["status"] == "fail"
    assert result["network_status"] == expected
    assert SENTINEL not in repr(result)


@pytest.mark.parametrize(
    ("probe_result", "expected"),
    [
        ({"status": "fail", "error": "timeout"}, "timeout"),
        ({"status": "fail", "error": "dns"}, "dns"),
        ({"status": "fail", "error": "http_404", "http_status": 404}, "http_404"),
    ],
)
def test_cache_miss_network_failures_are_sanitized(
    environment_module, monkeypatch, tmp_path, probe_result, expected
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path, cache_complete=False)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    monkeypatch.setattr(module, "_probe_pinned_metadata", lambda *args: probe_result)

    report = module.collect_environment()

    assert report["ok"] is False
    assert [item["status"] for item in report["models"][:2]] == ["download_required", "download_required"]
    assert all(item["network_status"] == expected for item in report["models"][:2])
    assert all(item["status"] == "blocked" for item in report["models"][2:])
    assert all("error" not in item for item in report["models"])


def test_offline_cache_miss_never_calls_network(environment_module, monkeypatch, tmp_path):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path, cache_complete=False)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    called = []
    monkeypatch.setattr(module, "_probe_pinned_metadata", lambda *args: called.append(args))

    report = module.collect_environment(offline=True)

    assert report["ok"] is False
    assert called == []
    assert all(item["network_attempted"] is False for item in report["models"])
    assert all(item["network_status"] == "offline" for item in report["models"][:2])
    assert all(item["network_status"] == "python_blocked" for item in report["models"][2:])


@pytest.mark.parametrize("timeout", [0, -1, 10.0001, float("nan"), float("inf")])
def test_collect_environment_rejects_invalid_timeout_values(environment_module, timeout):
    with pytest.raises(ValueError):
        environment_module.collect_environment(network_timeout=timeout)


def test_credential_and_proxy_sentinels_never_reach_text_json_or_cli_streams(
    environment_module, monkeypatch, tmp_path, capsys
):
    module = environment_module
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", SENTINEL)
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
    monkeypatch.setenv("HTTPS_PROXY", "https://user:" + SENTINEL + "@proxy.invalid")
    monkeypatch.setenv("HTTP_PROXY", "http://user:" + SENTINEL + "@proxy.invalid")
    monkeypatch.setattr(module, "_package_version", lambda _name: (_ for _ in ()).throw(RuntimeError(SENTINEL)))
    monkeypatch.setattr(module, "_import_package", lambda _name: (_ for _ in ()).throw(ImportError(SENTINEL)))
    monkeypatch.setattr(module, "_effective_dataset_paths", lambda: (_ for _ in ()).throw(RuntimeError(SENTINEL)))
    monkeypatch.setattr(module, "_effective_cache_root", lambda: tmp_path / "cache")
    probe_calls = []

    def credential_probe(*args):
        probe_calls.append(args)
        return {"status": "fail", "error": SENTINEL}

    monkeypatch.setattr(module, "_probe_pinned_metadata", credential_probe)

    report = module.collect_environment()
    text = module.render_text(report)
    rendered = module.render_json(report)
    assert module.main(["--json", "--offline"]) == 1
    streams = capsys.readouterr()

    assert len(probe_calls) == 2
    for value in (text, rendered, streams.out, streams.err):
        assert SENTINEL not in value
        assert "RuntimeError" not in value
        assert "ImportError" not in value


def test_cli_exit_codes_for_invalid_timeout_failed_report_and_passed_report(
    environment_module, monkeypatch, capsys
):
    module = environment_module
    with pytest.raises(SystemExit) as error:
        module.main(["--network-timeout", "0"])
    assert error.value.code == 2
    capsys.readouterr()

    monkeypatch.setattr(module, "collect_environment", lambda **_kwargs: {"ok": False})
    assert module.main([]) == 1
    capsys.readouterr()
    monkeypatch.setattr(module, "collect_environment", lambda **_kwargs: {"ok": True})
    assert module.main(["--json"]) == 0
    assert capsys.readouterr().err == ""


def test_modern_models_are_python_blocked_without_optional_imports(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    optional_modules = (
        "chronos",
        "tirex",
        "models.Chronos2",
        "models.TiRex",
        "models.TimesFM",
        "transformers.models.timesfm2_5",
    )
    for name in optional_modules:
        monkeypatch.delitem(sys.modules, name, raising=False)
    imported_names = []
    original_import = module._import_package

    def guarded_import(name):
        imported_names.append(name)
        if name in {"chronos", "tirex"}:
            raise AssertionError("optional model package imported")
        return original_import(name)

    monkeypatch.setattr(module, "_import_package", guarded_import)
    report = module.collect_environment(offline=True)

    by_name = {item["name"]: item for item in report["models"]}
    assert [by_name[name]["status"] for name in ("Chronos2", "TiRex", "TimesFM")] == [
        "blocked", "blocked", "blocked"
    ]
    assert all(by_name[name]["python_status"] == "blocked" for name in ("Chronos2", "TiRex", "TimesFM"))
    assert by_name["Chronos2"]["package_status"] == "missing"
    assert by_name["TiRex"]["package_status"] == "missing"
    assert by_name["TimesFM"]["package_status"] == "incompatible"
    assert imported_names == ["torch", "transformers", "peft"]
    assert all(name not in sys.modules for name in optional_modules)


def test_cache_head_reachability_does_not_promote_missing_files_to_pass(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path, cache_complete=False)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.8.18")
    monkeypatch.setattr(module.sys, "executable", "/opt/data/private/penv/time/bin/python")
    monkeypatch.setattr(
        module,
        "_probe_pinned_metadata",
        lambda *_args: {"status": "pass", "http_status": 200},
    )

    report = module.collect_environment()

    assert all(item["status"] == "download_required" for item in report["models"][:2])
    assert all(item["network_status"] == "pass" for item in report["models"][:2])
    assert all(item["download_required"] is True for item in report["models"])


@pytest.mark.parametrize(
    ("model_name", "package_name", "version", "expected_status", "expected_package_status"),
    [
        ("Chronos2", "chronos-forecasting", "1.9.0", "incompatible", "incompatible"),
        ("Chronos2", "chronos-forecasting", "2.0.0", "pass", "pass"),
        ("TiRex", "tirex-ts", "0.9.0", "incompatible", "incompatible"),
        ("TiRex", "tirex-ts", "1.0.0", "pass", "pass"),
        ("TimesFM", "transformers", "5.2.0", "incompatible", "incompatible"),
        ("TimesFM", "transformers", "5.3.0", "pass", "pass"),
    ],
)
def test_modern_package_floors_are_checked_by_metadata_on_compatible_python(
    environment_module,
    monkeypatch,
    tmp_path,
    model_name,
    package_name,
    version,
    expected_status,
    expected_package_status,
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.11.0")
    monkeypatch.setattr(module.sys, "executable", module.EXPECTED_INTERPRETER)
    versions = {
        "torch": "2.3.1+cu118",
        "transformers": "4.46.2",
        "peft": "0.13.2",
        "chronos-forecasting": "2.0.0",
        "tirex-ts": "1.0.0",
    }
    versions[package_name] = version
    monkeypatch.setattr(module, "_package_version", lambda name: versions[name])

    report = module.collect_environment(offline=True)
    item = next(item for item in report["models"] if item["name"] == model_name)
    assert item["status"] == expected_status
    assert item["package_status"] == expected_package_status


def test_modern_python_selects_modern_base_dependency_profile(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.11.0")
    monkeypatch.setattr(module.sys, "executable", module.EXPECTED_INTERPRETER)
    versions = {
        "torch": "2.4.1",
        "transformers": "5.3.0",
        "peft": "0.18.1",
    }
    monkeypatch.setattr(module, "_package_version", lambda name: versions[name])

    report = module.collect_environment(offline=True)

    assert [item["expected"] for item in report["packages"]] == [
        ">=2.4,<3", ">=5.3,<6", ">=0.18.1,<1"
    ]
    assert [item["status"] for item in report["packages"]] == ["pass", "pass", "pass"]
    assert report["interpreter"]["status"] == "pass"


def test_modern_peft_floor_rejects_legacy_release_on_modern_python(
    environment_module, monkeypatch, tmp_path
):
    module = environment_module
    _install_happy_fakes(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_runtime_python_version", lambda: "3.11.0")
    monkeypatch.setattr(module.sys, "executable", module.EXPECTED_INTERPRETER)
    versions = {
        "torch": "2.4.1",
        "transformers": "5.3.0",
        "peft": "0.18.0",
    }
    monkeypatch.setattr(module, "_package_version", lambda name: versions[name])

    report = module.collect_environment(offline=True)

    peft = next(item for item in report["packages"] if item["name"] == "peft")
    assert peft["expected"] == ">=0.18.1,<1"
    assert peft["status"] == "fail"
    assert peft["version_status"] == "fail"
    assert report["ok"] is False
