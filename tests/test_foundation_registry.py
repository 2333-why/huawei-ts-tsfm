from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, is_dataclass

import pytest


def test_registry_exposes_fixed_catalog_and_immutable_specs():
    from models import MODEL_NAMES, RUN_MODES, get_model_spec

    assert MODEL_NAMES == ("Sundial", "TimeMoE", "Chronos2", "TiRex", "TimesFM")
    assert RUN_MODES == ("zero_shot", "adapter", "full", "last_layer")

    expected = {
        "Sundial": {
            "model_id": "thuml/sundial-base-128m",
            "revision": "3212e42564493f520593e5414af4367fc4b49226",
            "entrypoint": "models.Sundial:SundialBackend",
            "last_layer_selector": "flow_loss",
        },
        "TimeMoE": {
            "model_id": "Maple728/TimeMoE-50M",
            "revision": "446753ee48ff3726d0606a81d0092d54acee995e",
            "entrypoint": "models.TimeMoE:TimeMoEBackend",
            "last_layer_selector": "lm_heads",
        },
        "Chronos2": {
            "model_id": "amazon/chronos-2",
            "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
            "entrypoint": "models.Chronos2:Chronos2Backend",
            "last_layer_selector": "output_patch_embedding",
            "supported_modes": ("zero_shot", "adapter", "full", "last_layer"),
            "adapter_target_modules": (
                "self_attention.q",
                "self_attention.v",
                "self_attention.k",
                "self_attention.o",
                "output_patch_embedding.output_layer",
            ),
        },
        "TiRex": {
            "model_id": "NX-AI/TiRex",
            "revision": "63c740922493f5fbe60b277609ec62babfba2762",
            "entrypoint": "models.TiRex:TiRexBackend",
            "last_layer_selector": None,
            "supported_modes": ("zero_shot",),
        },
        "TimesFM": {
            "model_id": "google/timesfm-2.5-200m-transformers",
            "revision": "5a9806b9b291fad9233b5249d88263f1846304d3",
            "entrypoint": "models.TimesFM:TimesFMBackend",
            "last_layer_selector": "output_projection_point",
            "supported_modes": ("zero_shot", "adapter", "full", "last_layer"),
            "adapter_target_modules": ("all-linear",),
        },
    }

    for name, values in expected.items():
        spec = get_model_spec(name)
        assert is_dataclass(spec)
        assert spec.name == name
        for field, value in values.items():
            assert getattr(spec, field) == value

        with pytest.raises(FrozenInstanceError):
            spec.revision = "changed"


def test_registry_import_has_no_optional_dependency_side_effect():
    probe = (
        "import sys\n"
        "import models.registry\n"
        "assert 'transformers' not in sys.modules\n"
        "assert 'peft' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_unknown_model_name_reports_available_catalog():
    from models import get_model_spec

    with pytest.raises(ValueError, match="Sundial.*TimeMoE.*Chronos2.*TiRex.*TimesFM"):
        get_model_spec("unknown")


@pytest.mark.parametrize(
    ("model", "modes"),
    [
        ("Sundial", ("zero_shot", "adapter", "full", "last_layer")),
        ("TimeMoE", ("zero_shot", "adapter", "full", "last_layer")),
        ("Chronos2", ("zero_shot", "adapter", "full", "last_layer")),
        ("TiRex", ("zero_shot",)),
        ("TimesFM", ("zero_shot", "adapter", "full", "last_layer")),
    ],
)
def test_model_modes_are_capability_driven_and_invalid_modes_fail_early(model, modes):
    from models.registry import get_model_modes, validate_model_mode

    assert get_model_modes(model) == modes
    for mode in modes:
        assert validate_model_mode(model, mode) is None

    invalid = "full" if model == "TiRex" else "unsupported"
    with pytest.raises(ValueError, match="valid modes|zero_shot"):
        validate_model_mode(model, invalid)
