from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, is_dataclass

import pytest


def test_registry_exposes_fixed_catalog_and_immutable_specs():
    from foundation_models import MODEL_NAMES, RUN_MODES, get_model_spec

    assert MODEL_NAMES == ("Sundial", "TimeMoE")
    assert RUN_MODES == ("zero_shot", "adapter", "full", "last_layer")

    expected = {
        "Sundial": {
            "model_id": "thuml/sundial-base-128m",
            "revision": "3212e42564493f520593e5414af4367fc4b49226",
            "entrypoint": "foundation_models.backends:SundialBackend",
            "last_layer_selector": "flow_loss",
        },
        "TimeMoE": {
            "model_id": "Maple728/TimeMoE-50M",
            "revision": "446753ee48ff3726d0606a81d0092d54acee995e",
            "entrypoint": "foundation_models.backends:TimeMoEBackend",
            "last_layer_selector": "lm_heads",
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
        "import foundation_models.registry\n"
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
    from foundation_models import get_model_spec

    with pytest.raises(ValueError, match="Sundial.*TimeMoE"):
        get_model_spec("unknown")
