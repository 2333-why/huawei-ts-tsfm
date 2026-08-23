"""Copy selected Time-Series-Library models and their isolated layers.

The source checkout is treated as read-only.  Copied modules use an isolated
``layers.tslib`` package so the project's existing ``layers`` package remains
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tslib_registry import SELECTED_MODEL_NAMES


SOURCE_ROOT = Path("/opt/data/private/code/Time-Series-Library")
PRESERVE_EXISTING = {"DLinear", "PatchTST", "TimesNet", "iTransformer"}
EXCLUDED = {
    "Chronos", "Chronos2", "KANAD", "Sundial", "TimeMoE", "TiRex", "TimesFM",
}


def _read_utf8(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_utf8(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _rewrite_layer_imports(text: str) -> str:
    return text.replace("from layers.", "from layers.tslib.").replace(
        "import layers.", "import layers.tslib."
    )


def _source_model_paths(source_root: Path) -> list[Path]:
    model_dir = source_root / "models"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Source model directory does not exist: {model_dir}")
    return sorted(path for path in model_dir.glob("*.py") if path.name != "__init__.py")


def _validate_source_inventory(source_root: Path) -> list[Path]:
    model_paths = _source_model_paths(source_root)
    source_names = {path.stem for path in model_paths}
    selected_names = set(SELECTED_MODEL_NAMES)
    expected_names = selected_names | EXCLUDED
    if source_names != expected_names:
        missing = sorted(expected_names - source_names)
        unexpected = sorted(source_names - expected_names)
        raise ValueError(
            "Source model inventory differs from the registry: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    overlap = selected_names & EXCLUDED
    if overlap:
        raise ValueError(f"Selected and excluded model names overlap: {sorted(overlap)!r}")
    unknown_preserved = PRESERVE_EXISTING - selected_names
    if unknown_preserved:
        raise ValueError(
            "Preserved models are not selected by the registry: "
            f"{sorted(unknown_preserved)!r}"
        )
    return model_paths


def _ensure_disjoint_paths(source_root: Path, destination_root: Path) -> None:
    source_path = source_root.resolve()
    destination_path = destination_root.resolve()
    if (
        source_path == destination_path
        or source_path in destination_path.parents
        or destination_path in source_path.parents
    ):
        raise ValueError(
            "Source and destination paths overlap: "
            f"source={source_path}, destination={destination_path}"
        )


def sync_models(source_root: Path, destination_root: Path) -> list[Path]:
    """Synchronize selected model and layer source text into the project."""

    source_root = Path(source_root)
    destination_root = Path(destination_root)
    _ensure_disjoint_paths(source_root, destination_root)
    model_paths = _validate_source_inventory(source_root)

    source_layers_dir = source_root / "layers"
    if not source_layers_dir.is_dir():
        raise FileNotFoundError(f"Source layer directory does not exist: {source_layers_dir}")
    layer_paths = sorted(source_layers_dir.glob("*.py"))

    written: list[Path] = []
    destination_layers_dir = destination_root / "layers" / "tslib"
    for source_path in layer_paths:
        destination_path = destination_layers_dir / source_path.name
        _write_utf8(destination_path, _rewrite_layer_imports(_read_utf8(source_path)))
        written.append(destination_path)

    destination_models_dir = destination_root / "models"
    for source_path in model_paths:
        model_name = source_path.stem
        if model_name not in SELECTED_MODEL_NAMES or model_name in PRESERVE_EXISTING:
            continue
        destination_path = destination_models_dir / source_path.name
        _write_utf8(destination_path, _rewrite_layer_imports(_read_utf8(source_path)))
        written.append(destination_path)
    return written


def main() -> None:
    for path in sync_models(SOURCE_ROOT, PROJECT_ROOT):
        print(path)


if __name__ == "__main__":
    main()
