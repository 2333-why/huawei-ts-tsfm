#!/usr/bin/env python3
"""Download every pinned TSFM checkpoint into the shared repository cache."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models.registry import MODEL_NAMES, get_model_spec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HF_HOME directory (default: $HF_HOME or <repo>/checkpoints_huggingface)",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=MODEL_NAMES,
        dest="models",
        help="download only this model; repeat for multiple models",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = args.cache_dir or Path(
        os.environ.get("HF_HOME", ROOT_DIR / "checkpoints_huggingface")
    )
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is missing; run: python -m pip install -r requirements-modern.txt"
        ) from exc

    selected = tuple(args.models or MODEL_NAMES)
    print(f"HF_HOME={cache_dir}", flush=True)
    for name in selected:
        spec = get_model_spec(name)
        print(
            f"Downloading {name}: {spec.model_id}@{spec.revision}",
            flush=True,
        )
        snapshot_path = snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
        )
        print(f"Ready {name}: {snapshot_path}", flush=True)

    print(f"Downloaded {len(selected)} pinned foundation-model checkpoint(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
