#!/usr/bin/env python3
"""Download pinned model weights and build a portable HF cache archive."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import tarfile
from pathlib import Path

from download_foundation_weights import main as download_weights


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT_DIR / "checkpoints_huggingface"
DEFAULT_ARCHIVE = ROOT_DIR / "huawei-ts-tsfm-hf-cache.tar.gz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="HF_HOME to download and archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="output .tar.gz path",
    )
    parser.add_argument(
        "--no-token-prompt",
        action="store_true",
        help="do not prompt when HF_TOKEN is absent",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    archive = args.output.expanduser().resolve()
    checksum = archive.with_name(archive.name + ".sha256")

    if archive.suffixes[-2:] != [".tar", ".gz"]:
        raise SystemExit("--output must end with .tar.gz")
    try:
        archive.relative_to(cache_dir)
    except ValueError:
        pass
    else:
        raise SystemExit("--output must be outside --cache-dir")

    prompted_token = False
    if not os.environ.get("HF_TOKEN") and not args.no_token_prompt:
        token = getpass.getpass("Hugging Face read token (input hidden): ").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            prompted_token = True

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    try:
        result = download_weights(["--cache-dir", str(cache_dir)])
    finally:
        if prompted_token:
            os.environ.pop("HF_TOKEN", None)
    if result:
        return result

    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating portable archive: {archive}", flush=True)
    with tarfile.open(
        archive,
        mode="w:gz",
        compresslevel=1,
        dereference=False,
    ) as bundle:
        bundle.add(cache_dir, arcname="checkpoints_huggingface", recursive=True)

    digest = _sha256(archive)
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(f"Archive ready: {archive}", flush=True)
    print(f"SHA256 ready: {checksum}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
