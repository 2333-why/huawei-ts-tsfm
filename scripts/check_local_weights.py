#!/usr/bin/env python3
"""Validate flat model folders downloaded in a browser and uploaded offline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.local_weights import (  # noqa: E402
    MODEL_WEIGHT_ENVS,
    WEIGHTS_ROOT_ENV,
    inspect_local_weights,
)
from models.registry import MODEL_NAMES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check browser-downloaded TSFM files without loading a model"
    )
    parser.add_argument(
        "--weights-root",
        help="root containing one subfolder for every selected registry model",
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="hash the selected large files (slower, recommended after bucket upload)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.weights_root:
        os.environ[WEIGHTS_ROOT_ENV] = str(Path(args.weights_root).expanduser())

    reports = [
        inspect_local_weights(name, verify_sha256=args.verify_sha256)
        for name in args.models
    ]
    ok = all(bool(item["complete"]) for item in reports)
    report = {
        "ok": ok,
        "weights_root_environment": WEIGHTS_ROOT_ENV,
        "model_environments": MODEL_WEIGHT_ENVS,
        "models": reports,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("local TSFM weights: {}".format("PASS" if ok else "FAIL"))
        for item in reports:
            details = []
            if item["missing_files"]:
                details.append("missing=" + ",".join(item["missing_files"]))
            if not item["size_ok"]:
                details.append(
                    "size={}/{}".format(item["actual_size"], item["expected_size"])
                )
            if item["checksum_ok"] is False:
                details.append("sha256=mismatch")
            suffix = "" if not details else " (" + "; ".join(details) + ")"
            print(
                "{}: {} directory={}{}".format(
                    item["name"],
                    "PASS" if item["complete"] else "FAIL",
                    item["directory"] or "not configured",
                    suffix,
                )
            )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
