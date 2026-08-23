#!/usr/bin/env python
"""Run one deterministic 16-model CUDA shard of the Task7 matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Keep direct ``python scripts/run_tslib_model_matrix.py`` invocation rooted at
# the repository, just like ``python -m scripts.run_tslib_model_matrix``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.tslib_registry import SELECTED_MODEL_NAMES
from scripts.tslib_model_matrix import run_matrix


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    names = SELECTED_MODEL_NAMES[args.shard * 16 : (args.shard + 1) * 16]
    report = run_matrix(
        names,
        device=torch.device("cuda:0"),
        physical_gpu=args.physical_gpu,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
