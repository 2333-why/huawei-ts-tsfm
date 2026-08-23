#!/usr/bin/env python3
"""Sequentially train then test FACTS on Luoyang followed by YLJ.

Each dataset runs Teacher -> Student/KD -> Student test in that order.  A
failure stops the sequence, preserving the first failing command and log.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PIPELINES = {
    "luoyang": "scripts/luoyang_pipeline.py",
    "ylj": "scripts/ylj_pipeline.py",
}


def run(command: list[str]) -> None:
    print("\n+", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=ROOT, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(PIPELINES),
                        default=["luoyang", "ylj"], help="execution order")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--skip-production-data-check", action="store_true",
        help="skip opening the configured production Parquet datasets",
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="run only the final Student test for each dataset")
    parser.add_argument("--skip-test", action="store_true",
                        help="train Teacher and Student/KD only")
    parser.add_argument(
        "--after-luoyang-teacher", action="store_true",
        help=(
            "require and reuse the newest contract-matching Luoyang Teacher "
            "checkpoint, then run Luoyang Student/test and the complete YLJ pipeline"
        ),
    )
    args = parser.parse_args()

    if not args.skip_preflight:
        preflight = [sys.executable, "-u", "scripts/dual_dataset_preflight.py"]
        if not args.skip_production_data_check:
            preflight.append("--production-data")
        preflight.append("--training-runtime")
        preflight.extend(["--datasets", *args.datasets])
        run(preflight)
    for dataset in args.datasets:
        pipeline = PIPELINES[dataset]
        print(f"\n===== {dataset.upper()} =====", flush=True)
        if not args.skip_train:
            skip_completed_teacher = (
                args.after_luoyang_teacher and dataset == "luoyang"
            )
            if skip_completed_teacher:
                print(
                    "[skip] LUOYANG Teacher: reusing the newest "
                    "contract-matching checkpoint",
                    flush=True,
                )
            else:
                run([sys.executable, "-u", pipeline, "teacher"])
            run([
                sys.executable, "-u", pipeline, "student",
                "--prefer-latest-checkpoint",
            ])
        if not args.skip_test:
            run([
                sys.executable, "-u", pipeline, "test",
                "--prefer-latest-checkpoint",
            ])
            run([
                sys.executable, "-u", "scripts/verify_official_outputs.py", dataset,
            ])
    print("\nAll requested dataset runs completed successfully.")


if __name__ == "__main__":
    main()
