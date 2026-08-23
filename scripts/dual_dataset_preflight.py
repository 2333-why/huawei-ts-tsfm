#!/usr/bin/env python3
"""Run no-training checks for the Luoyang and YLJ FACTS pipelines.

This is deliberately safe to run before a long GPU job: it compiles the
project, runs synthetic data/shape tests, and validates both generated command
lines without opening either production Parquet file or changing checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def validate_production_data(datasets=("luoyang", "ylj")) -> None:
    from data_provider.data_loader_luoyang import LuoyangParquetDataset
    from data_provider.data_loader_ylj import YLJParquetDataset

    specifications = (
        (
            "luoyang", LuoyangParquetDataset,
            ROOT / "configs" / "luoyang_parquet.json", (16, 9), (48, 1),
        ),
        (
            "ylj", YLJParquetDataset,
            ROOT / "configs" / "datasets" / "ylj.yaml", (16, 22), (16, 1),
        ),
    )
    for name, dataset_class, config_path, x_shape, y_shape in specifications:
        if name not in datasets:
            continue
        split_sizes = {}
        for flag in ("train", "val", "test"):
            kwargs = ({
                "load_images": True,
                "privileged_teacher": flag == "train",
            } if name == "luoyang" else {
                "privileged_teacher": flag == "train",
            })
            dataset = dataset_class(config_path=config_path, flag=flag, **kwargs)
            if len(dataset) == 0:
                raise ValueError(f"{name} {flag} split is empty")
            split_sizes[flag] = len(dataset)
            if flag == "test":
                sample = dataset[0]
                if sample[0].shape != x_shape or sample[1].shape != y_shape:
                    raise ValueError(
                        f"{name} shape mismatch: x={sample[0].shape}, "
                        f"y={sample[1].shape}"
                    )
                if not np.isfinite(sample[0]).all() or not np.isfinite(sample[1]).all():
                    raise ValueError(f"{name} sample contains non-finite values")
                metadata = sample[-1]
                if name == "luoyang":
                    offsets = metadata["image_minute_offsets"][metadata["image_mask"]]
                    if offsets.size and float(offsets.max()) > 0:
                        raise ValueError("luoyang sample contains a future image")
            if flag == "train":
                indices = np.linspace(
                    0, len(dataset) - 1, min(64, len(dataset)), dtype=int)
                future_timeseries_found = False
                future_image_found = name != "luoyang"
                for index in np.unique(indices):
                    privileged = dataset[int(index)]
                    if not privileged[-1]["privileged_teacher_enabled"]:
                        raise ValueError(
                            f"{name} Teacher privileged inputs are disabled")
                    if not np.isfinite(privileged[7]).all():
                        raise ValueError(
                            f"{name} Teacher future time series are non-finite")
                    future_timeseries_found |= not np.allclose(privileged[7], 0.0)
                    if name == "luoyang":
                        future_image_found |= bool(
                            privileged[-1]["teacher_future_image_mask"].any())
                    if future_timeseries_found and future_image_found:
                        break
                if not future_timeseries_found:
                    raise ValueError(
                        f"{name} sampled Teacher future time series are all zero")
                if not future_image_found:
                    raise ValueError(
                        "luoyang sampled Teacher windows contain no readable future image")
        print(f"[production-valid] {name} splits={split_sizes}", flush=True)


def validate_training_runtime(datasets=("luoyang", "ylj")) -> None:
    import json
    import torch
    from data_provider.data_loader_ylj import load_ylj_config

    luoyang = json.loads(
        (ROOT / "configs" / "luoyang_parquet.json").read_text(encoding="utf-8"))
    ylj = load_ylj_config(ROOT / "configs" / "datasets" / "ylj.yaml")
    configurations = {"luoyang": luoyang, "ylj": ylj}
    selected = [configurations[name] for name in datasets]
    configured = [int(value) for value in str(
        selected[0]["runtime"]["gpu_devices"]).split(",")]
    if any(str(config["runtime"]["gpu_devices"]) != str(
           selected[0]["runtime"]["gpu_devices"]) for config in selected[1:]):
        raise ValueError("Luoyang and YLJ gpu_devices differ")
    if len(configured) != 8 or len(set(configured)) != 8:
        raise ValueError(f"Exactly eight unique configured GPUs are required: {configured}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active Python environment")
    visible = torch.cuda.device_count()
    if visible < len(configured):
        raise RuntimeError(
            f"Only {visible} CUDA devices are visible, but configuration requires "
            f"{len(configured)}")
    for dataset in datasets:
        config = configurations[dataset]
        runtime = config["runtime"]
        for key in ("teacher_batch_size", "student_batch_size"):
            if int(runtime[key]) < len(configured):
                raise ValueError(
                    f"{dataset} {key}={runtime[key]} cannot feed all eight DataParallel GPUs")
    names = [torch.cuda.get_device_name(index) for index in range(len(configured))]
    print(
        f"[runtime-valid] cuda={torch.version.cuda} torch={torch.__version__} "
        f"visible_gpus={visible} configured={configured} names={names}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-pytest", action="store_true",
        help="only compile and validate commands; do not run synthetic tests",
    )
    parser.add_argument(
        "--production-data", action="store_true",
        help="open configured Parquet files and validate train/val/test samples",
    )
    parser.add_argument(
        "--training-runtime", action="store_true",
        help="require CUDA, eight visible GPUs and compatible global batch sizes",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=("luoyang", "ylj"),
        default=["luoyang", "ylj"], help="datasets covered by this preflight",
    )
    args = parser.parse_args()
    run([sys.executable, "-m", "compileall", "-q", "data_provider", "exp", "models", "scripts", "utils", "tests"])
    if not args.skip_pytest:
        test_files = [f"tests/test_{dataset}_parquet.py" for dataset in args.datasets]
        run([sys.executable, "-m", "pytest", "-q", *test_files])
    for dataset in args.datasets:
        pipeline = f"scripts/{dataset}_pipeline.py"
        run([sys.executable, pipeline, "validate-config"])
        run([sys.executable, pipeline, "dry-run"])
    if args.production_data:
        validate_production_data(args.datasets)
    if args.training_runtime:
        validate_training_runtime(args.datasets)
    print(
        "Preflight passed: selected dataset interfaces, shapes and "
        "train/test commands are valid."
    )


if __name__ == "__main__":
    main()
