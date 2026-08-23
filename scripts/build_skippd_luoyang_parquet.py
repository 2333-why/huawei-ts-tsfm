#!/usr/bin/env python3
"""Convert the SKIPP'D trainval HDF5 arrays to Luoyang-compatible Parquet.

The source contains one image and one PV value per irregular timestamp.  This
builder snaps timestamps to minute boundaries, keeps only close observations,
deduplicates those canonical minutes, and emits complete right-closed
five-minute rows.  Images are read from HDF5 only for selected output rows and
written losslessly to PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_HDF5 = Path("/opt/data/private/data/Skippd/2017_2019_images_pv_processed.hdf5")
DEFAULT_TIMES = Path("/opt/data/private/data/Skippd/times_trainval.npy")
DEFAULT_OUTPUT_ROOT = Path("/opt/data/private/data/Skippd/luoyang")
SOURCE_GROUP = "trainval"
IMAGE_SHAPE = (64, 64, 3)
SNAP_INTERVAL = pd.Timedelta(minutes=1)
SNAP_TOLERANCE = pd.Timedelta(seconds=20)
OUTPUT_INTERVAL = pd.Timedelta(minutes=5)


def _parse_naive_times(values: np.ndarray) -> np.ndarray:
    """Return parsed timestamps as int64 nanoseconds, rejecting timezone data."""
    if values.ndim != 1:
        raise ValueError(f"times array must be one-dimensional, got shape {values.shape}")
    parsed: list[int] = []
    for index, value in enumerate(values.tolist()):
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"times[{index}] is not parseable: {value!r}") from exc
        if pd.isna(timestamp):
            raise ValueError(f"times[{index}] is missing or NaT")
        if timestamp.tz is not None:
            raise ValueError(f"times[{index}] must be naive, got timezone {timestamp.tz}")
        parsed.append(int(timestamp.value))
    return np.asarray(parsed, dtype=np.int64)


def _validate_source(hdf5_path: Path, times_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate the trusted local source and return raw timestamp/PV arrays."""
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 source not found: {hdf5_path}")
    if not times_path.is_file():
        raise FileNotFoundError(f"times source not found: {times_path}")
    try:
        times_values = np.load(times_path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"could not load times source {times_path}") from exc
    raw_ns = _parse_naive_times(np.asarray(times_values))

    with h5py.File(hdf5_path, "r") as handle:
        if SOURCE_GROUP not in handle:
            raise ValueError(f"HDF5 source is missing group {SOURCE_GROUP!r}")
        group = handle[SOURCE_GROUP]
        if "images_log" not in group or "pv_log" not in group:
            raise ValueError(f"HDF5 group {SOURCE_GROUP!r} must contain images_log and pv_log")
        images = group["images_log"]
        pv_dataset = group["pv_log"]
        if len(images) != len(raw_ns) or len(pv_dataset) != len(raw_ns):
            raise ValueError(
                "HDF5 dataset lengths must match times: "
                f"images={len(images)}, pv={len(pv_dataset)}, times={len(raw_ns)}"
            )
        if tuple(images.shape[1:]) != IMAGE_SHAPE:
            raise ValueError(f"images_log must have shape [N,64,64,3], got {images.shape}")
        if images.dtype != np.dtype(np.uint8):
            raise ValueError(f"images_log must be uint8, got {images.dtype}")
        pv = np.asarray(pv_dataset[:])
        if not np.issubdtype(pv.dtype, np.number):
            raise ValueError(f"pv_log must be numeric, got {pv.dtype}")
        pv = pv.astype(np.float64, copy=False)
        if not np.isfinite(pv).all() or (pv <= 0).any():
            raise ValueError("pv_log must contain finite positive values")
        source_meta = {
            "group": SOURCE_GROUP,
            "rows": int(len(raw_ns)),
            "image_shape": list(images.shape),
            "image_dtype": str(images.dtype),
            "pv_dtype": str(pv_dataset.dtype),
        }
    return raw_ns, pv, source_meta


def _snap_and_dedupe(raw_ns: np.ndarray, pv: np.ndarray) -> tuple[pd.DataFrame, dict[str, int]]:
    """Snap raw timestamps and collapse canonical-minute duplicates deterministically."""
    interval_ns = int(SNAP_INTERVAL.value)
    tolerance_ns = int(SNAP_TOLERANCE.value)
    # Half-up is deterministic at the otherwise ambiguous 30-second midpoint.
    canonical_ns = ((raw_ns + interval_ns // 2) // interval_ns) * interval_ns
    adjustment_ns = canonical_ns - raw_ns
    keep = np.abs(adjustment_ns) <= tolerance_ns
    raw_indices = np.arange(raw_ns.size, dtype=np.int64)
    candidates = pd.DataFrame(
        {
            "canonical_ns": canonical_ns[keep],
            "raw_ns": raw_ns[keep],
            "raw_index": raw_indices[keep],
            "adjustment_ns": adjustment_ns[keep],
            "pv": pv[keep],
        }
    )
    if candidates.empty:
        raise ValueError("no source rows remain after timestamp tolerance filtering")
    candidates["abs_adjustment_ns"] = candidates["adjustment_ns"].abs()
    candidates = candidates.sort_values(
        ["canonical_ns", "abs_adjustment_ns", "raw_ns", "raw_index"],
        kind="mergesort",
    )
    duplicate_group_count = int((candidates.groupby("canonical_ns", sort=False).size() > 1).sum())
    duplicate_rows = int(len(candidates) - candidates["canonical_ns"].nunique())
    # The sorted first row is also the deterministic image/raw-index choice.
    canonical = (
        candidates.groupby("canonical_ns", sort=True, as_index=False)
        .agg(
            pv=("pv", "mean"),
            raw_ns=("raw_ns", "first"),
            raw_index=("raw_index", "first"),
            adjustment_ns=("adjustment_ns", "first"),
            source_rows=("raw_index", "size"),
        )
        .rename(columns={"pv": "final_power"})
    )
    canonical["canonical_timestamp"] = pd.to_datetime(canonical["canonical_ns"], unit="ns")
    stats = {
        "dropped_by_tolerance_rows": int((~keep).sum()),
        "kept_after_tolerance_rows": int(keep.sum()),
        "duplicate_canonical_minutes": duplicate_group_count,
        "duplicate_rows_collapsed": duplicate_rows,
        "canonical_minute_rows": int(len(canonical)),
    }
    return canonical, stats


def _aggregate_bins(canonical: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aggregate canonical minutes into complete right-labelled/right-closed bins."""
    interval_ns = int(OUTPUT_INTERVAL.value)
    canonical = canonical.sort_values("canonical_ns", kind="mergesort").reset_index(drop=True)
    # ceil-to-grid labels the five-minute interval by its right edge.  A row
    # already on the grid remains in that same right-closed bin.
    canonical["bin_ns"] = (
        (canonical["canonical_ns"].to_numpy(np.int64) + interval_ns - 1) // interval_ns
    ) * interval_ns
    grouped = canonical.groupby("bin_ns", sort=True)
    complete_bins: list[dict[str, Any]] = []
    incomplete_bins = 0
    for bin_ns, group in grouped:
        if len(group) != 5:
            incomplete_bins += 1
            continue
        last = group.iloc[-1]
        complete_bins.append(
            {
                "timestamp": pd.Timestamp(int(bin_ns)),
                "timestamp_ns": int(bin_ns),
                "final_power": float(group["final_power"].mean()),
                "image_timestamp": pd.Timestamp(int(last["canonical_ns"])),
                "image_timestamp_ns": int(last["canonical_ns"]),
                "raw_index": int(last["raw_index"]),
            }
        )
    output = pd.DataFrame(complete_bins)
    if output.empty:
        raise ValueError("timestamp aggregation produced no complete five-minute bins")
    output = output.sort_values("timestamp_ns", kind="mergesort").drop_duplicates(
        "timestamp_ns", keep="first"
    ).reset_index(drop=True)
    timestamp_ns = output["timestamp_ns"].to_numpy(np.int64)
    ramp = np.full(len(output), np.nan, dtype=np.float64)
    if len(output) > 1:
        previous = output["final_power"].to_numpy(np.float64)[:-1]
        contiguous = np.diff(timestamp_ns) == interval_ns
        values = output["final_power"].to_numpy(np.float64)
        ramp[1:] = np.where(contiguous, values[1:] - previous, np.nan)
    output["pv_ramp_5min"] = ramp
    stats = {
        "candidate_bins": int(len(grouped)),
        "incomplete_bins_dropped": int(incomplete_bins),
        "complete_bins": int(len(output)),
        "gap_count": int(np.sum(np.diff(timestamp_ns) != interval_ns)),
        "missing_ramp_count": int(np.isnan(ramp).sum()),
    }
    return output, stats


def _valid_existing_image(path: Path) -> bool:
    """Check that an existing image is a readable, exact-size RGB image."""
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == IMAGE_SHAPE[:2] and image.mode == "RGB"
    except (OSError, ValueError):
        return False


def _write_selected_images(
    output_root: Path,
    output: pd.DataFrame,
    hdf5_path: Path,
    chunk_span: int = 512,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Write selected HDF images without loading the full image dataset."""
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    relative_paths: list[str] = []
    timestamp_values: list[str] = []
    requests: list[tuple[int, Path]] = []
    for row in output.itertuples(index=False):
        image_timestamp = pd.Timestamp(row.image_timestamp)
        row_timestamp = pd.Timestamp(row.timestamp)
        if image_timestamp > row_timestamp:
            raise ValueError("selected image timestamp is after its Parquet row timestamp")
        relative = (
            image_timestamp.strftime("%Y/%m/%d/%Y%m%d_%H%M%S_")
            + f"{int(row.raw_index)}.png"
        )
        path = image_root / relative
        relative_paths.append(relative)
        timestamp_values.append(image_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        if not _valid_existing_image(path):
            requests.append((int(row.raw_index), path))

    # Reading a bounded HDF5 slice amortizes filesystem seeks while keeping
    # memory proportional to a few hundred 64x64 RGB images.
    requests.sort(key=lambda pair: pair[0])
    with h5py.File(hdf5_path, "r") as handle:
        images = handle[SOURCE_GROUP]["images_log"]
        position = 0
        while position < len(requests):
            start = requests[position][0]
            end_position = position + 1
            while end_position < len(requests):
                candidate = requests[end_position][0]
                if candidate - start >= chunk_span:
                    break
                end_position += 1
            end = requests[end_position - 1][0]
            block = np.asarray(images[start : end + 1])
            for raw_index, path in requests[position:end_position]:
                array = np.asarray(block[raw_index - start], dtype=np.uint8)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array, mode="RGB").save(path, format="PNG")
            position = end_position

    bytes_written = 0
    for relative in relative_paths:
        path = image_root / relative
        if not _valid_existing_image(path):
            raise ValueError(f"image export failed validation: {path}")
        bytes_written += path.stat().st_size
    return (
        [json.dumps([relative], separators=(",", ":")) for relative in relative_paths],
        [json.dumps([stamp], separators=(",", ":")) for stamp in timestamp_values],
        {"image_count": int(len(relative_paths)), "image_bytes": int(bytes_written), "images_rewritten": int(len(requests))},
    )


def _timestamp_text(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _write_manifest(
    output_root: Path,
    hdf5_path: Path,
    times_path: Path,
    source_meta: dict[str, Any],
    filter_stats: dict[str, int],
    aggregation_stats: dict[str, int],
    output: pd.DataFrame,
    image_stats: dict[str, int],
) -> dict[str, Any]:
    timestamps = pd.to_datetime(output["timestamp"])
    split_boundaries = {
        "train_start_timestamp": "2017-03-09 06:55:00",
        "test_start_timestamp": "2019-08-01 00:00:00",
        "test_end_exclusive_timestamp": "2019-10-26 18:05:00",
    }
    counts = {
        "source_rows": int(source_meta["rows"]),
        "dropped_by_tolerance": int(filter_stats["dropped_by_tolerance_rows"]),
        "kept_after_tolerance": int(filter_stats["kept_after_tolerance_rows"]),
        "duplicate_canonical_minutes": int(filter_stats["duplicate_canonical_minutes"]),
        "duplicate_rows_collapsed": int(filter_stats["duplicate_rows_collapsed"]),
        "canonical_minute_rows": int(filter_stats["canonical_minute_rows"]),
        "candidate_bins": int(aggregation_stats["candidate_bins"]),
        "incomplete_bins_dropped": int(aggregation_stats["incomplete_bins_dropped"]),
        "complete_bins": int(aggregation_stats["complete_bins"]),
        "output_rows": int(len(output)),
        "image_count": int(image_stats["image_count"]),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "SkippdLuoyangParquet",
        "provenance": {
            "source_hdf5": str(hdf5_path.resolve()),
            "source_times": str(times_path.resolve()),
            "source_group": SOURCE_GROUP,
            "capacity_kw": 30.1,
        },
        "source": source_meta,
        "filter": {
            "snap_interval_minutes": 1,
            "snap_tolerance_seconds": 20,
            "timestamp_policy": "nearest minute, half-up at 30 seconds; keep absolute adjustment <=20 seconds",
            **filter_stats,
        },
        "aggregation": {
            "interval_minutes": 5,
            "label": "right",
            "closed": "right",
            "complete_bin_rule": "exactly five distinct canonical minute rows",
            **aggregation_stats,
        },
        "output": {
            "parquet_file": str((output_root / "skippd_luoyang.parquet").resolve()),
            "columns": ["timestamp", "final_power", "pv_ramp_5min", "asi_path", "asi_path_timestamps"],
            "rows": int(len(output)),
            "missing_ramp_count": int(aggregation_stats["missing_ramp_count"]),
            "gap_count": int(aggregation_stats["gap_count"]),
            "image_count": int(image_stats["image_count"]),
            "image_bytes": int(image_stats["image_bytes"]),
            "images_rewritten": int(image_stats["images_rewritten"]),
            "time_range": {
                "start": _timestamp_text(timestamps.iloc[0]),
                "end": _timestamp_text(timestamps.iloc[-1]),
            },
        },
        "counts": counts,
        "dropped_by_tolerance": int(filter_stats["dropped_by_tolerance_rows"]),
        "duplicate_rows": int(filter_stats["duplicate_rows_collapsed"]),
        "gap_count": int(aggregation_stats["gap_count"]),
        "missing_ramp_count": int(aggregation_stats["missing_ramp_count"]),
        "timestamp_policy": {
            "raw_times_are": "naive local time",
            "canonical_image_timestamp": "snapped canonical minute",
            "image_must_not_be_future": True,
            "right_labelled_right_closed_bins": True,
        },
        "split_boundaries": split_boundaries,
    }
    manifest_path = output_root / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build(
    hdf5: str | Path = DEFAULT_HDF5,
    times: str | Path = DEFAULT_TIMES,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Build the SKIPP'D Luoyang Parquet artifact and return its manifest."""
    hdf5_path = Path(hdf5).resolve()
    times_path = Path(times).resolve()
    output_path = Path(output_root).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    raw_ns, pv, source_meta = _validate_source(hdf5_path, times_path)
    canonical, filter_stats = _snap_and_dedupe(raw_ns, pv)
    output, aggregation_stats = _aggregate_bins(canonical)
    paths, timestamps, image_stats = _write_selected_images(output_path, output, hdf5_path)

    parquet = output[["timestamp", "final_power", "pv_ramp_5min"]].copy()
    parquet["asi_path"] = paths
    parquet["asi_path_timestamps"] = timestamps
    parquet = parquet[["timestamp", "final_power", "pv_ramp_5min", "asi_path", "asi_path_timestamps"]]
    parquet_path = output_path / "skippd_luoyang.parquet"
    parquet.to_parquet(parquet_path, index=False, engine="pyarrow")
    manifest = _write_manifest(
        output_path,
        hdf5_path,
        times_path,
        source_meta,
        filter_stats,
        aggregation_stats,
        output,
        image_stats,
    )
    # Return the manifest plus a compact output summary for CLI callers.
    return manifest


build_parquet = build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--times", type=Path, default=DEFAULT_TIMES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.hdf5, args.times, args.output_root), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
