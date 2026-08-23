"""Privacy-preserving, dataset-agnostic training diagnostics.

The monitor deliberately writes only aggregate statistics.  Forecast values,
timestamps, image paths and learned representations are never persisted.
Inputs to :meth:`TrainingMonitor.record_forecast` are expected to be in model
space, where power is divided by the site's rated power.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

try:  # torch is available in training, but keeping import failure explicit helps tooling.
    import torch
except ImportError:  # pragma: no cover - the repository itself requires torch.
    torch = None


_PATH_KEY = re.compile(
    r"(^|_)(path|paths|root|roots|file|files|filename|filenames|filepath|"
    r"filepaths|directory|directories|dir|dirs)($|_)",
    re.IGNORECASE,
)
_PRIVATE_ARRAY_KEYS = re.compile(
    r"(^|_)(predictions?|targets?|truths?|timestamps?|sample_times?|issue_time|"
    r"target_time|image_paths?|embeddings?|features?)(_|$)", re.IGNORECASE
)
_SECRET_KEYS = re.compile(
    r"(^|_)(password|passwd|secret|token|api_key|private_key)(_|$)", re.IGNORECASE
)
_PRIVATE_VALUE_KEYS = re.compile(
    r"^(timestamp|timestamps|sample_time|sample_times|issue_time|issue_time_ns|"
    r"target_time|image_path|image_paths)$",
    re.IGNORECASE,
)
_RAW_PAYLOAD_KEYS = re.compile(
    r"^(pred|preds|prediction|predictions|y_pred|output|outputs|target|targets|"
    r"truth|truths|true|trues|y_true|label|labels|feature|features|embedding|"
    r"embeddings|timestamp|timestamps|sample_time|sample_times|issue_time|"
    r"issue_time_ns|target_time|image_path|image_paths)$",
    re.IGNORECASE,
)
_SAFE_PAYLOAD_TEXT_KEYS = re.compile(
    r"^(stage|status|phase|split|role|units|.*(?:strategy|decision|reason|unit|mode))$",
    re.IGNORECASE,
)
_AGGREGATE_SCALAR_KEYS = re.compile(
    r"(^epoch$|_epoch$|(^|_)(count|samples?|steps?|batches?|horizons?|minutes?|"
    r"index|dimension|elements?|loss|rmse|mae|bias|corr|correlation|mean|std|"
    r"rate|fraction|ratio|norm|score|delta|scale|weight|threshold|seconds?|sec|"
    r"memory|bytes|power|enabled|saved|eligible|triggered|accepted|recorded|"
    r"training|suppressed|counter|completed|interventions?|bacc|auprc|accuracy|"
    r"recall|specificity|median|worst)(_|$)|^(task|auxiliary|kd_similarity|"
    r"kd_intervention|kd_extreme_response|image|weather|learning_rate|lr)$)",
    re.IGNORECASE,
)
_STRUCTURAL_SCALAR_KEYS = re.compile(
    r"(^epoch$|_epoch$|(^|_)(count|samples?|steps?|batches?|horizons?|minutes?|"
    r"index|dimension|elements?|seconds?|sec|memory|bytes|enabled|saved|eligible|"
    r"triggered|accepted|recorded|training|suppressed|counter|completed|"
    r"interventions?)(_|$)|^(learning_rate|lr)$)",
    re.IGNORECASE,
)
_MIN_PRIVACY_COUNT = 100
_MIN_PRIVACY_DAYS = 5


def _arg(args: Any, name: str, default: Any = None) -> Any:
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def _to_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _safe_name(value: Any, maximum: int = 96) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return "unnamed"
    if len(text) > maximum or "/" in text or "\\" in text:
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"hashed_{digest}"
    return text


def _safe_path(value: Any) -> Dict[str, bool]:
    del value
    return {"redacted": True}


def _is_path_key(key: Any, value: Any = None) -> bool:
    """Recognize path fields without treating words such as direction as paths."""
    text = str(key)
    if _PATH_KEY.search(text):
        if isinstance(value, (str, Path, Mapping)):
            return True
        if isinstance(value, np.ndarray):
            return value.dtype.kind in "OSU"
        if isinstance(value, (list, tuple, set)):
            return any(isinstance(item, (str, Path)) for item in value)
        return False
    lowered = text.lower()
    return (
        (lowered == "checkpoint" or lowered.endswith("_checkpoint"))
        and isinstance(value, (str, Path))
    )


def _value_count(value: Any) -> int:
    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.numel())
    try:
        return int(np.asarray(value).size)
    except Exception:
        try:
            return int(len(value))
        except (TypeError, ValueError):
            return 1


def _sanitize_aggregate_payload(
    value: Any, *, key: str = "", min_count: int = _MIN_PRIVACY_COUNT,
    min_days: int = _MIN_PRIVACY_DAYS, suppress_metrics: bool = False,
) -> Any:
    """Suppress raw aliases and low-count sequences accepted by public writers."""
    if key and _RAW_PAYLOAD_KEYS.match(key):
        return {"suppressed": True, "count": _value_count(value)}
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        finite_count = int(value.size)
        if np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_:
            finite_count = int(np.isfinite(value.astype(np.float64, copy=False)).sum())
        if finite_count < int(min_count):
            return {
                "privacy_suppressed": True,
                "count": int(value.size),
                "finite_count": finite_count,
            }
        return value
    if isinstance(value, Mapping):
        local_count = None
        for count_key in (
            "valid_target_count", "valid_count", "sample_count",
            "processed_samples", "count",
        ):
            candidate = _finite_float(value.get(count_key))
            if candidate is not None:
                local_count = candidate
                break
        local_suppression = suppress_metrics or (
            local_count is not None and 0 < local_count < int(min_count)
        )
        has_day_count = "distinct_day_count" in value
        local_day_count = _finite_float(value.get("distinct_day_count"))
        local_suppression = local_suppression or (
            has_day_count
            and (local_day_count is None or local_day_count < int(min_days))
        )
        result = {
            str(child_key): _sanitize_aggregate_payload(
                item, key=str(child_key), min_count=min_count,
                min_days=min_days, suppress_metrics=local_suppression,
            )
            for child_key, item in value.items()
            if not _SECRET_KEYS.search(str(child_key))
        }
        for raw_count_key, raw_count in value.items():
            count_key = str(raw_count_key)
            if (
                not count_key.endswith("_count")
                or count_key == "distinct_day_count"
                or count_key.endswith("_distinct_day_count")
            ):
                continue
            paired_count = _finite_float(raw_count)
            if paired_count is None or paired_count >= int(min_count):
                continue
            prefix = count_key[:-len("_count")]
            for metric_key in list(result):
                if metric_key.endswith("_count"):
                    continue
                comparable = metric_key
                for qualifier in ("parent_", "candidate_", "oracle_gate_"):
                    if comparable.startswith(qualifier):
                        comparable = comparable[len(qualifier):]
                        break
                if (
                    comparable == prefix
                    or comparable.startswith(prefix + "_")
                    or ("_" + prefix + "_") in comparable
                ):
                    result[metric_key] = None
            result[f"{prefix}_privacy_suppressed"] = True
        for raw_day_key, raw_day_count in value.items():
            day_key = str(raw_day_key)
            suffix = "_distinct_day_count"
            if not day_key.endswith(suffix):
                continue
            paired_day_count = _finite_float(raw_day_count)
            if (
                paired_day_count is not None
                and paired_day_count >= int(min_days)
            ):
                continue
            prefix = day_key[:-len(suffix)]
            for metric_key in list(result):
                if metric_key.endswith("_count"):
                    continue
                comparable = metric_key
                for qualifier in ("parent_", "candidate_", "oracle_gate_"):
                    if comparable.startswith(qualifier):
                        comparable = comparable[len(qualifier):]
                        break
                if (
                    comparable == prefix
                    or comparable.startswith(prefix + "_")
                    or ("_" + prefix + "_") in comparable
                ):
                    result[metric_key] = None
            result[f"{prefix}_privacy_suppressed"] = True
        if local_suppression and (local_count is not None or has_day_count):
            result["privacy_suppressed"] = True
        return result
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if key == "warnings" and len(values) <= 32 and all(
            isinstance(item, str) for item in values
        ):
            return [_safe_name(item) for item in values]
        try:
            array = np.asarray(values)
        except Exception:
            array = np.asarray([], dtype=np.float64)
        if array.dtype.kind in "biufc" and array.ndim > 0:
            return _sanitize_aggregate_payload(
                array, key=key, min_count=min_count, min_days=min_days,
                suppress_metrics=suppress_metrics,
            )
        return {"suppressed": True, "count": len(values)}
    if isinstance(value, str):
        return (
            _safe_name(value)
            if _SAFE_PAYLOAD_TEXT_KEYS.match(key)
            else {"suppressed": True, "count": 1}
        )
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        if key and not _AGGREGATE_SCALAR_KEYS.search(key):
            return {"suppressed": True, "count": 1}
        if (
            suppress_metrics
            and key
            and not _STRUCTURAL_SCALAR_KEYS.search(key)
        ):
            return None
    return value


def _json_safe(value: Any, key: str = "", *, summarize_arrays: bool = True) -> Any:
    """Convert values to strict JSON while suppressing sample-level payloads."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Path):
        return _safe_path(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite_float(value)
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_safe(value.item(), key=key)
        if _PRIVATE_ARRAY_KEYS.search(key):
            return {"suppressed": True, "count": int(value.size)}
        if not summarize_arrays and value.size <= 32:
            return [_json_safe(item, key=key) for item in value.tolist()]
        if np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_:
            finite = value.astype(np.float64, copy=False).reshape(-1)
            finite = finite[np.isfinite(finite)]
            if not finite.size:
                return {"count": int(value.size), "finite_count": 0}
            return {
                "count": int(value.size),
                "finite_count": int(finite.size),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
            }
        return {"suppressed": True, "count": int(value.size)}
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for raw_key, item in value.items():
            child_key = str(raw_key)
            if _SECRET_KEYS.search(child_key):
                continue
            if _PRIVATE_VALUE_KEYS.match(child_key):
                try:
                    count = int(np.asarray(item).size)
                except Exception:
                    count = 1
                result[_safe_name(child_key)] = {"suppressed": True, "count": count}
                continue
            if _PRIVATE_ARRAY_KEYS.search(child_key) and isinstance(
                item, (list, tuple, set, np.ndarray)
            ):
                try:
                    count = int(np.asarray(item).size)
                except Exception:
                    count = 0
                result[_safe_name(child_key)] = {"suppressed": True, "count": count}
                continue
            if _is_path_key(child_key, item):
                if isinstance(item, Mapping) and item.get("redacted") is True:
                    result[_safe_name(child_key)] = {"redacted": True}
                elif isinstance(item, (str, Path)):
                    result[_safe_name(child_key)] = _safe_path(item)
                else:
                    try:
                        count = len(item)
                    except (TypeError, ValueError):
                        count = 1
                    result[_safe_name(child_key)] = {
                        "redacted": True,
                        "count": int(count),
                    }
            else:
                result[_safe_name(child_key)] = _json_safe(
                    item, key=child_key, summarize_arrays=summarize_arrays
                )
        return result
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if _PRIVATE_ARRAY_KEYS.search(key):
            return {"suppressed": True, "count": len(values)}
        if len(values) > 32:
            return {"suppressed": True, "count": len(values)}
        return [_json_safe(item, key=key, summarize_arrays=summarize_arrays) for item in values]
    return _safe_name(type(value).__name__)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _json_safe(payload), handle, ensure_ascii=False, indent=2,
                sort_keys=True, allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        _json_safe(payload), ensure_ascii=False, sort_keys=True, allow_nan=False,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


_INVALID_DAY = np.iinfo(np.int64).min


def _sample_day_ids(value: Any, sample_count: int) -> np.ndarray:
    """Convert per-sample timestamps to opaque day ids without persisting them."""
    result = np.full(max(0, int(sample_count)), _INVALID_DAY, dtype=np.int64)
    if value is None or not result.size:
        return result
    try:
        array = _to_numpy(value)
        if array.ndim == 0:
            array = np.repeat(array.reshape(1), result.size)
        elif array.shape[0] == result.size:
            array = array.reshape(result.size, -1)[:, 0]
        elif array.size == result.size:
            array = array.reshape(-1)
        elif array.size == 1:
            array = np.repeat(array.reshape(1), result.size)
        else:
            return result

        if np.issubdtype(array.dtype, np.datetime64):
            timestamps = array.astype("datetime64[ns]")
            valid = ~np.isnat(timestamps)
        elif np.issubdtype(array.dtype, np.number):
            numeric = array.astype(np.float64, copy=False)
            valid = np.isfinite(numeric)
            timestamps = np.full(array.shape, np.datetime64("NaT", "ns"))
            timestamps[valid] = numeric[valid].astype(np.int64).astype("datetime64[ns]")
        else:
            timestamps = array.astype("datetime64[ns]")
            valid = ~np.isnat(timestamps)
        if np.any(valid):
            result[valid] = timestamps[valid].astype("datetime64[D]").astype(np.int64)
    except (TypeError, ValueError, OverflowError):
        pass
    return result


def _day_gate(count: int, day_ids: Iterable[int], min_count: int, min_days: int) -> bool:
    return (
        int(count) >= max(1, int(min_count))
        and (int(min_days) <= 0 or len(set(day_ids)) >= int(min_days))
    )


class _DistributionAccumulator:
    """Online moments plus a fixed reservoir and independent sample/day support."""

    def __init__(self, reservoir_size: int, rng: np.random.Generator):
        self.reservoir_size = max(1, int(reservoir_size))
        self.rng = rng
        self.count = 0
        self.missing_count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.reservoir: list[float] = []
        self.support_sample_count = 0
        self.missing_support_sample_count = 0
        self.total_sample_count = 0
        self.support_day_ids: set[int] = set()
        self.missing_day_ids: set[int] = set()
        self.total_day_ids: set[int] = set()

    def update(
        self, values: Any, valid_mask: Any = None, sample_day_ids: Any = None
    ) -> None:
        shaped_values = _to_numpy(values, np.float64)
        if shaped_values.ndim == 0:
            shaped_values = shaped_values.reshape(1)
        value_shape = shaped_values.shape
        array = shaped_values.reshape(-1)
        finite = np.isfinite(array)
        if valid_mask is not None:
            mask = _to_numpy(valid_mask, bool)
            if mask.ndim and value_shape and mask.shape[0] == value_shape[0]:
                while mask.ndim < len(value_shape):
                    mask = np.expand_dims(mask, -1)
            try:
                mask = np.broadcast_to(mask, value_shape).reshape(-1)
            except ValueError:
                if mask.size != array.size:
                    raise ValueError("distribution mask is not broadcastable to values")
                mask = mask.reshape(-1)
            finite &= mask

        sample_count = int(value_shape[0])
        days = np.asarray(
            sample_day_ids
            if sample_day_ids is not None
            else np.full(sample_count, _INVALID_DAY, dtype=np.int64),
            dtype=np.int64,
        ).reshape(-1)
        if days.size != sample_count:
            days = np.full(sample_count, _INVALID_DAY, dtype=np.int64)
        finite_by_sample = finite.reshape(sample_count, -1).any(axis=1)
        missing_by_sample = (~finite).reshape(sample_count, -1).any(axis=1)
        self.total_sample_count += sample_count
        self.support_sample_count += int(finite_by_sample.sum())
        self.missing_support_sample_count += int(missing_by_sample.sum())
        self.total_day_ids.update(int(day) for day in days if day != _INVALID_DAY)
        self.support_day_ids.update(
            int(day)
            for day, supported in zip(days, finite_by_sample)
            if supported and day != _INVALID_DAY
        )
        self.missing_day_ids.update(
            int(day)
            for day, missing in zip(days, missing_by_sample)
            if missing and day != _INVALID_DAY
        )
        self.missing_count += int(array.size - finite.sum())
        selected = array[finite]
        if not selected.size:
            return

        batch_count = int(selected.size)
        batch_mean = float(selected.mean())
        batch_m2 = float(np.square(selected - batch_mean).sum())
        if self.count:
            combined = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean += delta * batch_count / combined
            self.m2 += batch_m2 + delta * delta * self.count * batch_count / combined
            self.count = combined
        else:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
        self.minimum = min(self.minimum, float(selected.min()))
        self.maximum = max(self.maximum, float(selected.max()))

        # Algorithm R. Memory is bounded by reservoir_size regardless of split size.
        seen_before = self.count - batch_count
        for offset, item in enumerate(selected):
            seen = seen_before + offset
            if len(self.reservoir) < self.reservoir_size:
                self.reservoir.append(float(item))
            else:
                replacement = int(self.rng.integers(0, seen + 1))
                if replacement < self.reservoir_size:
                    self.reservoir[replacement] = float(item)

    def summary(
        self, total: Optional[int] = None, min_count: int = 1, min_days: int = 0
    ) -> Dict[str, Any]:
        denominator = self.count + self.missing_count if total is None else int(total)
        required_count = max(1, int(min_count))
        required_days = max(0, int(min_days))
        value_allowed = (
            self.count >= required_count
            and self.support_sample_count >= required_count
            and _day_gate(
                self.support_sample_count, self.support_day_ids,
                required_count, required_days,
            )
        )
        valid_category_allowed = (
            self.support_sample_count == 0
            or _day_gate(
                self.support_sample_count, self.support_day_ids,
                required_count, required_days,
            )
        )
        missing_category_allowed = (
            self.missing_support_sample_count == 0
            or _day_gate(
                self.missing_support_sample_count, self.missing_day_ids,
                required_count, required_days,
            )
        )
        rare_category_present = (
            self.support_sample_count > 0
            and self.missing_support_sample_count > 0
            and (not valid_category_allowed or not missing_category_allowed)
        )
        value_allowed = value_allowed and not rare_category_present
        missing_rate_allowed = (
            denominator >= required_count
            and self.total_sample_count >= required_count
            and _day_gate(
                self.total_sample_count, self.total_day_ids,
                required_count, required_days,
            )
            and valid_category_allowed
            and missing_category_allowed
        )
        result = {
            "count": self.count if not rare_category_present else None,
            "missing_count": (
                max(0, denominator - self.count)
                if not rare_category_present else None),
            "total_count": max(0, denominator),
            "support_sample_count": (
                self.support_sample_count if not rare_category_present else None),
            "missing_support_sample_count": (
                self.missing_support_sample_count
                if not rare_category_present else None),
            "total_sample_count": self.total_sample_count,
            "distinct_day_count": (
                len(self.support_day_ids) if not rare_category_present else None),
            "missing_distinct_day_count": (
                len(self.missing_day_ids) if not rare_category_present else None),
            "total_distinct_day_count": len(self.total_day_ids),
            "missing_rate": (
                float(max(0, denominator - self.count) / denominator)
                if denominator and missing_rate_allowed else None
            ),
            "mean": None,
            "std": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
        }
        if value_allowed:
            samples = np.asarray(self.reservoir, dtype=np.float64)
            quantiles = np.quantile(
                samples, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            variance = self.m2 / self.count
            result.update({
                "mean": self.mean,
                "std": math.sqrt(max(0.0, variance)),
                "p01": float(quantiles[0]),
                "p05": float(quantiles[1]),
                "p25": float(quantiles[2]),
                "p50": float(quantiles[3]),
                "p75": float(quantiles[4]),
                "p95": float(quantiles[5]),
                "p99": float(quantiles[6]),
            })
        if not value_allowed or not missing_rate_allowed:
            result.update({
                "privacy_suppressed": True,
                "privacy_min_count": required_count,
                "privacy_min_days": required_days,
            })
        return result


class _MaskProfileAccumulator:
    """Per-position validity rates with rare-category and day suppression."""

    def __init__(self, name: str, feature_names: Optional[Sequence[str]] = None):
        self.name = _safe_name(name)
        self.feature_names = [_safe_name(value) for value in (feature_names or [])]
        self.shape: Optional[tuple[int, ...]] = None
        self.valid_counts: Optional[np.ndarray] = None
        self.total_counts: Optional[np.ndarray] = None
        self.valid_day_ids: list[set[int]] = []
        self.missing_day_ids: list[set[int]] = []
        self.total_day_ids: list[set[int]] = []

    def update(
        self, mask: Any, sample_day_ids: Any, sample_selector: Any = None
    ) -> None:
        array = _to_numpy(mask, bool)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(array.shape[0], 1)
        sample_count = int(array.shape[0])
        positions = tuple(int(value) for value in array.shape[1:]) or (1,)
        matrix = array.reshape(sample_count, -1)
        if self.shape is None:
            self.shape = positions
            self.valid_counts = np.zeros(matrix.shape[1], dtype=np.int64)
            self.total_counts = np.zeros(matrix.shape[1], dtype=np.int64)
            self.valid_day_ids = [set() for _ in range(matrix.shape[1])]
            self.missing_day_ids = [set() for _ in range(matrix.shape[1])]
            self.total_day_ids = [set() for _ in range(matrix.shape[1])]
        elif self.shape != positions:
            raise ValueError(f"{self.name} mask shape changed across batches")

        selector = np.ones(sample_count, dtype=bool)
        if sample_selector is not None:
            raw_selector = _to_numpy(sample_selector, bool).reshape(-1)
            if raw_selector.size == 1:
                selector[:] = bool(raw_selector[0])
            elif raw_selector.size == sample_count:
                selector = raw_selector
            else:
                raise ValueError(f"{self.name} selector is not sample-aligned")
        matrix = matrix[selector]
        days = np.asarray(sample_day_ids, dtype=np.int64).reshape(-1)
        if days.size != sample_count:
            days = np.full(sample_count, _INVALID_DAY, dtype=np.int64)
        days = days[selector]
        if not matrix.size:
            return
        assert self.valid_counts is not None and self.total_counts is not None
        self.valid_counts += matrix.sum(axis=0, dtype=np.int64)
        self.total_counts += matrix.shape[0]
        for index in range(matrix.shape[1]):
            column = matrix[:, index]
            self.total_day_ids[index].update(
                int(day) for day in days if day != _INVALID_DAY)
            self.valid_day_ids[index].update(
                int(day) for day, value in zip(days, column)
                if value and day != _INVALID_DAY)
            self.missing_day_ids[index].update(
                int(day) for day, value in zip(days, column)
                if not value and day != _INVALID_DAY)

    def summary(self, min_count: int, min_days: int) -> Dict[str, Any]:
        if self.shape is None or self.valid_counts is None or self.total_counts is None:
            return {"status": "not_observed", "positions": []}
        entries = {}
        for flat_index, (valid_count, total_count) in enumerate(
            zip(self.valid_counts, self.total_counts)
        ):
            valid_count = int(valid_count)
            total_count = int(total_count)
            missing_count = total_count - valid_count
            coordinates = np.unravel_index(flat_index, self.shape)
            entry: Dict[str, Any] = {
                "position_index": int(flat_index),
                "support_sample_count": total_count,
                "distinct_day_count": len(self.total_day_ids[flat_index]),
                "total_sample_count": total_count,
                "total_distinct_day_count": len(self.total_day_ids[flat_index]),
            }
            if len(self.shape) >= 1:
                entry["step_index"] = int(coordinates[0])
            if len(self.shape) >= 2:
                feature_index = int(coordinates[-1])
                entry["feature_index"] = feature_index
                if feature_index < len(self.feature_names):
                    entry["feature"] = self.feature_names[feature_index]
            total_allowed = _day_gate(
                total_count, self.total_day_ids[flat_index], min_count, min_days)
            valid_allowed = valid_count == 0 or _day_gate(
                valid_count, self.valid_day_ids[flat_index], min_count, min_days)
            missing_allowed = missing_count == 0 or _day_gate(
                missing_count, self.missing_day_ids[flat_index], min_count, min_days)
            if total_allowed and valid_allowed and missing_allowed:
                entry.update({
                    "valid_count": valid_count,
                    "missing_count": missing_count,
                    "valid_rate": valid_count / max(total_count, 1),
                    "missing_rate": missing_count / max(total_count, 1),
                })
            else:
                entry.update({
                    "valid_count": None,
                    "missing_count": None,
                    "valid_rate": None,
                    "missing_rate": None,
                    "privacy_suppressed": True,
                    "privacy_min_count": int(min_count),
                    "privacy_min_days": int(min_days),
                })
            entries[f"position_{flat_index:04d}"] = entry
        return {"shape": list(self.shape), "positions": entries}


class _CategoricalAccumulator:
    """Fixed-vocabulary categories with per-category sample/day support."""

    def __init__(self, labels: Sequence[str]):
        self.labels = tuple(str(label) for label in labels)
        self.counts = Counter()
        self.sample_counts = Counter()
        self.day_ids = {label: set() for label in self.labels}
        self.total_sample_count = 0
        self.total_day_ids: set[int] = set()

    def update(self, labels: Any, sample_day_ids: Any, valid_mask: Any = None) -> None:
        array = np.asarray(labels, dtype=object)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(array.shape[0], 1)
        matrix = array.reshape(array.shape[0], -1)
        valid = np.ones(matrix.shape, dtype=bool)
        if valid_mask is not None:
            raw_valid = _to_numpy(valid_mask, bool)
            while raw_valid.ndim < array.ndim:
                raw_valid = np.expand_dims(raw_valid, -1)
            valid = np.broadcast_to(raw_valid, array.shape).reshape(matrix.shape)
        days = np.asarray(sample_day_ids, dtype=np.int64).reshape(-1)
        if days.size != matrix.shape[0]:
            days = np.full(matrix.shape[0], _INVALID_DAY, dtype=np.int64)
        self.total_sample_count += matrix.shape[0]
        self.total_day_ids.update(int(day) for day in days if day != _INVALID_DAY)
        for row, row_valid, day in zip(matrix, valid, days):
            present = set()
            for raw_label, is_valid in zip(row, row_valid):
                if not is_valid:
                    continue
                label = str(raw_label)
                if label not in self.labels:
                    label = "other" if "other" in self.labels else self.labels[-1]
                self.counts[label] += 1
                present.add(label)
            for label in present:
                self.sample_counts[label] += 1
                if day != _INVALID_DAY:
                    self.day_ids[label].add(int(day))

    def summary(self, min_count: int, min_days: int) -> Dict[str, Any]:
        total = int(sum(self.counts.values()))
        overall_allowed = _day_gate(
            self.total_sample_count, self.total_day_ids, min_count, min_days)
        category_allowed = all(
            count == 0 or _day_gate(
                self.sample_counts[label], self.day_ids[label], min_count, min_days)
            for label, count in self.counts.items()
        )
        base = {
            "count": total,
            "support_sample_count": self.total_sample_count,
            "distinct_day_count": len(self.total_day_ids),
        }
        if not overall_allowed or not category_allowed:
            return {
                **base,
                "privacy_suppressed": True,
                "privacy_min_count": int(min_count),
                "privacy_min_days": int(min_days),
            }
        counts = {label: int(self.counts[label]) for label in self.labels if self.counts[label]}
        return {
            **base,
            "counts": counts,
            "rates": {label: count / max(total, 1) for label, count in counts.items()},
            "category_support_sample_count": {
                label: int(self.sample_counts[label]) for label in counts
            },
            "category_distinct_day_count": {
                label: len(self.day_ids[label]) for label in counts
            },
        }


def _scaled_distribution(summary: Mapping[str, Any], scale: float) -> Dict[str, Any]:
    output = dict(summary)
    for name in ("mean", "std", "p01", "p05", "p25", "p50", "p75", "p95", "p99"):
        if output.get(name) is not None:
            output[name] = float(output[name]) * scale
    return output


class DataQualityAccumulator:
    """Streaming bounded-memory accumulator used while scanning data loaders."""

    def __init__(
        self,
        split: str,
        feature_names: Optional[Sequence[str]] = None,
        reservoir_size: int = 4096,
        rated_power: float = 1.0,
        seed: int = 0,
        min_aggregate_count: int = 1,
        min_aggregate_days: int = 0,
        forecast_step_minutes: float = 1.0,
        clip_min: Optional[float] = 0.0,
        clip_max: Optional[float] = 1.0,
        direction_threshold: float = 0.02,
        large_ramp_threshold: float = 0.10,
        near_zero_threshold: float = 0.02,
    ):
        self.split = _safe_name(split)
        self.feature_names = [_safe_name(name) for name in (feature_names or [])]
        self.reservoir_size = max(1, int(reservoir_size))
        self.rated_power = float(rated_power)
        self.min_aggregate_count = max(1, int(min_aggregate_count))
        self.min_aggregate_days = max(0, int(min_aggregate_days))
        self.forecast_step_minutes = float(forecast_step_minutes)
        self.clip_min = _finite_float(clip_min) if clip_min is not None else None
        self.clip_max = _finite_float(clip_max) if clip_max is not None else None
        self.direction_threshold = max(0.0, float(direction_threshold))
        self.large_ramp_threshold = max(
            self.direction_threshold, float(large_ramp_threshold))
        self.near_zero_threshold = max(0.0, float(near_zero_threshold))
        self.rng = np.random.default_rng(int(seed))
        self.samples_seen = 0
        self.batches_seen = 0
        self.issue_time_valid_sample_count = 0
        self.issue_day_ids: set[int] = set()
        self.feature_distributions: list[_DistributionAccumulator] = []
        self.target = _DistributionAccumulator(self.reservoir_size, self.rng)
        self.current = _DistributionAccumulator(self.reservoir_size, self.rng)
        self.ramp = _DistributionAccumulator(self.reservoir_size, self.rng)
        self.input_valid_fraction = _DistributionAccumulator(
            self.reservoir_size, self.rng)
        self.target_valid_fraction = _DistributionAccumulator(
            self.reservoir_size, self.rng)
        self.image_coverage = _DistributionAccumulator(self.reservoir_size, self.rng)
        self.image_age = _DistributionAccumulator(self.reservoir_size, self.rng)
        self.image_states = _CategoricalAccumulator(
            ("none", "partial", "full", "other"))
        self.target_ranges = _CategoricalAccumulator((
            "below_floor", "near_zero", "low_output", "mid_output",
            "high_output", "above_capacity", "other",
        ))
        self.ramp_ranges = _CategoricalAccumulator((
            "large_decrease", "decrease", "stable", "increase",
            "large_increase", "other",
        ))
        self.input_profile = _MaskProfileAccumulator(
            "timeseries", self.feature_names)
        self.coverage_profiles: Dict[str, _MaskProfileAccumulator] = {}
        self.source_profiles: Dict[str, _CategoricalAccumulator] = {}
        self.target_horizons: list[_DistributionAccumulator] = []
        self.ramp_horizons: list[_DistributionAccumulator] = []
        self.absolute_ramp_horizons: list[_DistributionAccumulator] = []

    def _ensure_features(self, count: int) -> None:
        while len(self.feature_distributions) < count:
            self.feature_distributions.append(
                _DistributionAccumulator(self.reservoir_size, self.rng)
            )
        while len(self.feature_names) < count:
            self.feature_names.append(f"feature_{len(self.feature_names)}")
        self.input_profile.feature_names = list(self.feature_names)

    def _ensure_target_horizons(self, count: int) -> None:
        for collection in (
            self.target_horizons, self.ramp_horizons,
            self.absolute_ramp_horizons,
        ):
            while len(collection) < count:
                collection.append(
                    _DistributionAccumulator(self.reservoir_size, self.rng))

    @staticmethod
    def _batch_size(*values: Any) -> int:
        for value in values:
            if value is None:
                continue
            try:
                array = _to_numpy(value)
            except Exception:
                continue
            if array.ndim:
                return int(array.shape[0])
            return 1
        return 0

    def update(
        self,
        features: Any = None,
        feature_mask: Any = None,
        target: Any = None,
        current_power: Any = None,
        target_mask: Any = None,
        image_mask: Any = None,
        image_age_minutes: Any = None,
        source_codes: Any = None,
        issue_time_ns: Any = None,
        coverage_masks: Any = None,
        privileged_teacher_enabled: Any = None,
        history_images_enabled: Any = None,
        **metadata: Any,
    ) -> "DataQualityAccumulator":
        """Consume one batch and return self for convenient loader loops.

        Masks use ``True`` for a valid/observed value.  Common dataset metadata
        aliases are accepted through ``metadata``.
        """
        if isinstance(features, Mapping):
            batch = dict(features)
            features = batch.pop("features", batch.pop("seq_x", None))
            feature_mask = feature_mask if feature_mask is not None else batch.pop(
                "feature_mask", batch.pop("timeseries_mask", None)
            )
            target = target if target is not None else batch.pop("target", batch.pop("seq_y", None))
            current_power = current_power if current_power is not None else batch.pop(
                "current_power", None
            )
            target_mask = target_mask if target_mask is not None else batch.pop("target_mask", None)
            image_mask = image_mask if image_mask is not None else batch.pop("image_mask", None)
            image_age_minutes = image_age_minutes if image_age_minutes is not None else batch.pop(
                "image_age_minutes", batch.pop("image_minute_offsets", None)
            )
            source_codes = source_codes if source_codes is not None else batch.pop(
                "source_codes", batch.pop("forecast_source_codes", None)
            )
            issue_time_ns = issue_time_ns if issue_time_ns is not None else batch.pop(
                "issue_time_ns", batch.pop("issue_time", None))
            coverage_masks = coverage_masks if coverage_masks is not None else batch.pop(
                "coverage_masks", None)
            privileged_teacher_enabled = (
                privileged_teacher_enabled
                if privileged_teacher_enabled is not None
                else batch.pop("privileged_teacher_enabled", None)
            )
            history_images_enabled = (
                history_images_enabled
                if history_images_enabled is not None
                else batch.pop("history_images_enabled", None)
            )
            metadata = {**batch, **metadata}

        feature_mask = feature_mask if feature_mask is not None else metadata.get("timeseries_mask")
        target_mask = target_mask if target_mask is not None else metadata.get("target_mask")
        image_mask = image_mask if image_mask is not None else metadata.get("image_mask")
        if image_age_minutes is None:
            image_age_minutes = metadata.get("image_age_minutes")
        if image_age_minutes is None and metadata.get("image_minute_offsets") is not None:
            image_age_minutes = np.abs(_to_numpy(metadata["image_minute_offsets"], np.float64))
        source_codes = source_codes if source_codes is not None else metadata.get(
            "forecast_source_codes", metadata.get("source_code")
        )
        issue_time_ns = (
            issue_time_ns if issue_time_ns is not None
            else metadata.get("issue_time_ns", metadata.get("issue_time"))
        )
        privileged_teacher_enabled = (
            privileged_teacher_enabled
            if privileged_teacher_enabled is not None
            else metadata.get("privileged_teacher_enabled")
        )
        history_images_enabled = (
            history_images_enabled
            if history_images_enabled is not None
            else metadata.get("history_images_enabled")
        )

        self.batches_seen += 1
        batch_size = self._batch_size(features, target, current_power, image_mask)
        self.samples_seen += batch_size
        sample_days = _sample_day_ids(issue_time_ns, batch_size)
        valid_days = sample_days != _INVALID_DAY
        self.issue_time_valid_sample_count += int(valid_days.sum())
        self.issue_day_ids.update(int(day) for day in sample_days[valid_days])

        if features is not None:
            array = _to_numpy(features, np.float64)
            if array.ndim == 0:
                array = array.reshape(1, 1)
            elif array.ndim == 1:
                array = array.reshape(-1, 1)
            feature_count = int(array.shape[-1])
            self._ensure_features(feature_count)
            mask = None if feature_mask is None else _to_numpy(feature_mask, bool)
            if mask is not None:
                while mask.ndim < array.ndim:
                    mask = np.expand_dims(mask, -1)
                try:
                    mask = np.broadcast_to(mask, array.shape)
                except ValueError as exc:
                    raise ValueError("feature_mask is not broadcastable to features") from exc
            observed = np.isfinite(array) if mask is None else np.isfinite(array) & mask
            profile_mask = observed
            if profile_mask.ndim == 1:
                profile_mask = profile_mask.reshape(profile_mask.shape[0], 1)
            self.input_profile.update(profile_mask, sample_days)
            input_fraction = profile_mask.reshape(profile_mask.shape[0], -1).mean(axis=1)
            self.input_valid_fraction.update(input_fraction, sample_day_ids=sample_days)
            for index in range(feature_count):
                self.feature_distributions[index].update(
                    array[..., index], None if mask is None else mask[..., index],
                    sample_day_ids=sample_days,
                )

        if target is not None:
            target_array = _to_numpy(target, np.float64)
            target_valid = None if target_mask is None else _to_numpy(target_mask, bool)
            while target_valid is not None and target_valid.ndim < target_array.ndim:
                target_valid = np.expand_dims(target_valid, -1)
            self.target.update(
                target_array, target_valid, sample_day_ids=sample_days)
            if target_array.ndim == 1:
                target_matrix = target_array.reshape(-1, 1)
            elif target_array.shape[-1] == 1:
                target_matrix = target_array.reshape(target_array.shape[0], -1)
            else:
                target_matrix = target_array.reshape(target_array.shape[0], -1)
            if target_valid is None:
                target_valid_matrix = np.isfinite(target_matrix)
            else:
                target_valid_matrix = np.broadcast_to(
                    target_valid, target_array.shape).reshape(target_matrix.shape).copy()
                target_valid_matrix &= np.isfinite(target_matrix)
            self.target_valid_fraction.update(
                target_valid_matrix.mean(axis=1), sample_day_ids=sample_days)
            target_profile = self.coverage_profiles.setdefault(
                "targets", _MaskProfileAccumulator("targets"))
            target_profile.update(target_valid_matrix, sample_days)
            self._ensure_target_horizons(target_matrix.shape[1])
            for index in range(target_matrix.shape[1]):
                self.target_horizons[index].update(
                    target_matrix[:, index], target_valid_matrix[:, index],
                    sample_day_ids=sample_days)
            target_labels = np.full(target_matrix.shape, "other", dtype=object)
            if self.clip_min is not None:
                target_labels[target_matrix < self.clip_min] = "below_floor"
            lower = self.clip_min if self.clip_min is not None else 0.0
            target_labels[
                (target_matrix >= lower)
                & (target_matrix < self.near_zero_threshold)
            ] = "near_zero"
            target_labels[
                (target_matrix >= self.near_zero_threshold)
                & (target_matrix < 0.20)
            ] = "low_output"
            target_labels[(target_matrix >= 0.20) & (target_matrix < 0.80)] = "mid_output"
            upper = self.clip_max if self.clip_max is not None else math.inf
            target_labels[(target_matrix >= 0.80) & (target_matrix <= upper)] = "high_output"
            if self.clip_max is not None:
                target_labels[target_matrix > self.clip_max] = "above_capacity"
            self.target_ranges.update(
                target_labels, sample_days, valid_mask=target_valid_matrix)

            if current_power is not None:
                current_array = _to_numpy(current_power, np.float64)
                if current_array.ndim == 0:
                    current_array = current_array.reshape(1)
                while current_array.ndim > 1 and current_array.shape[-1] == 1:
                    current_array = current_array[..., 0]
                current_vector = current_array.reshape(-1)
                if current_vector.size == 1:
                    current_vector = np.full(target_matrix.shape[0], current_vector.item())
                if current_vector.size == target_matrix.shape[0]:
                    ramp = target_matrix - current_vector[:, None]
                    ramp_mask = None
                    if target_valid is not None:
                        try:
                            ramp_mask = np.broadcast_to(target_valid, target_array.shape).reshape(
                                target_matrix.shape
                            )
                        except ValueError:
                            ramp_mask = None
                    current_valid = metadata.get("current_power_valid")
                    if current_valid is not None:
                        cv = _to_numpy(current_valid, bool).reshape(-1)
                        if cv.size == ramp.shape[0]:
                            ramp_mask = cv[:, None] if ramp_mask is None else ramp_mask & cv[:, None]
                    self.ramp.update(
                        ramp, ramp_mask, sample_day_ids=sample_days)
                    ramp_valid = (
                        np.isfinite(ramp)
                        if ramp_mask is None
                        else np.isfinite(ramp) & ramp_mask
                    )
                    for index in range(ramp.shape[1]):
                        self.ramp_horizons[index].update(
                            ramp[:, index], ramp_valid[:, index],
                            sample_day_ids=sample_days)
                        self.absolute_ramp_horizons[index].update(
                            np.abs(ramp[:, index]), ramp_valid[:, index],
                            sample_day_ids=sample_days)
                    ramp_labels = np.full(ramp.shape, "stable", dtype=object)
                    ramp_labels[ramp <= -self.large_ramp_threshold] = "large_decrease"
                    ramp_labels[
                        (ramp > -self.large_ramp_threshold)
                        & (ramp < -self.direction_threshold)
                    ] = "decrease"
                    ramp_labels[
                        (ramp > self.direction_threshold)
                        & (ramp < self.large_ramp_threshold)
                    ] = "increase"
                    ramp_labels[ramp >= self.large_ramp_threshold] = "large_increase"
                    self.ramp_ranges.update(
                        ramp_labels, sample_days, valid_mask=ramp_valid)

        if current_power is not None:
            current_valid = metadata.get("current_power_valid")
            self.current.update(
                current_power, current_valid, sample_day_ids=sample_days)
            validity = (
                np.isfinite(_to_numpy(current_power, np.float64))
                if current_valid is None else _to_numpy(current_valid, bool)
            )
            current_profile = self.coverage_profiles.setdefault(
                "current_power", _MaskProfileAccumulator("current_power"))
            current_profile.update(validity, sample_days)

        image_selector = history_images_enabled
        if image_mask is not None and not (
            image_selector is not None
            and not np.any(_to_numpy(image_selector, bool))
        ):
            images = _to_numpy(image_mask, bool)
            if images.ndim == 0:
                images = images.reshape(1, 1)
            elif images.ndim == 1:
                images = (
                    images.reshape(-1, 1)
                    if images.shape[0] == batch_size else images.reshape(1, -1)
                )
            else:
                images = images.reshape(images.shape[0], -1)
            coverage = images.mean(axis=1, dtype=np.float64)
            selector = np.ones(images.shape[0], dtype=bool)
            if image_selector is not None:
                raw_selector = _to_numpy(image_selector, bool).reshape(-1)
                if raw_selector.size == 1:
                    selector[:] = bool(raw_selector[0])
                elif raw_selector.size == images.shape[0]:
                    selector = raw_selector
            image_profile = self.coverage_profiles.setdefault(
                "history_images", _MaskProfileAccumulator("history_images"))
            image_profile.update(images, sample_days, selector)
            image_days = sample_days[selector]
            selected_coverage = coverage[selector]
            self.image_coverage.update(
                selected_coverage, sample_day_ids=image_days)
            image_labels = np.full(selected_coverage.shape, "partial", dtype=object)
            image_labels[selected_coverage == 0.0] = "none"
            image_labels[selected_coverage == 1.0] = "full"
            self.image_states.update(image_labels, image_days)
            if image_age_minutes is not None:
                ages = np.abs(_to_numpy(image_age_minutes, np.float64))
                try:
                    valid_ages = np.broadcast_to(images, ages.shape)
                except ValueError:
                    valid_ages = None
                self.image_age.update(
                    ages[selector],
                    None if valid_ages is None else valid_ages[selector],
                    sample_day_ids=image_days,
                )
        elif image_age_minutes is not None:
            self.image_age.update(
                np.abs(_to_numpy(image_age_minutes, np.float64)),
                sample_day_ids=sample_days)

        if isinstance(coverage_masks, Mapping):
            allowed_masks = {
                "forecast_own_product", "power_history",
                "teacher_history_timeseries", "teacher_future_timeseries",
                "teacher_future_images",
            }
            for raw_name, values in coverage_masks.items():
                name = _safe_name(raw_name)
                if name not in allowed_masks or values is None:
                    continue
                selector = (
                    privileged_teacher_enabled
                    if name.startswith("teacher_") else None
                )
                profile = self.coverage_profiles.setdefault(
                    name, _MaskProfileAccumulator(name))
                profile.update(values, sample_days, selector)

        if source_codes is not None:
            self._update_source_codes(source_codes, sample_days)
        return self

    def _update_source_codes(self, source_codes: Any, sample_days: np.ndarray) -> None:
        if isinstance(source_codes, Mapping):
            expanded_items = []
            for raw_name, raw_values in source_codes.items():
                source_array = _to_numpy(raw_values)
                if (
                    str(raw_name) == "timeseries_source"
                    and source_array.ndim >= 1
                    and self.feature_names
                    and source_array.shape[-1] == len(self.feature_names)
                ):
                    expanded_items.extend(
                        (
                            f"timeseries:{self.feature_names[index]}",
                            source_array[..., index],
                        )
                        for index in range(source_array.shape[-1])
                    )
                else:
                    expanded_items.append((raw_name, source_array))
            items = expanded_items
        else:
            source_array = _to_numpy(source_codes)
            if (
                source_array.ndim >= 1
                and self.feature_names
                and source_array.shape[-1] == len(self.feature_names)
            ):
                items = (
                    (
                        f"timeseries:{self.feature_names[index]}",
                        source_array[..., index],
                    )
                    for index in range(source_array.shape[-1])
                )
            else:
                items = (("all", source_array),)
        for raw_name, raw_values in items:
            name = _safe_name(raw_name)
            values = _to_numpy(raw_values)
            labels = np.full(values.shape, "other", dtype=object)
            try:
                numeric = values.astype(np.float64, copy=False)
                labels[~np.isfinite(numeric)] = "missing"
                for code in range(4):
                    labels[np.isfinite(numeric) & (numeric == code)] = str(code)
            except (TypeError, ValueError, OverflowError):
                labels[:] = "other"
            profile = self.source_profiles.setdefault(
                name,
                _CategoricalAccumulator(("0", "1", "2", "3", "other", "missing")),
            )
            profile.update(labels, sample_days)

    def summary(self) -> Dict[str, Any]:
        feature_summary = {
            self.feature_names[index]: distribution.summary(
                min_count=self.min_aggregate_count,
                min_days=self.min_aggregate_days)
            for index, distribution in enumerate(self.feature_distributions)
        }

        def power_summary(distribution: _DistributionAccumulator) -> Dict[str, Any]:
            normalized = distribution.summary(
                min_count=self.min_aggregate_count,
                min_days=self.min_aggregate_days)
            return {
                **normalized,
                "original_units": _scaled_distribution(normalized, self.rated_power),
            }

        image_state_summary = self.image_states.summary(
            self.min_aggregate_count, self.min_aggregate_days)
        image_coverage_summary = self.image_coverage.summary(
            min_count=self.min_aggregate_count,
            min_days=self.min_aggregate_days)
        image_age_summary = self.image_age.summary(
            min_count=self.min_aggregate_count,
            min_days=self.min_aggregate_days)
        if "counts" not in image_state_summary:
            for distribution in (image_coverage_summary, image_age_summary):
                for field in (
                    "mean", "std", "p01", "p05", "p25", "p50", "p75",
                    "p95", "p99", "missing_rate",
                ):
                    if field in distribution:
                        distribution[field] = None
                distribution.update({
                    "privacy_suppressed": True,
                    "privacy_min_count": self.min_aggregate_count,
                    "privacy_min_days": self.min_aggregate_days,
                })
        image_samples = int(image_state_summary.get("support_sample_count", 0))
        if "counts" not in image_state_summary:
            image_counts = {
                "samples": image_samples,
                "none_count": None,
                "partial_count": None,
                "full_count": None,
                **{
                    key: value for key, value in image_state_summary.items()
                    if key not in {"counts", "rates"}
                },
            }
        else:
            state_counts = image_state_summary["counts"]
            image_counts = {
                "samples": image_samples,
                "none_count": int(state_counts.get("none", 0)),
                "partial_count": int(state_counts.get("partial", 0)),
                "full_count": int(state_counts.get("full", 0)),
                "state_rates": image_state_summary["rates"],
            }
        source_counts = {}
        source_rates = {}
        source_support = {}
        for name, profile in self.source_profiles.items():
            source = profile.summary(
                self.min_aggregate_count, self.min_aggregate_days)
            if "counts" not in source:
                suppressed = dict(source)
                source_counts[name] = suppressed
                source_rates[name] = suppressed
                source_support[name] = suppressed
                continue
            source_counts[name] = source["counts"]
            source_rates[name] = source["rates"]
            source_support[name] = {
                key: value for key, value in source.items()
                if key not in {"counts", "rates"}
            }

        target_by_horizon = {}
        for index, distribution in enumerate(self.target_horizons):
            target_summary = power_summary(distribution)
            ramp_summary = power_summary(self.ramp_horizons[index])
            absolute_ramp = self.absolute_ramp_horizons[index].summary(
                min_count=self.min_aggregate_count,
                min_days=self.min_aggregate_days)
            persistence_mae = absolute_ramp.get("mean")
            ramp_mean = ramp_summary.get("mean")
            ramp_std = ramp_summary.get("std")
            persistence_rmse = (
                math.sqrt(ramp_mean * ramp_mean + ramp_std * ramp_std)
                if ramp_mean is not None and ramp_std is not None else None
            )
            target_by_horizon[f"lead_{index + 1:03d}"] = {
                "lead_index": index + 1,
                "horizon_minutes": (index + 1) * self.forecast_step_minutes,
                "target": target_summary,
                "ramp": ramp_summary,
                "persistence_baseline": {
                    "support_sample_count": absolute_ramp.get(
                        "support_sample_count", 0),
                    "distinct_day_count": absolute_ramp.get(
                        "distinct_day_count", 0),
                    "mae_normalized": persistence_mae,
                    "rmse_normalized": persistence_rmse,
                    "mae_original_units": (
                        persistence_mae * self.rated_power
                        if persistence_mae is not None else None),
                    "rmse_original_units": (
                        persistence_rmse * self.rated_power
                        if persistence_rmse is not None else None),
                    **(
                        {
                            "privacy_suppressed": True,
                            "privacy_min_count": self.min_aggregate_count,
                            "privacy_min_days": self.min_aggregate_days,
                        }
                        if persistence_mae is None else {}
                    ),
                },
            }

        coverage_profiles = {
            "timeseries": self.input_profile.summary(
                self.min_aggregate_count, self.min_aggregate_days),
            **{
                name: profile.summary(
                    self.min_aggregate_count, self.min_aggregate_days)
                for name, profile in sorted(self.coverage_profiles.items())
            },
        }
        input_completeness_summary = self.input_valid_fraction.summary(
            min_count=self.min_aggregate_count,
            min_days=self.min_aggregate_days)
        target_completeness_summary = self.target_valid_fraction.summary(
            min_count=self.min_aggregate_count,
            min_days=self.min_aggregate_days)

        def couple_completeness_to_positions(
            distribution: Dict[str, Any], profile: Any
        ) -> None:
            positions = profile.get("positions", {}) if isinstance(profile, Mapping) else {}
            position_unsafe = (
                not isinstance(positions, Mapping)
                or any(
                    isinstance(entry, Mapping)
                    and entry.get("privacy_suppressed", False)
                    for entry in positions.values()
                )
            )
            if not position_unsafe:
                return
            for field in (
                "mean", "std", "p01", "p05", "p25", "p50", "p75", "p95",
                "p99", "missing_rate",
            ):
                if field in distribution:
                    distribution[field] = None
            distribution.update({
                "privacy_suppressed": True,
                "privacy_min_count": self.min_aggregate_count,
                "privacy_min_days": self.min_aggregate_days,
            })

        couple_completeness_to_positions(
            input_completeness_summary, coverage_profiles.get("timeseries", {}))
        couple_completeness_to_positions(
            target_completeness_summary, coverage_profiles.get("targets", {}))
        aggregate_allowed = _day_gate(
            self.issue_time_valid_sample_count, self.issue_day_ids,
            self.min_aggregate_count, self.min_aggregate_days)
        return _json_safe({
            "split": self.split,
            "samples_seen": self.samples_seen,
            "batches_seen": self.batches_seen,
            "privacy_support": {
                "valid_issue_time_sample_count": self.issue_time_valid_sample_count,
                "distinct_day_count": len(self.issue_day_ids),
                "min_sample_count": self.min_aggregate_count,
                "min_distinct_days": self.min_aggregate_days,
                "aggregate_allowed": aggregate_allowed,
            },
            "feature_count": len(self.feature_distributions),
            "features": feature_summary,
            "target": power_summary(self.target),
            "current_power": power_summary(self.current),
            "ramp": power_summary(self.ramp),
            "sample_completeness": {
                "input_valid_fraction": input_completeness_summary,
                "target_valid_fraction": target_completeness_summary,
            },
            "missingness_by_position": coverage_profiles,
            "horizon_label_profile": target_by_horizon,
            "target_operating_ranges": {
                "thresholds_normalized": {
                    "floor": self.clip_min,
                    "near_zero_upper": self.near_zero_threshold,
                    "low_output_upper": 0.20,
                    "mid_output_upper": 0.80,
                    "capacity": self.clip_max,
                },
                **self.target_ranges.summary(
                    self.min_aggregate_count, self.min_aggregate_days),
            },
            "ramp_ranges": {
                "thresholds_normalized": {
                    "direction": self.direction_threshold,
                    "large_ramp": self.large_ramp_threshold,
                },
                **self.ramp_ranges.summary(
                    self.min_aggregate_count, self.min_aggregate_days),
            },
            "images": {
                **image_counts,
                "coverage": image_coverage_summary,
                "age_minutes": image_age_summary,
            },
            "source_codes": source_counts,
            "source_code_rates": source_rates,
            "source_code_support": source_support,
            "source_code_legend": {
                "0": "raw_or_own_forecast",
                "1": "forward_fill_or_alternative_forecast",
                "2": "issue_time_persistence",
                "3": "training_mean",
                "other": "unknown_code",
                "missing": "nonfinite_code",
            },
            "reservoir_size": self.reservoir_size,
            "rated_power": self.rated_power,
        })

    finalize = summary


_HORIZON_FIELDS = (
    "phase", "epoch", "split", "role", "lead_index", "horizon_minutes", "count",
    "distinct_day_count", "privacy_suppressed",
    "rmse", "mae", "bias", "nrmse", "nmae", "corr",
    "persistence_count", "persistence_privacy_suppressed",
    "persistence_rmse", "persistence_mae",
    "persistence_skill_rmse", "persistence_skill_mae",
    "clipped_rmse", "clipped_mae", "clipped_bias", "clipped_nrmse", "clipped_nmae",
    "below_floor_rate", "above_capacity_rate", "out_of_bounds_rate",
    "invalid_prediction_count",
)

_SLICE_FIELDS = (
    "phase", "epoch", "split", "role", "slice_dimension", "slice_value", "count",
    "distinct_day_count",
    "rmse", "mae", "bias", "nrmse", "nmae", "corr",
    "persistence_rmse", "persistence_skill_rmse", "clipped_rmse", "clipped_mae",
)
_MONITOR_ARTIFACT_NAMES = {
    "run_manifest.json", "data_quality.json", "epochs.jsonl",
    "horizon_metrics.csv", "slice_metrics.csv", "distillation.jsonl",
    "summary.json",
}


class TrainingMonitor:
    """Write compact diagnostics that can safely be returned from private runs."""

    def __init__(
        self,
        run_dir: Any,
        stage: str,
        *,
        enabled: bool = True,
        rated_power: float = 1.0,
        units: str = "power_units",
        min_slice_count: int = 100,
        min_slice_days: int = _MIN_PRIVACY_DAYS,
        reservoir_size: int = 4096,
        forecast_step_minutes: float = 1.0,
        clip_min: Optional[float] = 0.0,
        clip_max: Optional[float] = 1.0,
        direction_threshold: float = 0.02,
        large_ramp_threshold: float = 0.10,
        near_zero_threshold: float = 0.02,
        safe_args: Optional[Mapping[str, Any]] = None,
        preserve_existing: bool = False,
    ):
        self.run_dir = Path(run_dir)
        self.stage = _safe_name(stage)
        self.enabled = bool(enabled)
        self.rated_power = float(rated_power)
        if not math.isfinite(self.rated_power) or self.rated_power <= 0:
            raise ValueError("monitor rated power must be finite and positive")
        self.units = _safe_name(units)
        self.min_slice_count = int(min_slice_count)
        self.min_slice_days = int(min_slice_days)
        if self.min_slice_count < _MIN_PRIVACY_COUNT:
            raise ValueError(
                f"monitor_min_slice_count must be at least {_MIN_PRIVACY_COUNT}"
            )
        if self.min_slice_days < _MIN_PRIVACY_DAYS:
            raise ValueError(
                f"monitor_min_slice_days must be at least {_MIN_PRIVACY_DAYS}"
            )
        self.reservoir_size = max(1, int(reservoir_size))
        self.forecast_step_minutes = float(forecast_step_minutes)
        self.clip_min = _finite_float(clip_min) if clip_min is not None else None
        self.clip_max = _finite_float(clip_max) if clip_max is not None else None
        if self.clip_min is not None and self.clip_max is not None and self.clip_min >= self.clip_max:
            raise ValueError("monitor_clip_min must be smaller than monitor_clip_max")
        self.direction_threshold = max(0.0, float(direction_threshold))
        self.large_ramp_threshold = max(self.direction_threshold, float(large_ramp_threshold))
        self.near_zero_threshold = max(0.0, float(near_zero_threshold))
        self.safe_args = dict(safe_args or {})
        self.preserve_existing = bool(preserve_existing)
        self._lock = threading.RLock()
        self._started = False
        self._data_quality: Dict[str, Any] = {}
        self._forecast_summaries: list[Dict[str, Any]] = []
        self._epoch_summaries: list[Dict[str, Any]] = []
        self._epoch_record_count = 0
        self._forecast_evaluation_count = 0
        self._existing_alerts: list[Dict[str, Any]] = []
        self._final_payload: Dict[str, Any] = {}

    @classmethod
    def from_args(
        cls, args: Any, run_dir: Any, stage: str, model: Any = None
    ) -> "TrainingMonitor":
        configured_dir = _arg(args, "monitor_dir", None)
        monitor_dir = Path(configured_dir) if configured_dir else Path(run_dir) / "monitor"
        if configured_dir:
            monitor_path = monitor_dir.resolve()
            training_path = Path(run_dir).resolve()
            if monitor_path == training_path or monitor_path in training_path.parents:
                raise ValueError(
                    "monitor_dir must be a dedicated directory, not the run directory "
                    "or one of its parents"
                )
        rated_power = _arg(args, "monitor_rated_power", None)
        if rated_power is None:
            for name in ("rated_power", "power_scale", "stanford_capacity_kw"):
                candidate = _finite_float(_arg(args, name, None))
                if candidate is not None and candidate > 0:
                    rated_power = candidate
                    break
        if rated_power is None:
            rated_power = 1.0

        forecast_step = _arg(args, "monitor_forecast_step_minutes", None)
        if forecast_step is None:
            forecast_step = _arg(args, "sample_interval_minutes", 1.0)

        safe_args = cls._select_safe_args(args)
        enabled_value = _arg(args, "monitor_enabled", None)
        if enabled_value is None:
            enabled_value = _arg(args, "monitor", False)
        monitor = cls(
            run_dir=monitor_dir,
            stage=stage,
            enabled=_as_bool(enabled_value),
            rated_power=float(rated_power),
            units=_arg(args, "monitor_units", _arg(args, "power_units", "power_units")),
            min_slice_count=int(_arg(args, "monitor_min_slice_count", 100)),
            min_slice_days=int(_arg(
                args, "monitor_min_slice_days", _MIN_PRIVACY_DAYS)),
            reservoir_size=int(_arg(args, "monitor_reservoir_size", 4096)),
            forecast_step_minutes=float(forecast_step),
            clip_min=_arg(args, "monitor_clip_min", 0.0),
            clip_max=_arg(args, "monitor_clip_max", 1.0),
            direction_threshold=float(_arg(args, "monitor_direction_threshold", 0.02)),
            large_ramp_threshold=float(_arg(args, "monitor_large_ramp_threshold", 0.10)),
            near_zero_threshold=float(_arg(args, "monitor_near_zero_threshold", 0.02)),
            safe_args=safe_args,
            preserve_existing=int(_arg(args, "is_training", 1) or 0) == 0,
        )
        if monitor.enabled and model is not None:
            # The full manifest is written by start_run once datasets are available.
            monitor.run_dir.mkdir(parents=True, exist_ok=True)
        return monitor

    @staticmethod
    def _select_safe_args(args: Any) -> Dict[str, Any]:
        values = dict(args) if isinstance(args, Mapping) else vars(args) if hasattr(args, "__dict__") else {}
        exact = {
            "task_name", "model_id", "model", "data", "features", "target", "freq",
            "seq_len", "label_len", "pred_len", "enc_in", "dec_in", "c_out", "d_model",
            "n_heads", "e_layers", "d_layers", "d_ff", "dropout", "batch_size",
            "train_epochs", "patience", "learning_rate", "lradj", "loss", "loss_type",
            "use_amp", "kd_type", "kd_loss_weight", "kd_loss_weight_sim",
            "kd_loss_weight_inter", "fusion_type", "image_encoder_type", "fold_index",
            "num_fold", "forecast_horizon_minutes", "sample_interval_minutes", "des",
        }
        selected: Dict[str, Any] = {}
        for key, value in values.items():
            if _SECRET_KEYS.search(str(key)):
                continue
            if not (str(key).startswith("monitor_") or key in exact):
                continue
            if _is_path_key(key, value):
                selected[str(key)] = _safe_path(value)
            elif isinstance(value, (type(None), bool, int, float, str)):
                selected[str(key)] = _json_safe(value, key=str(key))
            elif isinstance(value, (list, tuple)) and len(value) <= 16:
                selected[str(key)] = _json_safe(value, key=str(key), summarize_arrays=False)
        return selected

    @property
    def paths(self) -> Dict[str, Path]:
        return {
            "manifest": self.run_dir / "run_manifest.json",
            "data_quality": self.run_dir / "data_quality.json",
            "epochs": self.run_dir / "epochs.jsonl",
            "horizons": self.run_dir / "horizon_metrics.csv",
            "slices": self.run_dir / "slice_metrics.csv",
            "distillation": self.run_dir / "distillation.jsonl",
            "summary": self.run_dir / "summary.json",
        }

    @staticmethod
    def _model_summary(model: Any) -> Dict[str, Any]:
        if model is None or not hasattr(model, "parameters"):
            return {"class": None, "parameter_count": 0, "trainable_parameter_count": 0}
        parameters = list(model.parameters())
        return {
            "class": model.__class__.__name__,
            "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "trainable_parameter_count": int(
                sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
            ),
            "module_count": int(sum(1 for _ in model.modules())),
        }

    def _dataset_summary(self, dataset: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "class": dataset.__class__.__name__,
            "sample_count": int(len(dataset)) if hasattr(dataset, "__len__") else None,
        }
        for name in ("seq_len", "pred_len", "image_frames"):
            value = getattr(dataset, name, None)
            if isinstance(value, (int, np.integer)):
                result[name] = int(value)
        config = getattr(dataset, "config", None)
        if isinstance(config, Mapping):
            dimensions = config.get("dimensions", {})
            time_config = config.get("time", {})
            result["dataset"] = _safe_name(config.get("dataset", result["class"]))
            result["dimensions"] = {
                _safe_name(key): _json_safe(value)
                for key, value in dimensions.items()
                if isinstance(value, (bool, int, float, np.integer, np.floating))
            }
            for key in ("forecast_step_minutes", "sampling_interval_minutes"):
                value = _finite_float(time_config.get(key))
                if value is not None:
                    result[key] = value
            power = config.get("power", {})
            capacity = _finite_float(power.get("rated_power"))
            if capacity is not None and capacity > 0:
                result["rated_power"] = capacity
                result["units"] = _safe_name(power.get("units", self.units))
        audit = getattr(dataset, "candidate_audit", None)
        if isinstance(audit, Mapping):
            result["candidate_audit"] = {
                _safe_name(key): _json_safe(value)
                for key, value in audit.items()
                if isinstance(value, (bool, int, float, np.integer, np.floating))
            }
        sample_times = getattr(dataset, "sample_times", None)
        if sample_times is not None:
            try:
                timestamps = _to_numpy(sample_times).reshape(-1).astype("datetime64[ns]")
                valid = ~np.isnat(timestamps)
                timestamps = timestamps[valid]
                months = (
                    timestamps.astype("datetime64[M]").astype(np.int64) % 12 + 1
                    if timestamps.size else np.empty(0, dtype=np.int64)
                )
                hours = (
                    timestamps.astype("datetime64[h]").astype(np.int64) % 24
                    if timestamps.size else np.empty(0, dtype=np.int64)
                )
                day_ids = timestamps.astype("datetime64[D]").astype(np.int64)
                weekdays = (day_ids + 3) % 7
                distinct_days = int(
                    np.unique(day_ids).size
                )
                coverage = {
                    "valid_timestamp_count": int(timestamps.size),
                    "invalid_timestamp_count": int(valid.size - valid.sum()),
                    "distinct_day_count": distinct_days,
                }
                coverage_allowed = (
                    timestamps.size >= self.min_slice_count
                    and distinct_days >= self.min_slice_days
                )
                if not coverage_allowed:
                    coverage["privacy_suppressed"] = True
                else:
                    month_counts = {
                        f"{month:02d}": int(np.sum(months == month))
                        for month in range(1, 13)
                    }
                    hour_counts = {
                        f"{hour:02d}": int(np.sum(hours == hour))
                        for hour in range(24)
                    }
                    weekday_counts = {
                        str(day): int(np.sum(weekdays == day))
                        for day in range(7)
                    }
                    for name, histogram, groups in (
                        ("month_of_year_histogram", month_counts, months),
                        ("hour_of_day_histogram", hour_counts, hours),
                        ("day_of_week_histogram", weekday_counts, weekdays),
                    ):
                        support = [
                            (
                                count,
                                int(np.unique(day_ids[groups == int(label)]).size),
                            )
                            for label, count in histogram.items() if count
                        ]
                        coverage[name] = (
                            histogram
                            if all(
                                count >= self.min_slice_count
                                and days >= self.min_slice_days
                                for count, days in support
                            )
                            else {
                                "privacy_suppressed": True,
                                "privacy_min_count": self.min_slice_count,
                                "privacy_min_days": self.min_slice_days,
                            }
                        )
                    ordered = np.sort(timestamps.astype(np.int64))
                    gaps = np.diff(ordered).astype(np.float64) / 60_000_000_000.0
                    gap_days = day_ids[np.argsort(timestamps.astype(np.int64))][1:]
                    positive = gaps > 0
                    gap_distribution = _DistributionAccumulator(
                        self.reservoir_size, np.random.default_rng(0))
                    gap_distribution.update(
                        gaps, positive, sample_day_ids=gap_days)
                    coverage["sampling_gap_minutes"] = gap_distribution.summary(
                        min_count=self.min_slice_count,
                        min_days=self.min_slice_days)
                    coverage["duplicate_timestamp_count"] = int(
                        timestamps.size - np.unique(timestamps).size)
                result["time_coverage"] = coverage
            except (TypeError, ValueError, OverflowError):
                result["time_coverage"] = {"unavailable": True}
        candidate = result.get("candidate_audit", {})
        time_coverage = result.get("time_coverage", {})
        if (
            isinstance(candidate, Mapping)
            and isinstance(time_coverage, Mapping)
            and not time_coverage.get("privacy_suppressed", False)
        ):
            selected = _finite_float(candidate.get("selected_sample_count"))
            valid_candidates = _finite_float(candidate.get("valid_candidate_count"))
            theoretical = _finite_float(candidate.get("theoretical_issue_count"))
            observed = _finite_float(candidate.get("observed_issue_count"))
            continuous = _finite_float(candidate.get("continuous_window_count"))
            if selected is not None and selected >= self.min_slice_count:
                result["candidate_rates"] = {
                    "selected_from_valid_rate": (
                        selected / valid_candidates
                        if valid_candidates is not None and valid_candidates > 0 else None),
                    "valid_from_theoretical_rate": (
                        valid_candidates / theoretical
                        if valid_candidates is not None and theoretical else None),
                    "continuous_from_observed_rate": (
                        continuous / observed
                        if continuous is not None and observed else None),
                }
        return result

    def start_run(self, model: Any, datasets: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._started:
                self.register_datasets(datasets)
                return
            self.run_dir.mkdir(parents=True, exist_ok=True)
            entries = list(self.run_dir.iterdir())
            unexpected = sorted(
                path.name for path in entries
                if path.name not in _MONITOR_ARTIFACT_NAMES
            )
            if unexpected:
                raise ValueError(
                    "monitor_dir must contain only monitor artifacts; unexpected "
                    f"entries: {', '.join(unexpected[:5])}"
                )
            unsafe = sorted(
                path.name for path in entries
                if path.name in _MONITOR_ARTIFACT_NAMES
                and (path.is_symlink() or not path.is_file())
            )
            if unsafe:
                raise ValueError(
                    "monitor artifacts must be regular non-symlink files; unsafe "
                    f"entries: {', '.join(unsafe[:5])}"
                )
            preserving = self.preserve_existing and self.paths["manifest"].exists()
            if self.preserve_existing and entries and not preserving:
                raise ValueError(
                    "test-only monitor reuse requires a valid run_manifest.json"
                )
            dataset_summaries = {
                _safe_name(split): self._dataset_summary(dataset)
                for split, dataset in (datasets or {}).items()
            }
            capacities = {
                float(summary["rated_power"])
                for summary in dataset_summaries.values()
                if summary.get("rated_power") is not None
            }
            if len(capacities) == 1:
                self.rated_power = capacities.pop()
            steps = {
                float(summary["forecast_step_minutes"])
                for summary in dataset_summaries.values()
                if summary.get("forecast_step_minutes") is not None
            }
            if len(steps) == 1:
                self.forecast_step_minutes = steps.pop()
            units = {
                str(summary["units"])
                for summary in dataset_summaries.values()
                if summary.get("units")
            }
            if len(units) == 1:
                self.units = units.pop()
            manifest = {
                "schema_version": 1,
                "stage": self.stage,
                "created_unix_seconds": int(time.time()),
                "monitor": {
                    "rated_power": self.rated_power,
                    "units": self.units,
                    "min_slice_count": self.min_slice_count,
                    "min_slice_days": self.min_slice_days,
                    "reservoir_size": self.reservoir_size,
                    "forecast_step_minutes": self.forecast_step_minutes,
                    "clip_min_normalized": self.clip_min,
                    "clip_max_normalized": self.clip_max,
                    "direction_threshold_fraction": self.direction_threshold,
                    "large_ramp_threshold_fraction": self.large_ramp_threshold,
                    "near_zero_threshold_fraction": self.near_zero_threshold,
                    "privacy": "aggregate_only",
                },
                "arguments": self.safe_args,
                "model": self._model_summary(model),
                "datasets": dataset_summaries,
            }
            if preserving:
                existing_manifest = self._read_json(self.paths["manifest"])
                existing_monitor = (
                    existing_manifest.get("monitor", {})
                    if isinstance(existing_manifest, Mapping) else {}
                )
                if (
                    not isinstance(existing_manifest, Mapping)
                    or existing_manifest.get("schema_version") != 1
                    or not isinstance(existing_monitor, Mapping)
                    or existing_monitor.get("privacy") != "aggregate_only"
                ):
                    raise ValueError(
                        "test-only monitor reuse requires schema_version=1 and "
                        "monitor.privacy=aggregate_only"
                    )
                old_datasets = existing_manifest.get("datasets", {})
                if isinstance(old_datasets, Mapping):
                    manifest["datasets"] = {**old_datasets, **dataset_summaries}
                manifest["created_unix_seconds"] = existing_manifest.get(
                    "created_unix_seconds", manifest["created_unix_seconds"]
                )
                manifest["last_opened_unix_seconds"] = int(time.time())
            _atomic_json(self.paths["manifest"], manifest)

            if preserving:
                existing_quality = self._read_json(self.paths["data_quality"])
                if self.paths["data_quality"].exists() and (
                    not isinstance(existing_quality, Mapping)
                    or existing_quality.get("schema_version") != 1
                    or existing_quality.get("privacy") != "aggregate_only"
                ):
                    raise ValueError(
                        "preserved data_quality.json is not aggregate-only schema 1"
                    )
                if isinstance(existing_quality, Mapping):
                    self._data_quality = {
                        str(key): value
                        for key, value in existing_quality.items()
                        if key not in {
                            "schema_version", "privacy", "comparisons_to_train"
                        }
                    }
                existing_summary = self._read_json(self.paths["summary"])
                if self.paths["summary"].exists() and (
                    not isinstance(existing_summary, Mapping)
                    or existing_summary.get("schema_version") != 1
                    or existing_summary.get("privacy") != "aggregate_only"
                ):
                    raise ValueError(
                        "preserved summary.json is not aggregate-only schema 1"
                    )
                if isinstance(existing_summary, Mapping):
                    result = existing_summary.get("result", {})
                    if isinstance(result, Mapping):
                        self._final_payload = _sanitize_aggregate_payload(
                            dict(result), min_count=self.min_slice_count,
                            min_days=self.min_slice_days)
                    alerts = existing_summary.get("alerts", [])
                    if isinstance(alerts, list):
                        self._existing_alerts = alerts[-100:]
                self._epoch_record_count = self._count_jsonl(self.paths["epochs"])
                self._forecast_evaluation_count = self._count_forecast_evaluations(
                    self.paths["horizons"]
                )
                self._validate_preserved_text_artifacts()
            else:
                self._data_quality = {}
                self._epoch_record_count = 0
                self._forecast_evaluation_count = 0
                _atomic_json(self.paths["data_quality"], {
                    "schema_version": 1,
                    "privacy": "aggregate_only",
                })
                _atomic_text(self.paths["epochs"], "")
                _atomic_text(self.paths["distillation"], "")
                self._reset_csv(self.paths["horizons"], _HORIZON_FIELDS)
                self._reset_csv(self.paths["slices"], _SLICE_FIELDS)
                _atomic_json(self.paths["summary"], {
                    "schema_version": 1,
                    "privacy": "aggregate_only",
                    "stage": self.stage,
                    "status": "running",
                })
            if preserving:
                for name in ("epochs", "distillation"):
                    self.paths[name].touch(exist_ok=True)
                self._ensure_csv(self.paths["horizons"], _HORIZON_FIELDS)
                self._ensure_csv(self.paths["slices"], _SLICE_FIELDS)
            self._started = True

    def register_datasets(self, datasets: Mapping[str, Any]) -> None:
        """Merge aggregate dataset audit information into an active manifest."""
        if not self.enabled or not datasets:
            return
        with self._lock:
            manifest = self._read_json(self.paths["manifest"])
            if not isinstance(manifest, Mapping):
                return
            updated = dict(manifest)
            existing = manifest.get("datasets", {})
            existing = dict(existing) if isinstance(existing, Mapping) else {}
            existing.update({
                _safe_name(split): self._dataset_summary(dataset)
                for split, dataset in datasets.items()
            })
            updated["datasets"] = existing
            updated["last_dataset_registration_unix_seconds"] = int(time.time())
            _atomic_json(self.paths["manifest"], updated)

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError, TypeError):
            return None

    def _validate_preserved_text_artifacts(self) -> None:
        for path, expected in (
            (self.paths["horizons"], _HORIZON_FIELDS),
            (self.paths["slices"], _SLICE_FIELDS),
        ):
            if not path.exists() or not path.stat().st_size:
                continue
            try:
                with path.open(encoding="utf-8", newline="") as handle:
                    header = next(csv.reader(handle), [])
            except (OSError, csv.Error) as exc:
                raise ValueError(f"invalid preserved monitor CSV: {path.name}") from exc
            if tuple(header) != tuple(expected):
                raise ValueError(
                    f"preserved {path.name} has an unexpected schema"
                )

        for path in (self.paths["epochs"], self.paths["distillation"]):
            if not path.exists() or not path.stat().st_size:
                continue
            try:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        payload = json.loads(line)
                        if not isinstance(payload, Mapping):
                            raise ValueError("record is not a mapping")
                        safe = _json_safe(_sanitize_aggregate_payload(
                            dict(payload), min_count=self.min_slice_count,
                            min_days=self.min_slice_days,
                        ))
                        if safe != payload:
                            raise ValueError("record is not aggregate-safe")
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid preserved monitor JSONL: {path.name}"
                ) from exc

    @staticmethod
    def _count_jsonl(path: Path) -> int:
        if not path.exists():
            return 0
        try:
            with path.open(encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError:
            return 0

    @staticmethod
    def _count_forecast_evaluations(path: Path) -> int:
        if not path.exists() or not path.stat().st_size:
            return 0
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = csv.DictReader(handle)
                return len({
                    (row.get("phase"), row.get("epoch"), row.get("split"), row.get("role"))
                    for row in rows
                })
        except (OSError, csv.Error):
            return 0

    @staticmethod
    def _ensure_csv(path: Path, fieldnames: Sequence[str]) -> None:
        if path.exists() and path.stat().st_size:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _reset_csv(path: Path, fieldnames: Sequence[str]) -> None:
        header_buffer = []
        class _Buffer:
            def write(self, value: str) -> int:
                header_buffer.append(value)
                return len(value)
        csv.DictWriter(_Buffer(), fieldnames=fieldnames, lineterminator="\n").writeheader()
        _atomic_text(path, "".join(header_buffer))

    def new_data_quality_accumulator(
        self, split: str, feature_names: Optional[Sequence[str]] = None
    ) -> DataQualityAccumulator:
        return DataQualityAccumulator(
            split=split,
            feature_names=feature_names,
            reservoir_size=self.reservoir_size,
            rated_power=self.rated_power,
            seed=int(hashlib.sha256(str(split).encode("utf-8")).hexdigest()[:8], 16),
            min_aggregate_count=self.min_slice_count,
            min_aggregate_days=self.min_slice_days,
            forecast_step_minutes=self.forecast_step_minutes,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            direction_threshold=self.direction_threshold,
            large_ramp_threshold=self.large_ramp_threshold,
            near_zero_threshold=self.near_zero_threshold,
        )

    def result_value(self, key: str, default: Any = None) -> Any:
        """Return a previously finalized aggregate result after ``start_run``."""
        with self._lock:
            return self._final_payload.get(str(key), default)

    def record_data_quality(self, split: str, summary: Any) -> None:
        if not self.enabled:
            return
        if isinstance(summary, DataQualityAccumulator):
            summary = summary.summary()
        with self._lock:
            self._data_quality[_safe_name(split)] = _json_safe(
                self._suppress_low_count_distributions(summary))
            output = {
                "schema_version": 1,
                "privacy": "aggregate_only",
                **self._data_quality,
            }
            comparisons = self._data_quality_comparisons()
            if comparisons:
                output["comparisons_to_train"] = comparisons
            _atomic_json(self.paths["data_quality"], output)

    def _suppress_low_count_distributions(self, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        result = {
            key: self._suppress_low_count_distributions(item)
            for key, item in value.items()
        }
        count = _finite_float(result.get("count"))
        support_count = _finite_float(result.get("support_sample_count"))
        distinct_days = _finite_float(result.get("distinct_day_count"))
        total_support_count = _finite_float(result.get("total_sample_count"))
        total_distinct_days = _finite_float(result.get("total_distinct_day_count"))
        value_fields = {
            "mean", "std", "p01", "p05", "p25", "p50", "p75", "p95", "p99",
            "correlation",
            "mae_normalized", "rmse_normalized", "mae_original_units",
            "rmse_original_units",
        }
        coverage_fields = {"missing_rate", "valid_rate", "rate"}
        distribution_fields = value_fields | coverage_fields
        has_distribution_fields = bool(distribution_fields.intersection(result))
        statistical_support_invalid = (
            has_distribution_fields
            and (
                support_count is None
                or support_count < self.min_slice_count
                or distinct_days is None
                or distinct_days < self.min_slice_days
                or (count is not None and count < self.min_slice_count)
            )
        )
        missing_support_invalid = (
            bool(coverage_fields.intersection(result))
            and (
                (total_support_count if total_support_count is not None else support_count)
                is None
                or (total_support_count if total_support_count is not None else support_count)
                < self.min_slice_count
                or (total_distinct_days if total_distinct_days is not None else distinct_days)
                is None
                or (total_distinct_days if total_distinct_days is not None else distinct_days)
                < self.min_slice_days
            )
        )
        if (
            has_distribution_fields
            and (statistical_support_invalid or missing_support_invalid)
        ):
            fields_to_suppress = set()
            if statistical_support_invalid:
                fields_to_suppress.update(value_fields)
            if missing_support_invalid:
                fields_to_suppress.update(coverage_fields)
            for field in fields_to_suppress:
                if field in result:
                    result[field] = None
            result["privacy_suppressed"] = True
            result["privacy_min_count"] = self.min_slice_count
            result["privacy_min_days"] = self.min_slice_days
        return result

    @staticmethod
    def _distribution_shift(reference: Any, candidate: Any) -> Dict[str, Any]:
        if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
            return {}
        reference_mean = _finite_float(reference.get("mean"))
        candidate_mean = _finite_float(candidate.get("mean"))
        reference_std = _finite_float(reference.get("std"))
        candidate_std = _finite_float(candidate.get("std"))
        reference_missing = _finite_float(reference.get("missing_rate"))
        candidate_missing = _finite_float(candidate.get("missing_rate"))
        result: Dict[str, Any] = {}
        if reference_mean is not None and candidate_mean is not None:
            result["mean_delta"] = candidate_mean - reference_mean
            if reference_std is not None and reference_std > 1e-12:
                result["standardized_mean_delta"] = (
                    candidate_mean - reference_mean
                ) / reference_std
        if reference_std is not None and candidate_std is not None:
            result["std_ratio"] = candidate_std / reference_std if reference_std > 1e-12 else None
        if reference_missing is not None and candidate_missing is not None:
            result["missing_rate_delta"] = candidate_missing - reference_missing
        for quantile in ("p05", "p50", "p95"):
            reference_value = _finite_float(reference.get(quantile))
            candidate_value = _finite_float(candidate.get(quantile))
            if reference_value is not None and candidate_value is not None:
                result[f"{quantile}_delta"] = candidate_value - reference_value
        reference_p25 = _finite_float(reference.get("p25"))
        reference_p75 = _finite_float(reference.get("p75"))
        candidate_p25 = _finite_float(candidate.get("p25"))
        candidate_p75 = _finite_float(candidate.get("p75"))
        if None not in (
            reference_p25, reference_p75, candidate_p25, candidate_p75
        ):
            reference_iqr = reference_p75 - reference_p25
            candidate_iqr = candidate_p75 - candidate_p25
            result["iqr_ratio"] = (
                candidate_iqr / reference_iqr
                if reference_iqr > 1e-12 else None
            )
        return result

    @staticmethod
    def _categorical_rate_shift(reference: Any, candidate: Any) -> Dict[str, Any]:
        if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
            return {}
        allowed = {"0", "1", "2", "3", "other", "missing"}
        result = {}
        for label in sorted(allowed & set(reference) & set(candidate)):
            reference_value = _finite_float(reference.get(label))
            candidate_value = _finite_float(candidate.get(label))
            if reference_value is not None and candidate_value is not None:
                result[label] = candidate_value - reference_value
        return result

    def _data_quality_comparisons(self) -> Dict[str, Any]:
        train = self._data_quality.get("train")
        if not isinstance(train, Mapping):
            return {}
        output: Dict[str, Any] = {}
        train_features = train.get("features", {})
        for split, candidate in self._data_quality.items():
            if split == "train" or not isinstance(candidate, Mapping):
                continue
            split_result: Dict[str, Any] = {"features": {}}
            candidate_features = candidate.get("features", {})
            if isinstance(train_features, Mapping) and isinstance(candidate_features, Mapping):
                for feature in sorted(set(train_features) & set(candidate_features)):
                    shift = self._distribution_shift(
                        train_features[feature], candidate_features[feature]
                    )
                    if shift:
                        split_result["features"][feature] = shift
            for name in ("target", "current_power", "ramp"):
                shift = self._distribution_shift(train.get(name), candidate.get(name))
                if shift:
                    split_result[name] = shift
            train_completeness = train.get("sample_completeness", {})
            candidate_completeness = candidate.get("sample_completeness", {})
            if isinstance(train_completeness, Mapping) and isinstance(
                candidate_completeness, Mapping
            ):
                completeness_shift = {}
                for name in ("input_valid_fraction", "target_valid_fraction"):
                    shift = self._distribution_shift(
                        train_completeness.get(name),
                        candidate_completeness.get(name),
                    )
                    if shift:
                        completeness_shift[name] = shift
                if completeness_shift:
                    split_result["sample_completeness"] = completeness_shift

            train_horizons = train.get("horizon_label_profile", {})
            candidate_horizons = candidate.get("horizon_label_profile", {})
            if isinstance(train_horizons, Mapping) and isinstance(
                candidate_horizons, Mapping
            ):
                candidate_by_lead = {
                    item.get("lead_index"): item
                    for item in candidate_horizons.values()
                    if isinstance(item, Mapping)
                }
                horizon_shifts = {}
                for reference_horizon in train_horizons.values():
                    if not isinstance(reference_horizon, Mapping):
                        continue
                    lead = reference_horizon.get("lead_index")
                    candidate_horizon = candidate_by_lead.get(lead)
                    if not isinstance(candidate_horizon, Mapping):
                        continue
                    entry = {
                        "lead_index": lead,
                        "horizon_minutes": reference_horizon.get("horizon_minutes"),
                    }
                    for name in ("target", "ramp"):
                        shift = self._distribution_shift(
                            reference_horizon.get(name), candidate_horizon.get(name))
                        if shift:
                            entry[name] = shift
                    if len(entry) > 2:
                        horizon_shifts[f"lead_{int(lead):03d}"] = entry
                if horizon_shifts:
                    split_result["horizon_label_profile"] = horizon_shifts

            train_source_rates = train.get("source_code_rates", {})
            candidate_source_rates = candidate.get("source_code_rates", {})
            if isinstance(train_source_rates, Mapping) and isinstance(
                candidate_source_rates, Mapping
            ):
                source_shifts = {}
                for source in sorted(
                    set(train_source_rates) & set(candidate_source_rates)
                ):
                    shift = self._categorical_rate_shift(
                        train_source_rates[source], candidate_source_rates[source])
                    if shift:
                        source_shifts[source] = shift
                if source_shifts:
                    split_result["source_code_rate_delta"] = source_shifts
            output[split] = split_result
        return output

    def start_parameter_snapshot(self, model: Any) -> Optional[Dict[str, Any]]:
        if not self.enabled or model is None or not hasattr(model, "named_parameters"):
            return None
        parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        return {"started": time.monotonic(), "parameters": parameters}

    @staticmethod
    def _branch_name(parameter_name: str) -> str:
        pieces = parameter_name.split(".")
        if pieces and pieces[0] == "module":
            pieces = pieces[1:]
        return pieces[0] if pieces else "model"

    def finish_parameter_snapshot(
        self, model: Any, snapshot: Optional[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        empty = {
            "parameter_norm": None,
            "update_norm": None,
            "relative_update_norm": None,
            "updated_element_fraction": None,
            "snapshot_seconds": None,
            "branches": {},
        }
        if not self.enabled or not snapshot or model is None:
            return empty
        previous = snapshot.get("parameters", {})
        totals = {"parameter_sq": 0.0, "update_sq": 0.0, "elements": 0, "updated": 0}
        branches: Dict[str, Dict[str, float]] = {}
        with torch.no_grad() if torch is not None else _nullcontext():
            for name, parameter in model.named_parameters():
                if name not in previous:
                    continue
                current = parameter.detach().cpu()
                before = previous[name]
                if tuple(current.shape) != tuple(before.shape):
                    continue
                current64 = current.to(dtype=torch.float64)
                delta64 = current64 - before.to(dtype=torch.float64)
                parameter_sq = float(torch.sum(current64 * current64).item())
                update_sq = float(torch.sum(delta64 * delta64).item())
                elements = int(current.numel())
                updated = int(torch.count_nonzero(delta64).item())
                totals["parameter_sq"] += parameter_sq
                totals["update_sq"] += update_sq
                totals["elements"] += elements
                totals["updated"] += updated
                branch = branches.setdefault(
                    self._branch_name(name),
                    {"parameter_sq": 0.0, "update_sq": 0.0, "elements": 0, "updated": 0},
                )
                branch["parameter_sq"] += parameter_sq
                branch["update_sq"] += update_sq
                branch["elements"] += elements
                branch["updated"] += updated

        def finish(values: Mapping[str, float]) -> Dict[str, Any]:
            parameter_norm = math.sqrt(max(0.0, float(values["parameter_sq"])))
            update_norm = math.sqrt(max(0.0, float(values["update_sq"])))
            elements = int(values["elements"])
            return {
                "parameter_norm": parameter_norm,
                "update_norm": update_norm,
                "relative_update_norm": update_norm / parameter_norm if parameter_norm else None,
                "element_count": elements,
                "updated_element_fraction": float(values["updated"] / elements) if elements else None,
            }

        result = finish(totals)
        result["snapshot_seconds"] = max(0.0, time.monotonic() - float(snapshot.get("started", time.monotonic())))
        result["branches"] = {name: finish(values) for name, values in sorted(branches.items())}
        return _json_safe(result)

    def record_epoch(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        record = {"stage": self.stage, **dict(payload)}
        sample_count = _finite_float(record.get("processed_samples"))
        performance = record.get("performance")
        if sample_count is None and isinstance(performance, Mapping):
            sample_count = _finite_float(performance.get("samples"))
        suppress_metrics = bool(
            sample_count is not None and sample_count < self.min_slice_count)
        if suppress_metrics:
            record["privacy_suppressed"] = True
        safe = _json_safe(_sanitize_aggregate_payload(
            record, min_count=self.min_slice_count,
            min_days=self.min_slice_days,
            suppress_metrics=suppress_metrics))
        with self._lock:
            _append_jsonl(self.paths["epochs"], safe)
            self._epoch_summaries.append(safe)
            self._epoch_record_count += 1

    @staticmethod
    def _forecast_matrix(value: Any, name: str) -> np.ndarray:
        array = _to_numpy(value, np.float64)
        if array.ndim == 0:
            return array.reshape(1, 1)
        while array.ndim > 2 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim == 1:
            return array.reshape(-1, 1)
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            return array.reshape(array.shape[0], -1)
        raise ValueError(f"{name} must have sample and horizon dimensions")

    @staticmethod
    def _broadcast_mask(mask: Any, shape: Sequence[int]) -> np.ndarray:
        array = _to_numpy(mask, bool)
        while array.ndim > len(shape) and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim == 1 and array.shape[0] == shape[0]:
            array = array[:, None]
        try:
            return np.broadcast_to(array, shape).copy()
        except ValueError as exc:
            raise ValueError("target_mask is not broadcastable to forecast shape") from exc

    @staticmethod
    def _current_matrix(current_power: Any, shape: Sequence[int]) -> np.ndarray:
        current = _to_numpy(current_power, np.float64)
        while current.ndim > 1 and current.shape[-1] == 1:
            current = current[..., 0]
        if current.ndim == 0:
            return np.full(shape, float(current))
        if current.ndim == 1:
            if current.size == shape[0]:
                current = current[:, None]
            elif current.size == shape[1]:
                current = current[None, :]
        try:
            return np.broadcast_to(current, shape).copy()
        except ValueError as exc:
            raise ValueError("current_power is not broadcastable to forecast shape") from exc

    def _clip(self, prediction: np.ndarray) -> np.ndarray:
        lower = -np.inf if self.clip_min is None else self.clip_min
        upper = np.inf if self.clip_max is None else self.clip_max
        return np.clip(prediction, lower, upper)

    def _metric_block(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
        current: Optional[np.ndarray] = None,
        privacy_allowed: bool = True,
    ) -> Dict[str, Any]:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(target) & np.isfinite(prediction)
        count = int(valid.sum())
        privacy_suppressed = bool(
            count and (count < self.min_slice_count or not privacy_allowed)
        )
        if not count or privacy_suppressed:
            return {
                "count": 0, "rmse": None, "mae": None, "bias": None,
                "nrmse": None, "nmae": None, "corr": None,
                "persistence_count": 0, "persistence_rmse": None,
                "persistence_mae": None, "persistence_skill_rmse": None,
                "persistence_skill_mae": None,
                "privacy_suppressed": privacy_suppressed,
                "persistence_privacy_suppressed": privacy_suppressed,
                **({"count": count} if count else {}),
            }
        pred = prediction[valid]
        truth = target[valid]
        errors_normalized = pred - truth
        errors = errors_normalized * self.rated_power
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        mae = float(np.mean(np.abs(errors)))
        bias = float(np.mean(errors))
        corr = None
        if count >= 2 and float(np.std(pred)) > 0.0 and float(np.std(truth)) > 0.0:
            candidate = float(np.corrcoef(pred, truth)[0, 1])
            corr = candidate if math.isfinite(candidate) else None
        block: Dict[str, Any] = {
            "count": count,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "nrmse": rmse / self.rated_power,
            "nmae": mae / self.rated_power,
            "corr": corr,
            "privacy_suppressed": False,
            "persistence_count": 0,
            "persistence_privacy_suppressed": False,
            "persistence_rmse": None,
            "persistence_mae": None,
            "persistence_skill_rmse": None,
            "persistence_skill_mae": None,
        }
        if current is not None:
            persistence_valid = valid & np.isfinite(current)
            persistence_count = int(persistence_valid.sum())
            block["persistence_count"] = persistence_count
            if persistence_count and persistence_count < self.min_slice_count:
                block["persistence_privacy_suppressed"] = True
            elif persistence_count:
                baseline_error = (current[persistence_valid] - target[persistence_valid]) * self.rated_power
                model_error = (prediction[persistence_valid] - target[persistence_valid]) * self.rated_power
                baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_error))))
                baseline_mae = float(np.mean(np.abs(baseline_error)))
                comparable_rmse = float(np.sqrt(np.mean(np.square(model_error))))
                comparable_mae = float(np.mean(np.abs(model_error)))
                block["persistence_rmse"] = baseline_rmse
                block["persistence_mae"] = baseline_mae
                block["persistence_skill_rmse"] = (
                    1.0 - comparable_rmse / baseline_rmse if baseline_rmse > 0 else None
                )
                block["persistence_skill_mae"] = (
                    1.0 - comparable_mae / baseline_mae if baseline_mae > 0 else None
                )
        return block

    @staticmethod
    def _distinct_day_count(
        issue_dates: Optional[np.ndarray], valid: np.ndarray
    ) -> Optional[int]:
        if issue_dates is None:
            return None
        mask = np.asarray(valid, dtype=bool)
        selected_samples = mask if mask.ndim == 1 else np.any(mask, axis=1)
        selected_dates = issue_dates[selected_samples]
        selected_dates = selected_dates[~np.isnat(selected_dates)]
        return int(np.unique(selected_dates).size)

    @staticmethod
    def _metadata_per_sample(metadata: Mapping[str, Any], key: str, samples: int) -> Optional[np.ndarray]:
        value = metadata.get(key)
        if value is None:
            return None
        array = _to_numpy(value)
        if array.ndim == 0:
            return np.full(samples, array.item())
        if array.shape[0] != samples:
            return None
        return array

    @classmethod
    def _issue_dates(
        cls, metadata: Mapping[str, Any], samples: int
    ) -> Optional[np.ndarray]:
        issue_time = cls._metadata_per_sample(metadata, "issue_time_ns", samples)
        if issue_time is None:
            return None
        try:
            flattened = issue_time.reshape(samples, -1)[:, 0]
            if np.issubdtype(flattened.dtype, np.datetime64):
                datetimes = flattened.astype("datetime64[ns]")
            else:
                datetimes = flattened.astype(np.int64).astype("datetime64[ns]")
            return datetimes.astype("datetime64[D]")
        except (TypeError, ValueError, OverflowError):
            return None

    def _slice_definitions(
        self,
        target: np.ndarray,
        current: np.ndarray,
        base_valid: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> Iterable[tuple[str, str, np.ndarray]]:
        target_capacity_bins = (
            ("near_zero", -np.inf, self.near_zero_threshold),
            ("low", self.near_zero_threshold, 0.25),
            ("medium", 0.25, 0.75),
            ("high", 0.75, 0.95),
            ("at_or_above_capacity", 0.95, np.inf),
        )
        for label, lower, upper in target_capacity_bins:
            yield "target_capacity", label, base_valid & (target >= lower) & (target < upper)

        current_capacity_bins = (
            ("near_zero", -np.inf, self.near_zero_threshold),
            ("low", self.near_zero_threshold, 0.25),
            ("medium", 0.25, 0.75),
            ("high", 0.75, 0.95),
            ("at_or_above_capacity", 0.95, np.inf),
        )
        for label, lower, upper in current_capacity_bins:
            yield "current_capacity", label, base_valid & (current >= lower) & (current < upper)

        ramp = target - current
        magnitude = np.abs(ramp)
        yield "ramp_magnitude", "stable", base_valid & (magnitude < self.direction_threshold)
        yield "ramp_magnitude", "moderate", base_valid & (magnitude >= self.direction_threshold) & (magnitude < self.large_ramp_threshold)
        yield "ramp_magnitude", "large", base_valid & (magnitude >= self.large_ramp_threshold)
        yield "direction", "down", base_valid & (ramp <= -self.direction_threshold)
        yield "direction", "stable", base_valid & (magnitude < self.direction_threshold)
        yield "direction", "up", base_valid & (ramp >= self.direction_threshold)
        yield "near_zero", "yes", base_valid & (target < self.near_zero_threshold)
        yield "near_zero", "no", base_valid & (target >= self.near_zero_threshold)

        samples = target.shape[0]
        image_mask = self._metadata_per_sample(metadata, "image_mask", samples)
        if image_mask is not None:
            if image_mask.ndim == 1:
                coverage = image_mask.astype(np.float64)
            else:
                coverage = image_mask.reshape(samples, -1).astype(bool).mean(axis=1)
            categories = {
                "missing": coverage == 0.0,
                "partial": (coverage > 0.0) & (coverage < 1.0),
                "complete": coverage == 1.0,
            }
            for label, sample_mask in categories.items():
                yield "image_coverage", label, base_valid & sample_mask[:, None]

        missing_fraction = self._metadata_per_sample(metadata, "input_missing_fraction", samples)
        if missing_fraction is None:
            timeseries_mask = self._metadata_per_sample(metadata, "timeseries_mask", samples)
            if timeseries_mask is not None:
                if timeseries_mask.ndim == 1:
                    missing_fraction = 1.0 - timeseries_mask.astype(np.float64)
                else:
                    missing_fraction = 1.0 - timeseries_mask.reshape(samples, -1).astype(bool).mean(axis=1)
        if missing_fraction is not None:
            missing_fraction = np.asarray(missing_fraction, np.float64).reshape(samples, -1).mean(axis=1)
            groups = {
                "none": missing_fraction == 0.0,
                "low": (missing_fraction > 0.0) & (missing_fraction < 0.10),
                "high": missing_fraction >= 0.10,
            }
            for label, sample_mask in groups.items():
                yield "input_missingness", label, base_valid & sample_mask[:, None]

        # A caller may provide a non-identifying categorical label such as day/night.
        period = self._metadata_per_sample(metadata, "period", samples)
        if period is None:
            period = self._metadata_per_sample(metadata, "day_night", samples)
        if period is not None:
            period = period.reshape(samples, -1)[:, 0]
            for raw_label in np.unique(period):
                label = _safe_name(raw_label, maximum=24)
                yield "period", label, base_valid & (period == raw_label)[:, None]
        else:
            # Timestamps are used only in memory to derive a coarse period and
            # are never included in returned summaries or output files.
            issue_dates = self._issue_dates(metadata, samples)
            issue_time = self._metadata_per_sample(metadata, "issue_time_ns", samples)
            if issue_dates is not None and issue_time is not None:
                try:
                    flattened = issue_time.reshape(samples, -1)[:, 0]
                    if np.issubdtype(flattened.dtype, np.datetime64):
                        datetimes = flattened.astype("datetime64[ns]")
                    else:
                        datetimes = flattened.astype(np.int64).astype("datetime64[ns]")
                    time_valid = ~np.isnat(datetimes)
                    hours = datetimes.astype("datetime64[h]").astype(np.int64) % 24
                    day = time_valid & (hours >= 6) & (hours < 18)
                    night = time_valid & ~day
                    yield "period", "day", base_valid & day[:, None]
                    yield "period", "night", base_valid & night[:, None]
                except (TypeError, ValueError, OverflowError):
                    pass

    def record_forecast(
        self,
        epoch: Any,
        split: str,
        role: str,
        predictions: Any,
        targets: Any,
        current_power: Any,
        target_mask: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        phase: str = "epoch",
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        prediction = self._forecast_matrix(predictions, "predictions")
        target = self._forecast_matrix(targets, "targets")
        if prediction.shape != target.shape:
            raise ValueError("predictions and targets must have identical shapes")
        current = self._current_matrix(current_power, target.shape)
        valid = np.ones(target.shape, dtype=bool) if target_mask is None else self._broadcast_mask(
            target_mask, target.shape
        )
        valid &= np.isfinite(target)
        metadata = metadata if isinstance(metadata, Mapping) else {}
        current_valid = metadata.get("current_power_valid")
        if current_valid is not None:
            current_valid_matrix = self._broadcast_mask(current_valid, target.shape)
            current = np.where(current_valid_matrix, current, np.nan)

        prediction_finite = np.isfinite(prediction)
        invalid_prediction_count = int(np.sum(valid & ~prediction_finite))
        metric_valid = valid & prediction_finite
        clipped = self._clip(prediction)
        below = np.zeros_like(prediction, dtype=bool) if self.clip_min is None else prediction < self.clip_min
        above = np.zeros_like(prediction, dtype=bool) if self.clip_max is None else prediction > self.clip_max
        bound_denominator = int(np.sum(valid & prediction_finite))

        context = {
            "phase": _safe_name(phase),
            "epoch": _json_safe(epoch),
            "split": _safe_name(split),
            "role": _safe_name(role),
        }
        issue_dates = self._issue_dates(metadata, target.shape[0])
        horizon_rows = []
        for lead in range(target.shape[1]):
            lead_metric_valid = metric_valid[:, lead]
            lead_day_count = self._distinct_day_count(
                issue_dates, lead_metric_valid)
            lead_privacy_allowed = (
                lead_day_count is not None
                and lead_day_count >= self.min_slice_days
            )
            pre = self._metric_block(
                prediction[:, lead], target[:, lead], lead_metric_valid,
                current[:, lead], privacy_allowed=lead_privacy_allowed,
            )
            post = self._metric_block(
                clipped[:, lead], target[:, lead], lead_metric_valid,
                current[:, lead], privacy_allowed=lead_privacy_allowed,
            )
            lead_valid = valid[:, lead]
            lead_finite = lead_valid & prediction_finite[:, lead]
            lead_denominator = int(lead_finite.sum())
            row = {
                **context,
                "lead_index": lead + 1,
                "horizon_minutes": (lead + 1) * self.forecast_step_minutes,
                "distinct_day_count": lead_day_count,
                **pre,
                "clipped_rmse": post["rmse"],
                "clipped_mae": post["mae"],
                "clipped_bias": post["bias"],
                "clipped_nrmse": post["nrmse"],
                "clipped_nmae": post["nmae"],
                "below_floor_rate": (
                    float(np.sum(lead_finite & below[:, lead]) / lead_denominator)
                    if lead_denominator >= self.min_slice_count
                    and lead_privacy_allowed else None
                ),
                "above_capacity_rate": (
                    float(np.sum(lead_finite & above[:, lead]) / lead_denominator)
                    if lead_denominator >= self.min_slice_count
                    and lead_privacy_allowed else None
                ),
                "out_of_bounds_rate": (
                    float(np.sum(lead_finite & (below[:, lead] | above[:, lead])) / lead_denominator)
                    if lead_denominator >= self.min_slice_count
                    and lead_privacy_allowed else None
                ),
                "invalid_prediction_count": int(np.sum(lead_valid & ~prediction_finite[:, lead])),
            }
            horizon_rows.append(row)

        slice_rows = []
        for dimension, value, slice_mask in self._slice_definitions(target, current, metric_valid, metadata):
            count = int(slice_mask.sum())
            if count < self.min_slice_count:
                continue
            distinct_day_count = self._distinct_day_count(issue_dates, slice_mask)
            if (
                distinct_day_count is None
                or distinct_day_count < self.min_slice_days
            ):
                continue
            pre = self._metric_block(prediction, target, slice_mask, current)
            post = self._metric_block(clipped, target, slice_mask, current)
            slice_rows.append({
                **context,
                "slice_dimension": dimension,
                "slice_value": value,
                "distinct_day_count": distinct_day_count,
                **pre,
                "clipped_rmse": post["rmse"],
                "clipped_mae": post["mae"],
            })

        overall_day_count = self._distinct_day_count(issue_dates, metric_valid)
        overall_privacy_allowed = (
            overall_day_count is not None
            and overall_day_count >= self.min_slice_days
        )
        overall = self._metric_block(
            prediction, target, metric_valid, current,
            privacy_allowed=overall_privacy_allowed)
        overall_clipped = self._metric_block(
            clipped, target, metric_valid, current,
            privacy_allowed=overall_privacy_allowed)
        aggregate_allowed = (
            bound_denominator >= self.min_slice_count
            and overall_privacy_allowed
        )
        below_rate = (
            float(np.sum(valid & prediction_finite & below) / bound_denominator)
            if aggregate_allowed else None)
        above_rate = (
            float(np.sum(valid & prediction_finite & above) / bound_denominator)
            if aggregate_allowed else None)
        out_rate = (
            float(np.sum(valid & prediction_finite & (below | above)) / bound_denominator)
            if aggregate_allowed else None)
        summary = _json_safe({
            **context,
            "shape": {"samples": target.shape[0], "horizons": target.shape[1]},
            "rated_power": self.rated_power,
            "units": self.units,
            "overall": overall,
            "overall_clipped": overall_clipped,
            "below_floor_rate": below_rate,
            "above_capacity_rate": above_rate,
            "out_of_bounds_rate": out_rate,
            "invalid_prediction_count": invalid_prediction_count,
            "valid_target_count": int(valid.sum()),
            "distinct_day_count": overall_day_count,
            "privacy_suppressed": not aggregate_allowed and bound_denominator > 0,
            "prediction_mean_normalized": float(prediction[metric_valid].mean()) if aggregate_allowed else None,
            "prediction_std_normalized": float(prediction[metric_valid].std()) if aggregate_allowed else None,
            "target_mean_normalized": float(target[metric_valid].mean()) if aggregate_allowed else None,
            "target_std_normalized": float(target[metric_valid].std()) if aggregate_allowed else None,
            "reported_slice_count": len(slice_rows),
            "min_slice_count": self.min_slice_count,
            "min_slice_days": self.min_slice_days,
        })
        with self._lock:
            self._append_csv(self.paths["horizons"], _HORIZON_FIELDS, horizon_rows)
            self._append_csv(self.paths["slices"], _SLICE_FIELDS, slice_rows)
            self._forecast_summaries.append(summary)
            self._forecast_evaluation_count += 1
        return summary

    @staticmethod
    def _append_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        TrainingMonitor._ensure_csv(path, fields)
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            for row in rows:
                safe = _json_safe(row)
                writer.writerow({key: "" if safe.get(key) is None else safe.get(key, "") for key in fields})
            handle.flush()
            os.fsync(handle.fileno())

    def record_distillation(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        safe = _json_safe(_sanitize_aggregate_payload(
            {"stage": self.stage, **dict(payload)},
            min_count=self.min_slice_count,
            min_days=self.min_slice_days,
        ))
        with self._lock:
            _append_jsonl(self.paths["distillation"], safe)

    @staticmethod
    def _nested_number(payload: Mapping[str, Any], *paths: str) -> Optional[float]:
        for path in paths:
            value: Any = payload
            for component in path.split("."):
                if not isinstance(value, Mapping) or component not in value:
                    value = None
                    break
                value = value[component]
            candidate = _finite_float(value)
            if candidate is not None:
                return candidate
        return None

    def _alerts(self) -> list[Dict[str, Any]]:
        alerts: list[Dict[str, Any]] = []
        for forecast in self._forecast_summaries:
            label = f"{forecast.get('split')}/{forecast.get('role')}/epoch={forecast.get('epoch')}"
            count = int(forecast.get("valid_target_count", 0) or 0)
            if count == 0:
                alerts.append({"code": "no_valid_targets", "context": label})
                continue
            out_rate = _finite_float(forecast.get("out_of_bounds_rate"))
            if out_rate is not None and out_rate >= 0.01:
                alerts.append({"code": "prediction_out_of_bounds", "context": label, "rate": out_rate})
            pred_std = _finite_float(forecast.get("prediction_std_normalized"))
            target_std = _finite_float(forecast.get("target_std_normalized"))
            if pred_std is not None and target_std is not None and target_std > 0.01 and pred_std < target_std * 0.05:
                alerts.append({"code": "prediction_collapse", "context": label, "prediction_std": pred_std, "target_std": target_std})
            skill = self._nested_number(forecast, "overall.persistence_skill_rmse")
            if skill is not None and skill <= 0:
                alerts.append({"code": "not_better_than_persistence", "context": label, "skill_rmse": skill})
        validation_history = []
        learning_rates = []
        for epoch in self._epoch_summaries:
            grad = self._nested_number(
                epoch,
                "gradient_norm",
                "gradient_norm_mean",
                "grad_norm",
                "grad.global_norm",
                "gradients.global_norm_mean",
                "gradients.pre_clip_norm_mean",
            )
            if grad is not None and grad <= 1e-12:
                alerts.append({"code": "vanishing_gradient", "epoch": epoch.get("epoch"), "gradient_norm": grad})
            nonfinite_gradients = self._nested_number(
                epoch,
                "nonfinite_gradient_steps",
                "gradients.nonfinite_count",
            )
            if nonfinite_gradients is not None and nonfinite_gradients > 0:
                alerts.append({
                    "code": "nonfinite_gradient",
                    "epoch": epoch.get("epoch"),
                    "step_count": int(nonfinite_gradients),
                })
            amp_skips = self._nested_number(epoch, "amp_skipped_steps", "amp.skipped_steps")
            if amp_skips is not None and amp_skips > 0:
                alerts.append({
                    "code": "amp_steps_skipped",
                    "epoch": epoch.get("epoch"),
                    "step_count": int(amp_skips),
                })
            relative_update = self._nested_number(
                epoch,
                "parameter_updates.relative_update_norm",
                "parameter_update.relative_update_norm",
            )
            if relative_update is not None and relative_update <= 1e-12:
                alerts.append({
                    "code": "negligible_parameter_update",
                    "epoch": epoch.get("epoch"),
                    "relative_update_norm": relative_update,
                })
            validation_loss = self._nested_number(
                epoch,
                "validation_loss",
                "validation.total_loss",
                "validation.task_loss",
            )
            if validation_loss is not None:
                validation_history.append((epoch.get("epoch"), validation_loss))
            learning_rate = self._nested_number(
                epoch, "learning_rate", "optimizer.learning_rate"
            )
            if learning_rate is not None:
                learning_rates.append((epoch.get("epoch"), learning_rate))

        if len(validation_history) >= 2:
            best_epoch, best_loss = min(validation_history, key=lambda item: item[1])
            last_epoch, last_loss = validation_history[-1]
            if best_loss > 0 and last_loss > best_loss * 1.20:
                alerts.append({
                    "code": "validation_degraded_after_best",
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                    "last_epoch": last_epoch,
                    "last_loss": last_loss,
                })
        initial_lr = _finite_float(self.safe_args.get("learning_rate"))
        if initial_lr is not None and initial_lr > 0 and learning_rates:
            last_epoch, last_lr = learning_rates[-1]
            if last_lr < initial_lr * 1e-4:
                alerts.append({
                    "code": "learning_rate_near_zero",
                    "epoch": last_epoch,
                    "learning_rate": last_lr,
                    "initial_learning_rate": initial_lr,
                })
        # Keep summary bounded even for very long runs.
        return alerts[-100:]

    def finalize(self, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        if payload:
            self._final_payload.update(_sanitize_aggregate_payload(
                dict(payload), min_count=self.min_slice_count,
                min_days=self.min_slice_days))
        alerts = [*self._existing_alerts, *self._alerts()]
        # Deduplicate strict-JSON alert objects while retaining chronological order.
        unique_alerts = []
        seen_alerts = set()
        for alert in alerts:
            safe_alert = _json_safe(alert)
            marker = json.dumps(safe_alert, sort_keys=True, allow_nan=False)
            if marker not in seen_alerts:
                seen_alerts.add(marker)
                unique_alerts.append(safe_alert)
        summary = _json_safe({
            "schema_version": 1,
            "privacy": "aggregate_only",
            "stage": self.stage,
            "status": self._final_payload.get("status", "completed"),
            "rated_power": self.rated_power,
            "units": self.units,
            "epochs_recorded": self._epoch_record_count,
            "forecast_evaluations_recorded": self._forecast_evaluation_count,
            "data_quality_splits": sorted(self._data_quality),
            "alerts": unique_alerts[-100:],
            "result": self._final_payload,
        })
        with self._lock:
            _atomic_json(self.paths["summary"], summary)
        return summary


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: Any) -> None:
        return None


__all__ = ["DataQualityAccumulator", "TrainingMonitor"]
