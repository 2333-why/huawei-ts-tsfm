"""Lazy construction of registered foundation-model backends."""

from __future__ import annotations

import importlib
from typing import Any, Optional

from .registry import FoundationModelSpec, get_model_spec


def _resolve_backend(spec: FoundationModelSpec) -> Any:
    entrypoint = spec.entrypoint
    if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
        raise ImportError(
            f"invalid backend entrypoint {entrypoint!r} for foundation model "
            f"{spec.name!r}; expected module:attribute"
        )
    module_name, attribute = entrypoint.split(":", 1)
    if not module_name or not attribute:
        raise ImportError(
            f"invalid backend entrypoint {entrypoint!r} for foundation model "
            f"{spec.name!r}; expected module:attribute"
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(
            f"could not import backend for foundation model {spec.name!r} from "
            f"{entrypoint!r}: {exc}"
        ) from exc
    try:
        backend = getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(
            f"backend entrypoint {entrypoint!r} for foundation model "
            f"{spec.name!r} does not define {attribute!r}"
        ) from exc
    if not callable(backend):
        raise ImportError(
            f"backend entrypoint {entrypoint!r} for foundation model "
            f"{spec.name!r} is not callable"
        )
    return backend


def build_backend(name: str, device: Any, revision: Optional[str] = None) -> Any:
    """Build a registered backend, resolving optional code only at build time."""

    spec = get_model_spec(name)
    backend_class = _resolve_backend(spec)
    resolved_revision = spec.revision if revision is None else revision
    try:
        return backend_class(
            spec=spec,
            device=device,
            revision=resolved_revision,
        )
    except ImportError as exc:
        raise ImportError(
            f"could not construct backend for foundation model {spec.name!r} "
            f"from {spec.entrypoint!r}: {exc}"
        ) from exc


__all__ = ["build_backend"]
