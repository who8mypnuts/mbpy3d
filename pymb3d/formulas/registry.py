"""The formula registry: formulas by name (case insensitive, with aliases)."""
from __future__ import annotations

import importlib
import pkgutil

from .dsl import FormulaSpec

_REGISTRY: dict[str, FormulaSpec] = {}
_ALIASES: dict[str, str] = {}
_LOADED = False


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def register(spec: FormulaSpec) -> None:
    _REGISTRY[_norm(spec.name)] = spec
    for a in spec.aliases:
        _ALIASES[_norm(a)] = _norm(spec.name)


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from . import builtin  # noqa: F401

    for m in pkgutil.iter_modules(builtin.__path__):
        importlib.import_module(f"{builtin.__name__}.{m.name}")


def lookup(name: str) -> FormulaSpec:
    _ensure_loaded()
    key = _norm(name)
    key = _ALIASES.get(key, key)
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown formula '{name}' (known: {', '.join(sorted(names()))})") from None


def has(name: str) -> bool:
    _ensure_loaded()
    key = _norm(name)
    return key in _REGISTRY or key in _ALIASES


def names() -> list[str]:
    _ensure_loaded()
    return sorted((s.name for s in _REGISTRY.values()), key=str.lower)


def all_specs() -> list[FormulaSpec]:
    _ensure_loaded()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def by_mb3d_id(fid: int) -> FormulaSpec | None:
    _ensure_loaded()
    for s in _REGISTRY.values():
        if s.mb3d_id == fid:
            return s
    return None
