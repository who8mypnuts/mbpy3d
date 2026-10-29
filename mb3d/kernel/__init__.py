"""Kernel: the DSL and the ray marching code written in it.

numba caches compiled kernels on disk, but does not notice when a function
another module depends on changes.  The cache directory is therefore named
after a hash of all kernel sources: any change starts a fresh cache.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_SOURCE_DIRS = [_PKG / "dsl", _PKG / "source", _PKG.parent / "formulas"]


def kernel_source_hash() -> str:
    h = hashlib.sha1()
    for d in _SOURCE_DIRS:
        for f in sorted(d.rglob("*.py")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def cache_dir() -> Path:
    base = os.environ.get("MB3D_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".cache" / "mb3d"
    d = root / "numba" / kernel_source_hash()
    d.mkdir(parents=True, exist_ok=True)
    return d


if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir())
