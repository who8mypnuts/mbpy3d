"""Loads a program's generated numba module from the cache directory (so
that numba's on-disk cache works and a formula combination compiles once
per machine)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import ModuleType

from .program import Program

_LOCK = threading.Lock()
_MODULES: dict[str, ModuleType] = {}


def cache_dir() -> Path:
    root = os.environ.get("PYMB3D_CACHE")
    if root:
        d = Path(root)
    elif sys.platform == "win32":
        d = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "pymb3d" / "cache"
    else:
        d = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pymb3d"
    d = d / "gen"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_program_module(program: Program) -> ModuleType:
    src = program.py_source()
    h = hashlib.sha1(src.encode()).hexdigest()[:16]
    with _LOCK:
        if h in _MODULES:
            return _MODULES[h]
        name = f"pymb3d_gen_{h}"
        path = cache_dir() / f"{name}.py"
        if not path.exists() or path.read_text() != src:
            path.write_text(src)
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _MODULES[h] = mod
        return mod
