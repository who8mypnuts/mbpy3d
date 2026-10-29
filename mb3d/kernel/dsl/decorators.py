"""The ``@kfunc`` decorator: one Python function, three back ends.

A kernel function is compiled by numba for the CPU (the module level name
is bound to the numba dispatcher, so kernel functions call each other
directly) and registered for the GLSL / OpenCL transpiler, which reads the
original Python source.
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Callable

import numba
import numpy as np

_REGISTRY: dict[str, "KernelFunction"] = {}


@dataclass
class KernelFunction:
    """A registered kernel function and its compiled forms."""

    name: str
    py_func: Callable
    dispatcher: object
    source: str = field(repr=False)
    module: str = ""

    def __call__(self, *args):
        return self.dispatcher(*args)

    @staticmethod
    def from_source(name: str, source: str, namespace: dict) -> "KernelFunction":
        """A kernel function from generated source (transpile only, not JIT compiled)."""
        ns = dict(namespace)
        exec(compile(source, f"<{name}>", "exec"), ns)
        return KernelFunction(name, ns[name], None, source, "<generated>")


def kfunc(func: Callable | None = None, *, cache: bool = True):
    """Register ``func`` as a kernel function and JIT it with numba."""

    def wrap(f: Callable):
        source = textwrap.dedent(inspect.getsource(f))
        dispatcher = numba.njit(cache=cache, nogil=True, error_model="numpy")(f)
        kf = KernelFunction(f.__name__, f, dispatcher, source, f.__module__)
        _REGISTRY[f.__name__] = kf
        dispatcher._kernel_function = kf  # type: ignore[attr-defined]
        return dispatcher

    return wrap(func) if func is not None else wrap


def registry() -> dict[str, KernelFunction]:
    return _REGISTRY


def lookup(name: str) -> KernelFunction | None:
    return _REGISTRY.get(name)


@numba.njit(cache=True, nogil=True)
def local_real(n):
    """A local array of reals (``REAL name[n]`` on the GPU)."""
    return np.zeros(n)


@numba.njit(cache=True, nogil=True)
def local_int(n):
    """A local array of ints (``int name[n]`` on the GPU)."""
    return np.zeros(n, dtype=np.int32)
