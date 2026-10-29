"""Type markers of the kernel DSL.

Kernel functions are ordinary Python functions annotated with these types.
They run unchanged under numba on the CPU and are transpiled to GLSL and
OpenCL C for the graphics card.  Only a small, C-like subset of Python is
allowed (see ``transpiler.py``).
"""

from __future__ import annotations

from dataclasses import dataclass


class Real(float):
    """A floating point scalar (double on CPU/OpenCL, float in GLSL)."""


class Int(int):
    """A 32 bit integer scalar."""


class Bool(int):
    """A boolean."""


@dataclass(frozen=True)
class ArrayType:
    """An array parameter: element type, fixed size and storage class."""

    elem: str  # "real" or "int"
    size: int
    const: bool  # read only context data (GLSL uniform, OpenCL __constant)

    @property
    def is_real(self) -> bool:
        return self.elem == "real"


class _ArrayFactory:
    """``RealArray[8]`` -> ArrayType("real", 8, False)."""

    def __init__(self, elem: str, const: bool):
        self._elem = elem
        self._const = const

    def __getitem__(self, size: int) -> ArrayType:
        return ArrayType(self._elem, int(size), self._const)


RealArray = _ArrayFactory("real", False)
IntArray = _ArrayFactory("int", False)
ConstRealArray = _ArrayFactory("real", True)
ConstIntArray = _ArrayFactory("int", True)

SCALAR_NAMES = {Real: "real", Int: "int", Bool: "bool"}


def scalar_name(annotation) -> str | None:
    """The DSL type name of a scalar annotation, None for arrays/unknown."""
    return SCALAR_NAMES.get(annotation)
