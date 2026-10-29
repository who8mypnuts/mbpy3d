"""The kernel DSL: write the math once, run it on numba, GLSL and OpenCL."""

from .decorators import KernelFunction, kfunc, local_int, local_real, lookup, registry
from .targets import GLSL, OPENCL_DOUBLE, OPENCL_FLOAT, Target
from .transpiler import TranspileError, transpile
from .types import Bool, ConstIntArray, ConstRealArray, Int, IntArray, Real, RealArray

__all__ = [
    "kfunc", "local_real", "local_int", "lookup", "registry", "KernelFunction",
    "GLSL", "OPENCL_DOUBLE", "OPENCL_FLOAT", "Target", "transpile", "TranspileError",
    "Real", "Int", "Bool", "RealArray", "IntArray", "ConstRealArray", "ConstIntArray",
]
