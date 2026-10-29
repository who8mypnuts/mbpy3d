"""Back end descriptions for the transpiler (GLSL and OpenCL C)."""

from __future__ import annotations

from dataclasses import dataclass, field

# math functions on reals: python name -> (glsl, opencl)
_REAL_MATH = {
    "sqrt": ("sqrt", "sqrt"),
    "sin": ("sin", "sin"),
    "cos": ("cos", "cos"),
    "tan": ("tan", "tan"),
    "asin": ("asin", "asin"),
    "acos": ("acos", "acos"),
    "atan": ("atan", "atan"),
    "atan2": ("atan", "atan2"),
    "exp": ("exp", "exp"),
    "log": ("log", "log"),
    "pow": ("pow", "pow"),
    "floor": ("floor", "floor"),
    "ceil": ("ceil", "ceil"),
    "fabs": ("abs", "fabs"),
    "abs": ("abs", "fabs"),
    "min": ("min", "fmin"),
    "max": ("max", "fmax"),
    "isnan": ("isnan", "isnan"),
    "isinf": ("isinf", "isinf"),
    "fmod": ("mod", "fmod"),
    "trunc": ("trunc", "trunc"),
    "round": ("round", "round"),
}


@dataclass(frozen=True)
class Target:
    """How a back end spells types, math functions and context arrays."""

    name: str
    real: str
    # const arrays are GLSL uniforms (dropped from parameter lists) or
    # explicit __constant pointer parameters (OpenCL)
    uniform_context: bool
    column: int  # index into _REAL_MATH tuples
    float_literal_limit: float | None  # clamp |exponent| for single precision
    prelude: str = ""
    extra_defines: dict[str, str] = field(default_factory=dict)

    def real_math(self, name: str) -> str:
        return _REAL_MATH[name][self.column]

    def scalar(self, t: str) -> str:
        return {"real": self.real, "int": "int", "bool": "bool"}[t]

    def cast(self, t: str, expr: str) -> str:
        if self.name == "glsl":
            return f"{self.scalar(t)}({expr})"
        return f"(({self.scalar(t)})({expr}))"

    def array_param(self, name: str, elem: str, size: int, const: bool) -> str:
        e = self.real if elem == "real" else "int"
        if self.name == "glsl":
            return f"inout {e} {name}[{size}]"
        if const:
            return f"__constant const {e}* {name}"
        return f"{e}* {name}"

    def local_array(self, name: str, elem: str, size: int) -> str:
        e = self.real if elem == "real" else "int"
        return f"{e} {name}[{size}];"

    def uniform_decl(self, name: str, elem: str, size: int) -> str:
        e = self.real if elem == "real" else "int"
        return f"uniform {e} {name}[{size}];"


GLSL = Target(
    name="glsl",
    real="float",
    uniform_context=True,
    column=0,
    float_literal_limit=1e-36,
    prelude="#version 330 core\n#define REAL float\n",
)

OPENCL_DOUBLE = Target(
    name="opencl",
    real="double",
    uniform_context=False,
    column=1,
    float_literal_limit=None,
    prelude=(
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
        "#define REAL double\n"
    ),
)

OPENCL_FLOAT = Target(
    name="opencl",
    real="float",
    uniform_context=False,
    column=1,
    float_literal_limit=1e-36,
    prelude="#define REAL float\n",
)
