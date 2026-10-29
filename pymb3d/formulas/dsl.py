"""The formula DSL.

A formula is an ordinary Python function operating on a :class:`State`::

    @formula(name="Amazing Box", params=[Param("scale", 2.0), ...], de="linear")
    def amazing_box(s, p):
        z = box_fold(s.z, p.fold)          # helpers are transpiled too
        r2 = dot(z, z)
        ...
        s.z = z * m + s.c
        s.dr = s.dr * fabs(m)

The same source is

* executed directly by Python (slow; the reference used by the tests),
* transpiled to C (GLSL for the OpenGL preview, OpenCL for precise renders),
* transpiled to scalar Python that numba compiles for the CPU backend.

Allowed subset: assignments, augmented assignments, ``if``/``elif``/``else``,
``for i in range(...)``, ``while``, ``break``/``continue``, conditional
expressions, comparisons, ``and``/``or``/``not``, arithmetic, ``**`` and the
functions exported from this module.  Types are ``real`` (float), ``int``,
``bool`` and ``vec3``; they are inferred, so no annotations are needed in
formulas (helpers annotate their parameters).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


# --------------------------------------------------------------------------- vec3
class vec3:
    """Reference vector type used when a formula runs as plain Python."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=None, z=None):
        if y is None and z is None:
            self.x = self.y = self.z = float(x)
        else:
            self.x, self.y, self.z = float(x), float(y), float(z)

    # arithmetic -------------------------------------------------------------
    def _bin(self, o, f):
        if isinstance(o, vec3):
            return vec3(f(self.x, o.x), f(self.y, o.y), f(self.z, o.z))
        o = float(o)
        return vec3(f(self.x, o), f(self.y, o), f(self.z, o))

    def _rbin(self, o, f):
        o = float(o)
        return vec3(f(o, self.x), f(o, self.y), f(o, self.z))

    def __add__(self, o):
        return self._bin(o, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, o):
        return self._bin(o, lambda a, b: a - b)

    def __rsub__(self, o):
        return self._rbin(o, lambda a, b: a - b)

    def __mul__(self, o):
        return self._bin(o, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self._bin(o, lambda a, b: a / b)

    def __rtruediv__(self, o):
        return self._rbin(o, lambda a, b: a / b)

    def __neg__(self):
        return vec3(-self.x, -self.y, -self.z)

    def __abs__(self):
        return vec3(abs(self.x), abs(self.y), abs(self.z))

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __repr__(self):
        return f"vec3({self.x}, {self.y}, {self.z})"

    def copy(self):
        return vec3(self.x, self.y, self.z)


real = float

#: named constants (see transpile.base.CONSTANTS)
TINY = 1e-40
PI = math.pi
TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------------- functions
def _map(f, a, b=None):
    if isinstance(a, vec3):
        if b is None:
            return vec3(f(a.x), f(a.y), f(a.z))
        if isinstance(b, vec3):
            return vec3(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z))
        return vec3(f(a.x, b), f(a.y, b), f(a.z, b))
    return f(a) if b is None else f(a, b)


def sqrt(a):
    return _map(math.sqrt, a)


def sin(a):
    return _map(math.sin, a)


def cos(a):
    return _map(math.cos, a)


def tan(a):
    return _map(math.tan, a)


def asin(a):
    return _map(lambda v: math.asin(max(-1.0, min(1.0, v))), a)


def acos(a):
    return _map(lambda v: math.acos(max(-1.0, min(1.0, v))), a)


def atan(a):
    return _map(math.atan, a)


def atan2(y, x):
    return math.atan2(y, x)


def sinh(a):
    return _map(math.sinh, a)


def cosh(a):
    return _map(math.cosh, a)


def tanh(a):
    return _map(math.tanh, a)


def exp(a):
    return _map(math.exp, a)


def log(a):
    return _map(math.log, a)


def pow(a, b):  # noqa: A001 - mirrors the shader function
    return _map(math.pow, a, b)


def fabs(a):
    return _map(abs, a)


def floor(a):
    return _map(lambda v: float(math.floor(v)), a)


def ceil(a):
    return _map(lambda v: float(math.ceil(v)), a)


def fract(a):
    return _map(lambda v: v - math.floor(v), a)


def mod(a, b):
    return _map(lambda x, y: x - y * math.floor(x / y), a, b)


def sign(a):
    return _map(lambda v: (v > 0) - (v < 0) + 0.0, a)


def fmin(a, b):
    return _map(min, a, b)


def fmax(a, b):
    return _map(max, a, b)


def clamp(a, lo, hi):
    return fmin(fmax(a, lo), hi)


def mix(a, b, t):
    return a + (b - a) * t


def sqr(a):
    return a * a


def length(v: vec3) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def dot(a: vec3, b: vec3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: vec3, b: vec3) -> vec3:
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def normalize(v: vec3) -> vec3:
    n = length(v)
    return v / n if n > 0 else v.copy()


#: functions the transpilers know, with their result type rule:
#: "same" = type of the first argument, or a fixed type name
FUNCTIONS: dict[str, str] = {
    "sqrt": "same", "sin": "same", "cos": "same", "tan": "same", "asin": "same", "acos": "same",
    "atan": "same", "atan2": "real", "sinh": "same", "cosh": "same", "tanh": "same", "exp": "same",
    "log": "same", "pow": "same", "fabs": "same", "abs": "same", "floor": "same", "ceil": "same",
    "fract": "same", "mod": "same", "sign": "same", "fmin": "same", "fmax": "same", "clamp": "same",
    "mix": "same", "sqr": "same", "length": "real", "dot": "real", "cross": "vec3",
    "normalize": "vec3", "vec3": "vec3", "int": "int", "float": "real", "real": "real",
}

__all__ = [
    "vec3", "real", "State", "Params", "Param", "FormulaSpec", "formula", "helper",
    "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "log",
    "pow", "fabs", "floor", "ceil", "fract", "mod", "sign", "fmin", "fmax", "clamp", "mix", "sqr",
    "length", "dot", "cross", "normalize", "FUNCTIONS", "TINY", "PI", "TWO_PI",
]


# --------------------------------------------------------------------------- state
class State:
    """The iteration state a formula reads and writes.

    z   the iterated point (vec3)
    w   4th component for 4D formulas (real)
    c   the constant added each iteration: the pixel position or the Julia seed
    cw  4th component of the constant
    dr  running derivative for the analytic distance estimate (starts at 1)
    r2  |z|^2 after the previous counted iteration (read only)
    """

    __slots__ = ("z", "w", "c", "cw", "dr", "r2")
    FIELD_TYPES = {"z": "vec3", "w": "real", "c": "vec3", "cw": "real", "dr": "real", "r2": "real"}

    def __init__(self, z: vec3, c: vec3, w: float = 0.0, cw: float = 0.0, dr: float = 1.0, r2: float | None = None):
        self.z = z.copy()
        self.c = c.copy()
        self.w = float(w)
        self.cw = float(cw)
        self.dr = float(dr)
        self.r2 = dot(z, z) if r2 is None else float(r2)


class Params:
    """Attribute access to a formula's option values."""

    def __init__(self, values: dict[str, float]):
        self.__dict__.update({k: float(v) for k, v in values.items()})


@dataclass
class Param:
    name: str
    default: float = 0.0
    #: "real", "int" or "bool" (ints/bools are rounded before use)
    kind: str = "real"
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    #: MB3D's option name, for .m3p import (defaults to ``name``)
    mb3d_name: str = ""
    #: a derived value computed from the other options on the host
    #: (``lambda values: ...``); not shown in the GUI, not stored in scenes
    derived: Callable[[dict[str, float]], float] | None = None

    @property
    def editable(self) -> bool:
        return self.derived is None


@dataclass
class FormulaSpec:
    """Meta data of a registered formula."""

    name: str
    func: Callable
    params: list[Param] = field(default_factory=list)
    #: analytic distance estimate the formula maintains in ``s.dr``:
    #: "none" (numerical gradient DE), "linear" (|z| / dr, box type) or
    #: "log" (0.5 |z| ln|z|^2 / dr, bulb type)
    de: str = "none"
    #: `dDEscale`: scale of the numerical / analytic DE
    de_scale: float = 1.0
    #: default bailout radius
    rstop: float = 16.0
    #: power for the smooth iteration count (name of a param or a number)
    si_pow: float | str = 2.0
    #: iterates w as a 4th dimension (|z|^2 includes w^2)
    four_d: bool = False
    category: str = "3D"
    description: str = ""
    #: MB3D's internal formula number (-1: none)
    mb3d_id: int = -1
    #: alternative names (MB3D .m3f names, abbreviations)
    aliases: tuple[str, ...] = ()
    #: counted as an iteration (False for pure transforms)
    counted: bool = True
    #: name of the option that is the formula's scale (DE / colour heuristics)
    scale_param: str | None = None
    #: `dADEscale`: scale of the analytic DE
    ade_scale: float = 1.0

    def defaults(self) -> dict[str, float]:
        """Default values of the editable options."""
        return {p.name: float(p.default) for p in self.params if p.editable}

    @property
    def editable_params(self) -> list[Param]:
        return [p for p in self.params if p.editable]

    def values(self, overrides: dict[str, float] | None = None) -> dict[str, float]:
        """All option values (editable ones from ``overrides`` or the
        defaults, then the derived ones)."""
        v = self.defaults()
        for k, x in (overrides or {}).items():
            if k in v:
                v[k] = float(x)
        for p in self.params:
            if p.editable and p.kind in ("int", "bool"):
                v[p.name] = float(round(v[p.name]))
        for p in self.params:
            if p.derived is not None:
                v[p.name] = float(p.derived(v))
        return v

    def smooth_pow(self, values: dict[str, float]) -> float:
        if isinstance(self.si_pow, str):
            return abs(float(values.get(self.si_pow, 2.0)))
        return float(self.si_pow)

    def run(self, s: State, values: dict[str, float] | None = None) -> None:
        """Run one iteration as plain Python (the reference)."""
        self.func(s, Params(self.values(values)))


@dataclass
class HelperSpec:
    name: str
    func: Callable
    arg_types: list[tuple[str, str]]
    return_type: str


_HELPERS: dict[str, HelperSpec] = {}


def helper(func: Callable) -> Callable:
    """Register a device helper function (transpiled once, callable from
    any formula).  Parameters must be annotated with ``real``/``float``,
    ``int`` or ``vec3``; the return type is inferred."""
    import inspect

    sig = inspect.signature(func)
    args = []
    for p in sig.parameters.values():
        ann = p.annotation
        if ann in (float, real) or ann == "real" or ann == "float":
            t = "real"
        elif ann is int or ann == "int":
            t = "int"
        elif ann is vec3 or ann == "vec3":
            t = "vec3"
        else:
            raise TypeError(f"helper {func.__name__}: parameter {p.name} needs a real/int/vec3 annotation")
        args.append((p.name, t))
    ret = sig.return_annotation
    if ret in (float, real) or ret in ("real", "float"):
        rt = "real"
    elif ret is int or ret == "int":
        rt = "int"
    elif ret is vec3 or ret == "vec3":
        rt = "vec3"
    else:
        raise TypeError(f"helper {func.__name__}: needs a return annotation (real/int/vec3)")
    _HELPERS[func.__name__] = HelperSpec(func.__name__, func, args, rt)
    return func


def helpers() -> dict[str, HelperSpec]:
    return _HELPERS


def formula(name: str, params: list[Param] | None = None, de: str = "none", de_scale: float = 1.0,
            rstop: float = 16.0, si_pow: float | str = 2.0, four_d: bool = False, category: str = "3D",
            description: str = "", mb3d_id: int = -1, aliases: tuple[str, ...] = (), counted: bool = True,
            scale_param: str | None = None, ade_scale: float = 1.0) -> Callable:
    """Register a formula (see the module docstring)."""
    from .registry import register

    def deco(func: Callable) -> Callable:
        spec = FormulaSpec(name=name, func=func, params=list(params or []), de=de, de_scale=de_scale, rstop=rstop,
                           si_pow=si_pow, four_d=four_d, category=category, description=description,
                           mb3d_id=mb3d_id, aliases=aliases, counted=counted, scale_param=scale_param,
                           ade_scale=ade_scale)
        register(spec)
        func.spec = spec  # type: ignore[attr-defined]
        return func

    return deco
