"""Shared parts of the transpilers: parsing a DSL function and inferring
the type (``real``, ``int``, ``bool`` or ``vec3``) of every expression."""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable

from ..dsl import FUNCTIONS, State, helpers

SCALAR = ("real", "int", "bool")

#: named constants of the DSL: name -> (C code, Python code)
CONSTANTS = {
    "TINY": ("TINY", "1e-40"),
    "PI": ("PI_", "3.141592653589793"),
    "TWO_PI": ("TWO_PI_", "6.283185307179586"),
}


class TranspileError(Exception):
    pass


def parse_function(func: Callable) -> ast.FunctionDef:
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            return node
    raise TranspileError(f"{func.__name__}: no function definition found")


def module_constants(func: Callable) -> dict[str, float]:
    """Numeric module level constants visible to ``func``."""
    g = getattr(func, "__globals__", {})
    return {k: v for k, v in g.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
            and not k.startswith("__")}


def widen(a: str, b: str) -> str:
    """The common type of two operands."""
    if "vec3" in (a, b):
        return "vec3"
    if "real" in (a, b):
        return "real"
    if "int" in (a, b):
        return "int"
    return "bool"


class TypeInfer:
    """Type inference over the DSL subset.  ``scope`` maps variable names to
    types; ``state_name``/``params_name`` are the formula's two arguments."""

    def __init__(self, scope: dict[str, str], state_name: str | None, params_name: str | None, fname: str,
                 consts: dict[str, float] | None = None):
        self.scope = scope
        self.state_name = state_name
        self.params_name = params_name
        self.fname = fname
        #: module level numeric constants the function may use
        self.consts = dict(consts or {})

    def err(self, node: ast.AST, msg: str) -> TranspileError:
        return TranspileError(f"{self.fname} line {getattr(node, 'lineno', '?')}: {msg}")

    def type_of(self, node: ast.AST) -> str:
        m = getattr(self, "t_" + type(node).__name__, None)
        if m is None:
            raise self.err(node, f"unsupported syntax {type(node).__name__}")
        return m(node)

    # ---------------------------------------------------------------- leaves
    def t_Constant(self, n: ast.Constant) -> str:
        if isinstance(n.value, bool):
            return "bool"
        if isinstance(n.value, int):
            return "int"
        if isinstance(n.value, float):
            return "real"
        raise self.err(n, f"unsupported constant {n.value!r}")

    def t_Name(self, n: ast.Name) -> str:
        if n.id in ("True", "False"):
            return "bool"
        if n.id in CONSTANTS:
            return "real"
        if n.id in self.scope:
            return self.scope[n.id]
        if n.id in self.consts:
            return "int" if isinstance(self.consts[n.id], int) else "real"
        if n.id in self.scope:
            return self.scope[n.id]
        raise self.err(n, f"unknown variable '{n.id}'")

    def t_Attribute(self, n: ast.Attribute) -> str:
        if isinstance(n.value, ast.Name):
            if n.value.id == self.state_name:
                if n.attr not in State.FIELD_TYPES:
                    raise self.err(n, f"unknown state field '{n.attr}'")
                return State.FIELD_TYPES[n.attr]
            if n.value.id == self.params_name:
                return "real"
        base = self.type_of(n.value)
        if base == "vec3" and n.attr in ("x", "y", "z"):
            return "real"
        raise self.err(n, f"unknown attribute '.{n.attr}' on {base}")

    # ------------------------------------------------------------ operators
    def t_BinOp(self, n: ast.BinOp) -> str:
        a, b = self.type_of(n.left), self.type_of(n.right)
        if isinstance(n.op, ast.Div):
            return "vec3" if "vec3" in (a, b) else "real"
        if isinstance(n.op, ast.Pow):
            return "vec3" if "vec3" in (a, b) else "real"
        if isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            t = widen(a, b)
            return "int" if t == "bool" else t
        if isinstance(n.op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
            if a not in ("int", "bool") or b not in ("int", "bool"):
                raise self.err(n, "bit operations need integer operands")
            return "int"
        raise self.err(n, f"unsupported operator {type(n.op).__name__}")

    def t_UnaryOp(self, n: ast.UnaryOp) -> str:
        t = self.type_of(n.operand)
        if isinstance(n.op, ast.Not):
            return "bool"
        if isinstance(n.op, (ast.USub, ast.UAdd)):
            return "int" if t == "bool" else t
        raise self.err(n, "unsupported unary operator")

    def t_Compare(self, n: ast.Compare) -> str:
        for c in [n.left] + n.comparators:
            if self.type_of(c) == "vec3":
                raise self.err(n, "comparisons of vectors are not supported")
        return "bool"

    def t_BoolOp(self, n: ast.BoolOp) -> str:
        return "bool"

    def t_IfExp(self, n: ast.IfExp) -> str:
        return widen(self.type_of(n.body), self.type_of(n.orelse))

    def t_Call(self, n: ast.Call) -> str:
        if not isinstance(n.func, ast.Name):
            raise self.err(n, "only plain function calls are supported")
        name = n.func.id
        arg_types = [self.type_of(a) for a in n.args]
        if name in helpers():
            return helpers()[name].return_type
        if name not in FUNCTIONS:
            raise self.err(n, f"unknown function '{name}'")
        rule = FUNCTIONS[name]
        if rule == "same":
            if not arg_types:
                raise self.err(n, f"{name}() needs arguments")
            if name in ("fmin", "fmax", "clamp", "mix", "pow", "mod"):
                t = "vec3" if "vec3" in arg_types else "real"
                if name in ("fmin", "fmax") and all(a == "int" for a in arg_types):
                    t = "int"
                return t
            t = arg_types[0]
            return "real" if t in ("int", "bool") and name not in ("abs", "fabs") else t
        return rule
