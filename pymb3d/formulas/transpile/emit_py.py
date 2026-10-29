"""DSL -> scalar Python for numba.

Vectors are split into three scalars (``l_v_x, l_v_y, l_v_z``), so the
generated code needs nothing but ``math`` and compiles with ``@njit``.

Naming: ``s_*`` the state components, ``l_*`` locals, ``p_*`` options,
``_tN`` temporaries, ``h_*`` helpers.
"""
from __future__ import annotations

import ast
from typing import Callable

from ..dsl import helpers
from .base import CONSTANTS, TranspileError, TypeInfer, module_constants, parse_function, widen

STATE_VARS = {
    "z": ("s_zx", "s_zy", "s_zz"),
    "w": "s_w",
    "c": ("s_cx", "s_cy", "s_cz"),
    "cw": "s_cw",
    "dr": "s_dr",
    "r2": "s_r2",
    "de": "s_de",
    "trap": "s_trap",
    "aux": "s_aux",
    "vary": "s_vary",
}
STATE_ARGS = "s_zx, s_zy, s_zz, s_w, s_cx, s_cy, s_cz, s_cw, s_dr, s_r2, s_de, s_trap, s_aux, s_vary"
STATE_RETURN = "s_zx, s_zy, s_zz, s_w, s_cx, s_cy, s_cz, s_dr, s_de, s_trap, s_vary"

PY_PRELUDE = '''
import math
from numba import njit


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def _sign(x):
    return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def _normalize(x, y, z):
    n = math.sqrt(x * x + y * y + z * z)
    if n > 0.0:
        return x / n, y / n, z / n
    return x, y, z

'''

MATH1 = {
    "sqrt": "math.sqrt", "sin": "math.sin", "cos": "math.cos", "tan": "math.tan", "asin": "math.asin",
    "acos": "math.acos", "atan": "math.atan", "sinh": "math.sinh", "cosh": "math.cosh", "tanh": "math.tanh",
    "exp": "math.exp", "log": "math.log",
}
BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%", ast.FloorDiv: "//", ast.Pow: "**", ast.BitAnd: "&",
          ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>"}
CMPOPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}

Code = str | tuple[str, str, str]


def is_atom(code: str) -> bool:
    c = code.strip("()")
    return c.replace("_", "").replace(".", "").replace("-", "").isalnum()


class PyEmitter:
    def __init__(self, func: Callable, slot: int | None = None, param_index: dict[str, int] | None = None,
                 param_base: int = 0):
        self.func = func
        self.node = parse_function(func)
        self.fname = func.__name__
        self.slot = slot
        self.param_index = param_index or {}
        self.param_base = param_base
        self.is_helper = slot is None
        self.out: list[str] = []
        self.indent = 1
        self.tmp = 0
        args = [a.arg for a in self.node.args.args]
        if self.is_helper:
            spec = helpers()[self.fname]
            self.ti = TypeInfer(dict(spec.arg_types), None, None, self.fname, module_constants(func))
            self.ret_type = spec.return_type
        else:
            if len(args) != 2:
                raise TranspileError(f"{self.fname}: a formula takes (state, params)")
            self.ti = TypeInfer({}, args[0], args[1], self.fname, module_constants(func))
            self.ret_type = "State"

    # ------------------------------------------------------------------ api
    def emit(self) -> str:
        body: list[str] = []
        self.out = body
        if not self.is_helper:
            for name in self.params_used():
                self.line(f"p_{name} = P[{self.param_base + self.param_index[name]}]")
        for st in self.node.body:
            self.stmt(st)
        if self.is_helper:
            spec = helpers()[self.fname]
            params = ", ".join(self.flat_names(f"l_{n}", t) for n, t in spec.arg_types)
            head = f"def h_{self.fname}({params}):"
        else:
            head = f"def formula_{self.slot}({STATE_ARGS}, P):"
            body.append(f"    return {STATE_RETURN}")
        return "\n".join(["@njit(cache=True, nogil=True, fastmath=False, error_model=\"numpy\")", head] + body)

    def params_used(self) -> list[str]:
        names = []
        for n in ast.walk(self.node):
            if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == self.ti.params_name
                    and n.attr not in names):
                if n.attr not in self.param_index:
                    raise self.ti.err(n, f"unknown parameter '{n.attr}'")
                names.append(n.attr)
        return names

    @staticmethod
    def flat_names(base: str, t: str) -> str:
        return f"{base}_x, {base}_y, {base}_z" if t == "vec3" else base

    # ---------------------------------------------------------------- helpers
    def line(self, s: str) -> None:
        self.out.append("    " * self.indent + s)

    def newtmp(self) -> str:
        self.tmp += 1
        return f"_t{self.tmp}"

    def atomize(self, t: str, code: Code) -> Code:
        """Store a compound expression in a temporary so it is evaluated once."""
        if t == "vec3":
            assert isinstance(code, tuple)
            if all(is_atom(c) for c in code):
                return code
            tmp = self.newtmp()
            self.line(f"{tmp}_x, {tmp}_y, {tmp}_z = {code[0]}, {code[1]}, {code[2]}")
            return (f"{tmp}_x", f"{tmp}_y", f"{tmp}_z")
        assert isinstance(code, str)
        if is_atom(code):
            return code
        tmp = self.newtmp()
        self.line(f"{tmp} = {code}")
        return tmp

    def broadcast(self, t: str, code: Code) -> tuple[str, str, str]:
        if t == "vec3":
            return code  # type: ignore[return-value]
        c = self.atomize(t, code)
        return (c, c, c)  # type: ignore[return-value]

    # -------------------------------------------------------------- statements
    def stmt(self, st: ast.stmt) -> None:
        m = getattr(self, "s_" + type(st).__name__, None)
        if m is None:
            raise self.ti.err(st, f"unsupported statement {type(st).__name__}")
        m(st)

    def s_Expr(self, st: ast.Expr) -> None:
        if isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
            return
        raise self.ti.err(st, "expression statements are not supported")

    def s_Pass(self, st: ast.Pass) -> None:
        self.line("pass")

    def s_Return(self, st: ast.Return) -> None:
        if self.is_helper:
            if st.value is None:
                raise self.ti.err(st, "helpers must return a value")
            t, code = self.expr(st.value, want=self.ret_type)
            self.line("return " + (", ".join(code) if t == "vec3" else code))
        else:
            if st.value is not None:
                raise self.ti.err(st, "formulas return nothing")
            self.line(f"return {STATE_RETURN}")

    def s_Assign(self, st: ast.Assign) -> None:
        if len(st.targets) != 1:
            raise self.ti.err(st, "one target per assignment")
        self.assign(st.targets[0], st.value, st)

    def s_AnnAssign(self, st: ast.AnnAssign) -> None:
        if st.value is None:
            raise self.ti.err(st, "annotated declarations need a value")
        self.assign(st.target, st.value, st)

    def s_AugAssign(self, st: ast.AugAssign) -> None:
        binop = ast.BinOp(left=st.target, op=st.op, right=st.value)
        ast.copy_location(binop, st)
        self.assign(st.target, binop, st)

    def assign(self, target: ast.expr, value: ast.expr, st: ast.stmt) -> None:
        if isinstance(target, ast.Name):
            t = self.ti.type_of(value)
            if target.id in self.ti.scope:
                old = self.ti.scope[target.id]
                if old != t:
                    if widen(old, t) in (old, "real") and "vec3" not in (old, t):
                        t = "real" if "real" in (old, t) else old
                        self.ti.scope[target.id] = t
                    else:
                        raise self.ti.err(st, f"variable '{target.id}' changes type from {old} to {t}")
            else:
                self.ti.scope[target.id] = t
            _, code = self.expr(value, want=t)
            if t == "vec3":
                self.line(f"l_{target.id}_x, l_{target.id}_y, l_{target.id}_z = {code[0]}, {code[1]}, {code[2]}")
            else:
                self.line(f"l_{target.id} = {code}")
        elif isinstance(target, ast.Attribute):
            t = self.ti.type_of(target)
            _, code = self.expr(value, want=t)
            names = self.lvalue(target)
            if t == "vec3":
                self.line(f"{', '.join(names)} = {', '.join(code)}")
            else:
                self.line(f"{names} = {code}")
        else:
            raise self.ti.err(st, "unsupported assignment target")

    def lvalue(self, target: ast.Attribute) -> Code:
        if isinstance(target.value, ast.Name) and target.value.id == self.ti.state_name:
            return STATE_VARS[target.attr]
        if isinstance(target.value, ast.Name) and target.value.id == self.ti.params_name:
            raise self.ti.err(target, "parameters are read only")
        if target.attr in ("x", "y", "z"):
            if isinstance(target.value, ast.Name):
                return f"l_{target.value.id}_{target.attr}"
            if isinstance(target.value, ast.Attribute):
                base = self.lvalue(target.value)
                return base["xyz".index(target.attr)]
        raise self.ti.err(target, "unsupported assignment target")

    def s_If(self, st: ast.If) -> None:
        _, cond = self.expr(st.test, want="bool")
        self.line(f"if {cond}:")
        self.block(st.body)
        if st.orelse:
            self.line("else:")
            self.block(st.orelse)

    def s_While(self, st: ast.While) -> None:
        self.line("while True:")
        self.indent += 1
        _, cond = self.expr(st.test, want="bool")
        self.line(f"if not {cond}:")
        self.line("    break")
        self.indent -= 1
        self.block(st.body)

    def s_For(self, st: ast.For) -> None:
        if not (isinstance(st.iter, ast.Call) and isinstance(st.iter.func, ast.Name) and st.iter.func.id == "range"
                and isinstance(st.target, ast.Name)):
            raise self.ti.err(st, "only 'for i in range(...)' loops are supported")
        self.ti.scope[st.target.id] = "int"
        args = ", ".join(self.expr(a, want="int")[1] for a in st.iter.args)
        self.line(f"for l_{st.target.id} in range({args}):")
        self.block(st.body)

    def s_Break(self, st: ast.Break) -> None:
        self.line("break")

    def s_Continue(self, st: ast.Continue) -> None:
        self.line("continue")

    def block(self, body: list[ast.stmt]) -> None:
        self.indent += 1
        if not body:
            self.line("pass")
        for st in body:
            self.stmt(st)
        self.indent -= 1

    # ------------------------------------------------------------ expressions
    def expr(self, node: ast.expr, want: str | None = None) -> tuple[str, Code]:
        t = self.ti.type_of(node)
        m = getattr(self, "e_" + type(node).__name__)
        code = m(node, t)
        if want is not None and want != t:
            if want == "real" and t in ("int", "bool"):
                code = f"float({code})"
                t = "real"
            elif want == "int" and t in ("real", "bool"):
                code = f"int({code})"
                t = "int"
            elif want == "bool" and t in ("int", "real"):
                code = f"({code} != 0)"
                t = "bool"
            elif want == "vec3" and t in ("int", "real"):
                code = self.broadcast(t, code)
                t = "vec3"
            elif want == "real" and t == "vec3":
                raise self.ti.err(node, "a vec3 cannot be used where a real is expected")
        return t, code

    def e_Constant(self, n: ast.Constant, t: str) -> str:
        if t == "bool":
            return "True" if n.value else "False"
        if t == "int":
            return str(int(n.value))
        return repr(float(n.value))

    def e_Name(self, n: ast.Name, t: str) -> Code:
        if n.id in ("True", "False"):
            return n.id
        if n.id in CONSTANTS:
            return CONSTANTS[n.id][1]
        if n.id not in self.ti.scope and n.id in self.ti.consts:
            return repr(self.ti.consts[n.id])
        if t == "vec3":
            return (f"l_{n.id}_x", f"l_{n.id}_y", f"l_{n.id}_z")
        return f"l_{n.id}"

    def e_Attribute(self, n: ast.Attribute, t: str) -> Code:
        if isinstance(n.value, ast.Name):
            if n.value.id == self.ti.state_name:
                return STATE_VARS[n.attr]
            if n.value.id == self.ti.params_name:
                return f"p_{n.attr}"
        _, base = self.expr(n.value)
        return base["xyz".index(n.attr)]

    def e_BinOp(self, n: ast.BinOp, t: str) -> Code:
        lt, rt = self.ti.type_of(n.left), self.ti.type_of(n.right)
        op = BINOPS[type(n.op)]
        if t == "vec3":
            _, a = self.expr(n.left)
            _, b = self.expr(n.right)
            a3 = self.broadcast(lt, a)
            b3 = self.broadcast(rt, b)
            return tuple(f"({x} {op} {y})" for x, y in zip(a3, b3))  # type: ignore[return-value]
        scalar_want = "int" if t == "int" else "real"
        _, a = self.expr(n.left, want=scalar_want if lt != "bool" else "int")
        _, b = self.expr(n.right, want=scalar_want if rt != "bool" else "int")
        return f"({a} {op} {b})"

    def e_UnaryOp(self, n: ast.UnaryOp, t: str) -> Code:
        if isinstance(n.op, ast.Not):
            _, a = self.expr(n.operand, want="bool")
            return f"(not {a})"
        _, a = self.expr(n.operand, want=t)
        sign = "-" if isinstance(n.op, ast.USub) else "+"
        if t == "vec3":
            return tuple(f"({sign}{c})" for c in a)  # type: ignore[return-value]
        return f"({sign}{a})"

    def e_Compare(self, n: ast.Compare, t: str) -> str:
        parts = []
        left = n.left
        for op, right in zip(n.ops, n.comparators):
            if type(op) not in CMPOPS:
                raise self.ti.err(n, "unsupported comparison")
            _, a = self.expr(left)
            _, b = self.expr(right)
            parts.append(f"({a} {CMPOPS[type(op)]} {b})")
            left = right
        return "(" + " and ".join(parts) + ")"

    def e_BoolOp(self, n: ast.BoolOp, t: str) -> str:
        op = " and " if isinstance(n.op, ast.And) else " or "
        return "(" + op.join(self.expr(v, want="bool")[1] for v in n.values) + ")"

    def e_IfExp(self, n: ast.IfExp, t: str) -> Code:
        _, c = self.expr(n.test, want="bool")
        c = self.atomize("bool", c)
        _, a = self.expr(n.body, want=t)
        _, b = self.expr(n.orelse, want=t)
        if t == "vec3":
            return tuple(f"({x} if {c} else {y})" for x, y in zip(a, b))  # type: ignore[return-value]
        return f"({a} if {c} else {b})"

    def e_Call(self, n: ast.Call, t: str) -> Code:
        name = n.func.id  # type: ignore[attr-defined]
        if name in helpers():
            spec = helpers()[name]
            if len(n.args) != len(spec.arg_types):
                raise self.ti.err(n, f"{name}() takes {len(spec.arg_types)} arguments")
            flat: list[str] = []
            for a, (_, at) in zip(n.args, spec.arg_types):
                _, code = self.expr(a, want=at)
                flat.extend(code if at == "vec3" else [code])
            call = f"h_{name}({', '.join(flat)})"
            if spec.return_type == "vec3":
                tmp = self.newtmp()
                self.line(f"{tmp}_x, {tmp}_y, {tmp}_z = {call}")
                return (f"{tmp}_x", f"{tmp}_y", f"{tmp}_z")
            return call
        arg_types = [self.ti.type_of(a) for a in n.args]
        if name == "vec3":
            if len(n.args) == 1:
                _, a = self.expr(n.args[0], want="real")
                return self.broadcast("real", a)
            if len(n.args) != 3:
                raise self.ti.err(n, "vec3() takes 1 or 3 arguments")
            return tuple(self.expr(a, want="real")[1] for a in n.args)  # type: ignore[return-value]
        if name == "int":
            return f"int({self.expr(n.args[0])[1]})"
        if name in ("float", "real"):
            return f"float({self.expr(n.args[0])[1]})"
        if name in ("length", "dot", "cross", "normalize"):
            vs = [self.atomize("vec3", self.expr(a, want="vec3")[1]) for a in n.args]
            a = vs[0]
            if name == "length":
                return f"math.sqrt({a[0]}*{a[0]} + {a[1]}*{a[1]} + {a[2]}*{a[2]})"
            if name == "normalize":
                tmp = self.newtmp()
                self.line(f"{tmp}_x, {tmp}_y, {tmp}_z = _normalize({a[0]}, {a[1]}, {a[2]})")
                return (f"{tmp}_x", f"{tmp}_y", f"{tmp}_z")
            b = vs[1]
            if name == "dot":
                return f"({a[0]}*{b[0]} + {a[1]}*{b[1]} + {a[2]}*{b[2]})"
            return (f"({a[1]}*{b[2]} - {a[2]}*{b[1]})", f"({a[2]}*{b[0]} - {a[0]}*{b[2]})",
                    f"({a[0]}*{b[1]} - {a[1]}*{b[0]})")
        # element-wise functions
        if t == "vec3":
            args = [self.broadcast(at, self.expr(a)[1]) for a, at in zip(n.args, arg_types)]
            return tuple(self.scalar_call(name, [a[k] for a in args]) for k in range(3))  # type: ignore[return-value]
        want = "int" if t == "int" else "real"
        args = [self.expr(a, want=want if at != "bool" else "int")[1] for a, at in zip(n.args, arg_types)]
        return self.scalar_call(name, args)

    def scalar_call(self, name: str, a: list[str]) -> str:
        if name in MATH1:
            return f"{MATH1[name]}({a[0]})"
        if name in ("abs", "fabs"):
            return f"abs({a[0]})"
        if name == "atan2":
            return f"math.atan2({a[0]}, {a[1]})"
        if name == "pow":
            return f"({a[0]} ** {a[1]})"
        if name == "floor":
            return f"float(math.floor({a[0]}))"
        if name == "ceil":
            return f"float(math.ceil({a[0]}))"
        if name == "fract":
            x = self.atomize("real", a[0])
            return f"({x} - math.floor({x}))"
        if name == "mod":
            x = self.atomize("real", a[0])
            y = self.atomize("real", a[1])
            return f"({x} - {y} * math.floor({x} / {y}))"
        if name == "sign":
            return f"_sign({a[0]})"
        if name == "fmin":
            return f"min({a[0]}, {a[1]})"
        if name == "fmax":
            return f"max({a[0]}, {a[1]})"
        if name == "clamp":
            return f"min(max({a[0]}, {a[1]}), {a[2]})"
        if name == "mix":
            x = self.atomize("real", a[0])
            return f"({x} + ({a[1]} - {x}) * {a[2]})"
        if name == "sqr":
            x = self.atomize("real", a[0])
            return f"({x} * {x})"
        raise TranspileError(f"{self.fname}: unknown function '{name}'")
