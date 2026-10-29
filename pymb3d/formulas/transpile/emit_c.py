"""DSL -> C-like code, valid as GLSL and as OpenCL C with the preludes in
``pymb3d/kernels/clike``.

Conventions of the generated code
* ``real`` / ``real3`` / ``int`` / ``bool`` types, ``V3(a,b,c)`` constructor
* every function takes the hidden ``FN_ARGS`` parameter (the parameter
  buffer on OpenCL, nothing on GLSL) and passes ``FN_PASS`` on
* a formula is ``State formula_<slot>(FN_ARGS State s)``; its options are
  read with ``FPARAM(slot, index)``
"""
from __future__ import annotations

import ast
from typing import Callable

from ..dsl import helpers
from .base import CONSTANTS, TranspileError, TypeInfer, module_constants, parse_function, widen

C_TYPES = {"real": "real", "vec3": "real3", "int": "int", "bool": "bool"}

# DSL function -> C function (same name unless listed)
C_FUNCS = {
    "abs": "fabs", "fabs": "fabs", "fmin": "fmin", "fmax": "fmax", "fract": "fractx", "mod": "modx",
    "atan2": "atan2", "clamp": "clamp", "mix": "mix", "sign": "sign", "length": "length", "dot": "dot",
    "cross": "cross", "normalize": "normalize", "sqrt": "sqrt", "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "exp": "exp", "log": "log", "pow": "pow", "floor": "floor", "ceil": "ceil",
}

BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%", ast.FloorDiv: "/", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
          ast.LShift: "<<", ast.RShift: ">>"}
CMPOPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}


def strip_parens(code: str) -> str:
    """Removes one pair of enclosing parentheses when they span the code."""
    while code.startswith("(") and code.endswith(")"):
        depth = 0
        for i, ch in enumerate(code):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(code) - 1:
                    return code
        code = code[1:-1]
    return code


def real_literal(v: float) -> str:
    s = repr(float(v))
    if "e" in s or "E" in s:
        mant, ex = s.split("e")
        if "." not in mant:
            mant += ".0"
        return f"{mant}e{ex}"
    if "." not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s


class CEmitter:
    def __init__(self, func: Callable, slot: int | None = None, param_index: dict[str, int] | None = None):
        self.func = func
        self.node = parse_function(func)
        self.fname = func.__name__
        self.slot = slot
        self.param_index = param_index or {}
        self.is_helper = slot is None
        self.decls: dict[str, str] = {}
        self.out: list[str] = []
        self.indent = 1
        self.tmp = 0
        args = [a.arg for a in self.node.args.args]
        if self.is_helper:
            spec = helpers()[self.fname]
            scope = dict(spec.arg_types)
            self.ti = TypeInfer(scope, None, None, self.fname, module_constants(func))
            self.ret_type = spec.return_type
        else:
            if len(args) != 2:
                raise TranspileError(f"{self.fname}: a formula takes (state, params)")
            self.ti = TypeInfer({}, args[0], args[1], self.fname, module_constants(func))
            self.ret_type = "State"
        self.params_in_scope = set(self.ti.scope)

    # ------------------------------------------------------------------ api
    def emit(self) -> str:
        body: list[str] = []
        saved = self.out
        self.out = body
        for st in self.node.body:
            self.stmt(st)
        self.out = saved
        if self.is_helper:
            spec = helpers()[self.fname]
            params = ", ".join(f"{C_TYPES[t]} l_{n}" for n, t in spec.arg_types)
            head = f"{C_TYPES[self.ret_type]} h_{self.fname}(FN_ARGS {params})"
        else:
            head = f"State formula_{self.slot}(FN_ARGS State s)"
        lines = [head + " {"]
        for name, t in self.decls.items():
            lines.append(f"    {C_TYPES[t]} l_{name};")
        lines.extend(body)
        if not self.is_helper:
            lines.append("    return s;")
        lines.append("}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- helpers
    def line(self, s: str) -> None:
        self.out.append("    " * self.indent + s)

    def newtmp(self, t: str) -> str:
        self.tmp += 1
        name = f"_t{self.tmp}"
        self.decls[name] = t
        return name

    def declare(self, name: str, t: str, node: ast.AST) -> None:
        if name in self.ti.scope:
            return
        self.decls[name] = t
        self.ti.scope[name] = t

    def assign_type(self, name: str, t: str, node: ast.AST) -> str:
        """The type a variable keeps when ``t`` is assigned to it."""
        if name not in self.ti.scope:
            self.declare(name, t, node)
            return t
        old = self.ti.scope[name]
        if old == t or (old == "real" and t in ("int", "bool")) or (old == "int" and t == "bool"):
            return old
        if old in ("int", "bool") and t == "real":
            raise self.ti.err(node, f"variable '{name}' changes type from {old} to real; use a float literal first")
        raise self.ti.err(node, f"variable '{name}' changes type from {old} to {t}")

    # -------------------------------------------------------------- statements
    def stmt(self, st: ast.stmt) -> None:
        m = getattr(self, "s_" + type(st).__name__, None)
        if m is None:
            raise self.ti.err(st, f"unsupported statement {type(st).__name__}")
        m(st)

    def s_Expr(self, st: ast.Expr) -> None:
        if isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
            return  # docstring
        raise self.ti.err(st, "expression statements are not supported")

    def s_Pass(self, st: ast.Pass) -> None:
        pass

    def s_Return(self, st: ast.Return) -> None:
        if self.is_helper:
            if st.value is None:
                raise self.ti.err(st, "helpers must return a value")
            t, code = self.expr(st.value, want=self.ret_type)
            self.line(f"return {code};")
        else:
            if st.value is not None:
                raise self.ti.err(st, "formulas return nothing")
            self.line("return s;")

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
            t = self.assign_type(target.id, self.ti.type_of(value), st)
            _, code = self.expr(value, want=t)
            self.line(f"l_{target.id} = {code};")
        elif isinstance(target, ast.Attribute):
            t = self.ti.type_of(target)
            _, code = self.expr(value, want=t)
            self.line(f"{self.lvalue(target)} = {code};")
        else:
            raise self.ti.err(st, "unsupported assignment target")

    def lvalue(self, target: ast.Attribute) -> str:
        if isinstance(target.value, ast.Name) and target.value.id == self.ti.state_name:
            return f"s.{target.attr}"
        if isinstance(target.value, ast.Name) and target.value.id == self.ti.params_name:
            raise self.ti.err(target, "parameters are read only")
        if target.attr in ("x", "y", "z"):
            if isinstance(target.value, ast.Name):
                return f"l_{target.value.id}.{target.attr}"
            if isinstance(target.value, ast.Attribute):
                return f"{self.lvalue(target.value)}.{target.attr}"
        raise self.ti.err(target, "unsupported assignment target")

    def s_If(self, st: ast.If) -> None:
        _, cond = self.expr(st.test, want="bool")
        self.line(f"if ({strip_parens(cond)}) {{")
        self.block(st.body)
        if st.orelse:
            self.line("} else {")
            self.block(st.orelse)
        self.line("}")

    def s_While(self, st: ast.While) -> None:
        # the condition may hoist temporaries: evaluate it inside the loop
        self.line("while (true) {")
        self.indent += 1
        _, cond = self.expr(st.test, want="bool")
        self.line(f"if (!({strip_parens(cond)})) break;")
        self.indent -= 1
        self.block(st.body)
        self.line("}")

    def s_For(self, st: ast.For) -> None:
        if not (isinstance(st.iter, ast.Call) and isinstance(st.iter.func, ast.Name) and st.iter.func.id == "range"
                and isinstance(st.target, ast.Name)):
            raise self.ti.err(st, "only 'for i in range(...)' loops are supported")
        args = st.iter.args
        var = st.target.id
        self.assign_type(var, "int", st)
        if len(args) == 1:
            lo, hi, step = "0", self.expr(args[0], want="int")[1], "1"
        elif len(args) == 2:
            lo, hi, step = self.expr(args[0], want="int")[1], self.expr(args[1], want="int")[1], "1"
        else:
            lo, hi, step = (self.expr(a, want="int")[1] for a in args[:3])
        self.line(f"for (l_{var} = {lo}; l_{var} < {hi}; l_{var} += {step}) {{")
        self.block(st.body)
        self.line("}")

    def s_Break(self, st: ast.Break) -> None:
        self.line("break;")

    def s_Continue(self, st: ast.Continue) -> None:
        self.line("continue;")

    def block(self, body: list[ast.stmt]) -> None:
        self.indent += 1
        for st in body:
            self.stmt(st)
        self.indent -= 1

    # ------------------------------------------------------------ expressions
    def expr(self, node: ast.expr, want: str | None = None) -> tuple[str, str]:
        """Returns (type, code); ``want`` converts scalars."""
        t = self.ti.type_of(node)
        m = getattr(self, "e_" + type(node).__name__)
        code = m(node, t)
        if want is not None and want != t:
            if want == "real" and t in ("int", "bool"):
                if isinstance(node, ast.Constant):
                    code = real_literal(float(node.value))
                else:
                    code = f"TO_REAL({code})"
                t = "real"
            elif want == "int" and t == "real":
                code = f"TO_INT({code})"
                t = "int"
            elif want == "int" and t == "bool":
                code = f"TO_INT({code})"
                t = "int"
            elif want == "bool" and t in ("int", "real"):
                code = f"({code} != 0)"
                t = "bool"
            elif want == "vec3" and t in ("int", "real"):
                code = f"V3s({code})"
                t = "vec3"
            elif want == "real" and t == "vec3":
                raise self.ti.err(node, "a vec3 cannot be used where a real is expected")
        return t, code

    def e_Constant(self, n: ast.Constant, t: str) -> str:
        if t == "bool":
            return "true" if n.value else "false"
        if t == "int":
            return str(int(n.value))
        return real_literal(n.value)

    def e_Name(self, n: ast.Name, t: str) -> str:
        if n.id == "True":
            return "true"
        if n.id == "False":
            return "false"
        if n.id in CONSTANTS:
            return CONSTANTS[n.id][0]
        if n.id not in self.ti.scope and n.id in self.ti.consts:
            v = self.ti.consts[n.id]
            return str(int(v)) if isinstance(v, int) else real_literal(v)
        return f"l_{n.id}"

    def e_Attribute(self, n: ast.Attribute, t: str) -> str:
        if isinstance(n.value, ast.Name):
            if n.value.id == self.ti.state_name:
                return f"s.{n.attr}"
            if n.value.id == self.ti.params_name:
                if n.attr not in self.param_index:
                    raise self.ti.err(n, f"unknown parameter '{n.attr}'")
                return f"FPARAM({self.slot}, {self.param_index[n.attr]})"
        _, base = self.expr(n.value)
        return f"({base}).{n.attr}"

    def e_BinOp(self, n: ast.BinOp, t: str) -> str:
        lt, rt = self.ti.type_of(n.left), self.ti.type_of(n.right)
        if isinstance(n.op, ast.Pow):
            base_t = "vec3" if lt == "vec3" else "real"
            _, base = self.expr(n.left, want=base_t)
            if isinstance(n.right, ast.Constant) and isinstance(n.right.value, int) and 1 <= n.right.value <= 4:
                k = n.right.value
                if k == 1:
                    return f"({base})"
                tmp = self.newtmp(base_t)
                self.line(f"{tmp} = {base};")
                return "(" + "*".join([tmp] * k) + ")"
            _, ex = self.expr(n.right, want=base_t if base_t == "vec3" and rt == "vec3" else "real")
            return f"pow({base}, {ex})"
        if isinstance(n.op, ast.Div):
            want = "vec3" if t == "vec3" else "real"
            _, a = self.expr(n.left, want=want if lt == "vec3" else "real")
            _, b = self.expr(n.right, want=want if rt == "vec3" else "real")
            return f"({a} / {b})"
        if isinstance(n.op, ast.FloorDiv):
            if t != "int":
                _, a = self.expr(n.left, want="real")
                _, b = self.expr(n.right, want="real")
                return f"floor({a} / {b})"
            _, a = self.expr(n.left, want="int")
            _, b = self.expr(n.right, want="int")
            return f"({a} / {b})"
        if isinstance(n.op, ast.Mod):
            if t == "int":
                _, a = self.expr(n.left, want="int")
                _, b = self.expr(n.right, want="int")
                return f"({a} % {b})"
            _, a = self.expr(n.left, want="vec3" if lt == "vec3" else "real")
            _, b = self.expr(n.right, want="vec3" if rt == "vec3" else "real")
            return f"modx({a}, {b})"
        op = BINOPS[type(n.op)]
        scalar_want = "int" if t == "int" else "real"
        _, a = self.expr(n.left, want="vec3" if lt == "vec3" else scalar_want)
        _, b = self.expr(n.right, want="vec3" if rt == "vec3" else scalar_want)
        return f"({a} {op} {b})"

    def e_UnaryOp(self, n: ast.UnaryOp, t: str) -> str:
        if isinstance(n.op, ast.Not):
            _, a = self.expr(n.operand, want="bool")
            return f"(!{a})"
        _, a = self.expr(n.operand, want=t)
        return f"(-{a})" if isinstance(n.op, ast.USub) else f"(+{a})"

    def e_Compare(self, n: ast.Compare, t: str) -> str:
        parts = []
        left = n.left
        for op, right in zip(n.ops, n.comparators):
            ct = widen(self.ti.type_of(left), self.ti.type_of(right))
            ct = "real" if ct == "real" else ("int" if ct == "int" else "bool")
            _, a = self.expr(left, want=ct)
            _, b = self.expr(right, want=ct)
            if type(op) not in CMPOPS:
                raise self.ti.err(n, "unsupported comparison")
            parts.append(f"({a} {CMPOPS[type(op)]} {b})")
            left = right
        return parts[0] if len(parts) == 1 else "(" + " && ".join(parts) + ")"

    def e_BoolOp(self, n: ast.BoolOp, t: str) -> str:
        op = " && " if isinstance(n.op, ast.And) else " || "
        return "(" + op.join(self.expr(v, want="bool")[1] for v in n.values) + ")"

    def e_IfExp(self, n: ast.IfExp, t: str) -> str:
        _, c = self.expr(n.test, want="bool")
        _, a = self.expr(n.body, want=t)
        _, b = self.expr(n.orelse, want=t)
        return f"({c} ? {a} : {b})"

    def e_Call(self, n: ast.Call, t: str) -> str:
        name = n.func.id  # type: ignore[attr-defined]
        if name in helpers():
            spec = helpers()[name]
            if len(n.args) != len(spec.arg_types):
                raise self.ti.err(n, f"{name}() takes {len(spec.arg_types)} arguments")
            args = [self.expr(a, want=at)[1] for a, (_, at) in zip(n.args, spec.arg_types)]
            return f"h_{name}(FN_PASS {', '.join(args)})"
        arg_types = [self.ti.type_of(a) for a in n.args]
        if name == "vec3":
            if len(n.args) == 1:
                _, a = self.expr(n.args[0], want="real")
                return f"V3s({a})"
            if len(n.args) != 3:
                raise self.ti.err(n, "vec3() takes 1 or 3 arguments")
            return "V3(" + ", ".join(self.expr(a, want="real")[1] for a in n.args) + ")"
        if name == "int":
            return f"TO_INT({self.expr(n.args[0])[1]})"
        if name in ("float", "real"):
            return f"TO_REAL({self.expr(n.args[0])[1]})"
        if name == "sqr":
            _, a = self.expr(n.args[0], want=t)
            if not a.replace("_", "").replace(".", "").isalnum():
                tmp = self.newtmp(t)
                self.line(f"{tmp} = {a};")
                a = tmp
            return f"({a} * {a})"
        if name in ("fmin", "fmax", "clamp", "mix", "pow", "mod", "atan2"):
            # vector / scalar mixes are allowed in both GLSL and OpenCL as
            # long as the first argument has the result type
            want_first = "vec3" if t == "vec3" else "real"
            args = [self.expr(n.args[0], want=want_first)[1]]
            for a, at in zip(n.args[1:], arg_types[1:]):
                args.append(self.expr(a, want="vec3" if at == "vec3" else "real")[1])
            if name == "mix" and t == "vec3":
                # mix(vec3, vec3, real): make both value arguments vectors
                args[1] = self.expr(n.args[1], want="vec3")[1]
            if name == "mod":
                return f"modx({args[0]}, {args[1]})"
            return f"{C_FUNCS[name]}({', '.join(args)})"
        if name in C_FUNCS:
            want = "vec3" if arg_types and arg_types[0] == "vec3" else "real"
            args = [self.expr(a, want=want)[1] for a in n.args]
            return f"{C_FUNCS[name]}({', '.join(args)})"
        raise self.ti.err(n, f"unknown function '{name}'")


def helpers_used(func: Callable, seen: set[str] | None = None) -> list[str]:
    """Helper names a function uses, dependencies first."""
    seen = set() if seen is None else seen
    order: list[str] = []
    node = parse_function(func)
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in helpers():
            name = call.func.id
            if name in seen:
                continue
            seen.add(name)
            order.extend(helpers_used(helpers()[name].func, seen))
            order.append(name)
    return order
