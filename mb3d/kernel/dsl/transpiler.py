"""Python subset -> C-like source (GLSL / OpenCL C).

Supported: typed scalar and array parameters, local scalars, local arrays
(``st = local_real(8)``), ``if/elif/else``, ``while``, ``for i in range()``,
``break``/``continue``/``return``, arithmetic, comparisons (chained),
``and/or/not``, ternaries, ``math.*`` calls, ``abs/min/max/int/float`` and
calls to other kernel functions.  Module level int/float constants become
``#define``s.
"""

from __future__ import annotations

import ast
import builtins
import math
from dataclasses import dataclass, field

from .decorators import KernelFunction, registry
from .targets import Target
from .types import ArrayType, scalar_name

_RANK = {"bool": 0, "int": 1, "real": 2}
# identifiers that are keywords or builtins of GLSL / OpenCL C
_RESERVED = {
    "out", "in", "inout", "input", "output", "sample", "filter", "buffer", "uniform", "varying", "const",
    "float", "double", "int", "void", "char", "short", "long", "unsigned", "signed", "bool", "true", "false",
    "struct", "union", "enum", "typedef", "static", "extern", "register", "volatile", "switch", "case",
    "default", "do", "goto", "sizeof", "asm", "new", "delete", "this", "class", "template", "namespace",
    "using", "public", "private", "protected", "half", "fixed", "attribute", "discard", "precision",
    "lowp", "mediump", "highp", "layout", "flat", "smooth", "centroid", "patch", "subroutine", "common",
    "partition", "active", "cast", "external", "interface", "kernel", "global", "local", "constant",
    "image", "texture", "main", "near", "far", "exp", "log", "sqrt", "abs", "min", "max", "pow", "sin",
    "cos", "tan", "floor", "ceil", "fract", "mod", "sign", "step", "mix", "clamp", "length", "distance",
    "dot", "cross", "normalize", "reflect", "refract", "radians", "degrees", "atan", "asin", "acos",
    "fma", "round", "trunc", "isnan", "isinf", "any", "all", "not", "select", "vec2", "vec3", "vec4", "mat3",
}


def _ident(name: str) -> str:
    return name + "_" if name in _RESERVED else name
_LOCAL_ARRAY_FUNCS = {"local_real": "real", "local_int": "int"}


class TranspileError(Exception):
    pass


@dataclass
class Signature:
    name: str
    params: list[tuple[str, object]]  # (name, "real"/"int"/"bool" or ArrayType)
    ret: str | None


@dataclass
class _FunctionContext:
    sig: Signature
    env: dict[str, object] = field(default_factory=dict)
    local_arrays: dict[str, tuple[str, int]] = field(default_factory=dict)
    constants: dict[str, object] = field(default_factory=dict)


def _join(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b


def _func_def(kf: KernelFunction) -> ast.FunctionDef:
    tree = ast.parse(kf.source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise TranspileError(f"no function definition in source of {kf.name}")


def signature_of(kf: KernelFunction) -> Signature:
    fd = _func_def(kf)
    params: list[tuple[str, object]] = []
    g = kf.py_func.__globals__
    for a in fd.args.args:
        ann = _eval_annotation(a.annotation, g, kf.name, a.arg)
        params.append((a.arg, ann))
    ret = None
    if fd.returns is not None and not (isinstance(fd.returns, ast.Constant) and fd.returns.value is None):
        r = _eval_annotation(fd.returns, g, kf.name, "return")
        if isinstance(r, ArrayType):
            raise TranspileError(f"{kf.name}: arrays cannot be returned")
        ret = r
    return Signature(kf.name, params, ret)


def _eval_annotation(node, g, fname, pname):
    if node is None:
        raise TranspileError(f"{fname}: parameter '{pname}' needs a type annotation")
    try:
        value = eval(compile(ast.Expression(node), "<ann>", "eval"), g)
    except Exception as e:  # pragma: no cover - programming error
        raise TranspileError(f"{fname}: bad annotation for '{pname}': {e}") from e
    if isinstance(value, ArrayType):
        return value
    s = scalar_name(value)
    if s is None:
        raise TranspileError(f"{fname}: unsupported annotation for '{pname}': {value!r}")
    return s


class Transpiler:
    """Transpiles a set of kernel functions for one target."""

    def __init__(self, target: Target, overrides: dict[str, KernelFunction] | None = None):
        self.target = target
        self.overrides = overrides or {}
        self.sigs: dict[str, Signature] = {}
        self.defines: dict[str, str] = {}
        self.uniforms: dict[str, tuple[str, int]] = {}

    # ------------------------------------------------------------------ API
    def transpile(self, entry_points: list[str], extra_source: str = "") -> str:
        funcs = self._collect(entry_points)
        for kf in funcs:
            self.sigs[kf.name] = signature_of(kf)
        bodies = [self._emit_function(kf) for kf in funcs]
        out = [self.target.prelude]
        for k, v in {**self.target.extra_defines, **self.defines}.items():
            out.append(f"#define {k} {v}")
        if self.target.uniform_context:
            for name, (elem, size) in self.uniforms.items():
                out.append(self.target.uniform_decl(name, elem, size))
        out.append("")
        for kf in funcs:
            out.append(self._prototype(self.sigs[kf.name]) + ";")
        out.append("")
        out.extend(bodies)
        out.append(extra_source)
        return "\n".join(out)

    # ------------------------------------------------------------ collection
    def _reg(self) -> dict[str, KernelFunction]:
        return {**registry(), **self.overrides}

    def _collect(self, entry_points: list[str]) -> list[KernelFunction]:
        reg = self._reg()
        order: list[KernelFunction] = []
        seen: set[str] = set()

        def visit(name: str):
            if name in seen:
                return
            kf = reg.get(name)
            if kf is None:
                raise TranspileError(f"unknown kernel function '{name}'")
            seen.add(name)
            for callee in _called_names(_func_def(kf)):
                if callee in reg:
                    visit(callee)
            order.append(kf)

        for e in entry_points:
            visit(e)
        return order

    # -------------------------------------------------------------- emission
    def _prototype(self, sig: Signature) -> str:
        t = self.target
        ret = t.scalar(sig.ret) if sig.ret else "void"
        params = []
        for name, ty in sig.params:
            if isinstance(ty, ArrayType):
                if ty.const and t.uniform_context:
                    self._note_uniform(name, ty)
                    continue
                params.append(t.array_param(_ident(name), ty.elem, ty.size, ty.const))
            else:
                params.append(f"{t.scalar(ty)} {_ident(name)}")
        return f"{ret} {sig.name}({', '.join(params)})"

    def _note_uniform(self, name: str, ty: ArrayType):
        name = _ident(name)
        prev = self.uniforms.get(name)
        if prev is not None and prev != (ty.elem, ty.size):
            raise TranspileError(f"context array '{name}' declared with different sizes")
        self.uniforms[name] = (ty.elem, ty.size)

    def _emit_function(self, kf: KernelFunction) -> str:
        fd = _func_def(kf)
        sig = self.sigs[kf.name]
        ctx = _FunctionContext(sig)
        for name, ty in sig.params:
            ctx.env[name] = ty
        ctx.constants = self._module_constants(kf, fd, ctx)
        self._infer_locals(fd.body, ctx)
        lines = [self._prototype(sig) + " {"]
        for name, ty in ctx.env.items():
            if name in dict(sig.params):
                continue
            if name in ctx.local_arrays:
                elem, size = ctx.local_arrays[name]
                lines.append("    " + self.target.local_array(_ident(name), elem, size))
            else:
                lines.append(f"    {self.target.scalar(ty)} {_ident(name)};")
        for stmt in fd.body:
            lines.extend(self._stmt(stmt, ctx, 1))
        lines.append("}")
        return "\n".join(lines)

    def _module_constants(self, kf, fd, ctx) -> dict[str, object]:
        g = kf.py_func.__globals__
        consts: dict[str, object] = {}
        params = {n for n, _ in ctx.sig.params}
        for node in ast.walk(fd):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                n = node.id
                if n in params or n in self._reg() or n in _LOCAL_ARRAY_FUNCS:
                    continue
                if n in g and isinstance(g[n], (bool, int, float)) and not callable(g[n]):
                    consts[n] = g[n]
                    self.defines[n] = self._literal(g[n])
        return consts

    # ------------------------------------------------------- type inference
    def _infer_locals(self, body, ctx: _FunctionContext):
        assigned = [n for n in ast.walk(ast.Module(body=body, type_ignores=[])) if isinstance(n, (ast.Assign, ast.AugAssign, ast.For))]
        # two passes so that a later real assignment promotes an int local
        for _ in range(2):
            for node in assigned:
                if isinstance(node, ast.For):
                    ctx.env[node.target.id] = "int"
                    continue
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if not isinstance(target, ast.Name):
                    continue
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in _LOCAL_ARRAY_FUNCS:
                    elem = _LOCAL_ARRAY_FUNCS[node.value.func.id]
                    size = node.value.args[0]
                    if not isinstance(size, ast.Constant):
                        raise TranspileError(f"{ctx.sig.name}: local array size must be a literal")
                    ctx.local_arrays[target.id] = (elem, int(size.value))
                    ctx.env[target.id] = ArrayType(elem, int(size.value), False)
                    continue
                try:
                    _, ty = self._expr(node.value, ctx)
                except TranspileError:
                    ty = "real"
                if not isinstance(ty, str):
                    raise TranspileError(f"{ctx.sig.name}: cannot assign an array to '{target.id}'")
                prev = ctx.env.get(target.id)
                if isinstance(prev, ArrayType):
                    raise TranspileError(f"{ctx.sig.name}: '{target.id}' is an array and cannot be assigned a scalar")
                if prev is None or prev in ctx.constants:
                    ctx.env[target.id] = ty
                elif isinstance(prev, str):
                    ctx.env[target.id] = _join(prev, ty)

    # ----------------------------------------------------------- statements
    def _stmt(self, node, ctx: _FunctionContext, depth: int) -> list[str]:
        ind = "    " * depth
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant):
                return []  # docstring
            code, _ = self._expr(node.value, ctx)
            return [f"{ind}{code};"]
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise TranspileError("one assignment target only")
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in ctx.local_arrays:
                return []
            tcode, tty = self._expr(t, ctx)
            code, ty = self._expr(node.value, ctx)
            return [f"{ind}{tcode} = {self._coerce(code, ty, tty)};"]
        if isinstance(node, ast.AugAssign):
            tcode, tty = self._expr(node.target, ctx)
            code, ty = self._expr(node.value, ctx)
            op = _BINOPS[type(node.op)]
            if op == "**":
                return [f"{ind}{tcode} = {self.target.real_math('pow')}({tcode}, {code});"]
            return [f"{ind}{tcode} {op}= {self._coerce(code, ty, tty)};"]
        if isinstance(node, ast.If):
            return self._if(node, ctx, depth, "if")
        if isinstance(node, ast.While):
            cond = self._cond(node.test, ctx)
            lines = [f"{ind}while ({cond}) {{"]
            for s in node.body:
                lines.extend(self._stmt(s, ctx, depth + 1))
            lines.append(f"{ind}}}")
            return lines
        if isinstance(node, ast.For):
            return self._for(node, ctx, depth)
        if isinstance(node, ast.Return):
            if node.value is None:
                return [f"{ind}return;"]
            code, ty = self._expr(node.value, ctx)
            return [f"{ind}return {self._coerce(code, ty, ctx.sig.ret)};"]
        if isinstance(node, ast.Break):
            return [f"{ind}break;"]
        if isinstance(node, ast.Continue):
            return [f"{ind}continue;"]
        if isinstance(node, ast.Pass):
            return []
        raise TranspileError(f"{ctx.sig.name}: unsupported statement {type(node).__name__}")

    def _cond(self, node, ctx) -> str:
        code, ty = self._expr(node, ctx)
        code = self._coerce(code, ty, "bool")
        while code.startswith("(") and code.endswith(")") and _balanced(code[1:-1]):
            code = code[1:-1]
        return code

    def _if(self, node: ast.If, ctx, depth, kw) -> list[str]:
        ind = "    " * depth
        cond = self._cond(node.test, ctx)
        lines = [f"{ind}{kw} ({cond}) {{"]
        for s in node.body:
            lines.extend(self._stmt(s, ctx, depth + 1))
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                sub = self._if(node.orelse[0], ctx, depth, "if")
                lines.append(f"{ind}}} else {sub[0].lstrip()}")
                lines.extend(sub[1:])
                return lines
            lines.append(f"{ind}}} else {{")
            for s in node.orelse:
                lines.extend(self._stmt(s, ctx, depth + 1))
        lines.append(f"{ind}}}")
        return lines

    def _for(self, node: ast.For, ctx, depth) -> list[str]:
        ind = "    " * depth
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"):
            raise TranspileError("only 'for i in range(...)' loops are supported")
        args = [self._expr(a, ctx)[0] for a in node.iter.args]
        var = _ident(node.target.id)
        if len(args) == 1:
            start, stop, step = "0", args[0], "1"
        elif len(args) == 2:
            start, stop, step = args[0], args[1], "1"
        else:
            start, stop, step = args
        cmp = "<" if not step.startswith("-") else ">"
        lines = [f"{ind}for ({var} = {start}; {var} {cmp} {stop}; {var} += {step}) {{"]
        for s in node.body:
            lines.extend(self._stmt(s, ctx, depth + 1))
        lines.append(f"{ind}}}")
        return lines

    # ---------------------------------------------------------- expressions
    def _coerce(self, code: str, ty, want) -> str:
        if want is None or ty == want or isinstance(want, ArrayType):
            return code
        if want == "real" and ty in ("int", "bool"):
            return self.target.cast("real", code)
        if want == "int" and ty == "real":
            return self.target.cast("int", code)
        if want == "int" and ty == "bool":
            return self.target.cast("int", code)
        if want == "bool" and ty in ("int", "real"):
            return f"({code} != 0)"
        return code

    def _literal(self, v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        lim = self.target.float_literal_limit
        if lim is not None and v != 0.0:
            if abs(v) < lim:
                v = math.copysign(lim, v)
            elif abs(v) > 1.0 / lim:
                v = math.copysign(1.0 / lim, v)
        s = repr(float(v))
        if s in ("inf", "-inf", "nan"):
            raise TranspileError(f"non finite literal {v}")
        if "e" not in s and "." not in s:
            s += ".0"
        return s

    def _expr(self, node, ctx: _FunctionContext):
        """Returns (code, type)."""
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                return self._literal(v), "bool"
            if isinstance(v, int):
                return str(v), "int"
            if isinstance(v, float):
                return self._literal(v), "real"
            raise TranspileError(f"unsupported literal {v!r}")
        if isinstance(node, ast.Name):
            n = node.id
            if n in ctx.env:
                return _ident(n), ctx.env[n]
            if n in ctx.constants:
                v = ctx.constants[n]
                return n, "bool" if isinstance(v, bool) else ("int" if isinstance(v, int) else "real")
            if n == "True":
                return "true", "bool"
            if n == "False":
                return "false", "bool"
            raise TranspileError(f"{ctx.sig.name}: unknown name '{n}'")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "math":
                if node.attr == "pi":
                    return self._literal(math.pi), "real"
                if node.attr == "e":
                    return self._literal(math.e), "real"
                if node.attr == "inf":
                    return "(1.0/0.0)", "real"
            raise TranspileError(f"{ctx.sig.name}: unsupported attribute {ast.unparse(node)}")
        if isinstance(node, ast.Subscript):
            base, bty = self._expr(node.value, ctx)
            if not isinstance(bty, ArrayType):
                raise TranspileError(f"{ctx.sig.name}: subscript of a non array '{base}'")
            idx, ity = self._expr(node.slice, ctx)
            return f"{base}[{self._coerce(idx, ity, 'int')}]", bty.elem
        if isinstance(node, ast.UnaryOp):
            code, ty = self._expr(node.operand, ctx)
            if isinstance(node.op, ast.USub):
                return f"(-{code})", ty
            if isinstance(node.op, ast.UAdd):
                return code, ty
            if isinstance(node.op, ast.Not):
                return f"(!{self._coerce(code, ty, 'bool')})", "bool"
            if isinstance(node.op, ast.Invert):
                return f"(~{code})", "int"
            raise TranspileError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            return self._binop(node, ctx)
        if isinstance(node, ast.BoolOp):
            op = " && " if isinstance(node.op, ast.And) else " || "
            parts = [self._coerce(*self._expr(v, ctx), "bool") for v in node.values]
            return "(" + op.join(parts) + ")", "bool"
        if isinstance(node, ast.Compare):
            left, lty = self._expr(node.left, ctx)
            parts = []
            for op, comp in zip(node.ops, node.comparators):
                right, rty = self._expr(comp, ctx)
                if lty == "real" or rty == "real":
                    left, right = self._coerce(left, lty, "real"), self._coerce(right, rty, "real")
                parts.append(f"({left} {_CMPOPS[type(op)]} {right})")
                left, lty = right, rty
            return "(" + " && ".join(parts) + ")", "bool"
        if isinstance(node, ast.IfExp):
            cond, cty = self._expr(node.test, ctx)
            a, aty = self._expr(node.body, ctx)
            b, bty = self._expr(node.orelse, ctx)
            ty = _join(aty, bty) if isinstance(aty, str) and isinstance(bty, str) else aty
            return f"({self._coerce(cond, cty, 'bool')} ? {self._coerce(a, aty, ty)} : {self._coerce(b, bty, ty)})", ty
        if isinstance(node, ast.Call):
            return self._call(node, ctx)
        raise TranspileError(f"{ctx.sig.name}: unsupported expression {ast.unparse(node)}")

    def _binop(self, node: ast.BinOp, ctx):
        t = self.target
        a, aty = self._expr(node.left, ctx)
        b, bty = self._expr(node.right, ctx)
        op = _BINOPS[type(node.op)]
        if not (isinstance(aty, str) and isinstance(bty, str)):
            raise TranspileError(f"{ctx.sig.name}: arithmetic on arrays")
        both_int = aty != "real" and bty != "real"
        if op == "**":
            return f"{t.real_math('pow')}({self._coerce(a, aty, 'real')}, {self._coerce(b, bty, 'real')})", "real"
        if op == "/":
            return f"({self._coerce(a, aty, 'real')} / {self._coerce(b, bty, 'real')})", "real"
        if op == "//":
            if both_int:
                return f"({a} / {b})", "int"
            return f"{t.real_math('floor')}({self._coerce(a, aty, 'real')} / {self._coerce(b, bty, 'real')})", "real"
        if op == "%":
            if both_int:
                return f"({a} % {b})", "int"
            return f"{t.real_math('fmod')}({self._coerce(a, aty, 'real')}, {self._coerce(b, bty, 'real')})", "real"
        if op in ("&", "|", "^", "<<", ">>"):
            return f"({a} {op} {b})", "int"
        if both_int:
            return f"({a} {op} {b})", "int"
        return f"({self._coerce(a, aty, 'real')} {op} {self._coerce(b, bty, 'real')})", "real"

    def _call(self, node: ast.Call, ctx):
        t = self.target
        f = node.func
        args = [self._expr(a, ctx) for a in node.args]
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "math":
            name = f.attr
            if name in ("isnan", "isinf"):
                return f"{t.real_math(name)}({self._coerce(*args[0], 'real')})", "bool"
            if name in ("floor", "ceil", "trunc"):
                return f"{t.real_math(name)}({self._coerce(*args[0], 'real')})", "real"
            try:
                fn = t.real_math(name)
            except KeyError as e:
                raise TranspileError(f"{ctx.sig.name}: unsupported math function {name}") from e
            return f"{fn}({', '.join(self._coerce(c, ty, 'real') for c, ty in args)})", "real"
        if isinstance(f, ast.Name):
            name = f.id
            if name in ("abs", "min", "max"):
                tys = [ty for _, ty in args]
                if "real" in tys:
                    return f"{t.real_math(name)}({', '.join(self._coerce(c, ty, 'real') for c, ty in args)})", "real"
                return f"{name}({', '.join(c for c, _ in args)})", "int"
            if name == "int":
                return t.cast("int", args[0][0]), "int"
            if name == "float":
                return t.cast("real", args[0][0]), "real"
            if name == "bool":
                return self._coerce(*args[0], "bool"), "bool"
            if name in self.sigs:
                return self._kernel_call(name, node, args, ctx)
            if name in self._reg():
                self.sigs[name] = signature_of(self._reg()[name])
                return self._kernel_call(name, node, args, ctx)
        raise TranspileError(f"{ctx.sig.name}: unsupported call {ast.unparse(node)}")

    def _kernel_call(self, name, node, args, ctx):
        sig = self.sigs[name]
        if len(args) != len(sig.params):
            raise TranspileError(f"{ctx.sig.name}: {name} expects {len(sig.params)} arguments")
        out = []
        for (code, ty), (_, pty) in zip(args, sig.params):
            if isinstance(pty, ArrayType):
                if pty.const and self.target.uniform_context:
                    continue
                out.append(code)
            else:
                out.append(self._coerce(code, ty, pty))
        return f"{name}({', '.join(out)})", sig.ret or "void"


_BINOPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}

_CMPOPS = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def _balanced(s: str) -> bool:
    """True when the parentheses of ``s`` never close below depth 0."""
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _called_names(fd: ast.FunctionDef) -> list[str]:
    return [n.func.id for n in ast.walk(fd) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]


def transpile(target: Target, entry_points: list[str], extra_source: str = "", overrides: dict[str, KernelFunction] | None = None) -> str:
    """Transpile the kernel functions reachable from ``entry_points``."""
    return Transpiler(target, overrides).transpile(entry_points, extra_source)


__all__ = ["transpile", "Transpiler", "TranspileError", "signature_of", "builtins"]
