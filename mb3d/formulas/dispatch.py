"""``apply_formula``: the generated dispatcher from formula id to kernel function.

The CPU gets one dispatcher with every registered formula (compiled once and
cached); a GPU shader gets one with only the formulas of its scene.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache

from mb3d.kernel import cache_dir
from mb3d.kernel.dsl import KernelFunction

from . import registry
from .spec import FormulaSpec

DISPATCH_NAME = "apply_formula"


def dispatcher_source(specs: list[FormulaSpec], decorate: bool = True) -> str:
    """DSL source of ``apply_formula`` for the given formulas."""
    lines = [
        "from mb3d.kernel.dsl import Bool, ConstRealArray, Int, RealArray, kfunc",
    ]
    for s in specs:
        kf = s.kernel._kernel_function
        lines.append(f"from {kf.module} import {kf.name}")
    lines.append("")
    if decorate:
        lines.append("@kfunc")
    lines.append(f"def {DISPATCH_NAME}(fid: Int, st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:")
    lines.append('    """Runs one iteration of formula ``fid`` on the state."""')
    kw = "if"
    for s in specs:
        lines.append(f"    {kw} fid == {s.fid}:")
        lines.append(f"        {registry.kernel_name(s)}(st, fp, b, ade)")
        kw = "elif"
    if not specs:
        lines.append("    return")
    lines.append("")
    return "\n".join(lines)


def gpu_dispatcher(specs: list[FormulaSpec]) -> KernelFunction:
    """A transpile-only ``apply_formula`` restricted to ``specs``."""
    src = dispatcher_source(specs, decorate=False)
    ns: dict = {}
    exec(compile("\n".join(l for l in src.splitlines() if l.startswith(("from", "import"))), "<imports>", "exec"), ns)
    body = "\n".join(l for l in src.splitlines() if not l.startswith(("from", "import")))
    return KernelFunction.from_source(DISPATCH_NAME, body, ns)


@lru_cache(maxsize=1)
def cpu_dispatcher():
    """The numba compiled ``apply_formula`` with every registered formula.

    Written as a real module file into the kernel cache directory, so that
    numba can cache it like the other kernel functions.
    """
    specs = registry.all_specs()
    src = dispatcher_source(specs)
    path = cache_dir() / "mb3d_apply_formula.py"
    if not path.exists() or path.read_text() != src:
        path.write_text(src)
    spec = importlib.util.spec_from_file_location("mb3d_apply_formula", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mb3d_apply_formula"] = module
    spec.loader.exec_module(module)
    return getattr(module, DISPATCH_NAME)
