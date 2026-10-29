"""Generates the hybrid iteration loops of a program, specialised for its
formula slots (no per-iteration slot dispatch, no dynamic parameter
indices), in the C-like dialect and in numba Python.

Two loops exist per hybrid part:

* ``iterate``: the alternating hybrid (`doHybridPas`): every slot runs its
  iteration count in turn, after the last slot the loop continues from the
  repeat slot; counted slots update |z|^2, the orbit trap and the bailout.
* ``difs_orbit``: the dIFS loop (`doHybridIFS3D`): shapes write a distance
  and a trap value; the smallest distance over the iterations wins.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoopSlot:
    index: int          # slot index in the program (formula_<index>)
    counted: bool
    param_iters: str    # expression giving the iteration count (C and Python share the P_ names)


def c_iterate(name: str, slots: list[LoopSlot], repeat_param: str, four_d: bool) -> str:
    L = [f"Orbit {name}(FN_ARGS real3 pos, int max_it, real rstop2) {{",
         "    Orbit o = orbit_init(FN_PASS pos);",
         f"    int repeat_from = TO_INT(PARAM({repeat_param}));",
         "    bool first = true;",
         "    int btmp;",
         "    int guard;"]
    for k, s in enumerate(slots):
        L.append(f"    int n{k} = TO_INT(PARAM({s.param_iters}));")
    L.append("    if (" + " + ".join(f"n{k}" for k in range(len(slots))) + " <= 0) return o;")
    L.append("    for (guard = 0; guard < 100000; guard += 1) {")
    for k, s in enumerate(slots):
        L.append(f"        if (first || {k} >= repeat_from) {{")
        L.append(f"            for (btmp = 0; btmp < n{k}; btmp += 1) {{")
        L.append("                o.rold = o.s.r2;")
        L.append(f"                o.s = formula_{s.index}(FN_PASS o.s);")
        L.append(f"                o.slot = {k};")
        if s.counted:
            L.append("                o.it += 1;")
            r2 = "dot(o.s.z, o.s.z) + o.s.w * o.s.w" if four_d else "dot(o.s.z, o.s.z)"
            L.append(f"                o.s.r2 = {r2};")
            L.append("                if (o.s.r2 < o.otrap) o.otrap = o.s.r2;")
            L.append("                if (o.it >= max_it || o.s.r2 > rstop2 || isnan(o.s.r2)) return o;")
        L.append("            }")
        L.append("        }")
    L.append("        first = false;")
    # a pass that counts nothing from repeat_from on would spin forever
    L.append("        if (repeat_from >= " + str(len(slots)) + ") return o;")
    L.append("    }")
    L.append("    return o;")
    L.append("}")
    return "\n".join(L)


def c_difs(name: str, slots: list[LoopSlot], repeat_param: str) -> str:
    exit_ = "{ o.it = it_min; o.otrap = dfree2; o.s.r2 = min_de; return o; }"
    L = [f"Orbit {name}(FN_ARGS real3 pos, int max_it, real rstopd, bool inside) {{",
         "    Orbit o = orbit_init(FN_PASS pos);",
         f"    int repeat_from = TO_INT(PARAM({repeat_param}));",
         "    bool first = true;",
         "    int btmp;",
         "    int guard;",
         "    real min_de = 65535.0;",
         "    real dfree2 = 0.0;",
         "    real de;",
         "    int it_min = 0;",
         "    int it = 0;",
         "    o.s.aux = rstopd;",
         "    o.s.de = 1e30;",
         "    o.s.trap = 0.0;"]
    for k, s in enumerate(slots):
        L.append(f"    int n{k} = TO_INT(PARAM({s.param_iters}));")
    L.append("    if (" + " + ".join(f"n{k}" for k in range(len(slots))) + " <= 0) " + exit_)
    L.append("    for (guard = 0; guard < 100000; guard += 1) {")
    for k, s in enumerate(slots):
        L.append(f"        if (first || {k} >= repeat_from) {{")
        L.append(f"            for (btmp = 0; btmp < n{k}; btmp += 1) {{")
        L.append(f"                o.s = formula_{s.index}(FN_PASS o.s);")
        L.append(f"                o.slot = {k};")
        if s.counted:
            L.append("                it += 1;")
            L.append("                de = o.s.de / o.s.dr;")
            L.append("                if (de < min_de) {")
            L.append("                    it_min = it;")
            L.append("                    dfree2 = o.s.trap;")
            L.append("                    min_de = de;")
            L.append("                    if (!inside && de < rstopd) " + exit_)
            L.append("                }")
            L.append("                if (it >= max_it) " + exit_)
        L.append("            }")
        L.append("        }")
    L.append("        first = false;")
    L.append("        if (repeat_from >= " + str(len(slots)) + ") " + exit_)
    L.append("    }")
    L.append("    " + exit_)
    L.append("}")
    return "\n".join(L)


PY_STATE = "s_zx, s_zy, s_zz, s_w, s_cx, s_cy, s_cz, s_cw, s_dr, s_r2, s_de, s_trap, s_aux, s_vary"
PY_CALL_RET = "s_zx, s_zy, s_zz, s_w, s_cx, s_cy, s_cz, s_dr, s_de, s_trap, s_vary"
DECO = '@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")'


PY_ITER_RET = "s_r2, s_dr, rold, otrap, it, slot, s_zx, s_zy, s_zz, s_cx, s_cy, s_cz"
PY_DIFS_RET = "min_de, it_min, dfree2, slot, s_zx, s_zy, s_zz, s_cx, s_cy, s_cz"


def py_iterate(name: str, slots: list[LoopSlot], repeat_param: str, four_d: bool) -> str:
    """Returns (r2, dr, rold, otrap, it, slot, zx, zy, zz, cx, cy, cz)."""
    L = [DECO, f"def {name}(px, py, pz, max_it, rstop2, P):",
         f"    {PY_STATE} = orbit_init(px, py, pz, P)",
         "    rold = s_r2", "    otrap = s_r2", "    it = 0", "    slot = 0",
         f"    repeat_from = int(P[{repeat_param}])", "    first = True"]
    for k, s in enumerate(slots):
        L.append(f"    n{k} = int(P[{s.param_iters}])")
    L.append("    if " + " + ".join(f"n{k}" for k in range(len(slots))) + " <= 0:")
    L.append("        return " + PY_ITER_RET)
    L.append("    for guard in range(100000):")
    for k, s in enumerate(slots):
        L.append(f"        if first or {k} >= repeat_from:")
        L.append(f"            for btmp in range(n{k}):")
        L.append("                rold = s_r2")
        L.append(f"                {PY_CALL_RET} = formula_{s.index}({PY_STATE}, P)")
        L.append(f"                slot = {k}")
        if s.counted:
            L.append("                it += 1")
            r2 = "s_zx * s_zx + s_zy * s_zy + s_zz * s_zz" + (" + s_w * s_w" if four_d else "")
            L.append(f"                s_r2 = {r2}")
            L.append("                if s_r2 < otrap:")
            L.append("                    otrap = s_r2")
            L.append("                if it >= max_it or s_r2 > rstop2 or math.isnan(s_r2):")
            L.append("                    return " + PY_ITER_RET)
        L.append("                pass")
    L.append("        first = False")
    L.append(f"        if repeat_from >= {len(slots)}:")
    L.append("            return " + PY_ITER_RET)
    L.append("    return " + PY_ITER_RET)
    return "\n".join(L)


def py_difs(name: str, slots: list[LoopSlot], repeat_param: str) -> str:
    """Returns (min_de, it_min, dfree2, slot, zx, zy, zz, cx, cy, cz)."""
    L = [DECO, f"def {name}(px, py, pz, max_it, rstopd, inside, P):",
         f"    {PY_STATE} = orbit_init(px, py, pz, P)",
         "    s_aux = rstopd", "    s_de = 1e30", "    s_trap = 0.0",
         "    min_de = 65535.0", "    dfree2 = 0.0", "    it_min = 0", "    it = 0", "    slot = 0",
         f"    repeat_from = int(P[{repeat_param}])", "    first = True"]
    for k, s in enumerate(slots):
        L.append(f"    n{k} = int(P[{s.param_iters}])")
    L.append("    if " + " + ".join(f"n{k}" for k in range(len(slots))) + " <= 0:")
    L.append("        return " + PY_DIFS_RET)
    L.append("    for guard in range(100000):")
    for k, s in enumerate(slots):
        L.append(f"        if first or {k} >= repeat_from:")
        L.append(f"            for btmp in range(n{k}):")
        L.append(f"                {PY_CALL_RET} = formula_{s.index}({PY_STATE}, P)")
        L.append(f"                slot = {k}")
        if s.counted:
            L.append("                it += 1")
            L.append("                de = s_de / s_dr")
            L.append("                if de < min_de:")
            L.append("                    it_min = it")
            L.append("                    dfree2 = s_trap")
            L.append("                    min_de = de")
            L.append("                    if (not inside) and de < rstopd:")
            L.append("                        return " + PY_DIFS_RET)
            L.append("                if it >= max_it:")
            L.append("                    return " + PY_DIFS_RET)
        L.append("                pass")
    L.append("        first = False")
    L.append(f"        if repeat_from >= {len(slots)}:")
    L.append("            return " + PY_DIFS_RET)
    L.append("    return " + PY_DIFS_RET)
    return "\n".join(L)


def c_iterate_stub(name: str) -> str:
    return (f"Orbit {name}(FN_ARGS real3 pos, int max_it, real rstop2) {{\n"
            "    return orbit_init(FN_PASS pos);\n}")


def c_difs_stub(name: str) -> str:
    return (f"Orbit {name}(FN_ARGS real3 pos, int max_it, real rstopd, bool inside) {{\n"
            "    Orbit o = orbit_init(FN_PASS pos);\n    o.s.r2 = 65535.0;\n    return o;\n}")


def py_iterate_stub(name: str) -> str:
    return "\n".join([DECO, f"def {name}(px, py, pz, max_it, rstop2, P):",
                      f"    {PY_STATE} = orbit_init(px, py, pz, P)",
                      "    rold = s_r2", "    otrap = s_r2", "    it = 0", "    slot = 0",
                      "    return " + PY_ITER_RET])


def py_difs_stub(name: str) -> str:
    return "\n".join([DECO, f"def {name}(px, py, pz, max_it, rstopd, inside, P):",
                      f"    {PY_STATE} = orbit_init(px, py, pz, P)",
                      "    min_de = 65535.0", "    it_min = 0", "    dfree2 = 0.0", "    slot = 0",
                      "    return " + PY_DIFS_RET])


def c_ipol(name: str, slots: list[LoopSlot], four_d: bool) -> str:
    """The interpolation hybrid (doInterpolHybridPas): both formulas are
    applied to the same point every iteration and blended by the weights,
    the length of the result blended as well."""
    a, b = slots[0].index, slots[1].index
    w2 = " + a.w * a.w" if four_d else ""
    w2b = " + b.w * b.w" if four_d else ""
    w2o = " + o.s.w * o.s.w" if four_d else ""
    return "\n".join([
        f"Orbit {name}(FN_ARGS real3 pos, int max_it, real rstop2) {{",
        "    Orbit o = orbit_init(FN_PASS pos);",
        "    real s1 = PARAM(P_IPOL_W1);",
        "    real s2 = PARAM(P_IPOL_W2);",
        "    State a;",
        "    State b;",
        "    real xx;",
        "    real yy;",
        "    int guard;",
        "    for (guard = 0; guard < 100000; guard += 1) {",
        "        o.rold = o.s.r2;",
        f"        a = formula_{a}(FN_PASS o.s);",
        f"        b = formula_{b}(FN_PASS o.s);",
        f"        xx = sqrt(dot(a.z, a.z){w2}) * s1 + sqrt(dot(b.z, b.z){w2b}) * s2;",
        "        o.s.z = a.z * s1 + b.z * s2;",
        "        o.s.w = a.w * s1 + b.w * s2;",
        "        o.s.dr = fabs(a.dr) * s1 + fabs(b.dr) * s2;",
        "        o.s.c = a.c;",
        "        o.s.cw = a.cw;",
        "        o.s.vary = a.vary;",
        f"        yy = xx / sqrt(dot(o.s.z, o.s.z){w2o} + TINY);",
        "        o.s.z = o.s.z * yy;",
        "        o.s.w = o.s.w * yy;",
        "        o.slot = 1;",
        "        o.it += 1;",
        f"        o.s.r2 = dot(o.s.z, o.s.z){w2o};",
        "        if (o.s.r2 < o.otrap) o.otrap = o.s.r2;",
        "        if (o.it >= max_it || o.s.r2 > rstop2 || isnan(o.s.r2)) return o;",
        "    }",
        "    return o;",
        "}",
    ])


def py_ipol(name: str, slots: list[LoopSlot], four_d: bool) -> str:
    a, b = slots[0].index, slots[1].index
    A = "a_zx, a_zy, a_zz, a_w, a_cx, a_cy, a_cz, a_dr, a_de, a_trap, a_vary"
    B = "b_zx, b_zy, b_zz, b_w, b_cx, b_cy, b_cz, b_dr, b_de, b_trap, b_vary"
    wa = " + a_w * a_w" if four_d else ""
    wb = " + b_w * b_w" if four_d else ""
    wo = " + s_w * s_w" if four_d else ""
    return "\n".join([
        DECO, f"def {name}(px, py, pz, max_it, rstop2, P):",
        f"    {PY_STATE} = orbit_init(px, py, pz, P)",
        "    rold = s_r2", "    otrap = s_r2", "    it = 0", "    slot = 1",
        "    s1 = P[P_IPOL_W1]", "    s2 = P[P_IPOL_W2]",
        "    for guard in range(100000):",
        "        rold = s_r2",
        f"        {A} = formula_{a}({PY_STATE}, P)",
        f"        {B} = formula_{b}({PY_STATE}, P)",
        f"        xx = math.sqrt(a_zx * a_zx + a_zy * a_zy + a_zz * a_zz{wa}) * s1 + math.sqrt(b_zx * b_zx + b_zy * b_zy + b_zz * b_zz{wb}) * s2",
        "        s_zx = a_zx * s1 + b_zx * s2",
        "        s_zy = a_zy * s1 + b_zy * s2",
        "        s_zz = a_zz * s1 + b_zz * s2",
        "        s_w = a_w * s1 + b_w * s2",
        "        s_dr = abs(a_dr) * s1 + abs(b_dr) * s2",
        "        s_cx, s_cy, s_cz, s_vary = a_cx, a_cy, a_cz, a_vary",
        f"        yy = xx / math.sqrt(s_zx * s_zx + s_zy * s_zy + s_zz * s_zz{wo} + 1e-40)",
        "        s_zx, s_zy, s_zz, s_w = s_zx * yy, s_zy * yy, s_zz * yy, s_w * yy",
        "        it += 1",
        f"        s_r2 = s_zx * s_zx + s_zy * s_zy + s_zz * s_zz{wo}",
        "        if s_r2 < otrap:",
        "            otrap = s_r2",
        "        if it >= max_it or s_r2 > rstop2 or math.isnan(s_r2):",
        "            return " + PY_ITER_RET,
        "    return " + PY_ITER_RET,
    ])
