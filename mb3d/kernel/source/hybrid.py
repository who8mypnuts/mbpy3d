"""The hybrid iteration loop (``doHybridPas`` / ``doHybridPasDE`` / ``doHybrid4DPas``).

``st`` holds the iteration record, ``ip``/``sp`` the slot configuration.
"""

import math

from mb3d.formulas.dispatch import cpu_dispatcher
from mb3d.kernel.dsl import Bool, ConstIntArray, ConstRealArray, Int, Real, RealArray, kfunc

from .layout import (
    CX,
    CY,
    CZ,
    FP_SLOT,
    IP_END_TO,
    IP_FID,
    IP_ITS,
    IP_JULIA,
    IP_REPEAT_FROM,
    IP_UNCOUNTED,
    IT,
    JW,
    JX,
    JY,
    JZ,
    OTRAP,
    ROLD,
    ROUT,
    SIT,
    SP_FHLN,
    SP_JU,
    SP_LN_RSTOP,
    SP_SMATRIX4,
    W,
    X,
    Y,
    Z,
)

apply_formula = cpu_dispatcher()


@kfunc
def hybrid_start(st: RealArray[16], sp: ConstRealArray[80], ip: ConstIntArray[32], four_d: Bool) -> None:
    """Sets the iterated vector and the constant J from the start position C."""
    cx = st[CX]
    cy = st[CY]
    cz = st[CZ]
    if four_d:
        # Rotate4Dex: the 3D point (w = 0) turned into 4D
        vx = cx * sp[SP_SMATRIX4 + 0] + cy * sp[SP_SMATRIX4 + 1] + cz * sp[SP_SMATRIX4 + 2]
        vy = cx * sp[SP_SMATRIX4 + 4] + cy * sp[SP_SMATRIX4 + 5] + cz * sp[SP_SMATRIX4 + 6]
        vz = cx * sp[SP_SMATRIX4 + 8] + cy * sp[SP_SMATRIX4 + 9] + cz * sp[SP_SMATRIX4 + 10]
        vw = cx * sp[SP_SMATRIX4 + 12] + cy * sp[SP_SMATRIX4 + 13] + cz * sp[SP_SMATRIX4 + 14]
        st[X] = vx
        st[Y] = vy
        st[Z] = vz
        st[W] = vw
        if ip[IP_JULIA] != 0:
            st[JX] = sp[SP_JU]
            st[JY] = sp[SP_JU + 1]
            st[JZ] = sp[SP_JU + 2]
            st[JW] = sp[SP_JU + 3]
        else:
            st[JX] = vx
            st[JY] = vy
            st[JZ] = vz
            st[JW] = vw
        st[ROUT] = vx * vx + vy * vy + vz * vz + vw * vw
    else:
        st[X] = cx
        st[Y] = cy
        st[Z] = cz
        st[W] = 0.0
        if ip[IP_JULIA] != 0:
            st[JX] = sp[SP_JU]
            st[JY] = sp[SP_JU + 1]
            st[JZ] = sp[SP_JU + 2]
        else:
            st[JX] = cx
            st[JY] = cy
            st[JZ] = cz
        st[JW] = sp[SP_JU + 3]
        st[ROUT] = cx * cx + cy * cy + cz * cz
    st[OTRAP] = st[ROUT]
    st[ROLD] = st[ROUT]


@kfunc
def hybrid_run(st: RealArray[16], sp: ConstRealArray[80], fp: ConstRealArray[96], ip: ConstIntArray[32],
               four_d: Bool, ade: Bool, max_it: Int, rstop: Real) -> Int:
    """The alternating hybrid loop; returns the slot of the last formula, st[IT] the count."""
    end_to = ip[IP_END_TO]
    repeat_from = ip[IP_REPEAT_FROM]
    n = 0
    btmp = ip[IP_ITS]
    it_result = 0
    uncounted_guard = 0
    while True:
        st[ROLD] = st[ROUT]
        guard = 0
        while btmp <= 0:
            n += 1
            if n > end_to:
                n = repeat_from
            btmp = ip[IP_ITS + n]
            guard += 1
            if guard > 64:
                st[IT] = float(it_result)
                return n
        apply_formula(ip[IP_FID + n], st, fp, n * FP_SLOT, ade)
        btmp -= 1
        if ip[IP_UNCOUNTED + n] != 0:
            uncounted_guard += 1
            if uncounted_guard > 100000:
                break
            continue
        it_result += 1
        x = st[X]
        y = st[Y]
        z = st[Z]
        rout = x * x + y * y + z * z
        if four_d:
            rout += st[W] * st[W]
        if math.isinf(rout):
            rout = 1e30
        st[ROUT] = rout
        if rout < st[OTRAP]:
            st[OTRAP] = rout
        if it_result >= max_it or rout > rstop:
            break
    st[IT] = float(it_result)
    return n


@kfunc
def smooth_iterations(st: RealArray[16], sp: ConstRealArray[80], n: Int) -> None:
    """``CalcSmoothIterations``: the fractional iteration count into st[SIT]."""
    rout = st[ROUT]
    it = st[IT]
    if rout <= 1.0:
        st[SIT] = it
    elif st[ROLD] < 1.0:
        d = math.log(0.5 * math.log(rout)) * sp[SP_FHLN + n]
        st[SIT] = it + sp[SP_LN_RSTOP] - d
    else:
        d = math.log(0.5 * math.log(rout))
        e = d - math.log(0.5 * math.log(st[ROLD]))
        st[SIT] = (sp[SP_LN_RSTOP] - d) / (e + 1e-30) + it


@kfunc
def mand_function(st: RealArray[16], sp: ConstRealArray[80], fp: ConstRealArray[96], ip: ConstIntArray[32],
                  four_d: Bool, max_it: Int, rstop: Real, calc_sit: Bool) -> None:
    """``mMandFunction``: iterate from C without a DE."""
    hybrid_start(st, sp, ip, four_d)
    n = hybrid_run(st, sp, fp, ip, four_d, False, max_it, rstop)
    if calc_sit:
        smooth_iterations(st, sp, n)


@kfunc
def mand_function_de(st: RealArray[16], sp: ConstRealArray[80], fp: ConstRealArray[96], ip: ConstIntArray[32],
                     max_it: Int, rstop: Real, calc_sit: Bool) -> Real:
    """``doHybridPasDE``: 3D hybrid with the running derivative in w; returns |v| / |w|."""
    hybrid_start(st, sp, ip, False)
    st[W] = 1.0
    n = hybrid_run(st, sp, fp, ip, False, True, max_it, rstop)
    if calc_sit:
        smooth_iterations(st, sp, n)
    return math.sqrt(st[ROUT]) / abs(st[W])
