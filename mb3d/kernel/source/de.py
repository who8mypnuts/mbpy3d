"""Distance estimation (``CalcDEanalytic`` / ``CalcDEnoADE``).

The DE is returned in units of the pixel step width, like MB3D's marcher
works with it.
"""

import math

from mb3d.kernel.dsl import Bool, ConstIntArray, ConstRealArray, Int, Real, RealArray, kfunc

from .hybrid import hybrid_run, hybrid_start, mand_function_de, smooth_iterations
from .layout import (
    CX,
    IP_DE_MODE,
    IP_MAX_IT,
    IT,
    OTRAP,
    ROUT,
    SIT,
    SP_D_DE_SCALE,
    SP_DE_OFFSET,
    SP_DE_OFFSET006,
    SP_RSTOP,
    SP_RSTOP3D,
)


@kfunc
def calc_de(st: RealArray[16], sp: ConstRealArray[80], fp: ConstRealArray[96], ip: ConstIntArray[32],
            ms_de_stop: Real, calc_sit: Bool) -> Real:
    """Distance estimate at the point st[CX..CZ]; sets st[IT], st[SIT], st[OTRAP]."""
    mode = ip[IP_DE_MODE]
    max_it = ip[IP_MAX_IT]
    rstop = sp[SP_RSTOP]
    if mode == 1:
        result = mand_function_de(st, sp, fp, ip, max_it, rstop, calc_sit) * sp[SP_D_DE_SCALE]
    else:
        four_d = mode == 2
        hybrid_start(st, sp, ip, four_d)
        n = hybrid_run(st, sp, fp, ip, four_d, False, max_it, rstop)
        if calc_sit:
            smooth_iterations(st, sp, n)
        buf_rout = st[ROUT]
        if buf_rout < 1e-30:
            result = 0.0
        else:
            buf_it = st[IT]
            buf_sit = st[SIT]
            buf_otrap = st[OTRAP]
            its = int(buf_it)
            rstop3d = sp[SP_RSTOP3D]
            off = sp[SP_DE_OFFSET]
            # the gradient relative to |v|^2: the squares stay in range in single precision
            inv = 1.0 / buf_rout
            g = 0.0
            for k in range(3):
                bufd = st[CX + k]
                st[CX + k] = bufd + off
                hybrid_start(st, sp, ip, four_d)
                hybrid_run(st, sp, fp, ip, four_d, False, its, rstop3d)
                d = (buf_rout - st[ROUT]) * inv
                g += d * d
                st[CX + k] = bufd
            result = math.log(buf_rout) * sp[SP_D_DE_SCALE] / (math.sqrt(g) + sp[SP_DE_OFFSET006] * inv)
            st[ROUT] = buf_rout
            st[IT] = buf_it
            st[SIT] = buf_sit
            st[OTRAP] = buf_otrap
    mn = ms_de_stop * 0.25
    if result < mn or math.isnan(result):
        result = mn
    return result
