"""Quaternion (4D) and Tricorn formulas (``formulas.pas``)."""

from mb3d.kernel.dsl import Bool, ConstRealArray, Int, RealArray, kfunc
from mb3d.kernel.source.layout import JW, JX, JY, JZ, W, X, Y, Z


@kfunc
def f_quaternion(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'Quaternion': v = v^2 + J in 4D; options (YW multiplier, W add)."""
    yw_mul = fp[b]
    w_add = fp[b + 1]
    x = st[X]
    y = st[Y]
    z = st[Z]
    w = st[W]
    st[X] = x * x - y * y - z * z - w * w + st[JX]
    st[Y] = 2.0 * (y * x + z * w) + st[JY]
    st[Z] = 2.0 * (z * x + yw_mul * y * w) + st[JZ]
    st[W] = 2.0 * (w * x + y * z) + w_add + st[JW]


@kfunc
def f_tricorn(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'Tricorn': options (Z multiplier, CZ multiplier)."""
    z_mul = fp[b]
    cz_mul = fp[b + 1]
    x = st[X]
    y = st[Y]
    z = st[Z]
    st[Z] = z * x * z_mul + st[JZ] * cz_mul
    st[X] = x * x - (y * y + z * z) + st[JX]
    st[Y] = 2.0 * (y * x) + st[JY]
