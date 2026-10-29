"""The triplex 'bulb' formulas: Integer Power 2..8, Real Power, Folding Int Pow.

Ported from ``formulas.pas`` (``HybridItIntPow2..P8``, ``HybridFloatPow``)
keeping the evaluation order of MB3D's assembler.
"""

import math

from mb3d.kernel.dsl import Bool, ConstRealArray, Int, Real, RealArray, kfunc
from mb3d.kernel.source.layout import JX, JY, JZ, ROUT, X, Y, Z

EPS_POW = 1e-30


@kfunc
def box_fold(x: Real, fold: Real) -> Real:
    """``x = abs(x + fold) - abs(x - fold) - x`` (the 'Tglad' fold)."""
    return abs(x + fold) - (abs(x - fold) + x)


@kfunc
def int_pow_step(st: RealArray[16], power: Int, zmul: Real) -> None:
    """One iteration of the integer power bulb: v = v^power + J."""
    x = st[X]
    y = st[Y]
    z = st[Z]
    if power <= 2:
        yy = y * y
        xx = x * x
        r = xx + yy
        st[Z] = (math.sqrt(r) * zmul * z) * 2.0 + st[JZ]
        a = (r - z * z) / (r + EPS_POW)
        st[X] = (xx - yy) * a + st[JX]
        st[Y] = (x * y * a) * 2.0 + st[JY]
    elif power == 3:
        sy = y * y
        sx = x * x
        r = sx + sy
        sz = z * z
        a = 1.0 - 3.0 * sz / (r + EPS_POW)
        st[X] = (sx - 3.0 * sy) * a * x + st[JX]
        st[Z] = st[JZ] - (sz - 3.0 * r) * z * zmul
        st[Y] = a * (3.0 * sx - sy) * y + st[JY]
    elif power == 4:
        sy = y * y
        sx = x * x
        r = sx + sy
        sz = z * z
        a = 1.0 + sz * (sz - 6.0 * r) / (r * r + EPS_POW)
        st[X] = (sx * (sx - 6.0 * sy) + sy * sy) * a + st[JX]
        st[Z] = math.sqrt(r) * (r - sz) * z * 4.0 * zmul + st[JZ]
        st[Y] = a * x * (sx - sy) * 4.0 * y + st[JY]
    elif power == 5:
        sy = y * y
        sx = x * x
        r = sx + sy
        sz = z * z
        a = 1.0 + (sz * sz - sz * r * 2.0) * 5.0 / (r * r + EPS_POW)
        st[Y] = (5.0 * sx * sx - sy * (10.0 * sx - sy)) * a * y + st[JY]
        st[Z] = (sz * (sz - 10.0 * r) + 5.0 * r * r) * z * zmul + st[JZ]
        st[X] = a * x * (sx * (sx - 10.0 * sy) + 5.0 * sy * sy) + st[JX]
    elif power == 6:
        sy = y * y
        sx = x * x
        r = sx + sy
        sz = z * z
        rr = r * r
        a = 1.0 - sz * (15.0 * rr + sz * (sz - r * 15.0)) / (rr * r + EPS_POW)
        st[Y] = (3.0 * sy * sy + sx * (3.0 * sx - 10.0 * sy)) * a * y * x * 2.0 + st[JY]
        st[Z] = (sz * (3.0 * sz - 10.0 * r) + 3.0 * rr) * math.sqrt(r) * z * zmul * 2.0 + st[JZ]
        st[X] = (sy * sy * (15.0 * sx - sy) + sx * sx * (sx - 15.0 * sy)) * a + st[JX]
    elif power == 7:
        sy = y * y
        sx = x * x
        r = sx + sy
        sz = z * z
        rr = r * r
        a = 1.0 - 7.0 * (sz * (3.0 * rr + sz * (sz - 5.0 * r))) / (rr * r + EPS_POW)
        st[Y] = (sx * (21.0 * sy * sy + sx * (7.0 * sx - 35.0 * sy)) - sy * sy * sy) * a * y + st[JY]
        st[Z] = st[JZ] - (sz * sz * sz - 7.0 * r * (rr + sz * (3.0 * sz - 5.0 * r))) * z * zmul
        st[X] = a * x * (sx * (35.0 * sy * sy + sx * (sx - 21.0 * sy)) - 7.0 * sy * sy * sy) + st[JX]
    else:
        xx = x * x
        yy = y * y
        zz = z * z
        r = xx + yy
        rr = r * r
        zzzz = zz * zz
        t = (zzzz - 6.0 * r * zz + rr) * (zz - r) * math.sqrt(r)
        st[Z] = -(t * z * 8.0 * zmul) + st[JZ]
        num = (rr * 70.0 + zzzz) * zzzz - 28.0 * zz * r * (zzzz + rr)
        a = num / (rr * rr + EPS_POW) + 1.0
        xxxx = xx * xx
        yyyy = yy * yy
        st[Y] = (yyyy * (7.0 * xx - yy) + xxxx * (xx - 7.0 * yy)) * 8.0 * x * y * a + st[JY]
        st[X] = a * (xxxx * xxxx + yyyy * (70.0 * xxxx + yyyy) - 28.0 * (xx * yy * (yyyy + xxxx))) + st[JX]


@kfunc
def f_integer_power(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'Integer Power': options (Integer power 2..8, Z multiplier)."""
    power = int(fp[b] + 0.5)
    if power < 2:
        power = 2
    if power > 8:
        power = 8
    int_pow_step(st, power, fp[b + 1])


@kfunc
def f_real_power(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'Real Power': options (Float power, Z multiplier); uses the stored |v|^2."""
    pw = fp[b]
    zmul = fp[b + 1]
    x = st[X]
    y = st[Y]
    z = st[Z]
    theta = math.atan2(y, x) * pw
    phi = math.atan2(z, math.sqrt(x * x + y * y)) * pw
    r = math.pow(st[ROUT], pw * 0.5)
    cp = math.cos(phi)
    st[X] = math.cos(theta) * cp * r + st[JX]
    st[Y] = math.sin(theta) * cp * r + st[JY]
    st[Z] = math.sin(phi) * r * zmul + st[JZ]


@kfunc
def f_folding_int_pow(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'Folding Int Pow': a box fold of x, y, z, then the integer power bulb."""
    fold = fp[b + 2]
    st[X] = box_fold(st[X], fold)
    st[Y] = box_fold(st[Y], fold)
    st[Z] = box_fold(st[Z], fold)
    power = int(fp[b] + 0.5)
    if power < 2:
        power = 2
    if power > 8:
        power = 8
    int_pow_step(st, power, fp[b + 1])
