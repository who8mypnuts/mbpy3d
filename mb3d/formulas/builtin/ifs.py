"""Kaleidoscopic IFS formulas: Menger3, Sierpinski3 and ABoxModKali.

Native ports of the ``.m3f`` formulas of the same names (Jesse / Kali).
"""

from mb3d.kernel.dsl import Bool, ConstRealArray, Int, RealArray, kfunc
from mb3d.kernel.source.layout import JX, JY, JZ, W, X, Y, Z

from .boxes import DEG, rotate_xyz


@kfunc
def sort_abs_xyz(st: RealArray[16]) -> None:
    """x, y, z := |x|, |y|, |z| sorted so that x >= y >= z (the cubic fold)."""
    x = abs(st[X])
    y = abs(st[Y])
    z = abs(st[Z])
    if x < y:
        t = x
        x = y
        y = t
    if x < z:
        t = x
        x = z
        z = t
    if y < z:
        t = y
        y = z
        z = t
    st[X] = x
    st[Y] = y
    st[Z] = z


@kfunc
def f_menger3(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'Menger3': options (Scale, CScale X, CScale Y, CScale Z, Rotation1 XYZ, Rotation2 XYZ)."""
    scale = fp[b]
    csx = fp[b + 1]
    csy = fp[b + 2]
    csz = fp[b + 3]
    sort_abs_xyz(st)
    rotate_xyz(st, fp[b + 4] * DEG, fp[b + 5] * DEG, fp[b + 6] * DEG)
    sm1 = scale - 1.0
    st[X] = scale * st[X] - csx * sm1
    st[Y] = scale * st[Y] - csy * sm1
    z = scale * st[Z]
    if z > 0.5 * csz * sm1:
        z = z - csz * sm1
    st[Z] = z
    rotate_xyz(st, fp[b + 7] * DEG, fp[b + 8] * DEG, fp[b + 9] * DEG)
    if ade:
        st[W] = st[W] * scale


@kfunc
def f_sierpinski3(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'Sierpinski3': tetrahedral folds, then scale; options like Menger3."""
    scale = fp[b]
    csx = fp[b + 1]
    csy = fp[b + 2]
    csz = fp[b + 3]
    x = st[X]
    y = st[Y]
    z = st[Z]
    if x + y < 0.0:
        t = -y
        y = -x
        x = t
    if x + z < 0.0:
        t = -z
        z = -x
        x = t
    if y + z < 0.0:
        t = -z
        z = -y
        y = t
    st[X] = x
    st[Y] = y
    st[Z] = z
    rotate_xyz(st, fp[b + 4] * DEG, fp[b + 5] * DEG, fp[b + 6] * DEG)
    sm1 = scale - 1.0
    st[X] = scale * st[X] - csx * sm1
    st[Y] = scale * st[Y] - csy * sm1
    st[Z] = scale * st[Z] - csz * sm1
    rotate_xyz(st, fp[b + 7] * DEG, fp[b + 8] * DEG, fp[b + 9] * DEG)
    if ade:
        st[W] = st[W] * scale


@kfunc
def f_abox_mod_kali(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'ABoxModKali': x = X add - |x| (y, z alike), sphere fold, scale, + C.

    Options: Scale, Min R, Z add, Y add, X add.
    """
    scale = fp[b]
    min_r = fp[b + 1]
    x = fp[b + 4] - abs(st[X])
    y = fp[b + 3] - abs(st[Y])
    z = fp[b + 2] - abs(st[Z])
    rr = x * x + y * y + z * z
    sqr_min_r = min_r * min_r
    if rr < sqr_min_r:
        m = scale / sqr_min_r
    elif rr < 1.0:
        m = scale / rr
    else:
        m = scale
    if ade:
        st[W] = st[W] * m
    st[X] = x * m + st[JX]
    st[Y] = y * m + st[JY]
    st[Z] = z * m + st[JZ]
