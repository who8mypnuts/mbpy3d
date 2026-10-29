"""Amazing Box (Mandelbox), Bulbox, Amazing Surf and ABoxMod1.

``HybridCube`` / ``HybridCubeDE`` / ``HybridSuperCube2`` of formulas.pas and
native ports of the popular Kali / Luca GN ``.m3f`` box formulas.
"""

import math

from mb3d.kernel.dsl import Bool, ConstRealArray, Int, Real, RealArray, kfunc
from mb3d.kernel.source.layout import JX, JY, JZ, ROUT, W, X, Y, Z

from .bulbs import box_fold


@kfunc
def amazing_box_step(st: RealArray[16], scale: Real, min_r: Real, fold: Real, ade: Bool) -> None:
    """One Mandelbox iteration; with ``ade`` the derivative in w is scaled too."""
    x = box_fold(st[X], fold)
    y = box_fold(st[Y], fold)
    z = box_fold(st[Z], fold)
    r = z * z + y * y + x * x
    sqr_min_r = min_r * min_r
    if r < sqr_min_r:
        mul = scale / sqr_min_r
    elif r < 1.0:
        mul = scale / r
    else:
        mul = scale
    if ade:
        st[W] = st[W] * mul
    st[Z] = z * mul + st[JZ]
    st[Y] = y * mul + st[JY]
    st[X] = x * mul + st[JX]


@kfunc
def f_amazing_box(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'Amazing Box': options (Scale, Min R, Fold)."""
    amazing_box_step(st, fp[b], fp[b + 1], fp[b + 2], ade)


@kfunc
def int_pow2_scale(st: RealArray[16], scaling: Real) -> None:
    """``HybridItIntPow2scale``: sine bulb power 2 with scaling (Bulbox)."""
    s = 1.0 / scaling
    x = st[X] * s
    y = st[Y] * s
    z = st[Z] * s
    xx = x * x
    yy = y * y
    r = xx + yy
    st[Z] = -((math.sqrt(r) * z) * 2.0) * scaling + st[JZ]
    a = (r - z * z) / (r + 1e-30)
    st[X] = (xx - yy) * a * scaling + st[JX]
    st[Y] = (x * y * a) * 2.0 * scaling + st[JY]


@kfunc
def f_bulbox(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'Bulbox' (``HybridSuperCube2``): bulb inside a radius, box outside, blended between."""
    scale = fp[b]
    min_r = fp[b + 1]
    fold = fp[b + 2]
    bulb_scale = fp[b + 3]
    rt1 = fp[b + 4] * fp[b + 4]
    rt2 = fp[b + 5] * fp[b + 5]
    rout = st[ROUT]
    if rout < rt1:
        if rout < rt2:
            int_pow2_scale(st, bulb_scale)
        else:
            m = (rout - rt2) / (rt1 - rt2)
            x0 = st[X]
            y0 = st[Y]
            z0 = st[Z]
            int_pow2_scale(st, bulb_scale)
            ox = st[X]
            oy = st[Y]
            oz = st[Z]
            st[X] = x0
            st[Y] = y0
            st[Z] = z0
            amazing_box_step(st, scale, min_r, fold, False)
            r1 = math.sqrt(st[X] * st[X] + st[Y] * st[Y] + st[Z] * st[Z])
            m2 = 1.0 - m
            r2 = r1 * m + m2 * math.sqrt(ox * ox + oy * oy + oz * oz)
            x1 = st[X] * m + m2 * ox
            y1 = st[Y] * m + m2 * oy
            z1 = st[Z] * m + m2 * oz
            r1 = r2 / math.sqrt(x1 * x1 + y1 * y1 + z1 * z1 + 1e-30)
            st[X] = x1 * r1
            st[Y] = y1 * r1
            st[Z] = z1 * r1
    else:
        amazing_box_step(st, scale, min_r, fold, False)


@kfunc
def rotate_xyz(st: RealArray[16], ax: Real, ay: Real, az: Real) -> None:
    """``BuildRotMatrix`` of MB3D's formula options applied to (x, y, z); radians."""
    if ax == 0.0 and ay == 0.0 and az == 0.0:
        return
    sx = math.sin(ax)
    cx = math.cos(ax)
    sy = math.sin(ay)
    cy = math.cos(ay)
    sz = math.sin(az)
    cz = math.cos(az)
    x = st[X]
    y = st[Y]
    z = st[Z]
    st[X] = cy * cz * x - cy * sz * y + sy * z
    st[Y] = (sx * sy * cz + cx * sz) * x + (cx * cz - sx * sy * sz) * y - sx * cy * z
    st[Z] = (sx * sz - cx * sy * cz) * x + (cx * sy * sz + sx * cz) * y + cx * cy * z


DEG = 0.017453292519943295


@kfunc
def f_amazing_surf(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'Amazing Surf' (Kali / Luca GN): box fold of x and y only, sphere fold, scale.

    Options: Scale, Min R, Fold, Rotation1 X/Y/Z (degrees), Scale vary,
    Sphere or Cylinder (1 = cylinder fold on x, y).
    """
    scale = fp[b]
    min_r = fp[b + 1]
    fold = fp[b + 2]
    vary = fp[b + 6]
    cyl = fp[b + 7]
    scale = scale + vary * (abs(scale) - 1.0)
    x = box_fold(st[X], fold)
    y = box_fold(st[Y], fold)
    z = st[Z]
    if cyl > 0.5:
        r = x * x + y * y
    else:
        r = x * x + y * y + z * z
    sqr_min_r = min_r * min_r
    if r < sqr_min_r:
        mul = scale / sqr_min_r
    elif r < 1.0:
        mul = scale / r
    else:
        mul = scale
    if ade:
        st[W] = st[W] * abs(mul)
    st[X] = x * mul
    st[Y] = y * mul
    st[Z] = z * mul
    rotate_xyz(st, fp[b + 3] * DEG, fp[b + 4] * DEG, fp[b + 5] * DEG)
    st[X] = st[X] + st[JX]
    st[Y] = st[Y] + st[JY]
    st[Z] = st[JZ] + st[Z]


@kfunc
def f_abox_mod1(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'ABoxMod1' (Luca GN): Mandelbox with separate fold modifiers per axis.

    Options: Scale, Min R, Fold, Scale vary, FoldXMod, FoldYMod, FoldZMod.
    """
    scale = fp[b]
    min_r = fp[b + 1]
    fold = fp[b + 2]
    vary = fp[b + 3]
    scale = scale + vary * (abs(scale) - 1.0)
    x = box_fold(st[X], fold + fp[b + 4])
    y = box_fold(st[Y], fold + fp[b + 5])
    z = box_fold(st[Z], fold + fp[b + 6])
    r = x * x + y * y + z * z
    sqr_min_r = min_r * min_r
    if r < sqr_min_r:
        mul = scale / sqr_min_r
    elif r < 1.0:
        mul = scale / r
    else:
        mul = scale
    if ade:
        st[W] = st[W] * abs(mul)
    st[X] = x * mul + st[JX]
    st[Y] = y * mul + st[JY]
    st[Z] = z * mul + st[JZ]
