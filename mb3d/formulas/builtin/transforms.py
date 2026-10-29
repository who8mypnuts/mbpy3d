"""Transformations (``DEoption = -1``): _Rotate, _Translate, _Scaling, _AbsY."""

from mb3d.kernel.dsl import Bool, ConstRealArray, Int, RealArray, kfunc
from mb3d.kernel.source.layout import W, X, Y, Z

from .boxes import DEG, rotate_xyz


@kfunc
def f_rotate(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'_Rotate': options (Rotation1 X, Y, Z in degrees)."""
    rotate_xyz(st, fp[b] * DEG, fp[b + 1] * DEG, fp[b + 2] * DEG)


@kfunc
def f_translate(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'_Translate': options (X add, Y add, Z add)."""
    st[X] = st[X] + fp[b]
    st[Y] = st[Y] + fp[b + 1]
    st[Z] = st[Z] + fp[b + 2]


@kfunc
def f_scaling(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'_Scaling': options (Scale)."""
    s = fp[b]
    st[X] = st[X] * s
    st[Y] = st[Y] * s
    st[Z] = st[Z] * s
    if ade:
        st[W] = st[W] * s


@kfunc
def f_abs_y(st: RealArray[16], fp: ConstRealArray[96], b: Int, ade: Bool) -> None:
    """'_AbsY': y := |y|."""
    st[Y] = abs(st[Y])
