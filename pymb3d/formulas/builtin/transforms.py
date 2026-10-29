"""Transforms of MB3D (DE option -1: not counted as iterations) and the
genIFS formula, ported from the machine code of their `.m3f` files."""
from ..dsl import *  # noqa: F401,F403
from ..dsl import Param, formula, vec3, real, TINY, TWO_PI
from .box import rot_apply, rotation_params

DEG2RAD = 0.017453293


@formula(
    name="_PolyFold-sym",
    params=[Param("order", 4.0, "real", "polygon order", mb3d_name="Order"),
            Param("shift_deg", 0.0, "real", "shift angle (degrees)", mb3d_name="Shift (deg)"),
            Param("shift_x", 0.0, "real", "shift x", mb3d_name="Shift x"),
            Param("shift_y", 0.0, "real", "shift y", mb3d_name="Shift y"),
            Param("skew_deg", 0.0, "real", "skew (degrees)", mb3d_name="Skew (deg)")],
    de="none",
    rstop=1024.0,
    category="transforms",
    description="MB3D's _PolyFold-sym: symmetric polygonal fold of the xy plane.",
    aliases=("polyfold-sym", "polyfold sym"),
    counted=False,
)
def polyfold_sym(s, p):
    y = s.z.y + p.shift_y
    x = s.z.x + p.shift_x
    a = atan2(y, x) + (p.skew_deg + p.shift_deg) * DEG2RAD
    a = a / TWO_PI * p.order
    n = int(floor(a + 0.5))
    a2 = -float(n) / p.order * TWO_PI + p.shift_deg * DEG2RAD
    sn = sin(a2)
    cs = cos(a2)
    nx = y * sn - x * cs
    ny = x * sn + y * cs
    if (n & 1) != 0:
        ny = -ny
    s.z = vec3(nx - p.shift_x, ny - p.shift_y, s.z.z)


@formula(
    name="_SphereFolding1",
    params=[Param("r_fold", 2.0, "real", "fold radius squared", mb3d_name="R fold")],
    de="none",
    rstop=1024.0,
    category="transforms",
    description="MB3D's _SphereFolding1: points beyond the fold radius are reflected inward.",
    aliases=("spherefolding1",),
    counted=False,
)
def sphere_folding1(s, p):
    r2 = s.r2
    v = p.r_fold
    if r2 >= v:
        s.z = s.z * ((2.0 * v - r2) / r2)
    elif r2 < -v:
        s.z = s.z * ((-2.0 * v - r2) / r2)


@formula(
    name="genIFS",
    params=[Param("scale", -1.6, "real", "scale", mb3d_name="Scale"),
            Param("abs_x", 1, "bool", "fold x (abs)", mb3d_name="Abs X"),
            Param("abs_y", 1, "bool", "fold y (abs)", mb3d_name="Abs Y"),
            Param("abs_z", 0, "bool", "fold z (abs)", mb3d_name="Abs Z"),
            Param("offset_x", -1.0, "real", "offset x", mb3d_name="Offset X"),
            Param("offset_y", -1.0, "real", "offset y", mb3d_name="Offset Y"),
            Param("offset_z", 0.0, "real", "offset z", mb3d_name="Offset Z"),
            *rotation_params("rot")],
    de="linear",
    de_scale=0.2,
    ade_scale=0.5,
    rstop=16.0,
    category="IFS",
    description="MB3D's genIFS: folds, offsets, rotation, scale and the constant (a general IFS step).",
    aliases=("genifs",),
    scale_param="scale",
)
def gen_ifs(s, p):
    x = s.z.x
    y = s.z.y
    z = s.z.z
    if p.abs_x != 0.0:
        x = fabs(x)
    if p.abs_y != 0.0:
        y = fabs(y)
    if p.abs_z != 0.0:
        z = fabs(z)
    v = vec3(x + p.offset_x, y + p.offset_y, z + p.offset_z)
    v = rot_apply(v, p.rot_m0, p.rot_m1, p.rot_m2, p.rot_m3, p.rot_m4, p.rot_m5, p.rot_m6, p.rot_m7, p.rot_m8)
    s.dr = s.dr * fabs(p.scale)
    s.z = v * p.scale + s.c


@formula(
    name="_RotateC",
    params=[*rotation_params("rot", "rotation C")],
    de="none",
    rstop=1024.0,
    category="transforms",
    description="MB3D's _RotateC: rotates the constant C added by the formulas.",
    aliases=("rotatec",),
    counted=False,
)
def rotate_c(s, p):
    s.c = rot_apply(s.c, p.rot_m0, p.rot_m1, p.rot_m2, p.rot_m3, p.rot_m4, p.rot_m5, p.rot_m6, p.rot_m7, p.rot_m8)


@formula(
    name="_Translate",
    params=[Param("x_add", 0.0, "real", "X add", mb3d_name="X add"),
            Param("y_add", 0.0, "real", "Y add", mb3d_name="Y add"),
            Param("z_add", 0.0, "real", "Z add", mb3d_name="Z add")],
    de="none",
    rstop=1024.0,
    category="transforms",
    description="MB3D's _Translate: moves the point.",
    aliases=("translate",),
    counted=False,
)
def translate(s, p):
    s.z = s.z + vec3(p.x_add, p.y_add, p.z_add)
