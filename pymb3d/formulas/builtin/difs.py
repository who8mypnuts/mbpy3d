"""dIFS formulas (MB3D's distance estimated IFS, DE options 20 / 21).

A dIFS hybrid iterates *transforms* (which move the point and accumulate
the scale in ``s.dr``) and *shapes* (which write the distance of the point
to a primitive into ``s.de`` and a colour value into ``s.trap``).  The
loop (``difs_orbit``) keeps the smallest ``de / dr`` over the iterations.

The shapes follow the machine code of `SphereIFS.m3f`, `boxIFS.m3f`,
`cylinderIFS.m3f` and `transformIFS.m3f` (their option order is kept, so
`.m3p` files load).  Orbit trap options: 0 the DE stop, 1..3 the x, y, z
coordinate, each times "OTrap scale" plus "OTrap offset".
"""
from ..dsl import *  # noqa: F401,F403
from ..dsl import Param, formula, helper, vec3, real, TINY
from .box import rot_apply, rotation_params


@helper
def difs_trap(opt: int, p: vec3, aux: real, scale: real, offset: real) -> real:
    """The orbit trap value of a dIFS shape."""
    v = aux
    if opt == 1:
        v = p.x
    elif opt == 2:
        v = p.y
    elif opt == 3:
        v = p.z
    return v * scale + offset


def _common_params(x_add: float = 3.0) -> list:
    return [Param("scale", 2.0, "real", "scale applied before the shape", mb3d_name="Scale"),
            Param("z_add", 0.0, "real", "Z add", mb3d_name="Z add"),
            Param("y_add", 0.0, "real", "Y add", mb3d_name="Y add"),
            Param("x_add", x_add, "real", "X add", mb3d_name="X add")]


def _trap_params() -> list:
    return [Param("apply", 1, "bool", "apply scale and add to the point (IFS) or only for this shape",
                  mb3d_name="Apply scale+add"),
            Param("otrap_option", 0, "int", "orbit trap: 0 DE stop, 1 x, 2 y, 3 z", 0, 3, "OTrap option (0..3)"),
            Param("otrap_offset", 1.0, "real", "orbit trap offset", mb3d_name="OTrap offset"),
            Param("otrap_scale", 1.0, "real", "orbit trap scale", mb3d_name="OTrap scale")]


@formula(
    name="SphereIFS",
    params=[Param("radius", 1.0, "real", "radius", mb3d_name="Radius")] + _common_params(3.0) + [
        Param("z_size", 1.0, "real", "Z size", mb3d_name="Z size"),
        Param("y_size", 1.0, "real", "Y size", mb3d_name="Y size"),
        Param("x_size", 1.0, "real", "X size", mb3d_name="X size"),
    ] + _trap_params() + [Param("inside_radius", -1.0, "real", "inside radius (hollow sphere)", mb3d_name="Inside radius")],
    de="difs_shape",
    rstop=1024.0,
    category="dIFS",
    description="Sphere (ellipsoid) dIFS shape, with an inner radius for hollow spheres.",
    aliases=("sphereifs",),
)
def sphere_ifs(s, p):
    q = s.z * p.scale + vec3(p.x_add, p.y_add, p.z_add)
    if p.apply != 0.0:
        s.z = q
        s.dr = s.dr * p.scale
    rx = 1.0 / (p.x_size if fabs(p.x_size) > TINY else TINY)
    ry = 1.0 / (p.y_size if fabs(p.y_size) > TINY else TINY)
    rz = 1.0 / (p.z_size if fabs(p.z_size) > TINY else TINY)
    r = length(vec3(q.x * rx, q.y * ry, q.z * rz))
    de = fmax(r - p.radius, p.inside_radius - r) / fmax(fmax(fabs(rx), fabs(ry)), fabs(rz))
    if p.apply == 0.0:
        de = de / p.scale
    s.de = de
    s.trap = difs_trap(int(p.otrap_option), s.z, s.aux, p.otrap_scale, p.otrap_offset)


@formula(
    name="boxIFS",
    params=[Param("z_halfwidth", 1.0, "real", "Z halfwidth", mb3d_name="Z halfwidth"),
            Param("y_halfwidth", 1.0, "real", "Y halfwidth", mb3d_name="Y halfwidth"),
            Param("x_halfwidth", 1.0, "real", "X halfwidth", mb3d_name="X halfwidth")] + _common_params(2.0) + [
        Param("trapez", 0.0, "real", "x halfwidth grows with z", mb3d_name="Trapez"),
        Param("twist", 0.0, "real", "twist angle per z", mb3d_name="Twist"),
    ] + _trap_params(),
    de="difs_shape",
    rstop=1024.0,
    category="dIFS",
    description="Box dIFS shape with trapez and twist options.",
    aliases=("boxifs",),
)
def box_ifs(s, p):
    q = s.z * p.scale + vec3(p.x_add, p.y_add, p.z_add)
    if p.apply != 0.0:
        s.z = q
        s.dr = s.dr * p.scale
    if p.twist != 0.0:
        a = p.twist * q.z
        sa = sin(a)
        ca = cos(a)
        q = vec3(q.x * sa + q.y * ca, q.x * ca - q.y * sa, q.z)
    dx = fabs(q.x) - p.x_halfwidth - q.z * p.trapez
    dy = fabs(q.y) - p.y_halfwidth
    dz = fabs(q.z) - p.z_halfwidth
    de = fmax(fmax(dx, dy), dz)
    if p.apply == 0.0:
        de = de / p.scale
    s.de = de
    s.trap = difs_trap(int(p.otrap_option), s.z, s.aux, p.otrap_scale, p.otrap_offset)


@formula(
    name="cylinderIFS",
    params=[Param("radius", 1.0, "real", "radius", mb3d_name="Radius"),
            Param("x_halfwidth", 1.0, "real", "half length along x", mb3d_name="X halfwidth")] + _common_params(3.0) + [
        Param("barrel", 0.0, "real", "barrel: radius grows with x^2", mb3d_name="Barrel"),
        Param("tip", 0.0, "real", "tip: radius changes with x", mb3d_name="Tip"),
    ] + _trap_params() + [Param("inside_radius", -1.0, "real", "inside radius (tube)", mb3d_name="Inside radius")],
    de="difs_shape",
    rstop=1024.0,
    category="dIFS",
    description="Cylinder dIFS shape along the x axis with barrel, tip and an inner radius.",
    aliases=("cylinderifs",),
)
def cylinder_ifs(s, p):
    q = s.z * p.scale + vec3(p.x_add, p.y_add, p.z_add)
    if p.apply != 0.0:
        s.z = q
        s.dr = s.dr * p.scale
    r = sqrt(q.y * q.y + q.z * q.z)
    m = fmin(fabs(q.x), p.x_halfwidth)
    bump = p.barrel * m * m + p.tip * q.x
    de = fmax(fmax(r - p.radius + bump, p.inside_radius - r - bump), fabs(q.x) - p.x_halfwidth)
    if p.apply == 0.0:
        de = de / p.scale
    s.de = de
    s.trap = difs_trap(int(p.otrap_option), s.z, s.aux, p.otrap_scale, p.otrap_offset)


@formula(
    name="transformIFS",
    params=[Param("add_x", 0.0, "real", "Add X", mb3d_name="Add X"),
            Param("add_y", 0.0, "real", "Add Y", mb3d_name="Add Y"),
            Param("add_z", 0.0, "real", "Add Z", mb3d_name="Add Z"),
            *rotation_params("rot"),
            Param("abs_x", 0, "bool", "fold x (abs)", mb3d_name="Abs X"),
            Param("abs_y", 1, "bool", "fold y (abs)", mb3d_name="Abs Y"),
            Param("abs_z", 0, "bool", "fold z (abs)", mb3d_name="Abs Z"),
            Param("scale", 1.0, "real", "scale", mb3d_name="Scale")],
    de="difs_transform",
    rstop=1024.0,
    category="dIFS",
    description="dIFS transform: scale, translate, rotate and fold the point.",
    aliases=("transformifs",),
    counted=False,
)
def transform_ifs(s, p):
    q = s.z * p.scale + vec3(p.add_x, p.add_y, p.add_z)
    s.dr = s.dr * p.scale
    q = rot_apply(q, p.rot_m0, p.rot_m1, p.rot_m2, p.rot_m3, p.rot_m4, p.rot_m5, p.rot_m6, p.rot_m7, p.rot_m8)
    x = q.x
    y = q.y
    z = q.z
    if p.abs_x != 0.0:
        x = fabs(x)
    if p.abs_y != 0.0:
        y = fabs(y)
    if p.abs_z != 0.0:
        z = fabs(z)
    s.z = vec3(x, y, z)
