"""Kaleidoscopic IFS formulas: Menger sponge ('Menger3'), Sierpinski
tetrahedron ('Sierpinski3') and the octahedral KIFS."""
from ..dsl import *  # noqa: F401,F403
from ..dsl import Param, formula, helper, vec3, real, TINY
from .box import rot_apply, rotation4d_params, rotation_params


@helper
def sort_desc3(v: vec3) -> vec3:
    """Components sorted so that x >= y >= z (the Menger fold)."""
    x = v.x
    y = v.y
    z = v.z
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
    return vec3(x, y, z)


@formula(
    name="Menger3",
    params=[Param("scale", 3.0, "real", "scale", mb3d_name="Scale"),
            Param("cscale_x", 1.0, "real", "C scale X", mb3d_name="CScale X"),
            Param("cscale_y", 1.0, "real", "C scale Y", mb3d_name="CScale Y"),
            Param("cscale_z", 1.0, "real", "C scale Z", mb3d_name="CScale Z"),
            *rotation_params("rot1", "rotation 1"),
            *rotation_params("rot2", "rotation 2")],
    de="linear",
    de_scale=0.2,
    rstop=1024.0,
    category="IFS",
    description="Menger sponge as a kaleidoscopic IFS with two rotations (MB3D's Menger3).",
    aliases=("menger", "menger sponge"),
    scale_param="scale",
)
def menger3(s, p):
    z = fabs(s.z)
    z = rot_apply(z, p.rot1_m0, p.rot1_m1, p.rot1_m2, p.rot1_m3, p.rot1_m4, p.rot1_m5, p.rot1_m6, p.rot1_m7, p.rot1_m8)
    z = sort_desc3(z)
    z = rot_apply(z, p.rot2_m0, p.rot2_m1, p.rot2_m2, p.rot2_m3, p.rot2_m4, p.rot2_m5, p.rot2_m6, p.rot2_m7, p.rot2_m8)
    sc = p.scale
    off = sc - 1.0
    x = z.x * sc - p.cscale_x * off
    y = z.y * sc - p.cscale_y * off
    zz = z.z * sc
    if zz >= 0.5 * p.cscale_z * off:
        zz = zz - p.cscale_z * off
    s.dr = s.dr * fabs(sc)
    s.z = vec3(x, y, zz)


@formula(
    name="Sierpinski3",
    params=[Param("scale", 2.0, "real", "scale", mb3d_name="Scale"),
            Param("offset", 1.0, "real", "offset of the fixed points"),
            *rotation_params("rot")],
    de="linear",
    de_scale=0.2,
    rstop=1024.0,
    category="IFS",
    description="Sierpinski tetrahedron (kaleidoscopic IFS).",
    aliases=("sierpinski", "sierpinski tetrahedron"),
    scale_param="scale",
)
def sierpinski3(s, p):
    z = s.z
    if z.x + z.y < 0.0:
        z = vec3(-z.y, -z.x, z.z)
    if z.x + z.z < 0.0:
        z = vec3(-z.z, z.y, -z.x)
    if z.y + z.z < 0.0:
        z = vec3(z.x, -z.z, -z.y)
    z = rot_apply(z, p.rot_m0, p.rot_m1, p.rot_m2, p.rot_m3, p.rot_m4, p.rot_m5, p.rot_m6, p.rot_m7, p.rot_m8)
    s.dr = s.dr * fabs(p.scale)
    s.z = z * p.scale - vec3(p.offset, p.offset, p.offset) * (p.scale - 1.0)


@formula(
    name="Octahedral KIFS",
    params=[Param("scale", 2.0, "real", "scale"),
            Param("offset", 1.0, "real", "offset"),
            *rotation_params("rot")],
    de="linear",
    de_scale=0.2,
    rstop=1024.0,
    category="IFS",
    description="Octahedral kaleidoscopic IFS (abs folds, diagonal swaps).",
    aliases=("kifs", "octkifs"),
    scale_param="scale",
)
def octahedral_kifs(s, p):
    z = fabs(s.z)
    if z.x < z.y:
        z = vec3(z.y, z.x, z.z)
    if z.x < z.z:
        z = vec3(z.z, z.y, z.x)
    if z.y < z.z:
        z = vec3(z.x, z.z, z.y)
    z = rot_apply(z, p.rot_m0, p.rot_m1, p.rot_m2, p.rot_m3, p.rot_m4, p.rot_m5, p.rot_m6, p.rot_m7, p.rot_m8)
    s.dr = s.dr * fabs(p.scale)
    s.z = z * p.scale - vec3(p.offset, p.offset, 0.0) * (p.scale - 1.0)


@formula(
    name="Sphere Inversion",
    params=[Param("radius", 1.0, "real", "inversion radius"),
            Param("center_x", 0.0, "real", "center x"),
            Param("center_y", 0.0, "real", "center y"),
            Param("center_z", 0.0, "real", "center z")],
    de="none",
    rstop=1024.0,
    category="transforms",
    description="Inversion in a sphere (a transform to combine with other formulas in a hybrid).",
    counted=False,
)
def sphere_inversion(s, p):
    d = s.z - vec3(p.center_x, p.center_y, p.center_z)
    r2 = dot(d, d) + TINY
    s.z = d * (p.radius * p.radius / r2) + vec3(p.center_x, p.center_y, p.center_z)


@formula(
    name="MengerIFS",
    params=[Param("scale", 3.0, "real", "scale", mb3d_name="Scale"),
            Param("cscale_x", 1.0, "real", "C scale X", mb3d_name="CScale X"),
            Param("cscale_y", 1.0, "real", "C scale Y", mb3d_name="CScale Y"),
            Param("cscale_z", 0.5, "real", "C scale Z", mb3d_name="CScale Z"),
            *rotation_params("rot")],
    de="linear",
    de_scale=0.2,
    rstop=1024.0,
    category="IFS",
    description="MB3D's MengerIFS: a Menger version that harmonises better in hybrids than Menger3.",
    aliases=("mengerifs",),
    scale_param="scale",
)
def menger_ifs(s, p):
    z = sort_desc3(fabs(s.z))
    z = rot_apply(z, p.rot_m0, p.rot_m1, p.rot_m2, p.rot_m3, p.rot_m4, p.rot_m5, p.rot_m6, p.rot_m7, p.rot_m8)
    sc = p.scale
    off = sc - 1.0
    s.dr = s.dr * fabs(sc)
    x = z.x * sc - off * p.cscale_x
    y = z.y * sc - off * p.cscale_y
    oz = off * p.cscale_z
    zz = oz - fabs(z.z * sc - oz)
    s.z = vec3(x, y, zz)


@formula(
    name="_Rotate",
    params=[*rotation_params("rot", "rotation 1")],
    de="none",
    rstop=1024.0,
    category="transforms",
    description="MB3D's _Rotate transform: rotates the point (not counted as an iteration).",
    aliases=("rotate",),
    counted=False,
)
def rotate_transform(s, p):
    s.z = rot_apply(s.z, p.rot_m0, p.rot_m1, p.rot_m2, p.rot_m3, p.rot_m4, p.rot_m5, p.rot_m6, p.rot_m7, p.rot_m8)


@helper
def sort_desc4(x: real, y: real, z: real, w: real) -> vec3:
    """The three largest of four values in descending order (the 4D Menger
    fold); the smallest is recovered by the caller from the sum."""
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
    if x < w:
        t = x
        x = w
        w = t
    if y < w:
        t = y
        y = w
        w = t
    if z < w:
        t = z
        z = w
        w = t
    return vec3(x, y, z)


@formula(
    name="Menger4",
    params=[Param("scale", 3.0, "real", "scale", mb3d_name="Scale"),
            Param("cscale_x", 1.0, "real", "C scale X", mb3d_name="CScale X"),
            Param("cscale_y", 1.0, "real", "C scale Y", mb3d_name="CScale Y"),
            Param("cscale_z", 0.5, "real", "C scale Z", mb3d_name="CScale Z"),
            Param("cscale_w", 0.5, "real", "C scale W", mb3d_name="CScale W"),
            *rotation4d_params("rot")],
    de="linear",
    de_scale=0.2,
    ade_scale=0.5,
    rstop=1024.0,
    four_d=True,
    category="4D",
    description="MB3D's Menger4: the 4D Menger KIFS with a built-in 4D rotation (CScale W = 0 gives Menger3).",
    aliases=("menger4",),
    scale_param="scale",
)
def menger4(s, p):
    ax = fabs(s.z.x)
    ay = fabs(s.z.y)
    az = fabs(s.z.z)
    aw = fabs(s.w)
    top = sort_desc4(ax, ay, az, aw)
    wm = ax + ay + az + aw - top.x - top.y - top.z
    x = top.x
    y = top.y
    z = top.z
    nx = x * p.rot_m0 + y * p.rot_m1 + z * p.rot_m2 + wm * p.rot_m3
    ny = x * p.rot_m4 + y * p.rot_m5 + z * p.rot_m6 + wm * p.rot_m7
    nz = x * p.rot_m8 + y * p.rot_m9 + z * p.rot_m10 + wm * p.rot_m11
    nw = x * p.rot_m12 + y * p.rot_m13 + z * p.rot_m14 + wm * p.rot_m15
    sc = p.scale
    off = sc - 1.0
    s.dr = s.dr * fabs(sc)
    s.w = nw * sc - off * p.cscale_w
    # as in the machine code: CScale X applies to the second, CScale Y to the first component
    rx = nx * sc - off * p.cscale_y
    ry = ny * sc - off * p.cscale_x
    oz = off * p.cscale_z
    rz = oz - fabs(nz * sc - oz)
    s.z = vec3(rx, ry, rz)


@formula(
    name="SierpHilbert",
    params=[Param("scale", 3.0, "real", "scale", mb3d_name="Scale"),
            Param("cscale_x", 1.0, "real", "C scale X", mb3d_name="CScale X"),
            Param("cscale_y", 1.0, "real", "C scale Y", mb3d_name="CScale Y"),
            Param("cscale_z", 1.0, "real", "C scale Z", mb3d_name="CScale Z"),
            *rotation_params("rot1", "rotation 1"),
            *rotation_params("rot2", "rotation 2"),
            Param("edge1", 2.0, "real", "fold edge of y (0 = off)", mb3d_name="Edge1"),
            Param("edge2", 1.0, "real", "fold edge of x (0 = off)", mb3d_name="Edge2")],
    de="linear",
    de_scale=0.2,
    rstop=1024.0,
    category="IFS",
    description="MB3D's SierpHilbert: Sierpinski folds with edge folds and two rotations, a hybrid of the Sierpinski and Hilbert axioms.",
    aliases=("sierphilbert",),
    scale_param="scale",
)
def sierp_hilbert(s, p):
    v = rot_apply(s.z, p.rot1_m0, p.rot1_m1, p.rot1_m2, p.rot1_m3, p.rot1_m4, p.rot1_m5, p.rot1_m6, p.rot1_m7, p.rot1_m8)
    x = v.x
    y = v.y
    z = v.z
    if x + y < 0.0:
        t = x
        x = -y
        y = -t
    if x + z < 0.0:
        t = x
        x = -z
        z = -t
    if y + z < 0.0:
        t = y
        y = -z
        z = -t
    if p.edge1 > 0.0:
        y = p.edge1 - fabs(p.edge1 - y)
    if p.edge2 > 0.0:
        x = p.edge2 - fabs(p.edge2 - x)
    v = rot_apply(vec3(x, y, z), p.rot2_m0, p.rot2_m1, p.rot2_m2, p.rot2_m3, p.rot2_m4, p.rot2_m5, p.rot2_m6,
                  p.rot2_m7, p.rot2_m8)
    sc = p.scale
    s.dr = s.dr * fabs(sc)
    s.z = v * sc - vec3(p.cscale_x, p.cscale_y, p.cscale_z) * (sc - 1.0)
