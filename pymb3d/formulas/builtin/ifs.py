"""Kaleidoscopic IFS formulas: Menger sponge ('Menger3'), Sierpinski
tetrahedron ('Sierpinski3') and the octahedral KIFS."""
from ..dsl import *  # noqa: F401,F403
from ..dsl import Param, formula, helper, vec3, real, TINY
from .box import rotate_xyz_sc, rotation_params


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
    description="Menger sponge as a kaleidoscopic IFS with two rotations (after MB3D's Menger3).",
    aliases=("menger", "menger sponge"),
    scale_param="scale",
)
def menger3(s, p):
    z = fabs(s.z)
    z = rotate_xyz_sc(z, p.rot1_sx, p.rot1_cx, p.rot1_sy, p.rot1_cy, p.rot1_sz, p.rot1_cz)
    z = sort_desc3(fabs(z))
    z = rotate_xyz_sc(z, p.rot2_sx, p.rot2_cx, p.rot2_sy, p.rot2_cy, p.rot2_sz, p.rot2_cz)
    sc = p.scale
    off = sc - 1.0
    x = z.x * sc - p.cscale_x * off
    y = z.y * sc - p.cscale_y * off
    zz = z.z * sc
    if zz > 0.5 * p.cscale_z * off:
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
    z = rotate_xyz_sc(z, p.rot_sx, p.rot_cx, p.rot_sy, p.rot_cy, p.rot_sz, p.rot_cz)
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
    z = rotate_xyz_sc(z, p.rot_sx, p.rot_cx, p.rot_sy, p.rot_cy, p.rot_sz, p.rot_cz)
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
