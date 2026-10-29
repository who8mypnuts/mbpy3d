"""Box folding formulas: 'Amazing Box' (Mandelbox, `HybridCube`), 'Bulbox'
(`HybridSuperCube2`) and Kali's 'Amazing Surf'."""
from ..dsl import *  # noqa: F401,F403
from ..dsl import Param, formula, helper, vec3, real, TINY


@helper
def box_fold1(x: real, fold: real) -> real:
    """`x = abs(x + fold) - abs(x - fold) - x` (Tglad's fold)."""
    return fabs(x + fold) - (fabs(x - fold) + x)


@helper
def box_fold(z: vec3, fold: real) -> vec3:
    return vec3(box_fold1(z.x, fold), box_fold1(z.y, fold), box_fold1(z.z, fold))


@helper
def sphere_fold_mul(r2: real, scale: real, min_r: real) -> real:
    """The Mandelbox sphere fold multiplier (`HybridCube`)."""
    sqr_min_r = min_r * min_r
    m = scale
    if r2 < sqr_min_r:
        m = scale / sqr_min_r
    elif r2 < 1.0:
        m = scale / r2
    return m


@helper
def amazing_box_step(z: vec3, c: vec3, scale: real, min_r: real, fold: real) -> vec3:
    f = box_fold(z, fold)
    m = sphere_fold_mul(dot(f, f), scale, min_r)
    return f * m + c


@formula(
    name="Amazing Box",
    mb3d_id=4,
    params=[Param("scale", 2.0, "real", "scale", mb3d_name="Scale"),
            Param("min_r", 0.5, "real", "minimum radius of the sphere fold", mb3d_name="Min R"),
            Param("fold", 1.0, "real", "box fold", mb3d_name="Fold")],
    de="linear",
    de_scale=0.2,
    rstop=1024.0,
    category="3D boxes",
    description="Tom Lowe's Mandelbox (MB3D's 'Amazing Box').",
    aliases=("amazingbox", "mandelbox", "box", "abox"),
    scale_param="scale",
)
def amazing_box(s, p):
    f = box_fold(s.z, p.fold)
    m = sphere_fold_mul(dot(f, f), p.scale, p.min_r)
    s.dr = s.dr * fabs(m)
    s.z = f * m + s.c


@helper
def sine_bulb_pow2_scaled(z: vec3, c: vec3, scaling: real) -> vec3:
    """`HybridItIntPow2scale`: sine bulb power 2 with scaling (Bulbox)."""
    sc = 1.0 / scaling
    x = z.x * sc
    y = z.y * sc
    zz = z.z * sc
    xx = x * x
    yy = y * y
    r = xx + yy
    nz = -((sqrt(r) * zz) * 2.0) * scaling + c.z
    a = (r - zz * zz) / (r + TINY)
    nx = (xx - yy) * a * scaling + c.x
    ny = (x * y * a) * 2.0 * scaling + c.y
    return vec3(nx, ny, nz)


@formula(
    name="Bulbox",
    mb3d_id=5,
    params=[Param("scale", 2.0, "real", "box scale", mb3d_name="Box Scale"),
            Param("min_r", 0.5, "real", "box min R", mb3d_name="Box Min R"),
            Param("fold", 1.0, "real", "box fold", mb3d_name="Box fold"),
            Param("bulb_scale", 1.0, "real", "bulb scaling", mb3d_name="Bulb scaling"),
            Param("r_threshold", 2.0, "real", "box/bulb R threshold", mb3d_name="Box/Bulb R threshold"),
            Param("r_threshold2", 2.0, "real", "box/bulb R threshold 2", mb3d_name="Box/Bulb R threshold 2")],
    de="none",
    de_scale=0.2,
    rstop=1024.0,
    category="3D boxes",
    description="Sine bulb power 2 inside a radius, Amazing Box outside, blended in between (`HybridSuperCube2`).",
    scale_param="scale",
)
def bulbox(s, p):
    rt1 = p.r_threshold * p.r_threshold
    rt2 = p.r_threshold2 * p.r_threshold2
    rout = s.r2
    if rout < rt1:
        if rout < rt2:
            s.z = sine_bulb_pow2_scaled(s.z, s.c, p.bulb_scale)
        else:
            m = (rout - rt2) / (rt1 - rt2)
            out = sine_bulb_pow2_scaled(s.z, s.c, p.bulb_scale)
            box = amazing_box_step(s.z, s.c, p.scale, p.min_r, p.fold)
            m2 = 1.0 - m
            r2n = length(box) * m + m2 * length(out)
            v = box * m + out * m2
            r1 = r2n / sqrt(dot(v, v) + TINY)
            s.z = v * r1
    else:
        s.z = amazing_box_step(s.z, s.c, p.scale, p.min_r, p.fold)


@helper
def rotate_xyz_sc(v: vec3, sx: real, cx: real, sy: real, cy: real, sz: real, cz: real) -> vec3:
    """Rotate about X, then Y, then Z, given the sines and cosines of the
    angles (computed once on the host, see :func:`rotation_params`)."""
    y1 = cx * v.y - sx * v.z
    z1 = sx * v.y + cx * v.z
    x2 = cy * v.x + sy * z1
    z2 = -sy * v.x + cy * z1
    return vec3(cz * x2 - sz * y1, sz * x2 + cz * y1, z2)


def rotation_params(prefix: str, label: str = "rotation") -> list:
    """Three editable angle options (degrees) ``<prefix>_x/y/z`` plus the
    derived sines and cosines ``<prefix>_sx, _cx, ...`` the formula uses."""
    import math as _m

    out = [Param(f"{prefix}_{a}", 0.0, "real", f"{label} {a.upper()} (degrees)") for a in "xyz"]
    for a in "xyz":
        out.append(Param(f"{prefix}_s{a}", 0.0, derived=lambda v, k=f"{prefix}_{a}": _m.sin(_m.radians(v[k]))))
        out.append(Param(f"{prefix}_c{a}", 1.0, derived=lambda v, k=f"{prefix}_{a}": _m.cos(_m.radians(v[k]))))
    return out


@formula(
    name="Amazing Surf",
    params=[Param("scale", 1.5, "real", "scale", mb3d_name="Scale"),
            Param("min_r", 0.5, "real", "minimum radius", mb3d_name="Boxscale Min R"),
            Param("fold", 1.0, "real", "fold", mb3d_name="Fold"),
            *rotation_params("rot"),
            Param("scale_vary", 0.0, "real", "scale vary", mb3d_name="Scale vary"),
            Param("cylinder", 1, "bool", "cylinder (1) or sphere (0) inversion", mb3d_name="Sphere or Cylinder")],
    de="linear",
    de_scale=0.2,
    rstop=20.0,
    category="3D boxes",
    description="Kali's Amazing Surf: Tglad fold of x and y only, cylinder or sphere inversion, rotation.",
    aliases=("amazingsurf", "asurf"),
    scale_param="scale",
)
def amazing_surf(s, p):
    scale = p.scale + p.scale_vary * (fabs(p.scale) - 1.0)
    x = box_fold1(s.z.x, p.fold)
    y = box_fold1(s.z.y, p.fold)
    z = s.z.z
    rr = x * x + y * y
    if p.cylinder == 0.0:
        rr = rr + z * z
    sqr_min_r = p.min_r * p.min_r
    m = scale
    if rr < sqr_min_r:
        m = scale / sqr_min_r
    elif rr < 1.0:
        m = scale / rr
    s.dr = s.dr * fabs(m)
    v = vec3(x, y, z) * m + s.c
    s.z = rotate_xyz_sc(v, p.rot_sx, p.rot_cx, p.rot_sy, p.rot_cy, p.rot_sz, p.rot_cz)


@formula(
    name="ABox Mod",
    params=[Param("scale", 2.0, "real", "scale"),
            Param("fold", 1.0, "real", "box fold"),
            Param("min_r", 0.5, "real", "minimum radius"),
            Param("fold_z", 1.0, "real", "fold for z (0 = no z fold)"),
            Param("c_mul", 1.0, "real", "multiplier of the added constant"),
            Param("z_mul", 1.0, "real", "multiplier of z before the fold")],
    de="linear",
    de_scale=0.2,
    rstop=1024.0,
    category="3D boxes",
    description="Amazing Box variant with a separate z fold and multipliers (in the spirit of ABoxMod1).",
    scale_param="scale",
)
def abox_mod(s, p):
    x = box_fold1(s.z.x, p.fold)
    y = box_fold1(s.z.y, p.fold)
    z = s.z.z * p.z_mul
    if p.fold_z != 0.0:
        z = box_fold1(z, p.fold_z)
    f = vec3(x, y, z)
    m = sphere_fold_mul(dot(f, f), p.scale, p.min_r)
    s.dr = s.dr * fabs(m)
    s.z = f * m + s.c * p.c_mul
