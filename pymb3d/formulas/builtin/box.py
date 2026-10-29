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
def rot_apply(v: vec3, m0: real, m1: real, m2: real, m3: real, m4: real, m5: real, m6: real, m7: real,
              m8: real) -> vec3:
    """v' = M v with the rotation matrix rows (m0 m1 m2), (m3 m4 m5), (m6 m7 m8)."""
    return vec3(m0 * v.x + m1 * v.y + m2 * v.z, m3 * v.x + m4 * v.y + m5 * v.z, m6 * v.x + m7 * v.y + m8 * v.z)


def build_rot_matrix(xa: float, ya: float, za: float) -> list[float]:
    """MB3D's `BuildRotMatrix` of a `3SingleAngles` option (radians), row major."""
    import math as _m

    sx, cx = _m.sin(xa), _m.cos(xa)
    sy, cy = _m.sin(ya), _m.cos(ya)
    sz, cz = _m.sin(za), _m.cos(za)
    return [cy * cz, -cy * sz, sy,
            sx * sy * cz + cx * sz, cx * cz - sx * sy * sz, -sx * cy,
            sx * sz - cx * sy * cz, cx * sy * sz + sx * cz, cx * cy]


def rotation_params(prefix: str, label: str = "rotation") -> list:
    """Three editable angle options (degrees) ``<prefix>_x/y/z`` plus the
    nine derived matrix entries ``<prefix>_m0..m8`` (MB3D's 3SingleAngles)."""
    import math as _m

    out = [Param(f"{prefix}_{a}", 0.0, "real", f"{label} {a.upper()} (degrees)") for a in "xyz"]
    for k in range(9):
        out.append(Param(f"{prefix}_m{k}", 1.0 if k in (0, 4, 8) else 0.0,
                         derived=lambda v, k=k, p=prefix: build_rot_matrix(
                             _m.radians(v[f"{p}_x"]), _m.radians(v[f"{p}_y"]), _m.radians(v[f"{p}_z"]))[k]))
    return out


def ROT(prefix: str) -> str:
    """The argument list of :func:`rot_apply` for a rotation ``prefix`` (source text helper)."""
    return ", ".join(f"p.{prefix}_m{k}" for k in range(9))


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
    s.z = rot_apply(v, p.rot_m0, p.rot_m1, p.rot_m2, p.rot_m3, p.rot_m4, p.rot_m5, p.rot_m6, p.rot_m7, p.rot_m8)


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


@formula(
    name="ABoxModKali",
    params=[Param("scale", 1.5, "real", "scale", mb3d_name="Scale"),
            Param("min_r", 0.6, "real", "minimum radius", mb3d_name="Boxscale Min R"),
            Param("z_add", 0.5, "real", "Z add", mb3d_name="Z add"),
            Param("y_add", 0.5, "real", "Y add", mb3d_name="Y add"),
            Param("x_add", 0.5, "real", "X add", mb3d_name="X add")],
    de="linear",
    de_scale=0.15,
    ade_scale=0.5,
    rstop=1024.0,
    category="3D boxes",
    description="Kali's modified Amazing Box (MB3D's ABoxModKali): x = Xadd - |x| then the sphere fold.",
    aliases=("aboxmodkali",),
    scale_param="scale",
)
def abox_mod_kali(s, p):
    f = vec3(p.x_add - fabs(s.z.x), p.y_add - fabs(s.z.y), p.z_add - fabs(s.z.z))
    m = sphere_fold_mul(dot(f, f), p.scale, p.min_r)
    s.dr = s.dr * fabs(m)
    s.z = f * m + s.c


@helper
def fold_mod(x: real, fold: real, fmod: real) -> real:
    """ABoxMod1's fold: Fold - |(|x + FoldMod|) - Fold| - |FoldMod|."""
    return fold - fabs(fabs(x + fmod) - fold) - fabs(fmod)


@formula(
    name="ABoxMod1",
    params=[Param("scale", 2.0, "real", "scale", mb3d_name="Scale"),
            Param("min_r", 0.0, "real", "minimum radius", mb3d_name="Boxscale Min R"),
            Param("fold", 2.0, "real", "fold", mb3d_name="Fold"),
            Param("scale_vary", 0.0, "real", "scale changes every iteration by vary * (|scale| - 1)",
                  mb3d_name="Scale vary"),
            Param("fold_x_mod", 0.0, "real", "fold X modifier", mb3d_name="FoldXMod"),
            Param("fold_y_mod", 0.0, "real", "fold Y modifier", mb3d_name="FoldYMod"),
            Param("fold_z_mod", 0.0, "real", "fold Z modifier", mb3d_name="FoldZMod")],
    de="linear",
    de_scale=0.2,
    rstop=1024.0,
    category="3D boxes",
    description="MB3D's ABoxMod1 (after Kali): modified folds, a scale that may vary per iteration.",
    aliases=("aboxmod1",),
    scale_param="scale",
)
def abox_mod1(s, p):
    if s.vary == 0.0:
        s.vary = p.scale
    s.vary = s.vary + p.scale_vary * (fabs(s.vary) - 1.0)
    sc = s.vary
    f = vec3(fold_mod(s.z.x, p.fold, p.fold_x_mod), fold_mod(s.z.y, p.fold, p.fold_y_mod),
             fold_mod(s.z.z, p.fold, p.fold_z_mod))
    rr = dot(f, f)
    mr2 = p.min_r * p.min_r
    m = sc
    if rr < mr2:
        m = p.scale / mr2
    elif rr < 1.0:
        m = sc / rr
    s.dr = s.dr * fabs(m)
    s.z = f * m + s.c
