"""'Quaternion' (4D), 'Tricorn' and 'Folding Int Pow' of Mandelbulb3D."""
from ..dsl import *  # noqa: F401,F403
from ..dsl import Param, formula, vec3, real, TINY
from .box import box_fold
from .mandelbulb import int_pow_bulb, bulb_dr


@formula(
    name="Quaternion",
    mb3d_id=2,
    params=[Param("yw_mul", -1.0, "real", "YW multiplier", mb3d_name="YW multiplier"),
            Param("w_add", 0.0, "real", "W add", mb3d_name="W add")],
    de="log",
    si_pow=2.0,
    rstop=16.0,
    four_d=True,
    category="4D",
    description="Quaternion power 2 (x, y, z, w); the 4th component lives in s.w / s.cw.",
    aliases=("quat",),
)
def quaternion(s, p):
    x = s.z.x
    y = s.z.y
    z = s.z.z
    w = s.w
    r = sqrt(x * x + y * y + z * z + w * w)
    s.dr = 2.0 * r * s.dr + 1.0
    s.z = vec3(x * x - y * y - z * z - w * w + s.c.x,
               2.0 * (y * x + z * w) + s.c.y,
               2.0 * (z * x + p.yw_mul * y * w) + s.c.z)
    s.w = 2.0 * (w * x + y * z) + p.w_add + s.cw


@formula(
    name="Tricorn",
    mb3d_id=3,
    params=[Param("z_mul", -2.0, "real", "Z multiplier", mb3d_name="Z multiplier"),
            Param("cz_mul", 1.0, "real", "CZ multiplier", mb3d_name="CZ multiplier")],
    de="log",
    si_pow=2.0,
    rstop=16.0,
    category="3D bulbs",
    description="MB3D's Tricorn: (x^2 - y^2 - z^2, 2xy, x z zmul).",
)
def tricorn(s, p):
    x = s.z.x
    y = s.z.y
    z = s.z.z
    s.dr = 2.0 * length(s.z) * s.dr + 1.0
    s.z = vec3(x * x - (y * y + z * z) + s.c.x,
               2.0 * (y * x) + s.c.y,
               z * x * p.z_mul + s.c.z * p.cz_mul)


@formula(
    name="Folding Int Pow",
    mb3d_id=6,
    params=[Param("power", 2, "int", "integer power 2..8", 2, 8, "Integer power"),
            Param("z_mul", -1.0, "real", "Z multiplier", mb3d_name="Z multiplier"),
            Param("fold", 2.0, "real", "R fold", mb3d_name="R fold")],
    de="none",
    de_scale=0.5,
    si_pow="power",
    rstop=1024.0,
    category="3D bulbs",
    description="Box fold followed by the integer power bulb.",
    aliases=("foldingintpow", "foldintpow"),
)
def folding_int_pow(s, p):
    f = box_fold(s.z, p.fold)
    s.z = int_pow_bulb(f, s.c, int(p.power), p.z_mul)
