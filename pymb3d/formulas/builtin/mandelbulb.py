"""The triplex "sine" bulbs of Mandelbulb3D: 'Integer Power' (the fast
polynomial forms for powers 2..8, `fHIntFunctions`) and 'Real Power'
(`HybridFloatPow`), plus the common spherical-coordinate bulb."""
from ..dsl import *  # noqa: F401,F403
from ..dsl import Param, formula, helper, vec3, real, TINY


@helper
def bulb_dr(z: vec3, power: real, dr: real) -> real:
    """Running derivative for the logarithmic DE of a power-n bulb."""
    r = length(z)
    return power * pow(r, power - 1.0) * dr + 1.0


@helper
def sine_bulb_pow2(z: vec3, c: vec3, zmul: real) -> vec3:
    """`HybridItIntPow2`: the sine bulb of power 2."""
    x = z.x
    y = z.y
    zz = z.z
    yy = y * y
    xx = x * x
    r = xx + yy
    nz = (sqrt(r) * zmul * zz) * 2.0 + c.z
    a = (r - zz * zz) / (r + TINY)
    nx = (xx - yy) * a + c.x
    ny = (x * y * a) * 2.0 + c.y
    return vec3(nx, ny, nz)


@helper
def int_pow_bulb(z: vec3, c: vec3, power: int, zmul: real) -> vec3:
    """`fHIntFunctions[power]`: the polynomial triplex powers 2..8."""
    x = z.x
    y = z.y
    zc = z.z
    sx = x * x
    sy = y * y
    sz = zc * zc
    r = sx + sy
    out = vec3(0.0, 0.0, 0.0)
    if power <= 2:
        out = sine_bulb_pow2(z, c, zmul)
    elif power == 3:
        a = 1.0 - 3.0 * sz / (r + TINY)
        out = vec3((sx - 3.0 * sy) * a * x + c.x,
                   a * (3.0 * sx - sy) * y + c.y,
                   c.z - (sz - 3.0 * r) * zc * zmul)
    elif power == 4:
        a = 1.0 + sz * (sz - 6.0 * r) / (r * r + TINY)
        out = vec3((sx * (sx - 6.0 * sy) + sy * sy) * a + c.x,
                   a * x * (sx - sy) * 4.0 * y + c.y,
                   sqrt(r) * (r - sz) * zc * 4.0 * zmul + c.z)
    elif power == 5:
        a = 1.0 + (sz * sz - sz * r * 2.0) * 5.0 / (r * r + TINY)
        out = vec3(a * x * (sx * (sx - 10.0 * sy) + 5.0 * sy * sy) + c.x,
                   (5.0 * sx * sx - sy * (10.0 * sx - sy)) * a * y + c.y,
                   (sz * (sz - 10.0 * r) + 5.0 * r * r) * zc * zmul + c.z)
    elif power == 6:
        rr = r * r
        a = 1.0 - sz * (15.0 * rr + sz * (sz - r * 15.0)) / (rr * r + TINY)
        out = vec3((sy * sy * (15.0 * sx - sy) + sx * sx * (sx - 15.0 * sy)) * a + c.x,
                   (3.0 * sy * sy + sx * (3.0 * sx - 10.0 * sy)) * a * y * x * 2.0 + c.y,
                   (sz * (3.0 * sz - 10.0 * r) + 3.0 * rr) * sqrt(r) * zc * zmul * 2.0 + c.z)
    elif power == 7:
        rr = r * r
        a = 1.0 - 7.0 * (sz * (3.0 * rr + sz * (sz - 5.0 * r))) / (rr * r + TINY)
        out = vec3(a * x * (sx * (35.0 * sy * sy + sx * (sx - 21.0 * sy)) - 7.0 * sy * sy * sy) + c.x,
                   (sx * (21.0 * sy * sy + sx * (7.0 * sx - 35.0 * sy)) - sy * sy * sy) * a * y + c.y,
                   c.z - (sz * sz * sz - 7.0 * r * (rr + sz * (3.0 * sz - 5.0 * r))) * zc * zmul)
    else:
        rr = r * r
        zzzz = sz * sz
        t = (zzzz - 6.0 * r * sz + rr) * (sz - r) * sqrt(r)
        num = (rr * 70.0 + zzzz) * zzzz - 28.0 * sz * r * (zzzz + rr)
        a = num / (rr * rr + TINY) + 1.0
        xxxx = sx * sx
        yyyy = sy * sy
        out = vec3(a * (xxxx * xxxx + yyyy * (70.0 * xxxx + yyyy) - 28.0 * (sx * sy * (yyyy + xxxx))) + c.x,
                   (yyyy * (7.0 * sx - sy) + xxxx * (sx - 7.0 * sy)) * 8.0 * x * y * a + c.y,
                   -(t * zc * 8.0 * zmul) + c.z)
    return out


@formula(
    name="Integer Power",
    mb3d_id=0,
    params=[Param("power", 8, "int", "integer power 2..8", 2, 8, "Integer power"),
            Param("z_mul", -1.0, "real", "Z multiplier", mb3d_name="Z multiplier")],
    de="log",
    si_pow="power",
    rstop=16.0,
    category="3D bulbs",
    description="The classic Mandelbulb (triplex sine power 2..8), polynomial form of MB3D.",
    aliases=("mandelbulb", "intpow", "bulb"),
)
def integer_power(s, p):
    power = int(p.power)
    s.dr = bulb_dr(s.z, float(power), s.dr)
    s.z = int_pow_bulb(s.z, s.c, power, p.z_mul)


@formula(
    name="Real Power",
    mb3d_id=1,
    params=[Param("power", 8.0, "real", "real power", mb3d_name="Float power"),
            Param("z_mul", -1.0, "real", "Z multiplier", mb3d_name="Z multiplier")],
    de="log",
    si_pow="power",
    rstop=16.0,
    category="3D bulbs",
    description="Sine bulb with an arbitrary real power (`HybridFloatPow`).",
    aliases=("realpower", "floatpow"),
)
def real_power(s, p):
    x = s.z.x
    y = s.z.y
    z = s.z.z
    s.dr = bulb_dr(s.z, p.power, s.dr)
    theta = atan2(y, x) * p.power
    phi = atan2(z, sqrt(x * x + y * y)) * p.power
    r = pow(s.r2, p.power * 0.5)
    s.z = vec3(cos(theta) * cos(phi) * r + s.c.x,
               sin(theta) * cos(phi) * r + s.c.y,
               sin(phi) * r * p.z_mul + s.c.z)


@formula(
    name="Mandelbulb IQ",
    params=[Param("power", 8.0, "real", "power"),
            Param("phase_theta", 0.0, "real", "theta offset"),
            Param("phase_phi", 0.0, "real", "phi offset")],
    de="log",
    si_pow="power",
    rstop=16.0,
    category="3D bulbs",
    description="Spherical-coordinate bulb with theta measured from the z axis (as in most GLSL demos).",
    aliases=("iq bulb", "bulb iq"),
)
def mandelbulb_iq(s, p):
    r = length(s.z)
    s.dr = p.power * pow(r, p.power - 1.0) * s.dr + 1.0
    theta = acos(s.z.z / (r + TINY)) * p.power + p.phase_theta
    phi = atan2(s.z.y, s.z.x) * p.power + p.phase_phi
    zr = pow(r, p.power)
    s.z = vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta)) * zr + s.c
