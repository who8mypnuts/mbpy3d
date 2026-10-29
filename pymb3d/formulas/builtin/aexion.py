"""'Aexion C' (MB3D formula 9): a real power bulb whose constant C is
rotated by a power of its own angles every iteration."""
from ..dsl import *  # noqa: F401,F403
from ..dsl import Param, formula, vec3, real, TINY


@formula(
    name="Aexion C",
    mb3d_id=9,
    params=[Param("power", 8.0, "real", "float power", mb3d_name="Float power"),
            Param("z_mul", 1.0, "real", "Z multiplier", mb3d_name="Z multiplier"),
            Param("rot_c", 1, "bool", "enable rotate C", mb3d_name="Enable rotate C (0,1)"),
            Param("cond_phi", 1, "bool", "conditional phi", mb3d_name="Condi. Phi (0,1)"),
            Param("power_c", 8.0, "real", "float power C", mb3d_name="Float power C"),
            Param("cz_mul", 1.0, "real", "Cz multiplier", mb3d_name="Cz multiplier"),
            Param("powc_dist", 0, "bool", "PowC on dist Vec-C", mb3d_name="PowC on dist Vec-C (0,1)"),
            Param("mode", 0, "int", "mode 0..31", 0, 31, "Mode (0..31)")],
    de="none",
    si_pow="power",
    rstop=16.0,
    category="3D bulbs",
    description="Aexion's bulb with an iterated constant ('iterating C').",
    aliases=("aexionc", "aexion"),
)
def aexion_c(s, p):
    x = s.z.x
    y = s.z.y
    z = s.z.z
    r1 = x * x + y * y + z * z
    th = atan2(sqrt(x * x + z * z), y) * p.power
    ph = atan2(z, x) * p.power
    rp = pow(r1, p.power * 0.5)
    s.z = vec3(cos(ph) * cos(th) * rp + s.c.x,
               cos(th) * sin(ph) * rp + s.c.y,
               sin(th) * rp * p.z_mul + s.c.z)
    if p.rot_c != 0.0:
        pd = p.power_c
        if p.powc_dist != 0.0:
            pd = pd * length(s.z - s.c)
        r = length(s.c)
        mode = int(p.mode)
        a = s.c.y
        b = s.c.z
        cc = s.c.x
        if (mode & 16) != 0:
            a = s.z.y - s.c.y
            b = s.z.z - s.c.z
            cc = s.z.x - s.c.x
        ofs = 0
        ecx = 8
        if (mode & 8) != 0:
            tmp = a
            a = b
            b = tmp
            ecx = 16
        sv = sqrt(b * b + cc * cc)
        th2 = atan2(sv, a)
        if (mode & 1) != 0:
            th2 = atan2(a, sv)
        ph2 = atan2(b, cc)
        if (mode & 2) != 0:
            ofs = 24 - ecx
            ph2 = atan2(cc, b)
        if (mode & 4) != 0:
            tmp2 = th2
            th2 = ph2
            ph2 = tmp2
            ofs = ecx
        if p.cond_phi != 0.0:
            t = s.z.x
            if ofs == 8:
                t = s.z.y
            elif ofs == 16:
                t = s.z.z
            if t >= 0.0:
                ph2 = -ph2
        sa = sin(ph2 * pd)
        ca = cos(ph2 * pd)
        sb = sin(th2 * pd)
        cb = cos(th2 * pd)
        s.c = vec3(ca * cb * r, cb * sa * r, sb * r * p.cz_mul)
