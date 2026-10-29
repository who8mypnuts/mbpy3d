"""The ray marcher: one pixel from the start plane to the surface.

Port of the per pixel loop of ``TMandCalcThread.Execute`` (CalcThread.pas)
with ``RMdoBinSearch``, ``RMCalculateNormals`` and ``RMdoColor``; plus the
optional DE ambient occlusion and a soft shadow ray of this port.
All distances are in units of the pixel step width.
"""

import math

from mb3d.kernel.dsl import Bool, ConstIntArray, ConstRealArray, Int, Real, RealArray, kfunc, local_real

from .de import calc_de
from .layout import (
    CX,
    CY,
    CZ,
    IP_AO_SAMPLES,
    IP_BIN_STEPS,
    IP_COLOR_OPTION,
    IP_CUT_OPTIONS,
    IP_FIRST_STEP_RANDOM,
    IP_MAX_IT,
    IP_MIN_IT,
    IP_OPTIC,
    IP_SHADOW,
    IT,
    O_AO,
    O_DEPTH,
    O_HIT,
    O_ITS,
    O_NX,
    O_NY,
    O_NZ,
    O_OTRAP,
    O_ROUGH,
    O_SHADOW,
    O_SIG,
    O_STEPS,
    OTRAP,
    ROLD,
    ROUT,
    SIT,
    SP_AO_LEN,
    SP_COL_VAR,
    SP_COLOR_MUL,
    SP_CUT_POS,
    SP_D_COL_PLUS,
    SP_DE_STOP,
    SP_DE_STOP_FACTOR,
    SP_FOV_Y,
    SP_FOVX_MUL,
    SP_FOVX_OFF,
    SP_HEIGHT,
    SP_MCTS_M,
    SP_MH04ZSD,
    SP_MS_DE_SUB,
    SP_PL_OPTIC_Z,
    SP_RSTOP,
    SP_SHADOW_DIR,
    SP_STEP_WIDTH,
    SP_VG,
    SP_YSTART,
    SP_Z_STEP_DIV,
    SP_ZEND,
    X,
    Y,
    Z,
)


@kfunc
def pixel_ray(ix: Int, iy: Int, sp: ConstRealArray[80], ip: ConstIntArray[32], ray: RealArray[6]) -> None:
    """``RMCalculateVgradsFOV`` + ``RMCalculateStartPos``: origin (0..2) and step vector (3..5)."""
    cafy = (float(iy) / sp[SP_HEIGHT] - 0.5) * sp[SP_FOV_Y]
    cafx = (sp[SP_FOVX_OFF] - float(ix + 1)) * sp[SP_FOVX_MUL]
    optic = ip[IP_OPTIC]
    if optic == 1:
        vx = -cafx
        vy = cafy
        vz = sp[SP_PL_OPTIC_Z]
        n = 1.0 / math.sqrt(vx * vx + vy * vy + vz * vz)
        vx *= n
        vy *= n
        vz *= n
    elif optic == 2:
        vx = -math.sin(cafx) * math.cos(cafy)
        vy = math.sin(cafy)
        vz = math.cos(cafy) * math.cos(cafx)
    else:
        vx = -math.sin(cafx)
        vy = math.sin(cafy)
        vz = math.cos(cafy) * math.cos(cafx)
        n = 1.0 / math.sqrt(vx * vx + vy * vy + vz * vz)
        vx *= n
        vy *= n
        vz *= n
    # RotateVectorReverse: v' = sum_i v[i] * row_i (rows scaled to the step width)
    ray[3] = vx * sp[SP_VG + 0] + vy * sp[SP_VG + 3] + vz * sp[SP_VG + 6]
    ray[4] = vx * sp[SP_VG + 1] + vy * sp[SP_VG + 4] + vz * sp[SP_VG + 7]
    ray[5] = vx * sp[SP_VG + 2] + vy * sp[SP_VG + 5] + vz * sp[SP_VG + 8]
    if optic == 2:
        ray[0] = sp[SP_YSTART]
        ray[1] = sp[SP_YSTART + 1]
        ray[2] = sp[SP_YSTART + 2]
    else:
        fx = float(ix)
        fy = float(iy)
        ray[0] = sp[SP_YSTART] + sp[SP_VG + 0] * fx + sp[SP_VG + 3] * fy
        ray[1] = sp[SP_YSTART + 1] + sp[SP_VG + 1] * fx + sp[SP_VG + 4] * fy
        ray[2] = sp[SP_YSTART + 2] + sp[SP_VG + 2] * fx + sp[SP_VG + 5] * fy


@kfunc
def pixel_random(ix: Int, iy: Int) -> Real:
    """A per pixel pseudo random number in [0, 1) (no int32 overflow)."""
    n = (ix * 7919 + iy * 65521 + 1) & 0x7FFF
    n = (n * 40503 + 7) & 0x7FFF
    n = (n * 40503 + 13) & 0x7FFF
    return float(n) / 32768.0


@kfunc
def cut_plane_start(st: RealArray[16], sp: ConstRealArray[80], ip: ConstIntArray[32], ray: RealArray[6]) -> Real:
    """``RMmaxLengthToCutPlane``: steps from the ray start to the cutting planes (1e20: cut away)."""
    opts = ip[IP_CUT_OPTIONS]
    length = 0.0
    for k in range(3):
        if ((opts >> k) & 1) == 0:
            continue
        v = ray[3 + k]
        if abs(v) <= 1e-20:
            continue
        side = ((opts >> (4 + k)) & 1) != 0
        neg = v < 0.0
        if side == neg:
            length = 1e20
        else:
            t = (sp[SP_CUT_POS + k] - ray[k]) / v
            if t > length:
                length = t
    return length


@kfunc
def march_stop(sp: ConstRealArray[80], mzz: Real) -> Real:
    """The DE stop threshold grows with the distance (``bVaryDEstopOnFOV``)."""
    return sp[SP_DE_STOP] * (1.0 + mzz * sp[SP_DE_STOP_FACTOR])


@kfunc
def move_ray(st: RealArray[16], ray: RealArray[6], d: Real) -> None:
    st[CX] += ray[3] * d
    st[CY] += ray[4] * d
    st[CZ] += ray[5] * d


@kfunc
def surface_color(st: RealArray[16], sp: ConstRealArray[80], ip: ConstIntArray[32]) -> Real:
    """``RMdoColor``: the orbit trap / 2nd colour value 0..32767."""
    opt = ip[IP_COLOR_OPTION]
    if opt == 1:
        s = math.log(st[ROUT] / (st[ROLD] + 1.0)) * sp[SP_COLOR_MUL]
    elif opt == 2:
        s = (math.atan2(st[Y] - st[CY], st[X] - st[CX]) + math.pi) * 5200.0
    elif opt == 3:
        s = (math.atan2(st[Z] - st[CZ], st[X] - st[CX]) + math.pi) * 5200.0
    elif opt == 4:
        s = (math.atan2(st[Z] - st[CZ], st[Y] - st[CY]) + math.pi) * 5200.0
    else:
        s = st[OTRAP] * 4096.0
    if math.isnan(s) or s < 0.0:
        s = 0.0
    if s > 32767.0:
        s = 32767.0
    return s


@kfunc
def calculate_normals(st: RealArray[16], sp: ConstRealArray[80], fp: ConstRealArray[96], ip: ConstIntArray[32],
                      mzz: Real, ms_de_stop: Real, res: RealArray[12]) -> Real:
    """``RMCalculateNormals``: the view space normal from the DE gradient; returns the smooth iterations."""
    noffset = sp[SP_DE_STOP]
    if noffset > 1.0:
        noffset = 1.0
    noffset = noffset * (1.0 + mzz * sp[SP_DE_STOP_FACTOR]) * 0.15
    cx = st[CX]
    cy = st[CY]
    cz = st[CZ]
    calc_de(st, sp, fp, ip, ms_de_stop, True)
    nn = st[SIT]
    nx = 0.0
    ny = 0.0
    nz = 0.0
    for k in range(3):
        ax = sp[SP_VG + 3 * k] * noffset
        ay = sp[SP_VG + 3 * k + 1] * noffset
        az = sp[SP_VG + 3 * k + 2] * noffset
        st[CX] = cx + ax
        st[CY] = cy + ay
        st[CZ] = cz + az
        a = calc_de(st, sp, fp, ip, ms_de_stop, False)
        st[CX] = cx - ax
        st[CY] = cy - ay
        st[CZ] = cz - az
        b = calc_de(st, sp, fp, ip, ms_de_stop, False)
        d = (a - b) * 0.5
        if k == 0:
            nx = d
        elif k == 1:
            ny = d
        else:
            nz = d
    st[CX] = cx
    st[CY] = cy
    st[CZ] = cz
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-30 or math.isnan(length):
        nx = 0.0
        ny = 0.0
        nz = -1.0
    else:
        nx /= length
        ny /= length
        nz /= length
    res[O_NX] = nx
    res[O_NY] = ny
    res[O_NZ] = nz
    return nn


@kfunc
def world_normal(sp: ConstRealArray[80], res: RealArray[12], nw: RealArray[3]) -> None:
    """The unit world space normal of the view space normal in ``res``."""
    inv = 1.0 / sp[SP_STEP_WIDTH]
    for k in range(3):
        nw[k] = (res[O_NX] * sp[SP_VG + k] + res[O_NY] * sp[SP_VG + 3 + k] + res[O_NZ] * sp[SP_VG + 6 + k]) * inv


@kfunc
def ambient_occlusion(st: RealArray[16], sp: ConstRealArray[80], fp: ConstRealArray[96], ip: ConstIntArray[32],
                      ms_de_stop: Real, res: RealArray[12]) -> Real:
    """DE based ambient occlusion along the normal (0 = dark, 1 = open)."""
    samples = ip[IP_AO_SAMPLES]
    nw = local_real(3)
    world_normal(sp, res, nw)
    cx = st[CX]
    cy = st[CY]
    cz = st[CZ]
    sw = sp[SP_STEP_WIDTH]
    delta = sp[SP_AO_LEN] / float(samples)
    occ = 0.0
    weight = 1.0
    for i in range(1, samples + 1):
        t = delta * float(i)
        st[CX] = cx + nw[0] * t * sw
        st[CY] = cy + nw[1] * t * sw
        st[CZ] = cz + nw[2] * t * sw
        d = calc_de(st, sp, fp, ip, ms_de_stop, False)
        occ += (t - d) / t * weight
        weight *= 0.6
    st[CX] = cx
    st[CY] = cy
    st[CZ] = cz
    ao = 1.0 - occ * 0.9
    if ao < 0.0:
        ao = 0.0
    if ao > 1.0:
        ao = 1.0
    return ao


@kfunc
def shadow_ray(st: RealArray[16], sp: ConstRealArray[80], fp: ConstRealArray[96], ip: ConstIntArray[32],
               ms_de_stop: Real, res: RealArray[12]) -> Real:
    """A soft shadow ray from the surface towards the light (1 = lit)."""
    nw = local_real(3)
    world_normal(sp, res, nw)
    sw = sp[SP_STEP_WIDTH]
    lx = sp[SP_SHADOW_DIR]
    ly = sp[SP_SHADOW_DIR + 1]
    lz = sp[SP_SHADOW_DIR + 2]
    cx = st[CX] + nw[0] * ms_de_stop * 2.0 * sw
    cy = st[CY] + nw[1] * ms_de_stop * 2.0 * sw
    cz = st[CZ] + nw[2] * ms_de_stop * 2.0 * sw
    ox = st[CX]
    oy = st[CY]
    oz = st[CZ]
    t = ms_de_stop * 2.0
    lit = 1.0
    max_t = sp[SP_ZEND] * 0.5
    for i in range(96):
        st[CX] = cx + lx * t * sw
        st[CY] = cy + ly * t * sw
        st[CZ] = cz + lz * t * sw
        d = calc_de(st, sp, fp, ip, ms_de_stop, False)
        if d < ms_de_stop * 0.5:
            lit = 0.0
            break
        s = 8.0 * d / t
        if s < lit:
            lit = s
        t += d
        if t > max_t:
            break
    st[CX] = ox
    st[CY] = oy
    st[CZ] = oz
    if lit < 0.0:
        lit = 0.0
    return lit


@kfunc
def march_pixel(ix: Int, iy: Int, sp: ConstRealArray[80], fp: ConstRealArray[96], ip: ConstIntArray[32],
                st: RealArray[16], res: RealArray[12]) -> None:
    """One pixel: the ray march, the binary search, normal, colour values and depth into ``res``."""
    ray = local_real(6)
    pixel_ray(ix, iy, sp, ip, ray)
    st[CX] = ray[0]
    st[CY] = ray[1]
    st[CZ] = ray[2]
    for k in range(12):
        res[k] = 0.0
    res[O_AO] = 1.0
    res[O_SHADOW] = 1.0
    res[O_NZ] = -1.0
    mzz = 0.0
    de_stop = sp[SP_DE_STOP]
    ms_de_stop = de_stop
    zend = sp[SP_ZEND]
    max_it = ip[IP_MAX_IT]
    min_it = ip[IP_MIN_IT]
    z_step_div = sp[SP_Z_STEP_DIV]
    first_step = ip[IP_FIRST_STEP_RANDOM] != 0
    if ip[IP_CUT_OPTIONS] != 0:
        dt1 = cut_plane_start(st, sp, ip, ray)
        if dt1 > zend - mzz:
            return
        mzz = dt1
        move_ray(st, ray, dt1)
        ms_de_stop = march_stop(sp, mzz)
    dtmp = calc_de(st, sp, fp, ip, ms_de_stop, False)
    it_res = int(st[IT])
    if it_res >= max_it or dtmp < ms_de_stop:
        # inside the set at the start plane (or the cutting plane)
        res[O_HIT] = 2.0
        res[O_DEPTH] = mzz
        res[O_OTRAP] = surface_color(st, sp, ip)
        t = st[ROUT] / sp[SP_RSTOP]
        if t < 0.0:
            t = 0.0
        if t > 1.0:
            t = 1.0
        res[O_SIG] = 32768.0 + 32767.0 * t
        res[O_ITS] = st[IT]
        return
    step_count = 0.0
    rsf_mul = 1.0
    last_step = dtmp * z_step_div
    last_de = dtmp
    hit = False
    guard = 0
    while True:
        guard += 1
        if guard > 300000:
            break
        if it_res >= max_it:
            dt1 = -0.5 * last_step
            mzz += dt1
            move_ray(st, ray, dt1)
            ms_de_stop = march_stop(sp, mzz)
            dtmp = calc_de(st, sp, fp, ip, ms_de_stop, False)
            it_res = int(st[IT])
            last_step = -dt1
        if it_res < min_it or (it_res < max_it and dtmp >= ms_de_stop):
            last_de = dtmp
            d = (dtmp - sp[SP_MS_DE_SUB] * ms_de_stop) * z_step_div * rsf_mul
            if d < 0.11:
                d = 0.11
            s1 = ms_de_stop
            if s1 < 0.4:
                s1 = 0.4
            s1 *= sp[SP_MH04ZSD]
            if s1 < d:
                step_count += s1 / d
                d = s1
            else:
                step_count += 1.0
            if first_step:
                first_step = False
                d *= pixel_random(ix, iy)
            mzz += d
            if mzz > zend:
                break
            last_step = d
            move_ray(st, ray, d)
            ms_de_stop = march_stop(sp, mzz)
            dtmp = calc_de(st, sp, fp, ip, ms_de_stop, False)
            it_res = int(st[IT])
            if dtmp > last_de + last_step:
                dtmp = last_de + last_step
            if last_de > dtmp + 1e-30:
                t = last_step / (last_de - dtmp)
                if t < 1.0:
                    rsf_mul = t
                    if rsf_mul < 0.5:
                        rsf_mul = 0.5
                else:
                    rsf_mul = 1.0
            else:
                rsf_mul = 1.0
        else:
            hit = True
            break
    res[O_STEPS] = step_count
    if not hit:
        return
    # ---- the surface: binary search (RMdoBinSearch)
    de_limited = it_res < max_it or dtmp < ms_de_stop
    itmp = ip[IP_BIN_STEPS]
    dt1 = last_step * -0.5
    while abs(dtmp - ms_de_stop) > 0.001 and itmp > 0:
        mzz += dt1
        move_ray(st, ray, dt1)
        ms_de_stop = march_stop(sp, mzz)
        itmp -= 1
        if itmp <= 0:
            break
        dtmp = calc_de(st, sp, fp, ip, ms_de_stop, False)
        it_res = int(st[IT])
        if it_res >= max_it:
            dt1 = -abs(dt1)
        elif dtmp < ms_de_stop:
            dt1 = abs(dt1) * -0.55
        else:
            dt1 = abs(dt1) * 0.55
    # ---- normal, colour, depth
    nn = calculate_normals(st, sp, fp, ip, mzz, ms_de_stop, res)
    if de_limited:
        dsm = de_stop
        if ms_de_stop > dsm:
            dsm = ms_de_stop
        v = 32767.0 - (nn + sp[SP_D_COL_PLUS] + sp[SP_COL_VAR] * math.log(dsm * sp[SP_STEP_WIDTH])) * sp[SP_MCTS_M]
    else:
        v = 32767.0 - nn * sp[SP_MCTS_M]
    if math.isnan(v) or v < 0.0:
        v = 0.0
    if v > 32767.0:
        v = 32767.0
    res[O_SIG] = v
    res[O_OTRAP] = surface_color(st, sp, ip)
    res[O_ITS] = st[IT]
    res[O_HIT] = 1.0
    if mzz < 0.0:
        mzz = 0.0
    res[O_DEPTH] = mzz
    res[O_ROUGH] = 0.0
    if ip[IP_AO_SAMPLES] > 0:
        res[O_AO] = ambient_occlusion(st, sp, fp, ip, ms_de_stop, res)
    if ip[IP_SHADOW] != 0:
        res[O_SHADOW] = shadow_ray(st, sp, fp, ip, ms_de_stop, res)
