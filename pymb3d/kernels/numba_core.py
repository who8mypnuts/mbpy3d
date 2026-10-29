"""The numba ray marching core: the Python twin of ``clike/march_core.c``.

``NUMBA_CORE`` is source text appended to the generated formula module
(see :meth:`Program.py_source`); it uses the generated loops ``iterate``,
``iterate_p2``, ``difs_orbit``, ``difs_orbit_p2`` and the ``P_*`` / ``G_*``
constants defined there.  Keep the functions in step with the C core.

A DE result is the tuple (de, it, r2, rold, otrap, slot, sit, inside,
zx, zy, zz, cx, cy, cz, dr).
"""

NUMBA_CORE = r'''

@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def orbit_init(px, py, pz, P):
    zx, zy, zz, w = px, py, pz, 0.0
    if P[P_FOUR_D] > 0.5:
        zx = px * P[P_ROT4D + 0] + py * P[P_ROT4D + 1] + pz * P[P_ROT4D + 2]
        zy = px * P[P_ROT4D + 4] + py * P[P_ROT4D + 5] + pz * P[P_ROT4D + 6]
        zz = px * P[P_ROT4D + 8] + py * P[P_ROT4D + 9] + pz * P[P_ROT4D + 10]
        w = px * P[P_ROT4D + 12] + py * P[P_ROT4D + 13] + pz * P[P_ROT4D + 14]
    if P[P_JULIA] > 0.5:
        cx, cy, cz, cw = P[P_JULIA_C], P[P_JULIA_C + 1], P[P_JULIA_C + 2], P[P_JULIA_C + 3]
    else:
        cx, cy, cz, cw = zx, zy, zz, w
    r2 = zx * zx + zy * zy + zz * zz + w * w
    return zx, zy, zz, w, cx, cy, cz, cw, 1.0, r2, 1e30, 0.0, 0.0, 0.0


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def smooth_iterations(r2, rold, it, slot, mode, P):
    fhln = P[P_SLOTS + SLOT_STRIDE * slot + 2]
    ln_rstop = P[P_LN_RSTOP]
    if mode == 3:
        return float(it)
    if r2 <= 1.0:
        return float(it)
    if rold < 1.0:
        d = math.log(0.5 * math.log(r2)) * fhln
        return float(it) + ln_rstop - d
    d = math.log(0.5 * math.log(r2))
    e = d - math.log(0.5 * math.log(rold))
    return (ln_rstop - d) / (e + 1e-40) + float(it)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def de_numeric(px, py, pz, it, r2_0, part, de_scale, P):
    if r2_0 < 1e-30:
        return 0.0
    off = P[P_DE_OFFSET_PX] * P[P_STEP_WIDTH]
    g = 0.0
    if part == 0:
        r2k = iterate(px + off, py, pz, it, 1e30, P)[0]
    else:
        r2k = iterate_p2(px + off, py, pz, it, 1e30, P)[0]
    d = r2_0 - r2k
    g += d * d
    if part == 0:
        r2k = iterate(px, py + off, pz, it, 1e30, P)[0]
    else:
        r2k = iterate_p2(px, py + off, pz, it, 1e30, P)[0]
    d = r2_0 - r2k
    g += d * d
    if part == 0:
        r2k = iterate(px, py, pz + off, it, 1e30, P)[0]
    else:
        r2k = iterate_p2(px, py, pz + off, it, 1e30, P)[0]
    d = r2_0 - r2k
    g += d * d
    return r2_0 * math.log(r2_0) * de_scale * P[P_DE_OFFSET_PX] / (math.sqrt(g) + P[P_DE_OFFSET_PX] * 0.06)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def de_part(px, py, pz, ds, part, P):
    """The DE of one hybrid part: (de, it, r2, rold, otrap, slot, sit, inside, zx, zy, zz, cx, cy, cz, dr)."""
    if part == 0:
        max_it = int(P[P_MAX_IT])
        inside_it = int(P[P_INSIDE_IT])
        mode = int(P[P_DE_MODE])
        de_scale = P[P_DE_SCALE]
    else:
        max_it = int(P[P_MAX_IT2])
        inside_it = int(P[P_INSIDE_IT2])
        mode = int(P[P_DE_MODE2])
        de_scale = P[P_DE_SCALE2]
    sw = P[P_STEP_WIDTH]
    if mode == 3:
        if part == 0:
            min_de, it, otrap, slot, zx, zy, zz, cx, cy, cz = difs_orbit(px, py, pz, max_it, ds * sw * 1.03, False, P)
        else:
            min_de, it, otrap, slot, zx, zy, zz, cx, cy, cz = difs_orbit_p2(px, py, pz, max_it, ds * sw * 1.03, False, P)
        r2 = min_de
        rold = min_de
        dr = 1.0
        de = min_de * de_scale / sw
    else:
        if part == 0:
            r2, dr, rold, otrap, it, slot, zx, zy, zz, cx, cy, cz = iterate(px, py, pz, max_it, P[P_RSTOP2], P)
        else:
            r2, dr, rold, otrap, it, slot, zx, zy, zz, cx, cy, cz = iterate_p2(px, py, pz, max_it, P[P_RSTOP2], P)
        if it >= inside_it:
            de = 0.0
        elif mode == 1:
            de = math.sqrt(r2) / abs(dr) * de_scale / sw
        elif mode == 2:
            rr = math.sqrt(r2)
            de = 0.5 * math.log(r2) * rr / abs(dr) * de_scale / sw
        else:
            de = de_numeric(px, py, pz, it, r2, part, de_scale, P)
    if not (de >= ds * 0.25):
        de = ds * 0.25
    sit = 0.0
    if int(P[P_DECOMB_KIND]) >= 4:
        sit = smooth_iterations(r2, rold, it, slot, mode, P)
    return de, it, r2, rold, otrap, slot, sit, it >= inside_it, zx, zy, zz, cx, cy, cz, dr


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def calc_de(px, py, pz, ds, invert, P):
    a = de_part(px, py, pz, ds, 0, P)
    kind = int(P[P_DECOMB_KIND])
    mode1 = int(P[P_DE_MODE])
    last_difs = mode1 == 3
    if kind != 0:
        b = de_part(px, py, pz, ds, 1, P)
        de_a = a[0]
        rst = b[0]
        b_inside = b[7]
        if kind == 3:
            rst = ds * 4.0 - rst * 3.0
            b_inside = False
        if kind == 1:
            if rst < de_a:
                a = (rst, b[1], b[2], b[3], b[4], b[5], b[6], b_inside, b[8], b[9], b[10], b[11], b[12], b[13], b[14])
        elif kind == 2 or kind == 3:
            if rst > de_a:
                a = (rst, b[1], b[2], b[3], b[4], b[5], b[6], b_inside, b[8], b[9], b[10], b[11], b[12], b[13], b[14])
        elif kind == 6:
            mix_col = int(P[P_DECOMB_MIX_COL])
            sit = b[6]
            otrap = b[4]
            if mix_col == 0:
                sit = b[6] + a[6]
                otrap = (b[4] + a[4]) * 0.5
            elif mix_col == 1:
                sit = a[6]
                otrap = a[4]
            if mode1 == 1 or mode1 == 2:
                de = rst / abs(a[14])
            else:
                de = rst * abs(P[P_DECOMB_MIX_POW]) ** float(-1 - a[1])
            a = (de, b[1], b[2], b[3], otrap, b[5], sit, False, b[8], b[9], b[10], b[11], b[12], b[13], b[14])
        else:
            sm = P[P_DECOMB_SMOOTH]
            if kind == 4:
                res = min(rst - max(sm - de_a, 0.0), de_a - max(sm - rst, 0.0))
            else:
                rd1 = max(sm - de_a - ds, 0.0)
                d2 = max(sm - rst - ds, 0.0)
                res = min(rst - rd1 * (sm - d2) / sm, de_a - d2 * (sm - rd1) / sm)
            r2w = abs(rst)
            w = abs(res)
            f = 1.0 / (w + r2w + 1e-10)
            sit = (b[6] * w + a[6] * r2w) * f
            otrap = (b[4] * w + a[4] * r2w) * f
            if rst < de_a:
                a = (res, b[1], b[2], b[3], otrap, b[5], sit, b_inside, b[8], b[9], b[10], b[11], b[12], b[13], b[14])
            else:
                a = (res, a[1], a[2], a[3], otrap, a[5], sit, a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14])
        last_difs = int(P[P_DE_MODE2]) == 3
    if invert:
        de = 0.0 if a[7] else a[0]
        de = (ds * 2.0 - de) if last_difs else (ds * 4.0 - de * 3.0)
        a = (de, a[1], a[2], a[3], a[4], a[5], a[6], False, a[8], a[9], a[10], a[11], a[12], a[13], a[14])
    return a


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def pixel_ray(px, py, P):
    """Returns (ox, oy, oz, dx, dy, dz)."""
    cafy = (py / P[P_FULL_HEIGHT] - 0.5) * P[P_FOV_Y]
    cafx = (P[P_FOVX_OFF] - (px + 1.0)) * P[P_FOVX_MUL]
    vx = -math.sin(cafx)
    vy = math.sin(cafy)
    vz = math.cos(cafy) * math.cos(cafx)
    n = math.sqrt(vx * vx + vy * vy + vz * vz)
    vx, vy, vz = vx / n, vy / n, vz / n
    ox = P[P_YSTART] + P[P_VGRADS + 0] * px + P[P_VGRADS + 3] * py
    oy = P[P_YSTART + 1] + P[P_VGRADS + 1] * px + P[P_VGRADS + 4] * py
    oz = P[P_YSTART + 2] + P[P_VGRADS + 2] * px + P[P_VGRADS + 5] * py
    dx = P[P_VGRADS + 0] * vx + P[P_VGRADS + 3] * vy + P[P_VGRADS + 6] * vz
    dy = P[P_VGRADS + 1] * vx + P[P_VGRADS + 4] * vy + P[P_VGRADS + 7] * vz
    dz = P[P_VGRADS + 2] * vx + P[P_VGRADS + 5] * vy + P[P_VGRADS + 8] * vz
    return ox, oy, oz, dx, dy, dz


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def de_stop_at(t, P):
    return P[P_DE_STOP] * (1.0 + t * P[P_DE_STOP_FACTOR])


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def pixel_random(px, py):
    v = math.sin(px * 12.9898 + py * 78.233) * 43758.5453
    return v - math.floor(v)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def cut_plane_start(ox, oy, oz, dx, dy, dz, P):
    opts = int(P[P_CUT_OPTIONS])
    best = 0.0
    for k in range(3):
        if (opts & (1 << k)) == 0:
            continue
        dirk = dx if k == 0 else (dy if k == 1 else dz)
        posk = ox if k == 0 else (oy if k == 1 else oz)
        viewk = P[P_VGRADS + 6 + k]
        if abs(dirk) <= 1e-20:
            continue
        if (viewk > 0.0) != (dirk > 0.0):
            return 1e20
        d = (P[P_CUT_POS + k] - posk) / dirk
        if d > best:
            best = d
    return best


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def tangents(nx, ny, nz):
    """CreateXYVecsFromNormals: two tangent vectors (tx, ty) of a normal."""
    d = ny * ny + nx * nx
    if d < 1e-30:
        return 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
    a = math.acos(min(max(-nz / math.sqrt(d + nz * nz + 1e-40), -1.0), 1.0)) * 0.5
    sin_a = math.sin(a) / math.sqrt(d)
    cos_a = math.cos(a)
    dd = -ny * sin_a
    d1 = nx * sin_a
    return (1.0 - 2.0 * d1 * d1, 2.0 * dd * d1, 2.0 * d1 * cos_a,
            2.0 * dd * d1, 1.0 - 2.0 * dd * dd, -2.0 * dd * cos_a)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def de_at(px, py, pz, ds, invert, P):
    return calc_de(px, py, pz, ds, invert, P)[0]


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def surface_normal(px, py, pz, t, invert, P):
    """Returns (nx, ny, nz, roughness) with the normal in world space."""
    sm = int(P[P_SMOOTH_NORMALS])
    sw = P[P_STEP_WIDTH]
    noffset = min(P[P_DE_STOP], 1.0) * (1.0 + t * P[P_DE_STOP_FACTOR]) * 0.15
    ds = de_stop_at(t, P)
    vxx, vxy, vxz = P[P_VGRADS + 0] / sw, P[P_VGRADS + 1] / sw, P[P_VGRADS + 2] / sw
    vyx, vyy, vyz = P[P_VGRADS + 3] / sw, P[P_VGRADS + 4] / sw, P[P_VGRADS + 5] / sw
    vzx, vzy, vzz = P[P_VGRADS + 6] / sw, P[P_VGRADS + 7] / sw, P[P_VGRADS + 8] / sw
    nx = 0.0
    ny = 0.0
    nz = 0.0
    rough = 0.0
    h = noffset * sw
    if sm == 8:
        ssn = noffset * 1.3333 * sw
        for a in range(-2, 3):
            for b in range(-2, 3):
                for c in range(-2, 3):
                    if a == 0 and b == 0 and c == 0:
                        continue
                    qx = px + vzx * (a * ssn) + vyx * (b * ssn) + vxx * (c * ssn)
                    qy = py + vzy * (a * ssn) + vyy * (b * ssn) + vxy * (c * ssn)
                    qz = pz + vzz * (a * ssn) + vyz * (b * ssn) + vxz * (c * ssn)
                    d = de_at(qx, qy, qz, ds, invert, P)
                    if a != 0:
                        nz += d / a
                    if b != 0:
                        ny += d / b
                    if c != 0:
                        nx += d / c
        nx *= 0.0075
        ny *= 0.0075
        nz *= 0.0075
    else:
        nx = (de_at(px + vxx * h, py + vxy * h, pz + vxz * h, ds, invert, P)
              - de_at(px - vxx * h, py - vxy * h, pz - vxz * h, ds, invert, P)) * 0.5
        ny = (de_at(px + vyx * h, py + vyy * h, pz + vyz * h, ds, invert, P)
              - de_at(px - vyx * h, py - vyy * h, pz - vyz * h, ds, invert, P)) * 0.5
        nz = (de_at(px + vzx * h, py + vzy * h, pz + vzz * h, ds, invert, P)
              - de_at(px - vzx * h, py - vzy * h, pz - vzz * h, ds, invert, P)) * 0.5
    if sm > 0:
        noffset *= 2.0
        h = noffset * sw
        dnn = de_at(px, py, pz, ds, invert, P)
        if sm < 8:
            dnn += de_at(px - vxx * h, py - vxy * h, pz - vxz * h, ds, invert, P)
            dnn += de_at(px + vxx * h, py + vxy * h, pz + vxz * h, ds, invert, P)
            dnn += de_at(px - vyx * h, py - vyy * h, pz - vyz * h, ds, invert, P)
            dnn = (dnn + de_at(px + vyx * h, py + vyy * h, pz + vyz * h, ds, invert, P)) * 0.2
        ssn = noffset * 3.0 / (sm + 0.5)
        dm = sm * 2.0
        txx, txy, txz, tyx, tyy, tyz = tangents(nx, ny, nz)
        # the tangents are in view space: to world space
        wtx = vxx * txx + vyx * txy + vzx * txz
        wty = vxy * txx + vyy * txy + vzy * txz
        wtz = vxz * txx + vyz * txy + vzz * txz
        wux = vxx * tyx + vyx * tyy + vzx * tyz
        wuy = vxy * tyx + vyy * tyy + vzy * tyz
        wuz = vxz * tyx + vyz * tyy + vzz * tyz
        nn1 = 0.0
        ds1 = 0.0
        nn2 = 0.0
        ds2 = 0.0
        qx, qy, qz = px - wtx * (sm * ssn * sw), py - wty * (sm * ssn * sw), pz - wtz * (sm * ssn * sw)
        for k in range(-sm, sm + 1):
            if k != 0:
                d = (de_at(qx, qy, qz, ds, invert, P) - dnn) / k
                nn1 += d
                ds1 += d * d
            qx += wtx * (ssn * sw)
            qy += wty * (ssn * sw)
            qz += wtz * (ssn * sw)
        qx, qy, qz = px - wux * (sm * ssn * sw), py - wuy * (sm * ssn * sw), pz - wuz * (sm * ssn * sw)
        for k in range(-sm, sm + 1):
            if k != 0:
                d = (de_at(qx, qy, qz, ds, invert, P) - dnn) / k
                nn2 += d
                ds2 += d * d
            qx += wux * (ssn * sw)
            qy += wuy * (ssn * sw)
            qz += wuz * (ssn * sw)
        dsg = ds1 * dm - nn1 * nn1 + ds2 * dm - nn2 * nn2
        dt2 = noffset * 0.5 / (dm * ssn)
        rough = min(max(math.sqrt(max((dsg * 7.0 * dt2 * dt2 + 1e-40) / (nx * nx + ny * ny + nz * nz + 1e-40), 0.0)) - 0.05, 0.0), 1.0)
        if sm < 8:
            nx += nn1 * dt2
            ny += nn2 * dt2
    n2 = nx * nx + ny * ny + nz * nz
    if n2 < 1e-30:
        return -vzx, -vzy, -vzz, rough
    n = math.sqrt(n2)
    nx, ny, nz = nx / n, ny / n, nz / n
    return (vxx * nx + vyx * ny + vzx * nz, vxy * nx + vyy * ny + vzy * nz, vxz * nx + vyz * ny + vzz * nz, rough)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def de_ambient_occlusion(px, py, pz, nx, ny, nz, t, invert, P):
    samples = int(P[P_AO_SAMPLES])
    radius = P[P_AO_RADIUS] * P[P_STEP_WIDTH] * (1.0 + t * P[P_DE_STOP_FACTOR])
    occ = 0.0
    sca = 1.0
    falloff = P[P_AO_FALLOFF]
    ds = de_stop_at(t, P)
    for i in range(1, samples + 1):
        h = radius * float(i) / float(samples)
        d = calc_de(px + nx * h, py + ny * h, pz + nz * h, ds, invert, P)[0] * P[P_STEP_WIDTH]
        occ += (h - d) / h * sca
        sca *= falloff
    return min(max(1.0 - P[P_AO_STRENGTH] * occ / float(samples) * 2.0, 0.0), 1.0)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def shadow_factor(px, py, pz, nx, ny, nz, lx, ly, lz, t, maxlen, invert, P):
    sw = P[P_STEP_WIDTH]
    ds0 = de_stop_at(t, P)
    tt = ds0 * 2.0
    res = 1.0
    k = P[P_SHADOW_SOFTNESS]
    ox = px + nx * (ds0 * sw * 1.5)
    oy = py + ny * (ds0 * sw * 1.5)
    oz = pz + nz * (ds0 * sw * 1.5)
    for i in range(300):
        if tt > maxlen:
            break
        ds = de_stop_at(t + tt, P)
        de = calc_de(ox + lx * (tt * sw), oy + ly * (tt * sw), oz + lz * (tt * sw), ds, invert, P)[0]
        if de < ds * 0.5:
            return 0.0
        res = min(res, k * de / tt)
        step = max(de * P[P_Z_STEP_DIV] * 1.5, ds)
        tt += step
    return min(max(res, 0.0), 1.0)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def vol_visibility(px, py, pz, t, invert, P):
    li = int(P[P_VOL_LIGHT]) - 1
    if li < 0:
        return 0.0
    sw = P[P_STEP_WIDTH]
    lx = P[P_LIGHT_DIRS + 3 * li]
    ly = P[P_LIGHT_DIRS + 3 * li + 1]
    lz = P[P_LIGHT_DIRS + 3 * li + 2]
    maxlen = P[P_ZEND]
    if P[P_LIGHT_FLAGS + li] > 1.5:
        lx, ly, lz = lx - px, ly - py, lz - pz
        ln = math.sqrt(lx * lx + ly * ly + lz * lz)
        maxlen = ln / sw
        lx, ly, lz = lx / ln, ly / ln, lz / ln
    tt = de_stop_at(t, P)
    for i in range(48):
        if tt > maxlen:
            break
        ds = de_stop_at(t + tt, P)
        de = calc_de(px + lx * (tt * sw), py + ly * (tt * sw), pz + lz * (tt * sw), ds, invert, P)[0]
        if de < ds:
            return 0.0
        tt += max(de * P[P_Z_STEP_DIV] * 2.0, ds)
    return 1.0


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def bin_search(t, px, py, pz, r, dx, dy, dz, last_step, invert, P):
    steps = int(P[P_BIN_SEARCH])
    dt1 = last_step * -0.5
    for i in range(steps):
        t += dt1
        px += dx * dt1
        py += dy * dt1
        pz += dz * dt1
        ds = de_stop_at(t, P)
        r = calc_de(px, py, pz, ds, invert, P)
        if abs(r[0] - ds) <= 0.001:
            break
        if r[7]:
            dt1 = -abs(dt1)
        elif r[0] < ds:
            dt1 = abs(dt1) * -0.55
        else:
            dt1 = abs(dt1) * 0.55
    return t, px, py, pz, r


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def march_ray(ox, oy, oz, dx, dy, dz, t0, invert, pxl, pyl, P):
    """Returns (hit, at_start, t, x, y, z, steps, r, invert, vol) with r the DE result."""
    zend = P[P_ZEND]
    zsd = P[P_Z_STEP_DIV]
    mh04zsd = P[P_MH04ZSD]
    ms_de_sub = P[P_MS_DE_SUB]
    min_it = int(P[P_MIN_IT])
    rsf_mul = 1.0
    steps = 0.0
    first_step = P[P_FIRST_STEP_RANDOM] > 0.5
    t = t0
    px, py, pz = ox, oy, oz
    vol = P[P_VOL_LIGHT] > 0.5
    vol_every = max(1, int(P[P_VOL_STEP]))
    vis = 1.0
    vol_acc = 0.0
    nstep = 0
    if P[P_CUT_OPTIONS] > 0.5 and t0 == 0.0:
        d = cut_plane_start(ox, oy, oz, dx, dy, dz, P)
        if d > zend:
            r0 = calc_de(px, py, pz, 1.0, invert, P)
            return False, False, -1.0, px, py, pz, 0.0, r0, invert, 0.0
        if d > 0.0:
            t = d
            px, py, pz = ox + dx * d, oy + dy * d, oz + dz * d
    if t0 > zend:
        r0 = calc_de(px, py, pz, 1.0, invert, P)
        return False, False, -1.0, px, py, pz, 0.0, r0, invert, 0.0
    ds = de_stop_at(t, P)
    r = calc_de(px, py, pz, ds, invert, P)
    if t0 == 0.0 and int(P[P_INSIDE_MODE]) == 2 and (r[7] or r[0] < ds):
        invert = True
        r = calc_de(px, py, pz, ds, True, P)
    if r[7] or r[0] < ds:
        return True, True, t, px, py, pz, 0.0, r, invert, 0.0
    last_step = r[0] * zsd
    hit = False
    for guard in range(200000):
        if r[7]:
            d = -0.5 * last_step
            t += d
            px += dx * d
            py += dy * d
            pz += dz * d
            ds = de_stop_at(t, P)
            r = calc_de(px, py, pz, ds, invert, P)
            last_step = -d
        if r[1] < min_it or ((not r[7]) and r[0] >= ds):
            last_de = r[0]
            d = max(0.11, (r[0] - ms_de_sub * ds) * zsd * rsf_mul)
            s1 = max(ds, 0.4) * mh04zsd
            if s1 < d:
                steps += s1 / d
                d = s1
            else:
                steps += 1.0
            if first_step:
                first_step = False
                d *= pixel_random(pxl, pyl)
            if vol:
                if nstep % vol_every == 0:
                    vis = vol_visibility(px, py, pz, t, invert, P)
                nstep += 1
                vol_acc += vis * min(d, zend - t)
            t += d
            if t > zend:
                hit = False
                break
            last_step = d
            px += dx * d
            py += dy * d
            pz += dz * d
            ds = de_stop_at(t, P)
            r = calc_de(px, py, pz, ds, invert, P)
            de = r[0]
            if de > last_de + last_step:
                de = last_de + last_step
                r = (de, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14])
            if last_de > de + 1e-30:
                rsf_mul = min(1.0, max(0.5, last_step / (last_de - de)))
            else:
                rsf_mul = 1.0
        else:
            hit = True
            if int(P[P_BIN_SEARCH]) > 0 and ((not r[7]) or r[0] < ds):
                t, px, py, pz, r = bin_search(t, px, py, pz, r, dx, dy, dz, last_step, invert, P)
            break
    if invert:
        steps *= 200.0 * P[P_DE_STOP] / P[P_FULL_WIDTH]
    if not hit:
        return False, False, -1.0, px, py, pz, steps, r, invert, vol_acc
    return True, False, t, px, py, pz, steps, r, invert, vol_acc


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def march(pxl, pyl, P):
    ox, oy, oz, dx, dy, dz = pixel_ray(pxl, pyl, P)
    return march_ray(ox, oy, oz, dx, dy, dz, 0.0, int(P[P_INSIDE_MODE]) == 1, pxl, pyl, P)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def light_shadows(out, row, col, base, px, py, pz, nx, ny, nz, t, invert, P):
    nl = int(P[P_N_LIGHTS])
    for i in range(nl):
        if P[P_LIGHT_FLAGS + i] < 0.5:
            continue
        lx = P[P_LIGHT_DIRS + 3 * i]
        ly = P[P_LIGHT_DIRS + 3 * i + 1]
        lz = P[P_LIGHT_DIRS + 3 * i + 2]
        maxlen = P[P_ZEND]
        if P[P_LIGHT_FLAGS + i] > 1.5:
            lx, ly, lz = lx - px, ly - py, lz - pz
            ln = math.sqrt(lx * lx + ly * ly + lz * lz)
            maxlen = ln / P[P_STEP_WIDTH]
            lx, ly, lz = lx / ln, ly / ln, lz / ln
        if lx * nx + ly * ny + lz * nz <= 0.0:
            continue
        out[row, col, base + i] = shadow_factor(px, py, pz, nx, ny, nz, lx, ly, lz, t, maxlen, invert, P)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def color_index(r, ds, mode, P):
    if int(P[P_DECOMB_KIND]) >= 4:
        nn = r[6]
    else:
        nn = smooth_iterations(r[2], r[3], r[1], r[5], mode, P)
    if (not r[7]) or r[0] < ds:
        v = 32767.0 - (nn + P[P_COL_PLUS] + P[P_COL_VAR] * math.log(max(P[P_DE_STOP], ds) * P[P_STEP_WIDTH])) * P[P_MCTS_M]
    else:
        v = 32767.0 - nn * P[P_MCTS_M]
    if int(P[P_COLOR_OPTION]) == 5:
        rr = math.sqrt(r[8] * r[8] + r[9] * r[9] + r[10] * r[10] + 1e-100)
        v = (math.pi + math.asin(min(max(r[10] / rr, -1.0), 1.0)) * 2.0) * 5215.0
    return clip15(v)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def hash_uniform(x, y, k):
    h = ((x & 0xFFFFFFFF) * 73856093) & 0xFFFFFFFF
    h ^= ((y & 0xFFFFFFFF) * 19349663) & 0xFFFFFFFF
    h ^= ((k & 0xFFFFFFFF) * 83492791) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5bd1e995) & 0xFFFFFFFF
    h ^= h >> 15
    return float(h & 0xFFFFFF) / 16777216.0


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def mc_sky_visibility(px, py, pz, nx, ny, nz, t, pxl, pyl, invert, P):
    samples = int(P[P_MC_SAMPLES])
    seed = int(P[P_MC_SEED])
    sw = P[P_STEP_WIDTH]
    maxlen = P[P_MC_RANGE]
    if abs(ny) < 0.9:
        ax, ay, az = 0.0, 1.0, 0.0
    else:
        ax, ay, az = 1.0, 0.0, 0.0
    txx, txy, txz = _normalize(ny * az - nz * ay, nz * ax - nx * az, nx * ay - ny * ax)
    tyx, tyy, tyz = ny * txz - nz * txy, nz * txx - nx * txz, nx * txy - ny * txx
    ds0 = de_stop_at(t, P)
    ox = px + nx * (ds0 * sw * 1.5)
    oy = py + ny * (ds0 * sw * 1.5)
    oz = pz + nz * (ds0 * sw * 1.5)
    open_ = 0.0
    ipx = int(pxl)
    ipy = int(pyl)
    for i in range(samples):
        u1 = hash_uniform(ipx, ipy, seed * 1024 + i * 2)
        u2 = hash_uniform(ipx, ipy, seed * 1024 + i * 2 + 1)
        r = math.sqrt(u1)
        phi = 2.0 * math.pi * u2
        cz = math.sqrt(max(0.0, 1.0 - u1))
        dx = txx * (r * math.cos(phi)) + tyx * (r * math.sin(phi)) + nx * cz
        dy = txy * (r * math.cos(phi)) + tyy * (r * math.sin(phi)) + ny * cz
        dz = txz * (r * math.cos(phi)) + tyz * (r * math.sin(phi)) + nz * cz
        tt = ds0 * 2.0
        hit = False
        for j in range(200):
            if tt > maxlen:
                break
            ds = de_stop_at(t + tt, P)
            de = calc_de(ox + dx * (tt * sw), oy + dy * (tt * sw), oz + dz * (tt * sw), ds, invert, P)[0]
            if de < ds * 0.5:
                hit = True
                break
            tt += max(de * P[P_Z_STEP_DIV] * 1.5, ds)
        if not hit:
            open_ += 1.0
    return open_ / samples if samples > 0 else 1.0


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def clip15(v):
    return min(max(v, 0.0), 32767.0)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def orbit_trap_value(r, ds, mode, P):
    opt = int(P[P_COLOR_OPTION])
    otrap = r[4]
    zx, zy, zz, cx, cy, cz = r[8], r[9], r[10], r[11], r[12], r[13]
    if mode == 3:
        return clip15(otrap * 4096.0)
    if opt == 1:
        return clip15(math.log(r[2] / (r[3] + 1.0)) * P[P_COLOR_MUL])
    if opt == 2:
        return clip15((math.atan2(zy - cy, zx - cx) + math.pi) * 5200.0)
    if opt == 3:
        return clip15((math.atan2(zz - cz, zx - cx) + math.pi) * 5200.0)
    if opt == 4:
        return clip15((math.atan2(zz - cz, zy - cy) + math.pi) * 5200.0)
    if opt == 5:
        return clip15((math.atan2(zx, zy) + math.pi) * 5215.0)
    return clip15(otrap * 4096.0)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def slice_pixel(pxl, pyl, P, out, row, col):
    ox, oy, oz, dx, dy, dz = pixel_ray(pxl, pyl, P)
    mode = int(P[P_DE_MODE])
    which = int(P[P_SLICE_2D])
    t = 0.0
    if which == 2:
        t = P[P_ZEND] * 0.5
    elif which == 3:
        t = P[P_ZEND]
    px, py, pz = ox + dx * t, oy + dy * t, oz + dz * t
    ds = de_stop_at(t, P)
    r = calc_de(px, py, pz, ds, False, P)
    out[row, col, G_DEPTH] = t
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    out[row, col, G_NX] = -dx / n
    out[row, col, G_NY] = -dy / n
    out[row, col, G_NZ] = -dz / n
    out[row, col, G_ITER] = float(r[1])
    out[row, col, G_OTRAP] = orbit_trap_value(r, ds, mode, P)
    out[row, col, G_COLOR] = color_index(r, ds, mode, P)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def shade_pixel(pxl, pyl, P, out, row, col):
    mode = int(P[P_DE_MODE])
    for k in range(N_CHANNELS):
        out[row, col, k] = 0.0
    out[row, col, G_DEPTH] = -1.0
    out[row, col, G_NZ] = -1.0
    out[row, col, G_AO] = 1.0
    out[row, col, G_R_DEPTH] = -1.0
    out[row, col, G_R_NZ] = -1.0
    out[row, col, G_R_AO] = 1.0
    for i in range(MAX_LIGHTS):
        out[row, col, G_SHADOW + i] = 1.0
        out[row, col, G_R_SHADOW + i] = 1.0
    if int(P[P_SLICE_2D]) > 0:
        slice_pixel(pxl, pyl, P, out, row, col)
        return
    hit, at_start, t, px, py, pz, steps, r, invert, vol_acc = march(pxl, pyl, P)
    out[row, col, G_STEPS] = steps
    out[row, col, G_VOL] = vol_acc
    if not hit:
        return
    out[row, col, G_DEPTH] = t
    ds = de_stop_at(t, P)
    out[row, col, G_ITER] = float(r[1])
    out[row, col, G_OTRAP] = orbit_trap_value(r, ds, mode, P)
    out[row, col, G_COLOR] = color_index(r, ds, mode, P)
    if at_start and P[P_CUT_OPTIONS] < 0.5:
        ox, oy, oz, dx, dy, dz = pixel_ray(pxl, pyl, P)
        n = math.sqrt(dx * dx + dy * dy + dz * dz)
        nx, ny, nz = -dx / n, -dy / n, -dz / n
    else:
        nx, ny, nz, rough = surface_normal(px, py, pz, t, invert, P)
        out[row, col, G_ROUGH] = rough
    out[row, col, G_NX] = nx
    out[row, col, G_NY] = ny
    out[row, col, G_NZ] = nz
    if int(P[P_AO_MODE]) == 2:
        out[row, col, G_AO] = de_ambient_occlusion(px, py, pz, nx, ny, nz, t, invert, P)
    if int(P[P_MC_SAMPLES]) > 0:
        out[row, col, G_AO] = mc_sky_visibility(px, py, pz, nx, ny, nz, t, pxl, pyl, invert, P)
    if P[P_SHADOWS] > 0.5:
        light_shadows(out, row, col, G_SHADOW, px, py, pz, nx, ny, nz, t, invert, P)
    if P[P_REFLECT] > 0.0 and not at_start:
        ox, oy, oz, dx, dy, dz = pixel_ray(pxl, pyl, P)
        dn = math.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / dn, dy / dn, dz / dn
        dot_nd = 2.0 * (nx * dx + ny * dy + nz * dz)
        rx, ry, rz = dx - nx * dot_nd, dy - ny * dot_nd, dz - nz * dot_nd
        sw = P[P_STEP_WIDTH]
        eps = ds * sw * 2.0
        h2 = march_ray(px + nx * eps, py + ny * eps, pz + nz * eps, rx * sw, ry * sw, rz * sw, t + ds * 2.0, invert,
                       pxl, pyl, P)
        out[row, col, G_R_STEPS] = h2[6]
        if h2[0]:
            t2, qx, qy, qz, r2 = h2[2], h2[3], h2[4], h2[5], h2[7]
            out[row, col, G_R_DEPTH] = t2 - t
            ds2 = de_stop_at(t2, P)
            mx, my, mz, _rough = surface_normal(qx, qy, qz, t2, h2[8], P)
            out[row, col, G_R_NX] = mx
            out[row, col, G_R_NY] = my
            out[row, col, G_R_NZ] = mz
            out[row, col, G_R_OTRAP] = orbit_trap_value(r2, ds2, mode, P)
            out[row, col, G_R_COLOR] = color_index(r2, ds2, mode, P)
            if int(P[P_AO_MODE]) == 2:
                out[row, col, G_R_AO] = de_ambient_occlusion(qx, qy, qz, mx, my, mz, t2, h2[8], P)
            if P[P_SHADOWS] > 0.5:
                light_shadows(out, row, col, G_R_SHADOW, qx, qy, qz, mx, my, mz, t2, h2[8], P)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def finish_pixel(pxl, pyl, P, out, row, col):
    """Finishes a pixel marched in single precision elsewhere: from its
    depth the march continues in double (a few steps back), then the
    depth, normal, roughness, colour and orbit trap are replaced.  The
    AO, shadow, step and reflection channels stay."""
    d = out[row, col, G_DEPTH]
    if d < 0.0:
        return
    mode = int(P[P_DE_MODE])
    ox, oy, oz, dx, dy, dz = pixel_ray(pxl, pyl, P)
    ds = de_stop_at(d, P)
    t0 = max(0.0, d - 3.0 * ds)
    invert = int(P[P_INSIDE_MODE]) == 1
    hit, at_start, t, px, py, pz, steps, r, invert, vol_acc = march_ray(
        ox + dx * t0, oy + dy * t0, oz + dz * t0, dx, dy, dz, t0, invert, pxl, pyl, P)
    if not hit:
        out[row, col, G_DEPTH] = -1.0
        return
    out[row, col, G_DEPTH] = t
    ds = de_stop_at(t, P)
    out[row, col, G_ITER] = float(r[1])
    out[row, col, G_OTRAP] = orbit_trap_value(r, ds, mode, P)
    out[row, col, G_COLOR] = color_index(r, ds, mode, P)
    nx, ny, nz, rough = surface_normal(px, py, pz, t, invert, P)
    out[row, col, G_NX] = nx
    out[row, col, G_NY] = ny
    out[row, col, G_NZ] = nz
    out[row, col, G_ROUGH] = rough


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def finish_rows(P, out, y0, y1, abort):
    rx = P[P_RECT_X]
    ry = P[P_RECT_Y]
    step = P[P_PIXEL_STEP]
    w = out.shape[1]
    done = 0
    for y in range(y0, y1):
        if abort[0] != 0:
            break
        for x in range(w):
            finish_pixel(float(x) * step + rx, float(y) * step + ry, P, out, y, x)
        done += 1
    return done


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def render_rows(P, out, y0, y1, abort):
    """Rows y0..y1 (local to the rectangle) into out[y, x, channel].
    ``abort[0] != 0`` stops after the current row; returns the rows done."""
    rx = P[P_RECT_X]
    ry = P[P_RECT_Y]
    step = P[P_PIXEL_STEP]
    w = out.shape[1]
    done = 0
    for y in range(y0, y1):
        if abort[0] != 0:
            break
        for x in range(w):
            shade_pixel(float(x) * step + rx, float(y) * step + ry, P, out, y, x)
        done += 1
    return done


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def de_grid(P, x0, y0, z0, step, out, k0, k1, abort):
    """The distance estimate (world units) on a grid: out[k, j, i] for the
    z slices k0..k1 at (x0 + i step, y0 + j step, z0 + k step)."""
    sw = P[P_STEP_WIDTH]
    ny = out.shape[1]
    nx = out.shape[2]
    invert = int(P[P_INSIDE_MODE]) == 1
    for k in range(k0, k1):
        if abort[0] != 0:
            break
        z = z0 + k * step
        for j in range(ny):
            y = y0 + j * step
            for i in range(nx):
                r = calc_de(x0 + i * step, y, z, 1.0, invert, P)
                out[k, j, i] = -1.0 if r[7] else r[0] * sw


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def formula_test(P, slot, io):
    """Iterates one slot on a buffer of states (the transpiler tests)."""
    for i in range(io.shape[0]):
        zx, zy, zz, w, cx, cy, cz, dr, de, trap, vary = run_slot(
            slot, io[i, 0], io[i, 1], io[i, 2], io[i, 3], io[i, 4], io[i, 5], io[i, 6], io[i, 7], io[i, 8],
            io[i, 9], io[i, 10], io[i, 11], io[i, 12], io[i, 13], P)
        io[i, 0] = zx
        io[i, 1] = zy
        io[i, 2] = zz
        io[i, 3] = w
        io[i, 4] = cx
        io[i, 5] = cy
        io[i, 6] = cz
        io[i, 8] = dr
        io[i, 10] = de
        io[i, 11] = trap
        io[i, 13] = vary
'''
