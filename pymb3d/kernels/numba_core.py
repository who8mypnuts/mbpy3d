"""The numba ray marching core: the Python twin of ``clike/march_core.c``.

``NUMBA_CORE`` is source text appended to the generated formula module
(see :meth:`Program.py_source`); it uses ``run_slot`` and the ``P_*`` /
``G_*`` constants defined there.  Keep the functions in step with the C
core: same names, same order of operations.
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
    return zx, zy, zz, w, cx, cy, cz, cw, 1.0, r2


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def iterate(px, py, pz, max_it, rstop2, P):
    """The alternating hybrid loop.  Returns (r2, dr, rold, otrap, it, slot)."""
    zx, zy, zz, w, cx, cy, cz, cw, dr, r2 = orbit_init(px, py, pz, P)
    rold = r2
    otrap = r2
    it = 0
    slot = 0
    nslots = int(P[P_NSLOTS])
    repeat_from = int(P[P_REPEAT_FROM])
    four_d = P[P_FOUR_D] > 0.5
    n = 0
    btmp = int(P[P_SLOTS])
    if nslots < 1:
        return r2, dr, rold, otrap, it, slot
    total = 0
    while total < 100000:
        total += 1
        rold = r2
        guard = 0
        while btmp <= 0:
            n += 1
            if n >= nslots:
                n = repeat_from
            btmp = int(P[P_SLOTS + SLOT_STRIDE * n])
            guard += 1
            if guard > 64:
                return r2, dr, rold, otrap, it, slot
        zx, zy, zz, w, cx, cy, cz, dr = run_slot(n, zx, zy, zz, w, cx, cy, cz, cw, dr, r2, P)
        slot = n
        btmp -= 1
        if P[P_SLOTS + SLOT_STRIDE * n + 1] > 0.5:
            continue
        it += 1
        r2 = zx * zx + zy * zy + zz * zz
        if four_d:
            r2 += w * w
        if r2 < otrap:
            otrap = r2
        if it >= max_it or r2 > rstop2:
            return r2, dr, rold, otrap, it, slot
        if math.isnan(r2):
            return r2, dr, rold, otrap, it, slot
    return r2, dr, rold, otrap, it, slot


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def smooth_iterations(r2, rold, it, slot, P):
    fhln = P[P_SLOTS + SLOT_STRIDE * slot + 2]
    ln_rstop = P[P_LN_RSTOP]
    if r2 <= 1.0:
        return float(it)
    if rold < 1.0:
        d = math.log(0.5 * math.log(r2)) * fhln
        return float(it) + ln_rstop - d
    d = math.log(0.5 * math.log(r2))
    e = d - math.log(0.5 * math.log(rold))
    return (ln_rstop - d) / (e + 1e-40) + float(it)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def de_numeric(px, py, pz, it, r2_0, P):
    if r2_0 < 1e-30:
        return 0.0
    off = P[P_DE_OFFSET_PX] * P[P_STEP_WIDTH]
    g = 0.0
    r2k, _a, _b, _c, _d, _e = iterate(px + off, py, pz, it, 1e30, P)
    d = r2_0 - r2k
    g += d * d
    r2k, _a, _b, _c, _d, _e = iterate(px, py + off, pz, it, 1e30, P)
    d = r2_0 - r2k
    g += d * d
    r2k, _a, _b, _c, _d, _e = iterate(px, py, pz + off, it, 1e30, P)
    d = r2_0 - r2k
    g += d * d
    return r2_0 * math.log(r2_0) * P[P_DE_SCALE] * P[P_DE_OFFSET_PX] / (math.sqrt(g) + P[P_DE_OFFSET_PX] * 0.06)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def calc_de(px, py, pz, de_stop_here, P):
    """Returns (de, it, r2, rold, otrap, slot)."""
    max_it = int(P[P_MAX_IT])
    mode = int(P[P_DE_MODE])
    r2, dr, rold, otrap, it, slot = iterate(px, py, pz, max_it, P[P_RSTOP2], P)
    if it >= max_it:
        de = 0.0
    elif mode == 1:
        de = math.sqrt(r2) / abs(dr) * P[P_DE_SCALE] / P[P_STEP_WIDTH]
    elif mode == 2:
        rr = math.sqrt(r2)
        de = 0.5 * math.log(r2) * rr / abs(dr) * P[P_DE_SCALE] / P[P_STEP_WIDTH]
    else:
        de = de_numeric(px, py, pz, it, r2, P)
    if not (de >= de_stop_here * 0.25):
        de = de_stop_here * 0.25
    return de, it, r2, rold, otrap, slot


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
def surface_normal(px, py, pz, t, P):
    h = min(P[P_DE_STOP], 1.0) * (1.0 + t * P[P_DE_STOP_FACTOR]) * 0.15 * P[P_STEP_WIDTH]
    ds = de_stop_at(t, P)
    nx = calc_de(px + h, py, pz, ds, P)[0] - calc_de(px - h, py, pz, ds, P)[0]
    ny = calc_de(px, py + h, pz, ds, P)[0] - calc_de(px, py - h, pz, ds, P)[0]
    nz = calc_de(px, py, pz + h, ds, P)[0] - calc_de(px, py, pz - h, ds, P)[0]
    n2 = nx * nx + ny * ny + nz * nz
    if n2 < 1e-30:
        return 0.0, 0.0, -1.0
    n = math.sqrt(n2)
    return nx / n, ny / n, nz / n


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def de_ambient_occlusion(px, py, pz, nx, ny, nz, t, P):
    samples = int(P[P_AO_SAMPLES])
    radius = P[P_AO_RADIUS] * P[P_STEP_WIDTH] * (1.0 + t * P[P_DE_STOP_FACTOR])
    occ = 0.0
    sca = 1.0
    falloff = P[P_AO_FALLOFF]
    ds = de_stop_at(t, P)
    for i in range(1, samples + 1):
        h = radius * float(i) / float(samples)
        d = calc_de(px + nx * h, py + ny * h, pz + nz * h, ds, P)[0] * P[P_STEP_WIDTH]
        occ += (h - d) / h * sca
        sca *= falloff
    return min(max(1.0 - P[P_AO_STRENGTH] * occ / float(samples) * 2.0, 0.0), 1.0)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def shadow_factor(px, py, pz, nx, ny, nz, lx, ly, lz, t, P):
    sw = P[P_STEP_WIDTH]
    ds0 = de_stop_at(t, P)
    tt = ds0 * 2.0
    maxlen = P[P_ZEND]
    res = 1.0
    k = P[P_SHADOW_SOFTNESS]
    ox = px + nx * (ds0 * sw * 1.5)
    oy = py + ny * (ds0 * sw * 1.5)
    oz = pz + nz * (ds0 * sw * 1.5)
    for i in range(400):
        if tt > maxlen:
            break
        ds = de_stop_at(t + tt, P)
        de = calc_de(ox + lx * (tt * sw), oy + ly * (tt * sw), oz + lz * (tt * sw), ds, P)[0]
        if de < ds * 0.5:
            return 0.0
        res = min(res, k * de / tt)
        step = max(de * P[P_Z_STEP_DIV] * 1.5, ds)
        tt += step
    return min(max(res, 0.0), 1.0)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def bin_search(t, px, py, pz, de, it, r2, rold, otrap, slot, dx, dy, dz, last_step, P):
    steps = int(P[P_BIN_SEARCH])
    max_it = int(P[P_MAX_IT])
    dt1 = last_step * -0.5
    for i in range(steps):
        t += dt1
        px += dx * dt1
        py += dy * dt1
        pz += dz * dt1
        ds = de_stop_at(t, P)
        de, it, r2, rold, otrap, slot = calc_de(px, py, pz, ds, P)
        if abs(de - ds) <= 0.001:
            break
        if it >= max_it:
            dt1 = -abs(dt1)
        elif de < ds:
            dt1 = abs(dt1) * -0.55
        else:
            dt1 = abs(dt1) * 0.55
    return t, px, py, pz, de, it, r2, rold, otrap, slot


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def march(pxl, pyl, P):
    """Returns (hit, at_start, t, x, y, z, de, steps, it, r2, rold, otrap, slot)."""
    ox, oy, oz, dx, dy, dz = pixel_ray(pxl, pyl, P)
    zend = P[P_ZEND]
    zsd = P[P_Z_STEP_DIV]
    mh04zsd = P[P_MH04ZSD]
    ms_de_sub = P[P_MS_DE_SUB]
    max_it = int(P[P_MAX_IT])
    min_it = int(P[P_MIN_IT])
    rsf_mul = 1.0
    steps = 0.0
    first_step = P[P_FIRST_STEP_RANDOM] > 0.5
    t = 0.0
    px, py, pz = ox, oy, oz
    if P[P_CUT_OPTIONS] > 0.5:
        d = cut_plane_start(ox, oy, oz, dx, dy, dz, P)
        if d > zend:
            return False, False, -1.0, px, py, pz, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0
        if d > 0.0:
            t = d
            px, py, pz = ox + dx * d, oy + dy * d, oz + dz * d
    ds = de_stop_at(t, P)
    de, it, r2, rold, otrap, slot = calc_de(px, py, pz, ds, P)
    if it >= max_it or de < ds:
        return True, True, t, px, py, pz, de, 0.0, it, r2, rold, otrap, slot
    last_step = de * zsd
    hit = False
    for guard in range(200000):
        if it >= max_it:
            d = -0.5 * last_step
            t += d
            px += dx * d
            py += dy * d
            pz += dz * d
            ds = de_stop_at(t, P)
            de, it, r2, rold, otrap, slot = calc_de(px, py, pz, ds, P)
            last_step = -d
        if it < min_it or (it < max_it and de >= ds):
            last_de = de
            d = max(0.11, (de - ms_de_sub * ds) * zsd * rsf_mul)
            s1 = max(ds, 0.4) * mh04zsd
            if s1 < d:
                steps += s1 / d
                d = s1
            else:
                steps += 1.0
            if first_step:
                first_step = False
                d *= pixel_random(pxl, pyl)
            t += d
            if t > zend:
                hit = False
                break
            last_step = d
            px += dx * d
            py += dy * d
            pz += dz * d
            ds = de_stop_at(t, P)
            de, it, r2, rold, otrap, slot = calc_de(px, py, pz, ds, P)
            if de > last_de + last_step:
                de = last_de + last_step
            if last_de > de + 1e-30:
                rsf_mul = min(1.0, max(0.5, last_step / (last_de - de)))
            else:
                rsf_mul = 1.0
        else:
            hit = True
            if int(P[P_BIN_SEARCH]) > 0 and (it < max_it or de < ds):
                t, px, py, pz, de, it, r2, rold, otrap, slot = bin_search(
                    t, px, py, pz, de, it, r2, rold, otrap, slot, dx, dy, dz, last_step, P)
            break
    if not hit:
        return False, False, -1.0, px, py, pz, de, steps, it, r2, rold, otrap, slot
    return True, False, t, px, py, pz, de, steps, it, r2, rold, otrap, slot


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def shade_pixel(pxl, pyl, P, out, row, col):
    hit, at_start, t, px, py, pz, de, steps, it, r2, rold, otrap, slot = march(pxl, pyl, P)
    for k in range(N_CHANNELS):
        out[row, col, k] = 0.0
    out[row, col, G_DEPTH] = -1.0
    out[row, col, G_NZ] = -1.0
    out[row, col, G_STEPS] = steps
    out[row, col, G_AO] = 1.0
    for i in range(MAX_LIGHTS):
        out[row, col, G_SHADOW + i] = 1.0
    if not hit:
        return
    out[row, col, G_DEPTH] = t
    ds = de_stop_at(t, P)
    sit = smooth_iterations(r2, rold, it, slot, P)
    out[row, col, G_ITER] = float(it)
    out[row, col, G_DE] = de
    out[row, col, G_OTRAP] = otrap
    if it < int(P[P_MAX_IT]) or de < ds:
        out[row, col, G_COLOR] = sit + P[P_COL_PLUS] + P[P_COL_VAR] * math.log(max(P[P_DE_STOP], ds) * P[P_STEP_WIDTH])
    else:
        out[row, col, G_COLOR] = sit
    if at_start and P[P_CUT_OPTIONS] < 0.5:
        ox, oy, oz, dx, dy, dz = pixel_ray(pxl, pyl, P)
        n = math.sqrt(dx * dx + dy * dy + dz * dz)
        nx, ny, nz = -dx / n, -dy / n, -dz / n
    else:
        nx, ny, nz = surface_normal(px, py, pz, t, P)
    out[row, col, G_NX] = nx
    out[row, col, G_NY] = ny
    out[row, col, G_NZ] = nz
    if int(P[P_AO_MODE]) == 2:
        out[row, col, G_AO] = de_ambient_occlusion(px, py, pz, nx, ny, nz, t, P)
    if P[P_SHADOWS] > 0.5:
        nl = int(P[P_N_LIGHTS])
        for i in range(nl):
            if P[P_LIGHT_FLAGS + i] < 0.5:
                continue
            lx = P[P_LIGHT_DIRS + 3 * i]
            ly = P[P_LIGHT_DIRS + 3 * i + 1]
            lz = P[P_LIGHT_DIRS + 3 * i + 2]
            if lx * nx + ly * ny + lz * nz <= 0.0:
                continue
            out[row, col, G_SHADOW + i] = shadow_factor(px, py, pz, nx, ny, nz, lx, ly, lz, t, P)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def render_rows(P, out, y0, y1, abort):
    """Rows y0..y1 (local to the rectangle) into out[y, x, channel].
    ``abort[0] != 0`` stops after the current row; returns the rows done."""
    rx = P[P_RECT_X]
    ry = P[P_RECT_Y]
    w = out.shape[1]
    done = 0
    for y in range(y0, y1):
        if abort[0] != 0:
            break
        for x in range(w):
            shade_pixel(float(x) + rx, float(y) + ry, P, out, y, x)
        done += 1
    return done


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def formula_test(P, slot, io):
    """Iterates one slot on a buffer of states (the transpiler tests)."""
    for i in range(io.shape[0]):
        zx, zy, zz, w, cx, cy, cz, dr = run_slot(slot, io[i, 0], io[i, 1], io[i, 2], io[i, 3], io[i, 4], io[i, 5],
                                                 io[i, 6], io[i, 7], io[i, 8], io[i, 9], P)
        io[i, 0] = zx
        io[i, 1] = zy
        io[i, 2] = zz
        io[i, 3] = w
        io[i, 4] = cx
        io[i, 5] = cy
        io[i, 6] = cz
        io[i, 8] = dr
'''
