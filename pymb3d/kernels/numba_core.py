"""The numba ray marching core: the Python twin of ``clike/march_core.c``.

``NUMBA_CORE`` is source text appended to the generated formula module
(see :meth:`Program.py_source`); it uses the generated loops ``iterate``,
``iterate_p2``, ``difs_orbit``, ``difs_orbit_p2`` and the ``P_*`` / ``G_*``
constants defined there.  Keep the functions in step with the C core.

A DE result is the tuple (de, it, r2, rold, otrap, slot, sit, inside,
zx, zy, zz, cx, cy, cz).
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
    """The DE of one hybrid part: (de, it, r2, rold, otrap, slot, sit, inside, zx, zy, zz, cx, cy, cz)."""
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
    return de, it, r2, rold, otrap, slot, sit, it >= inside_it, zx, zy, zz, cx, cy, cz


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def calc_de(px, py, pz, ds, P):
    a = de_part(px, py, pz, ds, 0, P)
    kind = int(P[P_DECOMB_KIND])
    if kind == 0:
        return a
    b = de_part(px, py, pz, ds, 1, P)
    de_a = a[0]
    rst = b[0]
    b_inside = b[7]
    if kind == 3:
        rst = ds * 4.0 - rst * 3.0
        b_inside = False
    if kind == 1:
        if rst < de_a:
            return (rst, b[1], b[2], b[3], b[4], b[5], b[6], b_inside, b[8], b[9], b[10], b[11], b[12], b[13])
        return a
    if kind == 2 or kind == 3:
        if rst > de_a:
            return (rst, b[1], b[2], b[3], b[4], b[5], b[6], b_inside, b[8], b[9], b[10], b[11], b[12], b[13])
        return a
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
        return (res, b[1], b[2], b[3], otrap, b[5], sit, b_inside, b[8], b[9], b[10], b[11], b[12], b[13])
    return (res, a[1], a[2], a[3], otrap, a[5], sit, a[7], a[8], a[9], a[10], a[11], a[12], a[13])


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
def shadow_factor(px, py, pz, nx, ny, nz, lx, ly, lz, t, maxlen, P):
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
        de = calc_de(ox + lx * (tt * sw), oy + ly * (tt * sw), oz + lz * (tt * sw), ds, P)[0]
        if de < ds * 0.5:
            return 0.0
        res = min(res, k * de / tt)
        step = max(de * P[P_Z_STEP_DIV] * 1.5, ds)
        tt += step
    return min(max(res, 0.0), 1.0)


@njit(cache=True, nogil=True, fastmath=False, error_model="numpy")
def bin_search(t, px, py, pz, r, dx, dy, dz, last_step, P):
    steps = int(P[P_BIN_SEARCH])
    dt1 = last_step * -0.5
    for i in range(steps):
        t += dt1
        px += dx * dt1
        py += dy * dt1
        pz += dz * dt1
        ds = de_stop_at(t, P)
        r = calc_de(px, py, pz, ds, P)
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
def march(pxl, pyl, P):
    """Returns (hit, at_start, t, x, y, z, steps, r) with r the DE result."""
    ox, oy, oz, dx, dy, dz = pixel_ray(pxl, pyl, P)
    zend = P[P_ZEND]
    zsd = P[P_Z_STEP_DIV]
    mh04zsd = P[P_MH04ZSD]
    ms_de_sub = P[P_MS_DE_SUB]
    min_it = int(P[P_MIN_IT])
    rsf_mul = 1.0
    steps = 0.0
    first_step = P[P_FIRST_STEP_RANDOM] > 0.5
    t = 0.0
    px, py, pz = ox, oy, oz
    if P[P_CUT_OPTIONS] > 0.5:
        d = cut_plane_start(ox, oy, oz, dx, dy, dz, P)
        if d > zend:
            r0 = calc_de(px, py, pz, 1.0, P)
            return False, False, -1.0, px, py, pz, 0.0, r0
        if d > 0.0:
            t = d
            px, py, pz = ox + dx * d, oy + dy * d, oz + dz * d
    ds = de_stop_at(t, P)
    r = calc_de(px, py, pz, ds, P)
    if r[7] or r[0] < ds:
        return True, True, t, px, py, pz, 0.0, r
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
            r = calc_de(px, py, pz, ds, P)
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
            t += d
            if t > zend:
                hit = False
                break
            last_step = d
            px += dx * d
            py += dy * d
            pz += dz * d
            ds = de_stop_at(t, P)
            r = calc_de(px, py, pz, ds, P)
            de = r[0]
            if de > last_de + last_step:
                de = last_de + last_step
                r = (de, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13])
            if last_de > de + 1e-30:
                rsf_mul = min(1.0, max(0.5, last_step / (last_de - de)))
            else:
                rsf_mul = 1.0
        else:
            hit = True
            if int(P[P_BIN_SEARCH]) > 0 and ((not r[7]) or r[0] < ds):
                t, px, py, pz, r = bin_search(t, px, py, pz, r, dx, dy, dz, last_step, P)
            break
    if not hit:
        return False, False, -1.0, px, py, pz, steps, r
    return True, False, t, px, py, pz, steps, r


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
def shade_pixel(pxl, pyl, P, out, row, col):
    hit, at_start, t, px, py, pz, steps, r = march(pxl, pyl, P)
    mode = int(P[P_DE_MODE])
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
    if int(P[P_DECOMB_KIND]) >= 4:
        nn = r[6]
    else:
        nn = smooth_iterations(r[2], r[3], r[1], r[5], mode, P)
    out[row, col, G_ITER] = float(r[1])
    out[row, col, G_DE] = r[0]
    out[row, col, G_OTRAP] = orbit_trap_value(r, ds, mode, P)
    if (not r[7]) or r[0] < ds:
        v = 32767.0 - (nn + P[P_COL_PLUS] + P[P_COL_VAR] * math.log(max(P[P_DE_STOP], ds) * P[P_STEP_WIDTH])) * P[P_MCTS_M]
    else:
        v = 32767.0 - nn * P[P_MCTS_M]
    if int(P[P_COLOR_OPTION]) == 5:
        rr = math.sqrt(r[8] * r[8] + r[9] * r[9] + r[10] * r[10] + 1e-100)
        v = (math.pi + math.asin(min(max(r[10] / rr, -1.0), 1.0)) * 2.0) * 5215.0
    out[row, col, G_COLOR] = clip15(v)
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
            maxlen = P[P_ZEND]
            if P[P_LIGHT_FLAGS + i] > 1.5:
                lx, ly, lz = lx - px, ly - py, lz - pz
                ln = math.sqrt(lx * lx + ly * ly + lz * lz)
                maxlen = ln / P[P_STEP_WIDTH]
                lx, ly, lz = lx / ln, ly / ln, lz / ln
            if lx * nx + ly * ny + lz * nz <= 0.0:
                continue
            out[row, col, G_SHADOW + i] = shadow_factor(px, py, pz, nx, ny, nz, lx, ly, lz, t, maxlen, P)


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
