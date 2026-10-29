/*
 * The ray marching core, shared by the OpenGL (GLSL) and the OpenCL backends.
 *
 * Written in the common subset of GLSL and OpenCL C: no pointers, structs
 * passed and returned by value, the parameter block reached through
 * PARAM(i) and the hidden FN_ARGS / FN_PASS function parameter (see the
 * preludes).  The numba kernel (kernels/numba_core.py) follows the same
 * structure function by function.
 *
 * Pieces that precede this file: the prelude of the target, the #defines
 * of core/params.py, the generated formulas with run_slot().
 */

/* ------------------------------------------------------------------ orbit */

Orbit orbit_init(FN_ARGS real3 pos) {
    Orbit o;
    State s;
    real4 p4;
    s.z = pos;
    s.w = 0.0;
    if (PARAM(P_FOUR_D) > 0.5) {
        /* the 4D rotation of the start point (Rotate4Dex) */
        p4.x = pos.x * PARAM(P_ROT4D + 0) + pos.y * PARAM(P_ROT4D + 1) + pos.z * PARAM(P_ROT4D + 2);
        p4.y = pos.x * PARAM(P_ROT4D + 4) + pos.y * PARAM(P_ROT4D + 5) + pos.z * PARAM(P_ROT4D + 6);
        p4.z = pos.x * PARAM(P_ROT4D + 8) + pos.y * PARAM(P_ROT4D + 9) + pos.z * PARAM(P_ROT4D + 10);
        p4.w = pos.x * PARAM(P_ROT4D + 12) + pos.y * PARAM(P_ROT4D + 13) + pos.z * PARAM(P_ROT4D + 14);
        s.z = V3(p4.x, p4.y, p4.z);
        s.w = p4.w;
    }
    if (PARAM(P_JULIA) > 0.5) {
        s.c = V3(PARAM(P_JULIA_C), PARAM(P_JULIA_C + 1), PARAM(P_JULIA_C + 2));
        s.cw = PARAM(P_JULIA_C + 3);
    } else {
        s.c = s.z;
        s.cw = s.w;
    }
    s.dr = 1.0;
    s.r2 = dot(s.z, s.z) + s.w * s.w;
    o.s = s;
    o.rold = s.r2;
    o.otrap = s.r2;
    o.it = 0;
    o.slot = 0;
    return o;
}

/* The alternating hybrid loop (doHybridPas): slot n runs its iteration
 * count, then the next slot, after the last one the loop continues from
 * repeat_from.  Uncounted slots (transforms) do not count as iterations. */
Orbit iterate(FN_ARGS real3 pos, int max_it, real rstop2) {
    Orbit o = orbit_init(FN_PASS pos);
    int nslots = TO_INT(PARAM(P_NSLOTS));
    int repeat_from = TO_INT(PARAM(P_REPEAT_FROM));
    bool four_d = PARAM(P_FOUR_D) > 0.5;
    int n = 0;
    int btmp = TO_INT(PARAM(P_SLOTS + SLOT_STRIDE * 0));
    int guard;
    int total = 0;
    if (nslots < 1) return o;
    while (total < 100000) {
        total += 1;
        o.rold = o.s.r2;
        guard = 0;
        while (btmp <= 0) {
            n += 1;
            if (n >= nslots) n = repeat_from;
            btmp = TO_INT(PARAM(P_SLOTS + SLOT_STRIDE * n));
            guard += 1;
            if (guard > 64) return o;
        }
        o.s = run_slot(FN_PASS n, o.s);
        o.slot = n;
        btmp -= 1;
        if (PARAM(P_SLOTS + SLOT_STRIDE * n + 1) > 0.5) continue; /* uncounted */
        o.it += 1;
        o.s.r2 = dot(o.s.z, o.s.z) + (four_d ? o.s.w * o.s.w : 0.0);
        if (o.s.r2 < o.otrap) o.otrap = o.s.r2;
        if (o.it >= max_it || o.s.r2 > rstop2) return o;
        if (isnan(o.s.r2)) return o;
    }
    return o;
}

/* CalcSmoothIterations */
real smooth_iterations(FN_ARGS Orbit o) {
    real r2 = o.s.r2;
    real fhln = PARAM(P_SLOTS + SLOT_STRIDE * o.slot + 2);
    real ln_rstop = PARAM(P_LN_RSTOP);
    real d;
    real e;
    if (r2 <= 1.0) return TO_REAL(o.it);
    if (o.rold < 1.0) {
        d = log(0.5 * log(r2)) * fhln;
        return TO_REAL(o.it) + ln_rstop - d;
    }
    d = log(0.5 * log(r2));
    e = d - log(0.5 * log(o.rold));
    return (ln_rstop - d) / (e + TINY) + TO_REAL(o.it);
}

/* ------------------------------------------------------------ distance */

/* Numerical gradient DE (CalcDEnoADE): the change of |z|^2 after the same
 * number of iterations from three offset points. */
real de_numeric(FN_ARGS real3 pos, Orbit o) {
    real r2_0 = o.s.r2;
    real off = PARAM(P_DE_OFFSET_PX) * PARAM(P_STEP_WIDTH);
    real g = 0.0;
    real d;
    Orbit ok;
    if (r2_0 < 1e-30) return 0.0;
    ok = iterate(FN_PASS pos + V3(off, 0.0, 0.0), o.it, 1e30);
    d = r2_0 - ok.s.r2;
    g += d * d;
    ok = iterate(FN_PASS pos + V3(0.0, off, 0.0), o.it, 1e30);
    d = r2_0 - ok.s.r2;
    g += d * d;
    ok = iterate(FN_PASS pos + V3(0.0, 0.0, off), o.it, 1e30);
    d = r2_0 - ok.s.r2;
    g += d * d;
    return r2_0 * log(r2_0) * PARAM(P_DE_SCALE) * PARAM(P_DE_OFFSET_PX) / (sqrt(g) + PARAM(P_DE_OFFSET_PX) * 0.06);
}

/* The distance estimate at pos in pixel units, with the orbit it needed. */
DeResult calc_de(FN_ARGS real3 pos, real de_stop_here) {
    DeResult r;
    int max_it = TO_INT(PARAM(P_MAX_IT));
    int mode = TO_INT(PARAM(P_DE_MODE));
    real de;
    real rr;
    r.o = iterate(FN_PASS pos, max_it, PARAM(P_RSTOP2));
    if (r.o.it >= max_it) {
        de = 0.0;
    } else if (mode == 1) {
        de = sqrt(r.o.s.r2) / fabs(r.o.s.dr) * PARAM(P_DE_SCALE) / PARAM(P_STEP_WIDTH);
    } else if (mode == 2) {
        rr = sqrt(r.o.s.r2);
        de = 0.5 * log(r.o.s.r2) * rr / fabs(r.o.s.dr) * PARAM(P_DE_SCALE) / PARAM(P_STEP_WIDTH);
    } else {
        de = de_numeric(FN_PASS pos, r.o);
    }
    if (!(de >= de_stop_here * 0.25)) de = de_stop_here * 0.25; /* also NaN */
    r.de = de;
    return r;
}

/* --------------------------------------------------------------- camera */

Ray pixel_ray(FN_ARGS real px, real py) {
    Ray r;
    real cafy = (py / PARAM(P_FULL_HEIGHT) - 0.5) * PARAM(P_FOV_Y);
    real cafx = (PARAM(P_FOVX_OFF) - (px + 1.0)) * PARAM(P_FOVX_MUL);
    real3 v = V3(-sin(cafx), sin(cafy), cos(cafy) * cos(cafx));
    real3 vg0 = V3(PARAM(P_VGRADS + 0), PARAM(P_VGRADS + 1), PARAM(P_VGRADS + 2));
    real3 vg1 = V3(PARAM(P_VGRADS + 3), PARAM(P_VGRADS + 4), PARAM(P_VGRADS + 5));
    real3 vg2 = V3(PARAM(P_VGRADS + 6), PARAM(P_VGRADS + 7), PARAM(P_VGRADS + 8));
    v = normalize(v);
    r.origin = V3(PARAM(P_YSTART), PARAM(P_YSTART + 1), PARAM(P_YSTART + 2)) + vg0 * px + vg1 * py;
    r.dir = vg0 * v.x + vg1 * v.y + vg2 * v.z; /* length step_width: one pixel unit */
    return r;
}

real de_stop_at(FN_ARGS real t) {
    return PARAM(P_DE_STOP) * (1.0 + t * PARAM(P_DE_STOP_FACTOR));
}

/* Pseudo random 0..1 from the pixel position (the "first step random"). */
real pixel_random(real px, real py) {
    real v = sin(px * 12.9898 + py * 78.233) * 43758.5453;
    return v - floor(v);
}

/* RMmaxLengthToCutPlane: the cutting planes (bits 0..2 of cut_options: x,
 * y, z) cut away the side the camera looks from.  Returns the distance in
 * pixel units to the farthest plane in front of the start, or a value
 * beyond the far end when the ray never reaches the visible side. */
real cut_plane_start(FN_ARGS Ray ray) {
    int opts = TO_INT(PARAM(P_CUT_OPTIONS));
    real best = 0.0;
    real d;
    real dirk;
    real posk;
    real viewk;
    int k;
    for (k = 0; k < 3; k += 1) {
        if ((opts & (1 << k)) == 0) continue;
        dirk = (k == 0) ? ray.dir.x : ((k == 1) ? ray.dir.y : ray.dir.z);
        posk = (k == 0) ? ray.origin.x : ((k == 1) ? ray.origin.y : ray.origin.z);
        viewk = PARAM(P_VGRADS + 6 + k);
        if (fabs(dirk) <= 1e-20) continue;
        if ((viewk > 0.0) != (dirk > 0.0)) return 1e20;
        d = (PARAM(P_CUT_POS + k) - posk) / dirk;
        if (d > best) best = d;
    }
    return best;
}

/* ---------------------------------------------------------------- normal */

real3 surface_normal(FN_ARGS real3 pos, real t) {
    real h = fmin(PARAM(P_DE_STOP), 1.0) * (1.0 + t * PARAM(P_DE_STOP_FACTOR)) * 0.15 * PARAM(P_STEP_WIDTH);
    real ds = de_stop_at(FN_PASS t);
    real3 n;
    n.x = calc_de(FN_PASS pos + V3(h, 0.0, 0.0), ds).de - calc_de(FN_PASS pos - V3(h, 0.0, 0.0), ds).de;
    n.y = calc_de(FN_PASS pos + V3(0.0, h, 0.0), ds).de - calc_de(FN_PASS pos - V3(0.0, h, 0.0), ds).de;
    n.z = calc_de(FN_PASS pos + V3(0.0, 0.0, h), ds).de - calc_de(FN_PASS pos - V3(0.0, 0.0, h), ds).de;
    if (dot(n, n) < 1e-30) return V3(0.0, 0.0, -1.0);
    return normalize(n);
}

/* ------------------------------------------------------------------ AO */

/* Distance estimation ambient occlusion along the normal. */
real de_ambient_occlusion(FN_ARGS real3 pos, real3 n, real t) {
    int samples = TO_INT(PARAM(P_AO_SAMPLES));
    real radius = PARAM(P_AO_RADIUS) * PARAM(P_STEP_WIDTH) * (1.0 + t * PARAM(P_DE_STOP_FACTOR));
    real occ = 0.0;
    real sca = 1.0;
    real falloff = PARAM(P_AO_FALLOFF);
    real h;
    real d;
    int i;
    for (i = 1; i <= samples; i += 1) {
        h = radius * TO_REAL(i) / TO_REAL(samples);
        d = calc_de(FN_PASS pos + n * h, de_stop_at(FN_PASS t)).de * PARAM(P_STEP_WIDTH);
        occ += (h - d) / h * sca;
        sca *= falloff;
    }
    return clamp(1.0 - PARAM(P_AO_STRENGTH) * occ / TO_REAL(samples) * 2.0, 0.0, 1.0);
}

/* ---------------------------------------------------------------- shadow */

/* Soft shadow factor (1 = lit) towards a light direction (world space). */
real shadow_factor(FN_ARGS real3 pos, real3 n, real3 ldir, real t) {
    real sw = PARAM(P_STEP_WIDTH);
    real ds0 = de_stop_at(FN_PASS t);
    real tt = ds0 * 2.0;          /* pixel units along the light ray */
    real maxlen = PARAM(P_ZEND);
    real res = 1.0;
    real k = PARAM(P_SHADOW_SOFTNESS);
    real de;
    real ds;
    real step;
    int i;
    real3 p0 = pos + n * (ds0 * sw * 1.5);
    for (i = 0; i < 400; i += 1) {
        if (tt > maxlen) break;
        ds = de_stop_at(FN_PASS t + tt);
        de = calc_de(FN_PASS p0 + ldir * (tt * sw), ds).de;
        if (de < ds * 0.5) return 0.0;
        res = fmin(res, k * de / tt);
        step = fmax(de * PARAM(P_Z_STEP_DIV) * 1.5, ds);
        tt += step;
    }
    return clamp(res, 0.0, 1.0);
}

/* ---------------------------------------------------------------- march */

/* RMdoBinSearch: refine the hit between the last two positions. */
Hit bin_search(FN_ARGS Hit h, real3 dir, real last_step) {
    int steps = TO_INT(PARAM(P_BIN_SEARCH));
    real dt1 = last_step * -0.5;
    real ds;
    DeResult r;
    int i;
    for (i = 0; i < steps; i += 1) {
        h.t += dt1;
        h.pos += dir * dt1;
        ds = de_stop_at(FN_PASS h.t);
        r = calc_de(FN_PASS h.pos, ds);
        h.de = r.de;
        h.o = r.o;
        if (fabs(h.de - ds) <= 0.001) break;
        if (h.o.it >= TO_INT(PARAM(P_MAX_IT))) {
            dt1 = -fabs(dt1);
        } else if (h.de < ds) {
            dt1 = fabs(dt1) * -0.55;
        } else {
            dt1 = fabs(dt1) * 0.55;
        }
    }
    return h;
}

Hit march(FN_ARGS real px, real py) {
    Ray ray = pixel_ray(FN_PASS px, py);
    Hit h;
    real zend = PARAM(P_ZEND);
    real zsd = PARAM(P_Z_STEP_DIV);
    real mh04zsd = PARAM(P_MH04ZSD);
    real ms_de_sub = PARAM(P_MS_DE_SUB);
    int max_it = TO_INT(PARAM(P_MAX_IT));
    int min_it = TO_INT(PARAM(P_MIN_IT));
    real ds;
    real last_de;
    real last_step;
    real d;
    real s1;
    real rsf_mul = 1.0;
    real steps = 0.0;
    bool first_step = PARAM(P_FIRST_STEP_RANDOM) > 0.5;
    bool at_cut = false;
    DeResult r;
    int guard;
    h.hit = false;
    h.t = 0.0;
    h.steps = 0.0;
    h.pos = ray.origin;
    if (PARAM(P_CUT_OPTIONS) > 0.5) {
        d = cut_plane_start(FN_PASS ray);
        if (d > zend) return h;
        if (d > 0.0) {
            h.t = d;
            h.pos = ray.origin + ray.dir * d;
            at_cut = true;
        }
    }
    ds = de_stop_at(FN_PASS h.t);
    r = calc_de(FN_PASS h.pos, ds);
    h.de = r.de;
    h.o = r.o;
    if (h.o.it >= max_it || h.de < ds) {
        /* inside the set at the start plane / cutting plane */
        h.hit = true;
        h.at_start = true;
        return h;
    }
    h.at_start = false;
    last_step = h.de * zsd;
    for (guard = 0; guard < 200000; guard += 1) {
        if (h.o.it >= max_it) {
            /* overshot into the set: half a step back */
            d = -0.5 * last_step;
            h.t += d;
            h.pos += ray.dir * d;
            ds = de_stop_at(FN_PASS h.t);
            r = calc_de(FN_PASS h.pos, ds);
            h.de = r.de;
            h.o = r.o;
            last_step = -d;
        }
        if (h.o.it < min_it || (h.o.it < max_it && h.de >= ds)) {
            last_de = h.de;
            d = fmax(0.11, (h.de - ms_de_sub * ds) * zsd * rsf_mul);
            s1 = fmax(ds, 0.4) * mh04zsd;
            if (s1 < d) {
                steps += s1 / d;
                d = s1;
            } else {
                steps += 1.0;
            }
            if (first_step) {
                first_step = false;
                d *= pixel_random(px, py);
            }
            h.t += d;
            if (h.t > zend) {
                h.hit = false;
                break;
            }
            last_step = d;
            h.pos += ray.dir * d;
            ds = de_stop_at(FN_PASS h.t);
            r = calc_de(FN_PASS h.pos, ds);
            h.de = r.de;
            h.o = r.o;
            if (h.de > last_de + last_step) h.de = last_de + last_step;
            if (last_de > h.de + 1e-30) {
                rsf_mul = fmin(1.0, fmax(0.5, last_step / (last_de - h.de)));
            } else {
                rsf_mul = 1.0;
            }
        } else {
            h.hit = true;
            if (TO_INT(PARAM(P_BIN_SEARCH)) > 0 && (h.o.it < max_it || h.de < ds)) {
                h = bin_search(FN_PASS h, ray.dir, last_step);
            }
            break;
        }
    }
    h.steps = steps;
    return h;
}

/* The whole pixel: march, normal, colour values, AO and shadows. */
Pixel shade_pixel(FN_ARGS real px, real py) {
    Pixel g;
    Hit h = march(FN_PASS px, py);
    real ds;
    real sit;
    real3 n;
    real3 ldir;
    int i;
    int nl;
    g.depth = -1.0;
    g.normal = V3(0.0, 0.0, -1.0);
    g.color = 0.0;
    g.otrap = 0.0;
    g.steps = h.steps;
    g.ao = 1.0;
    for (i = 0; i < MAX_LIGHTS; i += 1) g.shadow[i] = 1.0;
    g.iter = 0.0;
    g.de = 0.0;
    if (!h.hit) return g;
    g.depth = h.t;
    ds = de_stop_at(FN_PASS h.t);
    sit = smooth_iterations(FN_PASS h.o);
    g.iter = TO_REAL(h.o.it);
    g.de = h.de;
    g.otrap = h.o.otrap;
    if (h.o.it < TO_INT(PARAM(P_MAX_IT)) || h.de < ds) {
        g.color = sit + PARAM(P_COL_PLUS) + PARAM(P_COL_VAR) * log(fmax(PARAM(P_DE_STOP), ds) * PARAM(P_STEP_WIDTH));
    } else {
        g.color = sit;
    }
    if (h.at_start) {
        /* the start plane / cutting plane: normal against the view */
        n = -normalize(pixel_ray(FN_PASS px, py).dir);
        if (PARAM(P_CUT_OPTIONS) > 0.5) n = surface_normal(FN_PASS h.pos, h.t);
    } else {
        n = surface_normal(FN_PASS h.pos, h.t);
    }
    g.normal = n;
    if (TO_INT(PARAM(P_AO_MODE)) == 2) g.ao = de_ambient_occlusion(FN_PASS h.pos, n, h.t);
    if (PARAM(P_SHADOWS) > 0.5) {
        nl = TO_INT(PARAM(P_N_LIGHTS));
        for (i = 0; i < nl; i += 1) {
            if (PARAM(P_LIGHT_FLAGS + i) < 0.5) continue;
            ldir = V3(PARAM(P_LIGHT_DIRS + 3 * i), PARAM(P_LIGHT_DIRS + 3 * i + 1), PARAM(P_LIGHT_DIRS + 3 * i + 2));
            if (dot(ldir, n) <= 0.0) continue;
            g.shadow[i] = shadow_factor(FN_PASS h.pos, n, ldir, h.t);
        }
    }
    return g;
}
