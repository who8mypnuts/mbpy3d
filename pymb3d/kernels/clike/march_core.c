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
 * of core/params.py, the generated formulas and the generated loops
 * iterate / iterate_p2 (the alternating hybrid of the first / second part)
 * and difs_orbit / difs_orbit_p2 (the dIFS loops), see kernels/loopgen.py.
 */

/* CalcSmoothIterations */
real smooth_iterations(FN_ARGS Orbit o, int mode) {
    real r2 = o.s.r2;
    real fhln = PARAM(P_SLOTS + SLOT_STRIDE * o.slot + 2);
    real ln_rstop = PARAM(P_LN_RSTOP);
    real d;
    real e;
    if (mode == 3) return TO_REAL(o.it);
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
real de_numeric(FN_ARGS real3 pos, Orbit o, int part, real de_scale) {
    real r2_0 = o.s.r2;
    real off = PARAM(P_DE_OFFSET_PX) * PARAM(P_STEP_WIDTH);
    real g = 0.0;
    real d;
    Orbit ok;
    if (r2_0 < 1e-30) return 0.0;
    ok = (part == 0) ? iterate(FN_PASS pos + V3(off, 0.0, 0.0), o.it, 1e30)
                     : iterate_p2(FN_PASS pos + V3(off, 0.0, 0.0), o.it, 1e30);
    d = r2_0 - ok.s.r2;
    g += d * d;
    ok = (part == 0) ? iterate(FN_PASS pos + V3(0.0, off, 0.0), o.it, 1e30)
                     : iterate_p2(FN_PASS pos + V3(0.0, off, 0.0), o.it, 1e30);
    d = r2_0 - ok.s.r2;
    g += d * d;
    ok = (part == 0) ? iterate(FN_PASS pos + V3(0.0, 0.0, off), o.it, 1e30)
                     : iterate_p2(FN_PASS pos + V3(0.0, 0.0, off), o.it, 1e30);
    d = r2_0 - ok.s.r2;
    g += d * d;
    return r2_0 * log(r2_0) * de_scale * PARAM(P_DE_OFFSET_PX) / (sqrt(g) + PARAM(P_DE_OFFSET_PX) * 0.06);
}

/* The distance estimate of one hybrid part at pos, in pixel units. */
DeResult de_part(FN_ARGS real3 pos, real ds, int part) {
    DeResult r;
    int max_it = TO_INT((part == 0) ? PARAM(P_MAX_IT) : PARAM(P_MAX_IT2));
    int inside_it = TO_INT((part == 0) ? PARAM(P_INSIDE_IT) : PARAM(P_INSIDE_IT2));
    int mode = TO_INT((part == 0) ? PARAM(P_DE_MODE) : PARAM(P_DE_MODE2));
    real de_scale = (part == 0) ? PARAM(P_DE_SCALE) : PARAM(P_DE_SCALE2);
    real sw = PARAM(P_STEP_WIDTH);
    real de;
    real rr;
    if (mode == 3) {
        r.o = (part == 0) ? difs_orbit(FN_PASS pos, max_it, ds * sw * 1.03, false)
                          : difs_orbit_p2(FN_PASS pos, max_it, ds * sw * 1.03, false);
        de = r.o.s.r2 * de_scale / sw;
    } else {
        r.o = (part == 0) ? iterate(FN_PASS pos, max_it, PARAM(P_RSTOP2))
                          : iterate_p2(FN_PASS pos, max_it, PARAM(P_RSTOP2));
        if (r.o.it >= inside_it) {
            de = 0.0;
        } else if (mode == 1) {
            de = sqrt(r.o.s.r2) / fabs(r.o.s.dr) * de_scale / sw;
        } else if (mode == 2) {
            rr = sqrt(r.o.s.r2);
            de = 0.5 * log(r.o.s.r2) * rr / fabs(r.o.s.dr) * de_scale / sw;
        } else {
            de = de_numeric(FN_PASS pos, r.o, part, de_scale);
        }
    }
    if (!(de >= ds * 0.25)) de = ds * 0.25; /* also NaN */
    r.de = de;
    /* the smooth iteration count is only needed here by the smooth minima */
    r.sit = (TO_INT(PARAM(P_DECOMB_KIND)) >= 4) ? smooth_iterations(FN_PASS r.o, mode) : 0.0;
    r.inside = r.o.it >= inside_it;
    return r;
}

/* CalcDEfull: the first part, or the combination of two hybrid parts. */
DeResult calc_de(FN_ARGS real3 pos, real ds) {
    DeResult a = de_part(FN_PASS pos, ds, 0);
    DeResult b;
    int kind = TO_INT(PARAM(P_DECOMB_KIND));
    real sm;
    real res;
    real rst;
    real rd1;
    real d2;
    real w;
    real r2;
    real f;
    if (kind == 0) return a;
    b = de_part(FN_PASS pos, ds, 1);
    rst = b.de;
    if (kind == 3) {
        /* the second part inverted */
        rst = ds * 4.0 - rst * 3.0;
        b.inside = false;
    }
    if (kind == 1) {
        return (rst < a.de) ? b : a;
    }
    if (kind == 2 || kind == 3) {
        if (rst > a.de) {
            b.de = rst;
            return b;
        }
        return a;
    }
    /* smooth minima (4, 5): the colours are blended by the distances */
    sm = PARAM(P_DECOMB_SMOOTH);
    res = a.de;
    if (kind == 4) {
        res = fmin(rst - fmax(sm - a.de, 0.0), a.de - fmax(sm - rst, 0.0));
    } else {
        rd1 = fmax(sm - a.de - ds, 0.0);
        d2 = fmax(sm - rst - ds, 0.0);
        res = fmin(rst - rd1 * (sm - d2) / sm, a.de - d2 * (sm - rd1) / sm);
    }
    r2 = fabs(rst);
    w = fabs(res);
    f = 1.0 / (w + r2 + 1e-10);
    if (rst < a.de) {
        b.sit = (b.sit * w + a.sit * r2) * f;
        b.o.otrap = (b.o.otrap * w + a.o.otrap * r2) * f;
        b.de = res;
        return b;
    }
    a.sit = (b.sit * w + a.sit * r2) * f;
    a.o.otrap = (b.o.otrap * w + a.o.otrap * r2) * f;
    a.de = res;
    return a;
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

/* Soft shadow factor (1 = lit) towards a light direction (world space),
 * at most maxlen pixel units far. */
real shadow_factor(FN_ARGS real3 pos, real3 n, real3 ldir, real t, real maxlen) {
    real sw = PARAM(P_STEP_WIDTH);
    real ds0 = de_stop_at(FN_PASS t);
    real tt = ds0 * 2.0;          /* pixel units along the light ray */
    real res = 1.0;
    real k = PARAM(P_SHADOW_SOFTNESS);
    real de;
    real ds;
    real step;
    int i;
    real3 p0 = pos + n * (ds0 * sw * 1.5);
    for (i = 0; i < 300; i += 1) {
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
    int i;
    for (i = 0; i < steps; i += 1) {
        h.t += dt1;
        h.pos += dir * dt1;
        ds = de_stop_at(FN_PASS h.t);
        h.r = calc_de(FN_PASS h.pos, ds);
        if (fabs(h.r.de - ds) <= 0.001) break;
        if (h.r.inside) {
            dt1 = -fabs(dt1);
        } else if (h.r.de < ds) {
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
    int min_it = TO_INT(PARAM(P_MIN_IT));
    real ds;
    real last_de;
    real last_step;
    real d;
    real s1;
    real rsf_mul = 1.0;
    real steps = 0.0;
    bool first_step = PARAM(P_FIRST_STEP_RANDOM) > 0.5;
    int guard;
    h.hit = false;
    h.at_start = false;
    h.t = 0.0;
    h.steps = 0.0;
    h.pos = ray.origin;
    if (PARAM(P_CUT_OPTIONS) > 0.5) {
        d = cut_plane_start(FN_PASS ray);
        if (d > zend) return h;
        if (d > 0.0) {
            h.t = d;
            h.pos = ray.origin + ray.dir * d;
        }
    }
    ds = de_stop_at(FN_PASS h.t);
    h.r = calc_de(FN_PASS h.pos, ds);
    if (h.r.inside || h.r.de < ds) {
        /* inside the set at the start plane / cutting plane */
        h.hit = true;
        h.at_start = true;
        return h;
    }
    last_step = h.r.de * zsd;
    for (guard = 0; guard < 200000; guard += 1) {
        if (h.r.inside) {
            /* overshot into the set: half a step back */
            d = -0.5 * last_step;
            h.t += d;
            h.pos += ray.dir * d;
            ds = de_stop_at(FN_PASS h.t);
            h.r = calc_de(FN_PASS h.pos, ds);
            last_step = -d;
        }
        if (h.r.o.it < min_it || (!h.r.inside && h.r.de >= ds)) {
            last_de = h.r.de;
            d = fmax(0.11, (h.r.de - ms_de_sub * ds) * zsd * rsf_mul);
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
            h.r = calc_de(FN_PASS h.pos, ds);
            if (h.r.de > last_de + last_step) h.r.de = last_de + last_step;
            if (last_de > h.r.de + 1e-30) {
                rsf_mul = fmin(1.0, fmax(0.5, last_step / (last_de - h.r.de)));
            } else {
                rsf_mul = 1.0;
            }
        } else {
            h.hit = true;
            if (TO_INT(PARAM(P_BIN_SEARCH)) > 0 && (!h.r.inside || h.r.de < ds)) {
                h = bin_search(FN_PASS h, ray.dir, last_step);
            }
            break;
        }
    }
    h.steps = steps;
    return h;
}

/* MB3D's 15 bit clip */
real clip15(real v) {
    return clamp(v, 0.0, 32767.0);
}

/* RMdoColor: the orbit trap colour value (the 2nd colour choice). */
real orbit_trap_value(FN_ARGS Orbit o, real ds, int mode) {
    int opt = TO_INT(PARAM(P_COLOR_OPTION));
    real3 z = o.s.z;
    real3 c = o.s.c;
    real r;
    if (mode == 3) return clip15(o.otrap * 4096.0);
    if (opt == 1) return clip15(log(o.s.r2 / (o.rold + 1.0)) * PARAM(P_COLOR_MUL));
    if (opt == 2) return clip15((atan2(z.y - c.y, z.x - c.x) + PI_) * 5200.0);
    if (opt == 3) return clip15((atan2(z.z - c.z, z.x - c.x) + PI_) * 5200.0);
    if (opt == 4) return clip15((atan2(z.z - c.z, z.y - c.y) + PI_) * 5200.0);
    if (opt == 5) return clip15((atan2(z.x, z.y) + PI_) * 5215.0);
    return clip15(o.otrap * 4096.0);
}

/* The whole pixel: march, normal, colour values, AO and shadows. */
Pixel shade_pixel(FN_ARGS real px, real py) {
    Pixel g;
    Hit h = march(FN_PASS px, py);
    real ds;
    real nn;
    real v;
    real r;
    real3 n;
    real3 ldir;
    real maxlen;
    int i;
    int nl;
    int mode = TO_INT(PARAM(P_DE_MODE));
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
    nn = (TO_INT(PARAM(P_DECOMB_KIND)) >= 4) ? h.r.sit : smooth_iterations(FN_PASS h.r.o, mode);
    g.iter = TO_REAL(h.r.o.it);
    g.de = h.r.de;
    g.otrap = orbit_trap_value(FN_PASS h.r.o, ds, mode);
    if (!h.r.inside || h.r.de < ds) {
        v = 32767.0 - (nn + PARAM(P_COL_PLUS) + PARAM(P_COL_VAR) * log(fmax(PARAM(P_DE_STOP), ds) * PARAM(P_STEP_WIDTH))) * PARAM(P_MCTS_M);
    } else {
        v = 32767.0 - nn * PARAM(P_MCTS_M);
    }
    if (TO_INT(PARAM(P_COLOR_OPTION)) == 5) {
        r = sqrt(dot(h.r.o.s.z, h.r.o.s.z) + 1e-100);
        v = (PI_ + asin(clamp(h.r.o.s.z.z / r, -1.0, 1.0)) * 2.0) * 5215.0;
    }
    g.color = clip15(v);
    if (h.at_start && PARAM(P_CUT_OPTIONS) < 0.5) {
        /* the start plane: normal against the view */
        n = -normalize(pixel_ray(FN_PASS px, py).dir);
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
            maxlen = PARAM(P_ZEND);
            if (PARAM(P_LIGHT_FLAGS + i) > 1.5) {
                /* positional light: towards its position, up to it */
                ldir = ldir - h.pos;
                maxlen = length(ldir) / PARAM(P_STEP_WIDTH);
                ldir = normalize(ldir);
            }
            if (dot(ldir, n) <= 0.0) continue;
            g.shadow[i] = shadow_factor(FN_PASS h.pos, n, ldir, h.t, maxlen);
        }
    }
    return g;
}
