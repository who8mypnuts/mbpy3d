/* The orbit start (before the generated loops, which call it). */

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
    s.de = 1e30;
    s.trap = 0.0;
    s.aux = 0.0;
    s.vary = 0.0;
    o.s = s;
    o.rold = s.r2;
    o.otrap = s.r2;
    o.it = 0;
    o.slot = 0;
    return o;
}

