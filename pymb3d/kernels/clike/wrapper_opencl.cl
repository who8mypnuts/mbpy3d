/* Kernel entry: one work item per pixel of the rectangle; the G-buffer as
 * N_CHANNELS reals per pixel, rows of the rectangle one after another. */
__kernel void march_kernel(__constant const real *P, __global real *out, const int rect_w, const int rect_h) {
    int lx = get_global_id(0);
    int ly = get_global_id(1);
    real px;
    real py;
    Pixel g;
    int o;
    int i;
    if (lx >= rect_w || ly >= rect_h) return;
    px = TO_REAL(lx) * PARAM(P_PIXEL_STEP) + PARAM(P_RECT_X);
    py = TO_REAL(ly) * PARAM(P_PIXEL_STEP) + PARAM(P_RECT_Y);
    g = shade_pixel(FN_PASS px, py);
    o = (ly * rect_w + lx) * N_CHANNELS;
    out[o + G_DEPTH] = g.depth;
    out[o + G_NX] = g.normal.x;
    out[o + G_NY] = g.normal.y;
    out[o + G_NZ] = g.normal.z;
    out[o + G_COLOR] = g.color;
    out[o + G_OTRAP] = g.otrap;
    out[o + G_STEPS] = g.steps;
    out[o + G_AO] = g.ao;
    for (i = 0; i < MAX_LIGHTS; i += 1) out[o + G_SHADOW + i] = g.shadow[i];
    out[o + G_ITER] = g.iter;
    out[o + G_ROUGH] = g.rough;
    out[o + G_VOL] = g.vol;
    out[o + G_R_DEPTH] = g.r_depth;
    out[o + G_R_NX] = g.r_normal.x;
    out[o + G_R_NY] = g.r_normal.y;
    out[o + G_R_NZ] = g.r_normal.z;
    out[o + G_R_COLOR] = g.r_color;
    out[o + G_R_OTRAP] = g.r_otrap;
    out[o + G_R_AO] = g.r_ao;
    for (i = 0; i < MAX_LIGHTS; i += 1) out[o + G_R_SHADOW + i] = g.r_shadow[i];
    out[o + G_R_STEPS] = g.r_steps;
    out[o + 31] = 0.0;
}

/* Iterates one formula slot on a buffer of states (the transpiler tests):
 * 14 reals per state: z, w, c, cw, dr, r2, de, trap, aux, vary. */
__kernel void formula_test_kernel(__constant const real *P, const int slot, __global real *io, const int n) {
    int i = get_global_id(0);
    State s;
    int o;
    if (i >= n) return;
    o = i * 14;
    s.z = V3(io[o], io[o + 1], io[o + 2]);
    s.w = io[o + 3];
    s.c = V3(io[o + 4], io[o + 5], io[o + 6]);
    s.cw = io[o + 7];
    s.dr = io[o + 8];
    s.r2 = io[o + 9];
    s.de = io[o + 10];
    s.trap = io[o + 11];
    s.aux = io[o + 12];
    s.vary = io[o + 13];
    s = run_slot(FN_PASS slot, s);
    io[o] = s.z.x;
    io[o + 1] = s.z.y;
    io[o + 2] = s.z.z;
    io[o + 3] = s.w;
    io[o + 4] = s.c.x;
    io[o + 5] = s.c.y;
    io[o + 6] = s.c.z;
    io[o + 7] = s.cw;
    io[o + 8] = s.dr;
    io[o + 9] = s.r2;
    io[o + 10] = s.de;
    io[o + 11] = s.trap;
    io[o + 12] = s.aux;
    io[o + 13] = s.vary;
}
