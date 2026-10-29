"""``CalcPixelColor2``: the light of every pixel from the G-buffer, vectorised.

Global lights (diffuse + specular), ambient top/bottom with ambient
occlusion, the surface palette and the interior colours, depth fog, dynamic
fog and gamma.  Not yet ported: light maps, background pictures, colour
maps, positional and visible lights.
"""

from __future__ import annotations

import numpy as np

from mb3d.params.scene import Scene
from mb3d.render.gbuffer import GBuffer

from .costab import cos_tab_val
from .light_values import LightVals, PaintCamera
from .palette import interior_colors, surface_colors

# MB3D's ambient shadow value of pixels without a calculated ambient occlusion
DEFAULT_AMB_SHADOW = 5000.0 / 16383.0


def _fast_int_pow(base: np.ndarray, expo: int) -> np.ndarray:
    """``FastIntPow``: base^expo for a power of two exponent, 0 for negative bases."""
    r = np.where(base < 0.0, 0.0, base)
    i = expo >> 1
    while i > 0:
        r = r * r
        i >>= 1
    return r


def paint(gbuf: GBuffer, scene: Scene, ao: np.ndarray | None = None, use_kernel_ao: bool = True,
          light_shadow: np.ndarray | None = None) -> np.ndarray:
    """Returns the painted image as (h, w, 3) uint8.

    ``ao``: an ambient occlusion map 0..1 (1 = open); by default the G-buffer's
    DE ambient occlusion when it was calculated.  ``light_shadow``: 0..1
    shadow of the first light (1 = lit), by default the G-buffer's.
    """
    h, w = gbuf.height, gbuf.width
    lv = LightVals.from_lighting(scene.lighting, scene.z_step_div, scene.dfog_on_it, scene.z_end - scene.z_start,
                                 scene.step_width(), scene.width)
    cam = PaintCamera.from_scene(scene)
    sqr = lv.sqr
    hit = gbuf.hit.astype(np.float32)
    surface = hit > 0.5
    zpos_word = cam.zpos_word(gbuf.depth.astype(np.float64), hit)
    view = cam.view_vectors()
    y_pos = (np.arange(h, dtype=np.float32) / h)[:, None, None]
    ys = y_pos * y_pos if lv.depth_func == 1 else y_pos if lv.depth_func == 0 else np.sqrt(np.maximum(y_pos, 0.0))
    dep_c = lv.depth_col2 + ys * (lv.depth_col - lv.depth_col2)
    dep_c = np.broadcast_to(dep_c, (h, w, 3)).astype(np.float32)
    z1 = ((8388352.0 - np.round(zpos_word * 256.0)) / cam.zc_mul + 1.0) ** 2 - 1.0
    z_pos = (z1 * cam.step_width / cam.zcorr + cam.zz_stmit_dif).astype(np.float32)

    # ---- ambient shadow
    if ao is None and use_kernel_ao and np.any(gbuf.ao < 1.0):
        ao = gbuf.ao
    shadow = (1.0 - np.clip(ao, 0.0, 1.0)).astype(np.float32) if ao is not None else np.full((h, w), DEFAULT_AMB_SHADOW, np.float32)
    if lv.s_amb_shad < 1.0:
        d_amb_sh = 1.0 - lv.s_amb_shad * shadow
    else:
        a = 1.0 - shadow
        d_amb_sh = a + (lv.s_amb_shad - 1.0) * (a * a - a)
    if sqr:
        d_amb_sh = (d_amb_sh * d_amb_sh + d_amb_sh) * 0.5
    d_rough = np.zeros((h, w), np.float32)
    lns = gbuf.normal.astype(np.float32)
    if light_shadow is None:
        light_shadow = gbuf.shadow.astype(np.float32)

    # ---- lights
    li_dif = np.zeros((h, w, 3), np.float32)
    li_spe = np.zeros((h, w, 3), np.float32)
    li_amb = np.zeros((h, w, 3), np.float32)
    n_dot_v = 2.0 * np.einsum("hwk,hwk->hw", lns, view)
    for li_index, lvl in enumerate(lv.lights):
        if lvl.positional:
            continue  # positional lights are not painted yet
        ln = lvl.ln
        dot = np.einsum("hwk,k->hw", lns, ln)
        d = cos_tab_val(lvl.diff_func, dot, d_rough, sqr)
        li_amb += lvl.col * d[..., None]
        lit = light_shadow if li_index == 0 else 1.0
        d_lit = d * d_amb_sh * lit
        li_dif += lvl.col * d_lit[..., None]
        ds = np.einsum("hwk,k->hw", view - lns * n_dot_v[..., None], ln)
        pf = float(lvl.pow_func)
        t2 = (1.0 + np.minimum(d_rough * 2.0, 1.0) * (1.0 / pf - 1.0)) * lv.s_spec
        li_amb += lvl.col * (t2 / pf)[..., None]
        spec = t2 * _fast_int_pow(ds, lvl.pow_func) * d_amb_sh * lit
        li_spe += lvl.col * np.where(ds > 0.0, spec, 0.0)[..., None]

    # ---- surface colours
    idif0 = lv.s_col_zmul * z_pos
    sig = gbuf.si_gradient.astype(np.float32)
    dif_s, spe_s = surface_colors(lv, np.minimum(sig, 32767.0), gbuf.otrap.astype(np.float32), idif0)
    inside = sig > 32767.0
    if np.any(inside):
        dif_i, spe_i = interior_colors(lv, sig, idif0)
        idif = np.where(inside[..., None], dif_i, dif_s)
        ispe = np.where(inside[..., None], spe_i, spe_s)
    else:
        idif, ispe = dif_s, spe_s
    ny = lns[..., 1]
    t2 = (ny * 0.5 + 0.5) * d_amb_sh
    t1 = d_amb_sh - t2
    iamb = (lv.amb_col * t1[..., None] + lv.amb_col2 * t2[..., None]) * idif
    dif_scaled = idif * lv.s_diff
    total = iamb + dif_scaled * li_dif + ispe * li_spe
    if not lv.ex_mode:
        l3 = iamb + dif_scaled * li_amb * (d_amb_sh * (0.11 if sqr else 0.3))[..., None]
        st = np.maximum(0.0, 1.0 - d_amb_sh)
        st2 = st * st * st * st
        refl = idif * (st2 * lv.s_ind_light_reflect)[..., None] + (st - st2)[..., None]
        out = total + l3 * idif * refl
    else:
        st = np.maximum(0.0, 1.0 - d_amb_sh)
        st = st * st * lv.s_ind_light_reflect
        luma = total[..., 0] * 0.3 + total[..., 1] * 0.59 + total[..., 2] * 0.11
        st2 = luma + 4.0
        out = total + total * idif * st[..., None]
        luma_o = np.maximum(out[..., 0] * 0.3 + out[..., 1] * 0.59 + out[..., 2] * 0.11, 0.0)
        out = out * ((st2 + (st2 + 128.0) * (st + 0.11) * 0.04) / (luma_o + 3.0))[..., None]
    # background pixels show the depth colour
    out = np.where(surface[..., None], out, dep_c)
    dtmp = np.where(surface, np.maximum(0.0, (zpos_word - 28000.0) * lv.s_depth + 1.0),
                    np.maximum(0.0, 1.0 - (60768.0 - zpos_word) * lv.s_depth)).astype(np.float32)

    # ---- depth fog and dynamic fog
    if sqr:
        t = (1.0 - dtmp) ** 2
        if lv.far_fog:
            t = t * t
        dtmp = np.where(dtmp < 1.0, 1.0 - t, dtmp)
    elif lv.far_fog:
        dtmp = np.where(dtmp < 1.0, 1.0 - (1.0 - dtmp) ** 2, dtmp)
    out = out * dtmp[..., None]
    ir = np.clip(gbuf.steps, 0.0, 1023.0).astype(np.float32)
    dfog = (ir - lv.s_shad - lv.s_shad_zmul * z_pos) * lv.s_shad_gr
    if lv.dfog_options & 2:
        dfog = np.maximum(dfog, 0.0)
    dtmp3 = np.minimum(ir * lv.s_dyn_fog_mul, 1.0) * dfog
    out = out + dep_c * np.maximum(0.0, 1.0 - dtmp)[..., None]
    if lv.dfog_options & 1:
        dfog = np.clip(dfog, 0.0, 1.0)
        dtmp3 = np.clip(dtmp3, 0.0, 1.0)
        out = out * (1.0 - dfog)[..., None]
    out = out + lv.dyn_fog_col * (dfog - dtmp3)[..., None] + lv.dyn_fog_col2 * dtmp3[..., None]

    # ---- to 8 bit: clipping and gamma
    if sqr:
        c = np.sqrt(np.clip(out * 255.0, 0.0, 65025.0))
    else:
        c = np.clip(out, 0.0, 255.0)
    if lv.gamma_h != 0:
        g = np.sqrt(c * 255.0) if lv.gamma_h > 0 else c * c / 255.0
        c = c + lv.s_gamma * (g - c)
    return np.clip(np.round(c), 0, 255).astype(np.uint8)
