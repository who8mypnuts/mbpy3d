"""Writes a Scene as a MB3D parameter file (``.m3p``, MandId 44).

The bytes of the file the scene was loaded from (``Scene.raw``) are the
template, so settings this port does not model survive a round trip.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .bytes_io import ByteWriter, to_short_float
from .lighting import Lighting
from .m3p_text import raw_to_text
from .scene import ADDON_SIZE, HEADER_SIZE, MAND_ID, CameraOptic, InsideMode, Scene

LIGHT_BASE = 432


def save(scene: Scene, path: str | Path) -> None:
    Path(path).write_bytes(to_bytes(scene))


def save_text(scene: Scene, path: str | Path) -> None:
    Path(path).write_text(to_text(scene), encoding="latin-1")


def to_text(scene: Scene) -> str:
    return raw_to_text(to_bytes(scene), scene.title or "Mandelbulb3D")


def _normalise_rows(m, n: float):
    out = []
    for row in m:
        length = (row[0] ** 2 + row[1] ** 2 + row[2] ** 2) ** 0.5 or 1.0
        out.append(tuple(v * n / length for v in row))
    return out


def to_bytes(sc: Scene) -> bytes:
    d = bytearray(sc.raw) if sc.raw is not None and len(sc.raw) >= HEADER_SIZE + ADDON_SIZE else bytearray(HEADER_SIZE + ADDON_SIZE)
    w = ByteWriter(d)
    w.i32(0, MAND_ID)
    w.i32(4, sc.width)
    w.i32(8, sc.height)
    w.i32(12, sc.iterations)
    ioptions = int(sc.first_step_random) | (int(sc.step_sub_de_stop) << 2) | (max(0, min(8, sc.smooth_normals)) << 6)
    w.u16(16, ioptions)
    w.u8(18, 0)  # rotation matrix, not a quaternion
    w.u8(19, sc.color_on_it)
    w.f64(20, sc.z_start)
    w.f64(28, sc.z_end)
    for k in range(3):
        w.f64(36 + 8 * k, sc.mid[k])
        w.f64(60 + 8 * k, sc.rot_4d[k])
    w.f64(84, sc.zoom)
    w.f64(92, sc.rstop if sc.rstop is not None else 0.0)
    w.f64(108, sc.fov_y)
    w.u8(126, sc.stereo_mode)
    w.u8(132, int(sc.normals_on_de))
    w.u8(134, max(0, min(255, sc.bin_search_steps)))
    w.u16(135, max(0, min(65535, sc.min_iterations)))
    w.i32(140, sc.stats.avg_de_steps10)
    w.i32(144, sc.stats.avg_its10)
    w.u8(148, int(sc.optic))
    if sc.ao is not None:
        a = sc.ao
        kind = 3 if sc.deao is not None else 0 if a.bits15 else 2 if a.random > 0 else 1
        bits = 1 | (int(a.t0) << 1) | (kind << 2)
        if sc.deao is not None:
            bits |= ((sc.deao.quality & 3) << 4) | (int(sc.deao.first_step_random) << 7)
            w.u8(188, sc.deao.dither)
            w.f32(374, sc.deao.max_len)
        w.u8(149, bits)
        w.f32(319, a.threshold)
        w.u8(127, int(round(a.border_mirror * 100.0)))
        w.u8(187, a.random)
    else:
        w.u8(149, 0)
    if sc.shadows is not None:
        h = sc.shadows
        w.u8(133, 1 | (int(h.set_cos) << 1) | ((h.lights & 0x3F) << 2))
        w.u8(139, int(h.soft))
        w.u16(224, to_short_float(h.soft_radius))
        w.f32(226, h.max_len_mul)
    else:
        w.u8(133, 0)
        w.f32(226, 1.0)
    w.u8(162, int(sc.vary_de_stop_on_fov))
    if sc.dof is not None:
        f = sc.dof
        w.f32(164, f.z_sharp)
        w.f32(168, f.clip_r)
        w.f32(172, f.aperture)
        w.u8(181, 1 | ((max(1, min(4, f.passes)) - 1) << 1) | (int(f.forward) << 3))
        w.f32(410, f.z_sharp2)
    else:
        w.u8(181, 0)
    w.u8(176, sc.cut_options & 7)
    for k in range(3):
        w.f64(346 + 8 * k, sc.cut_pos[k])
    w.f32(177, sc.de_stop)
    w.f32(182, sc.z_step_div)
    w.u8(190, int(sc.julia))
    for k in range(4):
        w.f64(191 + 8 * k, sc.julia_c[k])
    w.u8(223, min(255, sc.dfog_on_it))
    w.f32(242, sc.raystep_limiter)
    m = _normalise_rows(sc.vgrads, sc.step_width())
    for i in range(3):
        for j in range(3):
            w.f64(246 + (i * 3 + j) * 8, m[i][j])
    w.f32(338, sc.color_mul)
    w.u8(342, sc.color_option)
    if sc.raw is None:
        w.u8(343, 2 << 4)
    w.u8(344, int(sc.slice_2d == 0))
    w.u8(345, 2 if sc.slice_2d == 0 else sc.slice_2d)
    w.u8(331, int(sc.normals_on_zbuf))
    w.f32(230, sc.stereo_screen[0])
    w.f32(234, sc.stereo_screen[1])
    w.f32(238, sc.stereo_screen[2])
    w.f32(104, sc.decomb.mix_pow if sc.decomb else 2.0)
    w.f32(378, sc.decomb.smooth if sc.decomb else 0.5)
    w.i32(418, sc.decomb.iterations2 if sc.decomb else sc.iterations)
    w.i32(414, sc.stats.max_its)
    w.u8(422, sc.decomb.mix_color if sc.decomb else 0)
    if sc.raw is None:
        # Monte Carlo defaults of a fresh file
        w.f32(332, 0.5)
        w.u8(337, 1)
        w.f32(116, 1.5)
        w.f32(370, 0.2)
        w.f32(120, 1.0)
        w.u8(124, 2 | 16)
        w.u8(186, 3)
        w.u8(318, 32)
        w.u8(423, 128)
    _write_light(w, sc.lighting)
    _write_addon(w, sc)
    return bytes(d)


def _write_light(w: ByteWriter, l: Lighting) -> None:
    b = LIGHT_BASE
    w.i16(b, int(round(l.var_col_z)))
    w.u8(b + 2, int(max(0, min(255, round(l.roughness)))))
    w.rgb(b + 4, l.dyn_fog_col2)
    w.u8(b + 7, (int(l.bg_add_light) << 3) | int(l.internal_gamma2) | (int(l.yc_comb) << 2) | (int(l.bg_ambient) << 5))
    w.u8(b + 3, l.diff_map & 0xFF)
    ind = int(round(l.ind_light - 53.0))
    tbv = [
        (int(round(l.fog_offset)) & 0xFFFF) | (128 << 16),
        int(round(l.depth_fog)),
        int(round(l.diffuse)),
        int(round(l.dyn_fog)),
        (int(round(l.specular)) & 0xFFF) | (l.diff_map_offset[0] << 12) | (l.diff_map_offset[1] << 20),
        (int(round(l.ambient)) & 0xFFF) | (l.diff_map_rot << 20),
        int(round(l.color_start)),
        int(round(l.color_end)),
        (int(round(l.amb_shadow)) & 0xFF) | ((ind & 0xFF) << 8),
    ]
    for k, v in enumerate(tbv):
        w.i32(b + 8 + k * 4, v)
    tbo = (int(round(l.interior_start)) & 0x7F) | ((int(round(l.interior_end)) & 0x7F) << 7) | (7 << 20)
    tbo |= int(l.amb_rel_obj) << 29
    tbo |= (int(l.color_cycling) << 14) | (int(l.bg_direct) << 15) | (int(l.color_on_otrap) << 17) | (int(l.far_fog) << 18)
    tbo |= ((int(round(l.gamma)) & 0x3F) << 23) | ((l.depth_func & 3) << 30)
    if l.fine_col_adj is not None:
        tbo |= 0x10000
        w.u8(b + 48, l.fine_col_adj[0])
        w.u8(b + 49, l.fine_col_adj[1])
    w.u32(b + 44, tbo)
    w.u8(b + 50, l.bg_rot[0])
    w.u8(b + 51, l.bg_rot[1])
    w.u8(b + 67, l.bg_rot[2])
    w.rgb(b + 52, l.amb_top)
    w.rgb(b + 56, l.amb_bottom)
    w.rgb(b + 60, l.depth_col)
    w.rgb(b + 64, l.depth_col2)
    w.u8(b + 55, l.dyn_fog_col[0])
    w.u8(b + 59, l.dyn_fog_col[1])
    w.u8(b + 63, l.dyn_fog_col[2])
    for i, li in enumerate(l.lights[:6]):
        o = b + 68 + i * 32
        lopt = 2 if li.map > 0 else 0 if li.on else 1
        lopt |= (int(li.positional) << 2) | ((li.visible & 6) << 2) | (int(li.relative_to_object) << 5) | (int(not li.hs_enabled) << 6)
        w.u8(o, lopt)
        w.u8(o + 1, (li.spec_func & 7) | ((li.diff_func & 3) << 4) | ((li.visible & 8) << 4))
        w.u16(o + 2, to_short_float(li.amplitude))
        w.rgb(o + 4, li.color)
        w.d7b(o + 9, li.position[0] if li.positional else li.x_angle)
        w.d7b(o + 17, li.position[1] if li.positional else li.y_angle)
        w.d7b(o + 25, li.position[2] if li.positional else 0.0)
        if li.map > 0:
            w.u16(o + 7, li.map)
            w.u8(o + 9, li.map_rot[0])
            w.u8(o + 17, li.map_rot[1])
            w.u8(o + 25, li.map_rot[2])

    def light(i):
        return b + 68 + i * 32

    w.u8(light(0) + 16, 8)
    w.u8(light(0) + 24, l.dfog_options & 3)
    w.u8(light(1) + 16, (l.diff_map >> 8) & 0xFF)
    w.u8(light(1) + 24, l.diff_map_mode & 3)
    w.u8(light(2) + 16, l.diff_map_scale)
    w.u8(light(2) + 24, l.ex_mode)
    w.u8(light(3) + 24, int(l.no_col_ipol))
    w.u8(light(3) + 16, int(max(0, min(255, round(l.diffuse_shadowing * 256.0)))))
    w.u8(light(4) + 16, l.bg_brightness)
    for i, c in enumerate(l.palette[:10]):
        o = b + 260 + i * 10
        w.u16(o, c.position)
        w.rgb(o + 2, c.diffuse)
        w.u8(o + 5, 0)
        w.rgb(o + 6, c.specular)
        w.u8(o + 9, l.palette_alpha_of(i))
    for i, (pos, c) in enumerate(l.interior[:4]):
        o = b + 360 + i * 6
        w.u16(o, pos)
        w.rgb(o + 2, c)
        w.u8(o + 5, l.interior_spec[i])
    w.str(b + 384, 24, l.bg_image[:23])


def _write_addon(w: ByteWriter, sc: Scene) -> None:
    a = HEADER_SIZE
    d = w.d
    d[a:a + ADDON_SIZE] = bytes(ADDON_SIZE)
    w.u8(a, 16)
    used = [i for i, f in enumerate(sc.formulas) if f.iterations != 0]
    last = used[-1] if used else 0
    if sc.interpolation is not None:
        w.u8(a + 1, 1)
        w.u8(a + 5, 1)
        w.u16(a + 6, 0x151)
    elif sc.decomb is not None:
        c = sc.decomb
        w.u8(a + 1, 2)
        w.u8(a + 3, max(1, min(6, c.kind)) - 1)
        w.u8(a + 5, (c.end1 & 7) | (min(sc.repeat_from, c.end1) << 4))
        w.u16(a + 6, (c.start2 & 7) | ((c.end2 & 7) << 4) | ((c.repeat2 & 7) << 8))
    else:
        w.u8(a + 5, (last & 7) | (min(sc.repeat_from, last) << 4))
        w.u16(a + 6, 0x151)
    w.u8(a + 2, int(sc.disable_analytic_de) | {InsideMode.OUTSIDE: 0, InsideMode.INSIDE: 2, InsideMode.BOTH: 4}[sc.inside])
    w.u8(a + 4, 2 if sc.interpolation is not None else (last + 1 if used else 0))
    nf = 2 if sc.interpolation is not None else 6
    for i, f in enumerate(sc.formulas[:nf]):
        o = a + 8 + i * 188
        w.i32(o, f.iterations)
        if sc.interpolation is not None:
            struct.pack_into("<f", d, o, sc.interpolation[i])
        elif f.iterations == 0:
            continue
        w.i32(o + 4, f.fnr)
        w.i32(o + 8, f.option_count)
        w.str(o + 12, 32, f.name[:31])
        for k, v in enumerate(f.options[:16]):
            w.f64(o + 60 + 8 * k, v)
