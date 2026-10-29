"""Reads MB3D parameter files (``.m3p``, ``.m3i``, text parameters).

Port of ``LoadParameter`` (FileHandling.pas) for ``MandId >= 20``: the
``TMandHeader10`` (840 bytes with ``TLightingParas9``) followed by the
``THeaderCustomAddon`` (the six formula slots).
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

from .bytes_io import ByteReader, short_float
from .lighting import Light, Lighting, PaletteColor
from .m3p_text import find_text_params, raw_from_text
from .scene import (
    ADDON_SIZE,
    HEADER_SIZE,
    CalcStats,
    CameraOptic,
    DeaoParams,
    DeCombParams,
    DofParams,
    FormulaSlot,
    InsideMode,
    Scene,
    ShadowParams,
    SsaoParams,
)

INTERNAL_FORMULA_NAMES = {
    0: "Integer Power",
    1: "Real Power",
    2: "Quaternion",
    3: "Tricorn",
    4: "Amazing Box",
    5: "Bulbox",
    6: "Folding Int Pow",
    9: "Aexion C",
}

LIGHT_BASE = 432


class M3pError(ValueError):
    pass


def load(path: str | Path) -> Scene:
    """Loads a ``.m3p``/``.m3i`` file or a text file with MB3D text parameters."""
    path = Path(path)
    data = path.read_bytes()
    raw, title = raw_from_bytes(data, path.suffix.lower() == ".m3i")
    scene = parse(raw)
    scene.title = title or path.stem
    return scene


def raw_from_bytes(data: bytes, is_m3i: bool = False) -> tuple[bytes, str | None]:
    text = find_text_params(data)
    if text is not None:
        return raw_from_text(text)
    if len(data) < 4:
        raise M3pError("file too small")
    if is_m3i:
        return raw_from_m3i(data), None
    if struct.unpack_from("<i", data, 0)[0] < 20:
        raise M3pError("parameter files before MandId 20 (MB3D 1.7) are not supported")
    raw = bytearray(data[:HEADER_SIZE + ADDON_SIZE])
    raw += bytes(HEADER_SIZE + ADDON_SIZE - len(raw))
    return bytes(raw), None


def raw_from_m3i(data: bytes) -> bytes:
    """``.m3i`` image files: header, G-buffer (18 bytes per pixel), formula addon."""
    if len(data) < HEADER_SIZE:
        raise M3pError("file too small for a MB3D image")
    r = ByteReader(data)
    mand_id = r.i32(0)
    if mand_id < 20:
        raise M3pError("image files before MandId 20 are not supported")
    w, h = r.i32(4), r.i32(8)
    tiling = r.i32(428) if mand_id >= 35 else 0
    if tiling == 0:
        tw, th = w, h
    else:
        crop = max(1, (tiling >> 28) & 3)
        tx, ty = max(1, tiling & 0x7F), max(1, (tiling >> 7) & 0x7F)
        tw, th = w // tx + 2 * crop, h // ty + 2 * crop
    px = max(1, min(30000, tw)) * max(1, min(30000, th))
    off = HEADER_SIZE + px * 18
    raw = bytearray(data[:HEADER_SIZE])
    if len(data) >= off + 8:
        raw += data[off:min(len(data), off + ADDON_SIZE)]
    raw += bytes(HEADER_SIZE + ADDON_SIZE - len(raw))
    return bytes(raw[:HEADER_SIZE + ADDON_SIZE])


# ------------------------------------------------------------------ upgrades
def _upgrade_hybrid_options(d: bytearray, mand_id: int) -> None:
    a = HEADER_SIZE
    if mand_id < 33:
        d[418:422] = d[12:16]
    if len(d) < a + ADDON_SIZE or not (16 <= d[a] <= 99):
        return

    def its(i):
        return struct.unpack_from("<i", d, a + 8 + i * 188)[0]

    vl = min(d[343], 5)
    d[343] = vl
    j = 5
    while j > 0 and its(j) == 0:
        j -= 1
    opt1 = d[a + 1]
    opt3 = d[a + 3]
    hyb1 = 0 if opt1 != 0 else (j | (vl << 4))
    hyb2 = 0x151
    if opt1 == 2:
        hyb2 = 1 | (j << 4) | (max(vl, 1) << 8)
        d[a + 8:a + 12] = struct.pack("<i", 1)
        if opt3 != 2 and opt3 != 6:
            x, y = bytes(d[12:16]), bytes(d[418:422])
            d[12:16], d[418:422] = y, x
    else:
        opt3 = 0
    if opt3 in (2, 6):
        first = bytes(d[a + 8:a + 8 + 188])
        for i in range(j):
            d[a + 8 + i * 188:a + 8 + (i + 1) * 188] = d[a + 8 + (i + 1) * 188:a + 8 + (i + 2) * 188]
        d[a + 8 + j * 188:a + 8 + (j + 1) * 188] = first
        if opt3 == 6:
            opt3 = 5
        hyb1 = (max(j, 1) - 1) | ((max(vl, 1) - 1) << 4)
        hyb2 = j | (j << 4) | (j << 8)
    d[a + 3] = opt3
    d[a + 5] = hyb1 & 0xFF
    d[a + 6:a + 8] = struct.pack("<H", hyb2 & 0xFFFF)


def _upgrade_light(d: bytearray) -> int:
    """``UpdateLightParasAbove3``; returns the original light version."""
    b = LIGHT_BASE

    def light(i):
        return b + 68 + i * 32

    tbo = struct.unpack_from("<I", d, b + 44)[0]
    ver = (tbo >> 20) & 7
    verex = 0
    if ver < 4:
        d[b + 4], d[b + 5], d[b + 6] = d[b + 55], d[b + 59], d[b + 63]
    if ver < 5:
        tb6 = struct.unpack_from("<i", d, b + 20)[0]
        struct.pack_into("<i", d, b + 20, max(0, min(159, (tb6 - 53) * 2 + 53)))
        vz = struct.unpack_from("<h", d, b)[0]
        struct.pack_into("<h", d, b, max(-120, min(360, vz * 2)))
    if ver < 6:
        d[b + 3] = 0
        d[b + 2] = 255
        d[b + 67] = (d[b + 67] + 128) & 0xFF
        for i in range(6):
            d[light(i)] &= 0xFD
    if ver < 7:
        for i in range(6):
            o = light(i)
            d[o + 8] = 0
            d[o + 16] = 0
            d[o + 24] = 0
    else:
        verex = d[light(0) + 16]
    tbo = (tbo & 0xFF8FFFFF) | (7 << 20)
    struct.pack_into("<I", d, b + 44, tbo)
    if verex < 1:
        d[light(0) + 24] = 0
    if verex < 2:
        d[light(2) + 16] = 30
        d[light(1) + 24] = (d[b + 7] >> 1) & 1
        d[b + 7] &= 0xFB
    if verex < 3:
        d[light(3) + 16] = 128
        d[light(2) + 24] = 0
    if verex < 5:
        for i in range(4):
            d[b + 360 + i * 6 + 5] = 0
        for i in range(10):
            o = b + 260 + i * 10 + 6
            d[o + 3] = max(d[o], d[o + 1], d[o + 2])
    if verex < 6:
        from .bytes_io import to_short_float

        i = (d[b + 7] >> 4) & 7
        d[b + 7] = (d[b + 7] & 0x8F) | (16 if i != 0 else 0)
        for i in range(6):
            o = light(i)
            if d[o] & 6 != 4:
                struct.pack_into("<H", d, o + 2, to_short_float(1.0))
        d[light(4) + 16] = 40
        for i in range(4):
            o = b + 360 + i * 6 + 2
            y = (d[o] * 0.3 + d[o + 1] * 0.59 + d[o + 2] * 0.11) / 255.0
            d[o + 3] = int(round(y * 255.0))
    if verex < 7:
        for i in range(6):
            d[light(i) + 1] &= 0x3F
    if verex < 8:
        d[light(0) + 24] &= 1
        d[light(3) + 24] = 0
    d[light(0) + 16] = 8
    return ver


def upgraded_raw(data: bytes) -> bytearray:
    """The raw bytes brought to the current layout (MandId 44 semantics)."""
    d = bytearray(data)
    d += bytes(max(0, HEADER_SIZE + ADDON_SIZE - len(d)))
    mand_id = struct.unpack_from("<i", d, 0)[0]
    if mand_id < 42:
        _upgrade_hybrid_options(d, mand_id)
    if mand_id < 22:
        o = 432 + 44
        t = struct.unpack_from("<I", d, o)[0]
        t = (t & 0xE07FFFFF) | ((t & 0x1F000000) >> 1) | 0x10000000
        struct.pack_into("<I", d, o, t)
    _upgrade_light(d)
    return d


# --------------------------------------------------------------------- parse
def parse(data: bytes) -> Scene:
    if len(data) < HEADER_SIZE:
        raise M3pError("file too small for a MB3D parameter header")
    mand_id = struct.unpack_from("<i", data, 0)[0]
    if mand_id < 0:
        raise M3pError(f"parameter version {mand_id} is not a MB3D version")
    d = upgraded_raw(data)
    r = ByteReader(bytes(d))
    s = Scene()
    s.mand_id = mand_id
    s.raw = bytes(d)
    s.width = r.i32(4)
    s.height = r.i32(8)
    s.iterations = r.i32(12)
    ioptions = r.u16(16)
    new_options = r.u8(18)
    s.color_on_it = r.u8(19)
    s.z_start = r.f64(20)
    s.z_end = r.f64(28)
    s.mid = (r.f64(36), r.f64(44), r.f64(52))
    s.rot_4d = (r.f64(60), r.f64(68), r.f64(76))
    s.zoom = r.f64(84)
    if not (1 <= s.width < 32768 and 1 <= s.height < 32768):
        raise M3pError(f"invalid image size {s.width}x{s.height}")
    if not (math.isfinite(s.zoom) and s.zoom > 0.0):
        raise M3pError(f"invalid zoom {s.zoom}")
    rstop = r.f64(92)
    s.rstop = rstop if math.isfinite(rstop) and rstop > 1.0001 else None
    s.fov_y = r.f64(108)
    s.normals_on_de = r.u8(132) != 0
    s.bin_search_steps = r.u8(134)
    s.min_iterations = r.u16(135)
    s.stats = CalcStats(r.i32(140), r.i32(144), r.i32(414))
    s.optic = {1: CameraOptic.PLANAR, 2: CameraOptic.PANORAMA, 3: CameraOptic.PANORAMA}.get(r.u8(148) & 3, CameraOptic.COMMON)
    s.vary_de_stop_on_fov = r.u8(162) != 0
    s.de_stop = r.f32(177)
    s.z_step_div = max(0.001, min(1.0, r.f32(182)))
    s.julia = r.u8(190) != 0
    s.julia_c = (r.f64(191), r.f64(199), r.f64(207), r.f64(215))
    s.dfog_on_it = r.u8(223)
    s.raystep_limiter = r.f32(242)
    m = [[r.f64(246 + (i * 3 + j) * 8) for j in range(3)] for i in range(3)]
    if mand_id > 43 and (new_options & 1) != 0:
        m = _matrix_from_quat((m[0][0], m[0][1], m[0][2], m[1][0]))
    s.vgrads = tuple(tuple(row) for row in m)  # type: ignore[assignment]
    s.color_mul = r.f32(338)
    s.color_option = min(5, r.u8(342))
    if r.u8(344) == 0:
        s.slice_2d = {1: 1, 3: 3}.get(r.u8(345), 2)
    s.normals_on_zbuf = mand_id >= 27 and (r.u8(331) & 1) != 0
    s.stereo_mode = r.u8(126) if r.u8(126) in (1, 3, 4) else 0
    st = (r.f32(230), r.f32(234), r.f32(238))
    if all(math.isfinite(v) and v > 0.0 for v in st):
        s.stereo_screen = st
    s.smooth_normals = min(8, (ioptions >> 6) & 15)
    s.first_step_random = (ioptions & 1) != 0
    s.step_sub_de_stop = (ioptions & 4) != 0
    ao_bits = r.u8(149)
    if ao_bits & 1:
        kind = (ao_bits >> 2) & 3
        if kind == 3:
            s.deao = DeaoParams(quality=(ao_bits >> 4) & 3, dither=min(2, r.u8(188)),
                                max_len=max(0.01, r.f32(374)) if mand_id >= 30 else 1.0,
                                first_step_random=(ao_bits & 128) != 0)
        s.ao = SsaoParams(threshold=r.f32(319), border_mirror=min(0.9, r.u8(127) * 0.01),
                          t0=(ao_bits & 2) != 0, random=max(1, r.u8(187)) if kind == 2 else 0, bits15=kind == 0)
    else:
        s.ao = None
    hs = r.u8(133)
    if hs & 1 and (hs >> 2) != 0:
        s.shadows = ShadowParams(lights=hs >> 2, soft=mand_id >= 40 and (r.u8(139) & 1) != 0,
                                 soft_radius=max(0.01, min(20.0, short_float(r.u16(224)))) if mand_id >= 40 else 1.0,
                                 max_len_mul=max(0.01, r.f32(226)) if mand_id >= 25 else 1.0, set_cos=(hs & 2) != 0)
    s.cut_options = r.u8(176) & 7
    s.cut_pos = (r.f64(346), r.f64(354), r.f64(362))
    dt = r.u8(181)
    if dt & 1:
        zs = r.f32(164)
        s.dof = DofParams(z_sharp=zs, z_sharp2=r.f32(410) if mand_id >= 32 else zs,
                          clip_r=max(0.1, min(1000.0, r.f32(168))), aperture=max(0.0001, min(2.0, r.f32(172))),
                          passes=((dt >> 1) & 3) + 1, forward=((dt >> 3) & 1) != 0)
    s.lighting = parse_light(r)
    _parse_formulas(r, s, mand_id)
    _old_file_defaults(s, mand_id, r)
    return s


def _matrix_from_quat(q):
    return [
        [1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]), 2.0 * (q[0] * q[1] + q[2] * q[3]), 2.0 * (q[0] * q[2] - q[1] * q[3])],
        [2.0 * (q[0] * q[1] - q[2] * q[3]), 1.0 - 2.0 * (q[0] * q[0] + q[2] * q[2]), 2.0 * (q[2] * q[1] + q[0] * q[3])],
        [2.0 * (q[0] * q[2] + q[1] * q[3]), 2.0 * (q[1] * q[2] - q[0] * q[3]), 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1])],
    ]


def _parse_formulas(r: ByteReader, s: Scene, mand_id: int) -> None:
    a = HEADER_SIZE
    s.formulas = []
    if len(r.d) < HEADER_SIZE + ADDON_SIZE or not (16 <= r.d[a] <= 99):
        s.formulas = Scene().formulas
        return
    opt1, opt2, hyb1 = r.u8(a + 1), r.u8(a + 2), r.u8(a + 5)
    interp = (opt1 & 3) == 1
    if interp:
        s.interpolation = (r.f32(a + 8), r.f32(a + 8 + 188))
    if (opt1 & 3) == 2:
        hyb2 = r.u16(a + 6)

        def its(i):
            return r.i32(a + 8 + i * 188)

        x = 5
        while x > 0 and its(x) == 0:
            x -= 1
        start2 = max(1, min(hyb2 & 7, x))
        end2 = max(start2, x)
        repeat2 = max(start2, min((hyb2 >> 8) & 7, end2))
        x2 = 5
        while x2 > start2 and its(x2) <= 0:
            x2 -= 1
        repeat2 = min(repeat2, x2)
        s.decomb = DeCombParams(kind=max(1, min(6, r.u8(a + 3) + 1)), end1=start2 - 1, start2=start2, end2=end2,
                                repeat2=repeat2, iterations2=r.i32(418),
                                smooth=r.f32(378) if mand_id >= 31 else 0.5,
                                mix_pow=r.f32(104) if mand_id >= 39 else 2.0,
                                mix_color=r.u8(422) if mand_id >= 34 else 0)
    s.disable_analytic_de = (opt2 & 1) != 0
    s.inside = InsideMode.BOTH if opt2 & 4 else InsideMode.INSIDE if opt2 & 2 else InsideMode.OUTSIDE
    s.repeat_from = hyb1 >> 4
    for i in range(6):
        o = a + 8 + i * 188
        its_i = r.i32(o)
        fnr = r.i32(o + 4)
        nopt = max(0, min(16, r.i32(o + 8)))
        name = r.str(o + 12, 32)
        vals = [r.f64(o + 60 + 8 * k) for k in range(16)]
        vals = [v if math.isfinite(v) else 0.0 for v in vals]
        if interp:
            its_i = 1 if (i < 2 and fnr >= 0) else 0
        if its_i == 0:
            s.formulas.append(FormulaSlot("Integer Power", 0, 0, [8.0, -1.0] + [0.0] * 14, 2))
            continue
        if fnr in INTERNAL_FORMULA_NAMES:
            name = INTERNAL_FORMULA_NAMES[fnr]
            if fnr == 5 and nopt < 6:
                vals[5] = vals[4]
        elif 0 <= fnr < 20:
            name = name or f"internal #{fnr}"
        s.formulas.append(FormulaSlot(name, its_i, fnr, vals, nopt))
    while len(s.formulas) > 1 and s.formulas[-1].iterations == 0:
        s.formulas.pop()


def _old_file_defaults(s: Scene, mand_id: int, r: ByteReader) -> None:
    if mand_id < 21:
        if r.f32(182) < 0.01:
            s.z_step_div = 0.3
        s.optic = CameraOptic.COMMON
    if mand_id < 22:
        s.color_option = 0
        s.rot_4d = (0.0, 0.0, 0.0)
        s.color_mul = 1.0
        s.julia_c = (s.julia_c[0], s.julia_c[1], s.julia_c[2], 0.0)
    if mand_id < 23:
        s.raystep_limiter = 1.0
    if mand_id < 24:
        s.stereo_screen = (1.0, 2.0, 0.5)
    if mand_id < 26 and s.deao is not None:
        s.deao.dither = 0
    if mand_id < 28 and s.ao is not None:
        s.ao.random = min(1, s.ao.random)
    if mand_id < 29 and s.dof is not None:
        s.dof.clip_r, s.dof.aperture = 25.0, 0.02
    if mand_id < 35:
        s.dfog_on_it = 0
    if mand_id < 36:
        if s.ao is not None:
            s.ao.border_mirror = 0.0
        s.step_sub_de_stop = False


def parse_light(r: ByteReader) -> Lighting:
    b = LIGHT_BASE
    lt = Lighting()

    def tb(k):
        return r.i32(b + 8 + (k - 3) * 4)

    def light(i):
        return b + 68 + i * 32

    tbo = r.u32(b + 44)
    lt.var_col_z = float(r.i16(b))
    lt.roughness = float(r.u8(b + 2))
    lt.dyn_fog_col2 = r.rgb(b + 4)
    add_opt = r.u8(b + 7)
    lt.internal_gamma2 = (add_opt & 1) != 0
    lt.yc_comb = (add_opt & 4) != 0
    lt.bg_ambient = (add_opt & 32) != 0
    lt.amb_rel_obj = (tbo & 0x20000000) != 0
    lt.dfog_options = r.u8(light(0) + 24) & 3
    lt.ex_mode = r.u8(light(2) + 24)
    lt.no_col_ipol = (r.u8(light(3) + 24) & 1) != 0
    lt.diff_map = r.u8(b + 3) | (r.u8(light(1) + 16) << 8)
    lt.diff_map_mode = r.u8(light(1) + 24) & 3
    lt.diff_map_offset = ((tb(7) >> 12) & 0xFF, (tb(7) >> 20) & 0xFF)
    lt.diff_map_rot = (tb(8) >> 20) & 0xFF
    lt.diff_map_scale = r.u8(light(2) + 16)
    lt.fog_offset = float(tb(3) & 0xFFFF)
    lt.depth_fog = float(tb(4))
    lt.diffuse = float(tb(5))
    lt.dyn_fog = float(tb(6))
    lt.specular = float(tb(7) & 0xFFF)
    lt.ambient = float(tb(8) & 0xFFF)
    lt.color_start = float(tb(9))
    lt.color_end = float(tb(10))
    lt.amb_shadow = float(tb(11) & 0xFF)
    ind = (tb(11) >> 8) & 0xFF
    lt.ind_light = float(struct.unpack("<b", bytes([ind]))[0]) + 53.0
    lt.color_cycling = (tbo & 0x4000) != 0
    lt.color_on_otrap = ((tbo >> 17) & 1) != 0
    lt.far_fog = (tbo & 0x40000) != 0
    lt.depth_func = tbo >> 30
    lt.gamma = float((tbo >> 23) & 0x3F)
    lt.interior_start = float(tbo & 0x7F)
    lt.interior_end = float((tbo >> 7) & 0x7F)
    lt.fine_col_adj = (r.u8(b + 48), r.u8(b + 49)) if tbo & 0x10000 else None
    lt.amb_top = r.rgb(b + 52)
    lt.amb_bottom = r.rgb(b + 56)
    lt.depth_col = r.rgb(b + 60)
    lt.depth_col2 = r.rgb(b + 64)
    lt.dyn_fog_col = (r.u8(b + 55), r.u8(b + 59), r.u8(b + 63))
    lt.lights = []
    for i in range(6):
        o = light(i)
        lopt, lfunc = r.u8(o), r.u8(o + 1)
        on = (lopt & 3) == 0
        lt.lights.append(Light(
            on=on,
            positional=(lopt & 4) != 0,
            position=(r.d7b(o + 9), r.d7b(o + 17), r.d7b(o + 25)),
            visible=(((lopt >> 2) & 7) | ((lfunc & 128) >> 4)) & 14 if on else 0,
            color=r.rgb(o + 4),
            x_angle=r.d7b(o + 9),
            y_angle=r.d7b(o + 17),
            amplitude=short_float(r.u16(o + 2)),
            spec_func=lfunc & 7,
            diff_func=(lfunc >> 4) & 3,
            relative_to_object=((lopt >> 5) & 1) != 0,
            hs_enabled=((lopt >> 6) & 1) == 0,
            map=r.u16(o + 7) if (lopt & 3) == 2 else 0,
            map_rot=(r.u8(o + 9), r.u8(o + 17), r.u8(o + 25)),
        ))
    lt.diffuse_shadowing = r.u8(light(3) + 16) / 256.0
    lt.palette = []
    alpha = []
    for i in range(10):
        o = b + 260 + i * 10
        dif, spe = r.u32(o + 2), r.u32(o + 6)
        lt.palette.append(PaletteColor(r.u16(o), (dif & 0xFF, (dif >> 8) & 0xFF, (dif >> 16) & 0xFF),
                                       (spe & 0xFF, (spe >> 8) & 0xFF, (spe >> 16) & 0xFF)))
        alpha.append((spe >> 24) & 0xFF)
    if any(alpha[i] != max(lt.palette[i].specular) for i in range(10)):
        lt.palette_alpha = alpha
    lt.interior = []
    lt.interior_spec = []
    for i in range(4):
        o = b + 360 + i * 6
        c = r.u32(o + 2)
        lt.interior.append((r.u16(o), (c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF)))
        lt.interior_spec.append((c >> 24) & 0xFF)
    bg = r.str(b + 384, 24)
    if bg and all(32 <= ord(c) < 127 for c in bg):
        lt.bg_image = bg
        lt.bg_direct = (tbo & 0x8000) != 0
        lt.bg_rot = (r.u8(b + 50), r.u8(b + 51), r.u8(b + 67))
        lt.bg_brightness = r.u8(light(4) + 16)
        lt.bg_add_light = (add_opt & 8) != 0
    return lt
