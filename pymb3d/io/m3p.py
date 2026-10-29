"""Import of Mandelbulb3D parameter files (`.m3p`, MandId >= 20) and MB3D
text parameters (``Mandelbulb3Dv18{...}``).

Reads `TMandHeader10` (840 bytes with the light record `TLightingParas9`)
and `THeaderCustomAddon` (the hybrid formulas), following `LoadParameter`
in FileHandling.pas.  Settings this program does not model (DE
combinations, interpolation hybrids, Monte Carlo, DOF, volumetric light,
maps) are reported as warnings.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.camera import Camera
from ..core.lights import Light, Lighting, PaletteStop
from ..core.scene import AOSettings, DeCombination, FormulaSlot, Scene
from ..core.vec import normalise_matrix_to
from ..formulas import registry

HEADER_SIZE = 840
SLOT_SIZE = 188
ADDON_SIZE = 8 + 6 * SLOT_SIZE


class M3pError(Exception):
    pass


@dataclass
class M3pResult:
    scene: Scene
    warnings: list[str] = field(default_factory=list)
    mand_id: int = 0
    title: str = ""


class _Rd:
    def __init__(self, d: bytes):
        self.d = d

    def u8(self, o):
        return self.d[o]

    def u16(self, o):
        return struct.unpack_from("<H", self.d, o)[0]

    def i16(self, o):
        return struct.unpack_from("<h", self.d, o)[0]

    def i32(self, o):
        return struct.unpack_from("<i", self.d, o)[0]

    def u32(self, o):
        return struct.unpack_from("<I", self.d, o)[0]

    def f32(self, o):
        return float(struct.unpack_from("<f", self.d, o)[0])

    def f64(self, o):
        return struct.unpack_from("<d", self.d, o)[0]

    def rgb(self, o):
        return (self.d[o] / 255.0, self.d[o + 1] / 255.0, self.d[o + 2] / 255.0)

    def d7b(self, o):
        """`Double7B`: the 7 high bytes of a double."""
        return struct.unpack("<d", b"\x00" + self.d[o:o + 7])[0]

    def str(self, o, n):
        s = self.d[o:o + n]
        end = s.find(b"\x00")
        if end >= 0:
            s = s[:end]
        return s.decode("latin-1").strip()


def short_float(w: int) -> float:
    """`ShortFloatToSingle`: mantissa (shortint) * 10^(exponent - 1)."""
    m = struct.unpack("<b", bytes([w & 0xFF]))[0]
    e = struct.unpack("<b", bytes([(w >> 8) & 0xFF]))[0]
    e = max(-25, min(25, e))
    return float(m) * 10.0 ** (e - 1)


def matrix_from_quat(q) -> np.ndarray:
    return np.array([
        [1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]), 2.0 * (q[0] * q[1] + q[2] * q[3]), 2.0 * (q[0] * q[2] - q[1] * q[3])],
        [2.0 * (q[0] * q[1] - q[2] * q[3]), 1.0 - 2.0 * (q[0] * q[0] + q[2] * q[2]), 2.0 * (q[2] * q[1] + q[0] * q[3])],
        [2.0 * (q[0] * q[2] + q[1] * q[3]), 2.0 * (q[1] * q[2] - q[0] * q[3]), 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1])],
    ])


# ------------------------------------------------------------- text format
def four_chars_to_3_bytes(s: bytes) -> bytes | None:
    i = 0
    for j, c in enumerate(s[:4]):
        if 97 <= c <= 122:
            v = c - 59
        elif 65 <= c <= 90:
            v = c - 53
        elif 46 <= c <= 57:
            v = c - 46
        else:
            return None
        i |= v << (j * 6)
    return bytes([i & 0xFF, (i >> 8) & 0xFF, (i >> 16) & 0xFF])


def three_bytes_to_4_chars(b: bytes) -> str:
    i = b[0] | (b[1] << 8) | (b[2] << 16)
    out = ""
    for _ in range(4):
        c = i & 0x3F
        c += 46 if c < 12 else (53 if c < 38 else 59)
        out += chr(c)
        i >>= 6
    return out


def raw_from_text(text: str) -> tuple[bytes, str]:
    """`GetHeaderFromText`: decodes ``Mandelbulb3Dv18{...}{Titel: ...}``."""
    t = text.lstrip()
    open_ = t.find("{")
    if open_ < 0:
        raise M3pError("text parameters: missing '{'")
    num = ""
    for c in t[:open_]:
        if c == "." or c.isdigit():
            num += c
        else:
            num = ""
    try:
        ver = float(num) if num else 14.0
    except ValueError:
        ver = 13.0
    close = t.find("}", open_)
    if close < 0:
        raise M3pError("text parameters: missing '}'")
    body = bytes(c for c in t[open_ + 1:close].replace("\u2026", "...").encode("latin-1", "replace")
                 if c > 32 and c != 0x85)
    raw = bytearray()
    for k in range(0, len(body) - 3, 4):
        b = four_chars_to_3_bytes(body[k:k + 4])
        if b is None:
            raise M3pError(f"text parameters: bad characters {body[k:k + 4]!r}")
        raw += b
    if ver < 16.0:
        raise M3pError(f"text parameters of version {ver} are too old")
    if len(raw) < HEADER_SIZE:
        raise M3pError(f"text parameters too short ({len(raw)} of {HEADER_SIZE} header bytes)")
    raw += bytes(max(0, HEADER_SIZE + ADDON_SIZE - len(raw)))
    rest = t[close + 1:]
    title = "Mandelbulb3D"
    i = rest.find("{Titel: ")
    if i >= 0:
        r = rest[i + 8:]
        end = min([x for x in (r.find("}"), r.find("\n"), r.find("\r")) if x >= 0] or [len(r)])
        title = r[:end][:48]
    return bytes(raw[:HEADER_SIZE + ADDON_SIZE]), title


def raw_to_text(raw: bytes, title: str = "pymb3d") -> str:
    """`MakeTextparas`: the header as MB3D text parameters."""
    raw = bytes(raw) + bytes(max(0, HEADER_SIZE + ADDON_SIZE - len(raw)))
    a = HEADER_SIZE
    used = 0
    for i in range(6):
        if struct.unpack_from("<i", raw, a + 8 + i * SLOT_SIZE)[0] != 0:
            used = i + 1
    data = raw[:a + 8 + used * SLOT_SIZE]
    data += bytes((-len(data)) % 3)
    out = "".join(three_bytes_to_4_chars(data[k:k + 3]) for k in range(0, len(data), 3))
    lines = [out[k:k + 64] for k in range(0, len(out), 64)]
    return "Mandelbulb3Dv18{\n" + "\n".join(lines) + "\n}\n{Titel: " + title + "}\n"


# ---------------------------------------------------------------- upgrades
def _upgrade_hybrid_options(d: bytearray, mand_id: int) -> None:
    a = HEADER_SIZE
    if mand_id < 33:
        d[418:422] = d[12:16]
    if len(d) < a + ADDON_SIZE or not (16 <= d[a] <= 99):
        return
    vl = min(d[343], 5)
    d[343] = vl
    its = lambda i: struct.unpack_from("<i", d, a + 8 + i * SLOT_SIZE)[0]  # noqa: E731
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
        if opt3 not in (2, 6):
            d[12:16], d[418:422] = d[418:422], d[12:16]
    else:
        opt3 = 0
    if opt3 in (2, 6):
        first = bytes(d[a + 8:a + 8 + SLOT_SIZE])
        for i in range(j):
            d[a + 8 + i * SLOT_SIZE:a + 8 + (i + 1) * SLOT_SIZE] = d[a + 8 + (i + 1) * SLOT_SIZE:a + 8 + (i + 2) * SLOT_SIZE]
        d[a + 8 + j * SLOT_SIZE:a + 8 + (j + 1) * SLOT_SIZE] = first
        if opt3 == 6:
            opt3 = 5
        hyb1 = (max(j, 1) - 1) | ((max(vl, 1) - 1) << 4)
        hyb2 = j | (j << 4) | (j << 8)
    d[a + 3] = opt3
    d[a + 5] = hyb1 & 0xFF
    d[a + 6:a + 8] = struct.pack("<H", hyb2)


def _upgrade_light(d: bytearray) -> None:
    """`UpdateLightParasAbove3` (in place)."""
    b = 432
    light = lambda i: b + 68 + i * 32  # noqa: E731
    tbo = struct.unpack_from("<I", d, b + 44)[0]
    ver = (tbo >> 20) & 7
    verex = 0
    if ver < 4:
        d[b + 4], d[b + 5], d[b + 6] = d[b + 55], d[b + 59], d[b + 63]
    if ver < 5:
        tb6 = struct.unpack_from("<i", d, b + 20)[0]
        d[b + 20:b + 24] = struct.pack("<i", max(0, min(159, (tb6 - 53) * 2 + 53)))
        vz = struct.unpack_from("<h", d, b)[0]
        d[b:b + 2] = struct.pack("<h", max(-120, min(360, vz * 2)))
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
    d[b + 44:b + 48] = struct.pack("<I", tbo)
    if verex < 6:
        for i in range(6):
            o = light(i)
            if d[o] & 6 != 4:
                d[o + 2:o + 4] = struct.pack("<H", to_short_float(1.0))
    if verex < 7:
        for i in range(6):
            d[light(i) + 1] &= 0x3F


def to_short_float(v: float) -> int:
    if abs(v) < 1e-45 or not math.isfinite(v):
        return 0
    if abs(v) > 1e38:
        return 99 | (38 << 8)
    s, e = v, 0
    while abs(s) >= 9.95:
        s *= 0.1
        e += 1
    while abs(s) <= 0.995:
        s *= 10.0
        e -= 1
    m = int(round(s * 10.0))
    return (m & 0xFF) | ((e & 0xFF) << 8)


# ------------------------------------------------------------------ lights
def _parse_light(r: _Rd, w: list[str]) -> Lighting:
    b = 432
    l = Lighting()
    tb = lambda k: r.i32(b + 8 + (k - 3) * 4)  # noqa: E731
    tb_options = r.u32(b + 44)
    l.var_col_z = float(r.i16(b))
    l.fog_offset = float(tb(3) & 0xFFFF)
    l.depth_fog = float(tb(4))
    l.diffuse = float(tb(5))
    l.dyn_fog = float(tb(6))
    l.specular = float(tb(7) & 0xFFF)
    l.ambient = float(tb(8) & 0xFFF)
    l.color_start = float(tb(9))
    l.color_end = float(tb(10))
    l.amb_shadow = float(tb(11) & 0xFF)
    l.color_cycling = bool(tb_options & 0x4000)
    l.color_on_otrap = bool((tb_options >> 17) & 1)
    l.far_fog = bool(tb_options & 0x40000)
    l.gamma = float((tb_options >> 23) & 0x3F)
    l.no_color_interpolation = bool(r.u8(b + 68 + 3 * 32 + 24) & 1)
    l.roughness = float(r.u8(b + 2))
    add_opt = r.u8(b + 7)
    l.bg_ambient = bool(add_opt & 32)
    diff_map = r.u8(b + 3) | (r.u8(b + 68 + 32 + 16) << 8)
    if diff_map > 0:
        l.diffuse_map = str(diff_map)
        l.diffuse_map_mode = ("iter_otrap", "normals", "wrap", "wrap")[r.u8(b + 68 + 32 + 24) & 3]
        sc_byte = r.u8(b + 68 + 2 * 32 + 16)
        l.diffuse_map_scale = (1.2 if (r.u8(b + 68 + 32 + 24) & 3) else 1.05) ** (sc_byte - 30)
        l.diffuse_map_combine = bool(add_opt & 4)
    l.ambient_top = r.rgb(b + 52)
    l.ambient_bottom = r.rgb(b + 56)
    l.depth_color_top = r.rgb(b + 60)
    l.depth_color_bottom = r.rgb(b + 64)
    l.dyn_fog_color = (r.u8(b + 55) / 255.0, r.u8(b + 59) / 255.0, r.u8(b + 63) / 255.0)
    lights = []
    for i in range(6):
        o = b + 68 + i * 32
        lopt = r.u8(o)
        lfunc = r.u8(o + 1)
        on = (lopt & 3) == 0
        li = Light(
            enabled=on,
            positional=bool(lopt & 4),
            position=(r.d7b(o + 9), r.d7b(o + 17), r.d7b(o + 25)),
            color=r.rgb(o + 4),
            x_angle=r.d7b(o + 9),
            y_angle=r.d7b(o + 17),
            amplitude=short_float(r.u16(o + 2)),
            spec_func=lfunc & 7,
            diff_func=(lfunc >> 4) & 3,
            relative_to_object=bool((lopt >> 5) & 1),
            shadows=not ((lopt >> 6) & 1),
        )
        if (lopt & 3) == 2:
            # a light map instead of a light: the first one is used
            li.enabled = False
            if not l.light_map:
                l.light_map = str(r.u16(o + 7))
                l.light_map_intensity = li.amplitude
                l.light_map_rotation = tuple((r.u8(o + k) - 128) / 256.0 * 360.0 for k in (9, 17, 25))
        lights.append(li)
    l.lights = lights
    palette = []
    for i in range(10):
        o = b + 260 + i * 10
        dif = r.u32(o + 2)
        spe = r.u32(o + 6)
        palette.append(PaletteStop(r.u16(o), ((dif & 0xFF) / 255.0, ((dif >> 8) & 0xFF) / 255.0, ((dif >> 16) & 0xFF) / 255.0),
                                   ((spe & 0xFF) / 255.0, ((spe >> 8) & 0xFF) / 255.0, ((spe >> 16) & 0xFF) / 255.0)))
    l.palette = palette
    bg_name = r.str(b + 384, 24)
    if bg_name and all(c.isprintable() for c in bg_name):
        l.bg_image = bg_name
        l.bg_direct = bool(tb_options & 0x8000)
        l.bg_rotation = tuple((r.u8(k) - 128) / 256.0 * 360.0 for k in (b + 50, b + 51, b + 67))
        l.bg_brightness = 1.04 ** (r.u8(b + 68 + 4 * 32 + 16) - 40)
    return l


# ---------------------------------------------------------------- formulas
INTERNAL_FORMULAS = {0: "Integer Power", 1: "Real Power", 2: "Quaternion", 3: "Tricorn", 4: "Amazing Box",
                     5: "Bulbox", 6: "Folding Int Pow", 9: "Aexion C"}


def _formula_slot(fnr: int, name: str, vals: list[float], nopt: int, its: int, w: list[str]) -> FormulaSlot | None:
    if fnr < 20:
        if fnr not in INTERNAL_FORMULAS:
            w.append(f"internal formula #{fnr} ('{name}') is not supported; the slot is disabled")
            return None
        spec = registry.lookup(INTERNAL_FORMULAS[fnr])
        params = {}
        names = [p.name for p in spec.editable_params]
        for k, pname in enumerate(names[:nopt if nopt > 0 else len(names)]):
            params[pname] = vals[k]
        if fnr == 5 and nopt < 6:
            params["r_threshold2"] = vals[4]
        return FormulaSlot(spec.name, its, params)
    if not registry.has(name):
        w.append(f"custom formula '{name}' is not available; the slot is disabled")
        return None
    spec = registry.lookup(name)
    params = {p.name: vals[k] for k, p in enumerate(spec.editable_params) if k < nopt}
    return FormulaSlot(spec.name, its, params)


# ------------------------------------------------------------------- parse
def parse(data: bytes) -> M3pResult:
    if len(data) < HEADER_SIZE:
        raise M3pError("file too small for a MB3D parameter header")
    mand_id = struct.unpack_from("<i", data, 0)[0]
    if mand_id < 20:
        raise M3pError(f"parameter version {mand_id} is too old (MandId >= 20 needed)")
    d = bytearray(data) + bytearray(max(0, HEADER_SIZE + ADDON_SIZE - len(data)))
    if mand_id < 42:
        _upgrade_hybrid_options(d, mand_id)
    _upgrade_light(d)
    r = _Rd(bytes(d))
    w: list[str] = []
    s = Scene()
    s.width = r.i32(4)
    s.height = r.i32(8)
    s.iterations = r.i32(12)
    ioptions = r.u16(16)
    new_options = r.u8(18)
    z_start = r.f64(20)
    z_end = r.f64(28)
    mid = np.array([r.f64(36), r.f64(44), r.f64(52)])
    s.rot_4d = (r.f64(60), r.f64(68), r.f64(76))
    zoom = r.f64(84)
    if not (1 <= s.width < 32768 and 1 <= s.height < 32768):
        raise M3pError(f"invalid image size {s.width}x{s.height}")
    if not (math.isfinite(zoom) and zoom > 0):
        raise M3pError(f"invalid zoom {zoom}")
    s.rstop = r.f64(92)
    fov_y = r.f64(108)
    s.bin_search_steps = r.u8(134)
    s.min_iterations = r.u16(135)
    optic = r.u8(148) & 3
    if optic != 0:
        w.append("planar / panorama optics are rendered with the common perspective")
    s.vary_de_stop_on_fov = r.u8(162) != 0
    s.de_stop = r.f32(177)
    s.z_step_div = max(0.001, min(1.0, r.f32(182)))
    s.julia = r.u8(190) != 0
    s.julia_c = (r.f64(191), r.f64(199), r.f64(207), r.f64(215))
    s.raystep_limiter = r.f32(242)
    m = np.array([[r.f64(246 + (i * 3 + j) * 8) for j in range(3)] for i in range(3)])
    if mand_id > 43 and (new_options & 1):
        m = matrix_from_quat([m[0, 0], m[0, 1], m[0, 2], m[1, 0]])
    s.camera = Camera(mid=mid, zoom=zoom, fov_y=fov_y, z_start=z_start, z_end=z_end,
                      vgrads=normalise_matrix_to(1.0, m))
    s.color_option = min(r.u8(342), 5)
    s.color_mul = r.f32(338) if mand_id >= 22 else 1.0
    if r.u8(344) == 0:
        s.slice_2d = {1: 1, 3: 3}.get(r.u8(345), 2)
    s.first_step_random = bool(ioptions & 1)
    s.step_sub_de_stop = bool(ioptions & 4) and mand_id >= 36
    s.smooth_normals = min(8, (ioptions >> 6) & 15)
    ao_bits = r.u8(149)
    if ao_bits & 1:
        kind = (ao_bits >> 2) & 3
        s.ao = AOSettings(mode="de" if kind == 3 else "ssao", strength=1.0, radius=32.0, samples=5)
    else:
        s.ao = AOSettings(mode="none")
    hs = r.u8(133)
    s.shadows = bool(hs & 1) and (hs >> 2) != 0
    s.cut_options = r.u8(176) & 7
    s.cut_pos = (r.f64(346), r.f64(354), r.f64(362))
    if r.u8(181) & 1:
        s.post.dof = True
        s.post.dof_focus = max(0.0, min(1.0, r.f32(164)))
        s.post.dof_aperture = max(0.0001, min(2.0, r.f32(172))) * 100.0
        s.post.dof_forward_only = bool((r.u8(181) >> 3) & 1)
    if mand_id >= 44 and (r.u8(343) & 7):
        w.append("volumetric light is not supported")
    s.lighting = _parse_light(r, w)

    # ---- formulas (THeaderCustomAddon)
    s.slots = []
    a = HEADER_SIZE
    if len(d) >= HEADER_SIZE + ADDON_SIZE and 16 <= d[a] <= 99:
        opt1 = r.u8(a + 1)
        opt2 = r.u8(a + 2)
        hyb1 = r.u8(a + 5)
        if opt1 & 3 == 1:
            s.interpolation = (r.f32(a + 8), r.f32(a + 8 + SLOT_SIZE))
        elif opt1 & 3 == 2:
            # CheckHybridOptions
            hyb2 = r.u16(a + 6)
            its = lambda i: r.i32(a + 8 + i * SLOT_SIZE)  # noqa: E731
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
            kind = max(1, min(6, r.u8(a + 3) + 1))
            s.decomb = DeCombination(kind=kind, start2=start2, repeat_from2=repeat2, iterations2=max(1, r.i32(418)),
                                     smooth=(r.f32(378) if mand_id >= 31 else 0.5),
                                     mix_pow=(r.f32(104) if mand_id >= 39 else 2.0),
                                     mix_color=(r.u8(422) if mand_id >= 34 else 0))
        if opt2 & 1:
            s.analytic_de = False
        s.inside_mode = 2 if opt2 & 4 else (1 if opt2 & 2 else 0)
        s.repeat_from = hyb1 >> 4
        for i in range(6):
            o = a + 8 + i * SLOT_SIZE
            its = r.i32(o)
            fnr = r.i32(o + 4)
            nopt = max(0, min(16, r.i32(o + 8)))
            name = r.str(o + 12, 32)
            vals = [r.f64(o + 60 + 8 * k) for k in range(16)]
            interp = opt1 & 3 == 1
            if interp:
                its = 1 if (i < 2 and fnr >= 0) else 0
            if its == 0:
                continue
            if its < 0:
                w.append(f"slot {i + 1} ('{name}'): uncounted iterations are run as counted")
                its = -its
            slot = _formula_slot(fnr, name, vals, nopt, its, w)
            # a missing formula leaves an inactive slot so that the slot
            # indices of the hybrid options keep their meaning
            s.slots.append(slot if slot is not None else FormulaSlot("Integer Power", 0, {}))
    if not any(x.active for x in s.slots):
        w.append("no usable formula; using the default Mandelbulb")
        s.slots = [FormulaSlot("Integer Power", 1, {"power": 8})]
    s.repeat_from = min(s.repeat_from, max(0, len(s.slots) - 1))
    # the upgrades of LoadParameter for older files
    if mand_id < 21 and r.f32(182) < 0.01:
        s.z_step_div = 0.3
    if mand_id < 22:
        s.rot_4d = (0.0, 0.0, 0.0)
        s.julia_c = (s.julia_c[0], s.julia_c[1], s.julia_c[2], 0.0)
    if mand_id < 23:
        s.raystep_limiter = 1.0
    return M3pResult(s, w, mand_id)


def load(path: str | Path) -> M3pResult:
    path = Path(path)
    data = path.read_bytes()
    text_at = data.find(b"Mandelbulb3Dv")
    title = path.stem
    if text_at >= 0:
        raw, title = raw_from_text(data[text_at:].decode("latin-1"))
        res = parse(raw)
    else:
        res = parse(data)
    res.title = title
    res.scene.title = title
    return res


def load_text(text: str) -> M3pResult:
    raw, title = raw_from_text(text)
    res = parse(raw)
    res.title = title
    res.scene.title = title
    return res
