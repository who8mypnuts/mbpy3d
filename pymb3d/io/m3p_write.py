"""Export of a scene as a Mandelbulb3D parameter file (`.m3p`, MandId 44)
or as MB3D text parameters, following `write` of the Rust port / MB3D's
`TMandHeader10` layout.  Settings this program does not model keep MB3D's
neutral values; custom formulas are written with their MB3D names and the
option values in MB3D's order, so MB3D finds its own `.m3f` files."""
from __future__ import annotations

import math
import struct

import numpy as np

from ..core.scene import Scene
from ..core.vec import normalise_matrix_to
from ..formulas import registry
from .m3p import ADDON_SIZE, HEADER_SIZE, SLOT_SIZE, raw_to_text, to_short_float

INTERNAL_IDS = {"Integer Power": 0, "Real Power": 1, "Quaternion": 2, "Tricorn": 3, "Amazing Box": 4, "Bulbox": 5,
                "Folding Int Pow": 6, "Aexion C": 9}


def _rgb(c) -> bytes:
    return bytes(int(round(max(0.0, min(1.0, v)) * 255)) for v in c)


def write(scene: Scene) -> bytes:
    d = bytearray(HEADER_SIZE + ADDON_SIZE)

    def put(o, fmt, *v):
        struct.pack_into(fmt, d, o, *v)

    def d7b(o, v):
        d[o:o + 7] = struct.pack("<d", v)[1:8]

    sc = scene
    cam = sc.camera
    put(0, "<i", 44)
    put(4, "<i", sc.width)
    put(8, "<i", sc.height)
    put(12, "<i", sc.iterations)
    put(16, "<H", int(sc.first_step_random))
    put(20, "<d", cam.z_start)
    put(28, "<d", cam.z_end)
    for k in range(3):
        put(36 + 8 * k, "<d", float(cam.mid[k]))
        put(60 + 8 * k, "<d", float(sc.rot_4d[k]))
    put(84, "<d", cam.zoom)
    rstop = sc.rstop if sc.rstop else max([registry.lookup(s.name).rstop for s in sc.active_slots] + [16.0])
    put(92, "<d", rstop)
    put(104, "<f", 2.0)
    put(108, "<d", cam.fov_y)
    put(116, "<f", 1.5)
    put(120, "<f", 1.0)
    d[124] = 2 | 16
    d[132] = 1
    d[134] = max(0, min(255, sc.bin_search_steps))
    put(135, "<H", max(0, min(65535, sc.min_iterations)))
    d[148] = 0
    if sc.ao.mode != "none":
        kind = 3 if sc.ao.mode == "de" else 1
        d[149] = 1 | (kind << 2)
        put(319, "<f", 0.0)
        put(374, "<f", 1.0)
    if sc.shadows:
        lights_bits = sum(1 << i for i, li in enumerate(sc.lighting.lights[:6]) if li.enabled and li.shadows) & 0x3F
        d[133] = 1 | (lights_bits << 2)
    put(224, "<H", to_short_float(1.0))
    put(226, "<f", 1.0)
    d[162] = int(sc.vary_de_stop_on_fov)
    d[176] = sc.cut_options & 7
    for k in range(3):
        put(346 + 8 * k, "<d", float(sc.cut_pos[k]))
    put(177, "<f", sc.de_stop)
    put(182, "<f", sc.z_step_div)
    d[186] = 3
    d[190] = int(sc.julia)
    for k in range(4):
        put(191 + 8 * k, "<d", float(sc.julia_c[k]))
    put(230, "<f", 1.0)
    put(234, "<f", 2.0)
    put(238, "<f", 0.5)
    put(242, "<f", sc.raystep_limiter)
    m = normalise_matrix_to(cam.step_width(sc.width), cam.vgrads)
    for i in range(3):
        for j in range(3):
            put(246 + (i * 3 + j) * 8, "<d", float(m[i, j]))
    d[318] = 32
    put(338, "<f", sc.color_mul)
    d[342] = max(0, min(5, sc.color_option))
    d[343] = 2 << 4
    d[344] = 1
    d[345] = 2
    put(378, "<f", sc.decomb.smooth if sc.decomb.kind else 0.5)
    put(418, "<i", sc.decomb.iterations2 if sc.decomb.kind else sc.iterations)
    d[423] = 128

    # ---- TLightingParas9
    l = sc.lighting
    b = 432
    put(b, "<h", int(round(l.var_col_z)))
    d[b + 2] = 255
    d[b + 4:b + 7] = _rgb(l.dyn_fog_color)
    tbv = [
        (int(round(l.fog_offset)) & 0xFFFF) | (128 << 16),
        int(round(l.depth_fog)),
        int(round(l.diffuse)),
        int(round(l.dyn_fog)),
        int(round(l.specular)) & 0xFFF,
        int(round(l.ambient)) & 0xFFF,
        int(round(l.color_start)),
        int(round(l.color_end)),
        int(round(l.amb_shadow)) & 0xFF,
    ]
    for k, v in enumerate(tbv):
        put(b + 8 + k * 4, "<i", v)
    tbo = (60 & 0x7F) | ((60 & 0x7F) << 7) | (7 << 20)
    tbo |= int(l.color_cycling) << 14 | int(l.color_on_otrap) << 17 | int(l.far_fog) << 18
    tbo |= (int(round(l.gamma)) & 0x3F) << 23
    put(b + 44, "<I", tbo)
    d[b + 52:b + 55] = _rgb(l.ambient_top)
    d[b + 56:b + 59] = _rgb(l.ambient_bottom)
    d[b + 60:b + 63] = _rgb(l.depth_color_top)
    d[b + 64:b + 67] = _rgb(l.depth_color_bottom)
    fog = _rgb(l.dyn_fog_color)
    d[b + 55], d[b + 59], d[b + 63] = fog[0], fog[1], fog[2]
    for i, li in enumerate(l.lights[:6]):
        o = b + 68 + i * 32
        lopt = 0 if li.enabled else 1
        lopt |= int(li.positional) << 2 | int(li.relative_to_object) << 5 | int(not li.shadows) << 6
        d[o] = lopt
        d[o + 1] = (li.spec_func & 7) | ((li.diff_func & 3) << 4)
        put(o + 2, "<H", to_short_float(li.amplitude))
        d[o + 4:o + 7] = _rgb(li.color)
        if li.positional:
            d7b(o + 9, li.position[0])
            d7b(o + 17, li.position[1])
            d7b(o + 25, li.position[2])
        else:
            d7b(o + 9, li.x_angle)
            d7b(o + 17, li.y_angle)
    light = lambda i: b + 68 + i * 32  # noqa: E731
    d[light(0) + 16] = 8
    d[light(2) + 16] = 30
    d[light(3) + 24] = int(l.no_color_interpolation)
    d[light(3) + 16] = 0
    d[light(4) + 16] = 40
    for i, c in enumerate(l.palette[:10]):
        o = b + 260 + i * 10
        put(o, "<H", max(0, min(32767, c.position)))
        d[o + 2:o + 5] = _rgb(c.diffuse)
        d[o + 6:o + 9] = _rgb(c.specular)
        d[o + 9] = max(_rgb(c.specular))
    for i in range(4):
        o = b + 360 + i * 6
        pos, col = l.palette[min(i, 9)].position if i < 3 else 32700, l.palette[min(i, 9)].diffuse
        put(o, "<H", max(0, min(32767, pos)))
        d[o + 2:o + 5] = _rgb(col)
        d[o + 5] = 160

    # ---- THeaderCustomAddon
    a = HEADER_SIZE
    d[a] = 16
    slots = sc.slots[:6]
    last = max([i for i, s in enumerate(slots) if s.active] + [0])
    if sc.decomb.kind:
        d[a + 1] = 2
        d[a + 3] = max(1, min(6, sc.decomb.kind)) - 1
        end1 = max(0, sc.decomb.start2 - 1)
        d[a + 5] = (end1 & 7) | ((min(sc.repeat_from, end1) & 7) << 4)
        put(a + 6, "<H", (sc.decomb.start2 & 7) | ((last & 7) << 4) | ((sc.decomb.repeat_from2 & 7) << 8))
    else:
        d[a + 5] = (last & 7) | ((min(sc.repeat_from, last) & 7) << 4)
        put(a + 6, "<H", 0x151)
    d[a + 2] = int(not sc.analytic_de)
    d[a + 4] = last + 1
    for i, s in enumerate(slots):
        o = a + 8 + i * SLOT_SIZE
        if not s.active:
            continue
        spec = registry.lookup(s.name)
        fnr = INTERNAL_IDS.get(spec.name, 20)
        vals = spec.values(s.params)
        opts = [(p.name, vals[p.name], 2 if p.kind in ("int", "bool") else 0) for p in spec.editable_params][:16]
        put(o, "<i", s.iterations)
        put(o + 4, "<i", fnr)
        put(o + 8, "<i", len(opts))
        name = spec.name.encode("latin-1", "replace")[:31]
        d[o + 12:o + 12 + len(name)] = name
        for k, (_, v, t) in enumerate(opts):
            d[o + 44 + k] = t
            put(o + 60 + 8 * k, "<d", float(v))
    return bytes(d)


def save_m3p(scene: Scene, path) -> None:
    from pathlib import Path

    Path(path).write_bytes(write(scene))


def to_text(scene: Scene, title: str = "pymb3d") -> str:
    return raw_to_text(write(scene), title)
