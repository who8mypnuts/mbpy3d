"""Deferred shading of a G-buffer (numpy, vectorised over the image).

Follows `CalcPixelColor2` of Mandelbulb3D: up to six lights with MB3D's
four diffuse functions and Phong specular, ambient top/bottom colours with
ambient occlusion, the 10-stop palette on the smooth iteration count (or
the orbit trap), depth fog, dynamic fog and gamma.  Colours are computed on
MB3D's 0..255 scale and returned as float32 in 0..1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..core import params as PL
from ..core.camera import CameraSetup
from ..core.lights import Light, Lighting
from ..core.scene import Scene
from .maps import find_map, load_map, rotation_matrix, sample_sphere, sample_xy, small_blurred


@dataclass
class SceneMaps:
    background: np.ndarray | None = None
    bg_small: np.ndarray | None = None
    bg_rot: np.ndarray | None = None
    light: np.ndarray | None = None
    light_avg: np.ndarray | None = None
    light_rot: np.ndarray | None = None
    diffuse: np.ndarray | None = None


def load_scene_maps(scene: Scene) -> SceneMaps:
    """The images a scene's lighting refers to (missing files are ignored)."""
    l = scene.lighting
    m = SceneMaps()
    dirs = [scene.base_dir] if scene.base_dir else []
    dirs.append(".")
    if l.bg_image:
        p = find_map(l.bg_image, dirs)
        if p is not None:
            m.background = load_map(p)
            m.bg_rot = rotation_matrix(l.bg_rotation)
            if l.bg_ambient:
                m.bg_small = small_blurred(m.background)
    if l.light_map:
        p = find_map(l.light_map, dirs)
        if p is not None:
            m.light = load_map(p)
            m.light_avg = m.light.reshape(-1, 3).mean(axis=0)
            m.light_rot = rotation_matrix(l.light_map_rotation)
    if l.diffuse_map:
        p = find_map(l.diffuse_map, dirs)
        if p is not None:
            m.diffuse = load_map(p)
    return m


def diffuse_map_color(maps: "SceneMaps", scene: Scene, n_world: np.ndarray, d_rough: np.ndarray, color: np.ndarray,
                      otrap: np.ndarray, positions: np.ndarray | None) -> np.ndarray:
    """The diffuse colour from the map: by the normal, by the colour
    index / orbit trap, or wrapped around the object position."""
    l = scene.lighting
    mode = l.diffuse_map_mode
    scale = max(1e-6, l.diffuse_map_scale)
    if mode == "normals" or (mode == "wrap" and positions is None):
        c = sample_sphere(maps.diffuse, n_world)
    elif mode == "wrap":
        u = np.mod(positions[..., 0] * scale * 0.5, 1.0)
        v = np.mod(positions[..., 2] * scale * 0.5, 1.0)
        c = sample_xy(maps.diffuse, u, v, wrap=True)
    else:
        u = np.mod(color / 32768.0 * scale, 1.0)
        v = np.mod(otrap / 32768.0 * scale, 1.0)
        c = sample_xy(maps.diffuse, u, v, wrap=True)
    avg = maps.diffuse.reshape(-1, 3).mean(axis=0)
    return c * (1.0 - d_rough)[..., None] + avg * d_rough[..., None]


# ----------------------------------------------------------------- palette
def palette_tables(lighting: Lighting) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(positions 0..32767, diffuse colours, specular colours) of the palette
    stops, positions made strictly increasing for interpolation."""
    pos = np.array([float(p.position) for p in lighting.palette], dtype=np.float64)
    pos = np.maximum.accumulate(pos)
    for i in range(1, len(pos)):
        if pos[i] <= pos[i - 1]:
            pos[i] = pos[i - 1] + 1e-3
    dif = np.array([p.diffuse for p in lighting.palette], dtype=np.float32)
    spe = np.array([p.specular for p in lighting.palette], dtype=np.float32)
    return pos, dif, spe


def color_range(lighting: Lighting) -> tuple[float, float]:
    """`CalcSCstartAndSCmul`: the gradient window from the colour
    start/end sliders."""
    s_c_start = ((lighting.color_start + 30.0) / 90.0) ** 2 * 32767.0 - 10900.0
    s_c_mul = ((lighting.color_end + 30.0) / 90.0) ** 2 * 32767.0 - 10900.0 - s_c_start
    if abs(s_c_mul) > 0.001:
        s_c_mul = 2.0 / s_c_mul
    else:
        s_c_mul = -2000.0 if s_c_mul < 0 else 2000.0
    return s_c_start, s_c_mul


def gradient_index(gbuf: np.ndarray, scene: Scene) -> np.ndarray:
    """MB3D's `si_gradient` (0..32767), computed by the kernel."""
    return gbuf[..., PL.G_COLOR].astype(np.float64)


def palette_lookup(index: np.ndarray, lighting: Lighting) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse and specular colour (0..1) of every pixel from the palette
    position ``index`` (MB3D's `ir`, 0..32767)."""
    pos, dif, spe = palette_tables(lighting)
    if lighting.color_cycling:
        ir = np.mod(index, 32768.0)
    else:
        ir = index
    if lighting.no_color_interpolation:
        k = np.clip(np.searchsorted(pos, ir, side="right") - 1, 0, len(pos) - 1)
        return dif[k], spe[k]
    out_d = np.empty(ir.shape + (3,), dtype=np.float32)
    out_s = np.empty(ir.shape + (3,), dtype=np.float32)
    for c in range(3):
        out_d[..., c] = np.interp(ir, pos, dif[:, c], left=dif[0, c], right=dif[-1, c])
        out_s[..., c] = np.interp(ir, pos, spe[:, c], left=spe[0, c], right=spe[-1, c])
    return out_d, out_s


def surface_colors(gbuf: np.ndarray, scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """`CalcColors`: palette colours of the G-buffer."""
    l = scene.lighting
    s_c_start, s_c_mul = color_range(l)
    if l.color_on_otrap or scene.color_on_otrap:
        src = gbuf[..., PL.G_OTRAP].astype(np.float64)
    else:
        src = gradient_index(gbuf, scene)
    ir = ((src - s_c_start) * s_c_mul) * 16384.0
    return palette_lookup(ir, l)


# ------------------------------------------------------------ light helpers
@lru_cache(maxsize=1)
def cos_tables() -> np.ndarray:
    """`SetCosTabFunction`: MB3D's four diffuse functions sampled at 128
    points over the dot product, and their blurred versions (4..7) that the
    surface roughness blends in."""
    tabs = np.zeros((8, 128), dtype=np.float64)
    for i in range(128):
        d = 1.0 - (i - 2.0) / 60.0
        if d > 0.15:
            tabs[0, i] = (d - 0.08) * 1.0869565
        elif d <= 0.0:
            tabs[0, i] = 0.0
        else:
            tabs[0, i] = d ** max(1.0, (0.505 - d) * 3.8)
        dc = max(d, 0.0)
        tabs[1, i] = dc * dc
        tabs[2, i] = d * 0.5 + 0.5
        tabs[3, i] = (d * 0.5 + 0.5) ** 2
    for k in range(4):
        tmp = np.sqrt(np.maximum(tabs[k], 0.0))
        for j in range(128):
            e = 0.0
            for i in range(61):
                l = abs(j + i - 30)
                if l < 128:
                    e += tmp[l]
            a = e * 0.011 + (e * 0.007) ** 2
            tabs[k + 4, j] = a * a
    return tabs


def diffuse_function(kind: int, d: np.ndarray, rough: np.ndarray | float = 0.0) -> np.ndarray:
    """`GetCosTabVal`: the diffuse light function ``kind`` (0..3) of the
    dot product ``d``, blended towards its blurred version by ``rough``."""
    tabs = cos_tables()
    t = np.clip(62.0 - 60.0 * d, 0.0, 127.0)
    idx = np.arange(128, dtype=np.float64)
    a = np.interp(t, idx, tabs[kind & 3])
    if np.isscalar(rough) and rough == 0.0:
        return a
    b = np.interp(t, idx, tabs[(kind & 3) + 4])
    return a + rough * (b - a)


def light_vectors_view(light: Light, vgrads: np.ndarray) -> np.ndarray:
    """Unit vector towards the light in view space (right, up, forward)."""
    d = np.asarray(light.direction_view(), dtype=np.float64)
    if light.relative_to_object:
        # the angles describe a world space direction: rotate into the view
        unit = vgrads / np.linalg.norm(vgrads, axis=1, keepdims=True)
        d = unit @ d
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


@lru_cache(maxsize=8)
def view_directions(width: int, height: int, fov_y: float) -> np.ndarray:
    """Per pixel unit view direction in view space (H, W, 3), cached."""
    ix = np.arange(width, dtype=np.float64)
    iy = np.arange(height, dtype=np.float64)
    cafx = (0.5 * width - (ix + 1.0)) * (fov_y / height)
    cafy = (iy / height - 0.5) * fov_y
    vx = -np.sin(cafx)[None, :] * np.ones((height, 1))
    vy = np.sin(cafy)[:, None] * np.ones((1, width))
    vz = np.cos(cafy)[:, None] * np.cos(cafx)[None, :]
    v = np.stack([vx, vy, vz], axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def object_positions(depth: np.ndarray, scene: Scene, full_size: tuple[int, int], origin: tuple[int, int],
                     size: tuple[int, int]) -> np.ndarray:
    """World positions (H, W, 3) of the surface points of a G-buffer."""
    W_full, H_full = full_size
    ox, oy = origin
    W, H = size
    setup = CameraSetup.build(scene.camera, W_full, H_full, scene.raystep_limiter, scene.z_step_div,
                              scene.vary_de_stop_on_fov)
    ix = np.arange(W, dtype=np.float64) + ox
    iy = np.arange(H, dtype=np.float64) + oy
    cafx = (setup.fovx_off - (ix + 1.0)) * setup.fovx_mul
    cafy = (iy / H_full - 0.5) * setup.fov_y
    vx = -np.sin(cafx)[None, :] * np.ones((H, 1))
    vy = np.sin(cafy)[:, None] * np.ones((1, W))
    vz = np.cos(cafy)[:, None] * np.cos(cafx)[None, :]
    v = np.stack([vx, vy, vz], axis=-1)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    d = v @ setup.vgrads
    start = setup.ystart + setup.vgrads[0] * ix[None, :, None] + setup.vgrads[1] * iy[:, None, None]
    return start + d * np.maximum(depth, 0.0)[..., None]


def depth_word(depth: np.ndarray, scene: Scene, full_size: tuple[int, int]) -> np.ndarray:
    """MB3D's 16 bit depth (`CalcZposAndRough`): 32767 at the start plane,
    towards 0 at the far end, 32768 for the background."""
    W, H = full_size
    cam = scene.camera
    sw = cam.step_width(W)
    fov = math.radians(cam.fov_y)
    zcorr = math.sin(max(fov, 1.0) / H)
    zend = max(1e-10, (cam.z_end - cam.z_start) / sw)
    zc_mul = 32767.0 / (math.sqrt(zend * zcorr + 1.0) - 0.999999999)
    v = zc_mul * (np.sqrt(np.maximum(depth, 0.0) * zcorr + 1.0) - 1.0)
    word = np.clip(32767.0 - v, 0.0, 32767.0)
    word[depth < 0] = 32768.0
    return word


@dataclass
class ShadeOptions:
    #: apply the ambient occlusion channel
    use_ao: bool = True
    #: apply the shadow channels
    use_shadows: bool = True
    #: the full image size the G-buffer belongs to (tiles): (width, height)
    full_size: tuple[int, int] | None = None
    #: position of this G-buffer in the full image
    origin: tuple[int, int] = (0, 0)


@dataclass
class Layer:
    """One marched surface layer: the primary hits or the reflections."""

    depth: np.ndarray            # (H, W) pixel units, < 0 none
    normal: np.ndarray           # (H, W, 3) world space
    color: np.ndarray            # (H, W) si_gradient
    otrap: np.ndarray            # (H, W)
    ao: np.ndarray               # (H, W) 1 = open
    shadow: np.ndarray           # (H, W, 6) 1 = lit
    rough: np.ndarray            # (H, W) 0..1
    view: np.ndarray             # (H, W, 3) unit ray direction, world space
    position: np.ndarray | None  # (H, W, 3) hit positions (lazy when None)
    total_depth: np.ndarray      # (H, W) distance from the camera for the fog

    @property
    def hit(self) -> np.ndarray:
        return self.depth >= 0.0


def primary_layer(gbuf: np.ndarray, scene: Scene, opts: ShadeOptions, view_world: np.ndarray) -> Layer:
    depth = gbuf[..., PL.G_DEPTH].astype(np.float64)
    return Layer(depth, gbuf[..., PL.G_NX:PL.G_NZ + 1].astype(np.float64), gbuf[..., PL.G_COLOR].astype(np.float64),
                 gbuf[..., PL.G_OTRAP].astype(np.float64), gbuf[..., PL.G_AO].astype(np.float64),
                 gbuf[..., PL.G_SHADOW:PL.G_SHADOW + 6].astype(np.float64), gbuf[..., PL.G_ROUGH].astype(np.float64),
                 view_world, None, depth)


def reflection_layer(gbuf: np.ndarray, scene: Scene, opts: ShadeOptions, prim: Layer, positions: np.ndarray) -> Layer:
    depth = gbuf[..., PL.G_R_DEPTH].astype(np.float64)
    n = prim.normal
    ndotv = np.sum(n * prim.view, axis=-1, keepdims=True)
    rdir = prim.view - n * (2.0 * ndotv)
    rn = np.linalg.norm(rdir, axis=-1, keepdims=True)
    rdir = rdir / np.where(rn > 0, rn, 1.0)
    W = (opts.full_size or (gbuf.shape[1], gbuf.shape[0]))[0]
    sw = scene.camera.step_width(W)
    pos = positions + rdir * np.maximum(depth, 0.0)[..., None] * sw
    return Layer(depth, gbuf[..., PL.G_R_NX:PL.G_R_NZ + 1].astype(np.float64), gbuf[..., PL.G_R_COLOR].astype(np.float64),
                 gbuf[..., PL.G_R_OTRAP].astype(np.float64), gbuf[..., PL.G_R_AO].astype(np.float64),
                 gbuf[..., PL.G_R_SHADOW:PL.G_R_SHADOW + 6].astype(np.float64), np.zeros_like(depth),
                 rdir, pos, prim.depth + np.maximum(depth, 0.0))


def palette_colors_of(color: np.ndarray, otrap: np.ndarray, scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    l = scene.lighting
    s_c_start, s_c_mul = color_range(l)
    src = otrap if (l.color_on_otrap or scene.color_on_otrap) else color
    ir = ((src - s_c_start) * s_c_mul) * 16384.0
    d, sp = palette_lookup(ir, l)
    return d.astype(np.float64), sp.astype(np.float64)


def shade_layer(layer: Layer, scene: Scene, opts: ShadeOptions, maps: "SceneMaps", gbuf: np.ndarray,
                full_size: tuple[int, int], origin: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lights one layer.  Returns (colour 0..255 of the hit pixels, the
    specular palette colour, the surface positions used)."""
    l = scene.lighting
    H, W = layer.depth.shape
    vg = scene.camera.vgrads
    unit = vg / np.linalg.norm(vg, axis=1, keepdims=True)
    n_world = layer.normal
    n_view = n_world @ unit.T
    nn = np.linalg.norm(n_view, axis=-1, keepdims=True)
    n_view = n_view / np.where(nn > 0, nn, 1.0)
    occ = 1.0 - layer.ao if opts.use_ao else np.zeros((H, W))
    s_amb_shad = l.amb_shadow_mul
    if s_amb_shad < 1.0:
        d_amb_sh = 1.0 - s_amb_shad * occ
    else:
        a = 1.0 - occ
        d_amb_sh = a + (s_amb_shad - 1.0) * (a * a - a)
    d_amb_sh = np.clip(d_amb_sh, 0.0, 1.0)

    idif, ispe = palette_colors_of(layer.color, layer.otrap, scene)
    d_rough = layer.rough * (l.roughness / 255.0)
    positions = layer.position
    if maps.diffuse is not None:
        if positions is None and l.diffuse_map_mode == "wrap":
            positions = object_positions(layer.depth, scene, full_size, origin, (W, H))
        mapped = diffuse_map_color(maps, scene, n_world, d_rough, layer.color, layer.otrap, positions)
        if l.diffuse_map_combine:
            luma_m = mapped @ np.array([0.3, 0.59, 0.11])
            luma_p = idif @ np.array([0.3, 0.59, 0.11])
            idif = idif * (luma_m / (0.01 + luma_p))[..., None]
        else:
            idif = mapped

    s_diff = l.diffuse_mul
    s_spec = l.specular_mul
    li_dif = np.zeros((H, W, 3))
    li_spe = np.zeros((H, W, 3))
    li_amb = np.zeros((H, W, 3))
    view_world = layer.view
    ndotv = np.sum(n_world * view_world, axis=-1, keepdims=True)
    refl = view_world - n_world * (2.0 * ndotv)
    for i, light in enumerate(l.lights[: PL.MAX_LIGHTS]):
        if not light.enabled:
            continue
        col = np.asarray(light.color, dtype=np.float64) * 255.0 * light.amplitude
        if light.positional:
            if positions is None:
                positions = object_positions(layer.depth, scene, full_size, origin, (W, H))
            lv = np.asarray(light.position, dtype=np.float64) - positions
            d2 = np.sum(lv * lv, axis=-1, keepdims=True)
            ln = lv / np.sqrt(np.maximum(d2, 1e-30))
            att = (1.3 / (d2[..., 0] + 0.01))
            dotp = np.sum(n_world * ln, axis=-1)
            ds = np.sum(refl * ln, axis=-1)
        else:
            ln = light_vectors_view(light, vg) @ unit
            att = 1.0
            dotp = n_world @ ln
            ds = refl @ ln
        d = diffuse_function(light.diff_func & 3, dotp, d_rough) * att
        li_amb += col * d[..., None]
        sh = layer.shadow[..., i] if opts.use_shadows else np.ones((H, W))
        li_dif += col * (d * sh * d_amb_sh)[..., None]
        pf = float(2 << (light.spec_func & 7))
        pos = ds > 0.0
        t2 = (1.0 + np.minimum(d_rough * 2.0, 1.0) * (1.0 / pf - 1.0)) * s_spec
        li_amb += col * (t2 / pf)[..., None]
        spec = np.where(pos, np.maximum(ds, 0.0) ** pf, 0.0) * t2 * sh * d_amb_sh * att
        li_spe += col * spec[..., None]
    if maps.light is not None:
        px = sample_sphere(maps.light, n_world, maps.light_rot) * 255.0 * l.light_map_intensity
        c = maps.light_avg * 255.0 * l.light_map_intensity * (1.0 - d_rough)[..., None] + px * d_rough[..., None]
        li_amb += c
        li_dif += c * d_amb_sh[..., None]

    amb_mul = l.ambient_mul
    amb_top = np.asarray(l.ambient_top) * 255.0 * amb_mul
    amb_bot = np.asarray(l.ambient_bottom) * 255.0 * amb_mul
    ny = n_view[..., 1]
    t2 = (ny * 0.5 + 0.5) * d_amb_sh
    t1 = d_amb_sh - t2
    if maps.bg_small is not None and l.bg_ambient:
        iamb = sample_sphere(maps.bg_small, n_world, maps.bg_rot) * 255.0 * l.bg_brightness * amb_mul
        iamb = iamb * d_amb_sh[..., None] * idif
    else:
        iamb = (amb_top * t1[..., None] + amb_bot * t2[..., None]) * idif
    dif_s = idif * s_diff
    total = iamb + dif_s * li_dif + ispe * li_spe
    # CalcTotalLight1
    l3 = iamb + dif_s * li_amb * (d_amb_sh * 0.3)[..., None]
    st = np.maximum(0.0, 1.0 - d_amb_sh)
    st2 = st ** 4
    s_ind = (53.0 * 0.022) ** 2
    reflm = idif * (st2 * s_ind)[..., None] + (st - st2)[..., None]
    out = total + l3 * idif * reflm
    return out, ispe, positions


def background_of(view_world: np.ndarray, maps: "SceneMaps", scene: Scene, dep_c: np.ndarray,
                  full_size: tuple[int, int], origin: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    l = scene.lighting
    if maps.background is None:
        return dep_c
    W, H = size
    full_w, full_h = full_size
    ox, oy = origin
    if l.bg_direct:
        xs = (np.arange(W, dtype=np.float64) + ox + 0.5) / full_w
        ys = (np.arange(H, dtype=np.float64) + oy + 0.5) / full_h
        bg = sample_xy(maps.background, np.broadcast_to(xs[None, :], (H, W)), np.broadcast_to(ys[:, None], (H, W)),
                       wrap=False)
    else:
        bg = sample_sphere(maps.background, view_world, maps.bg_rot)
    return bg * 255.0 * l.bg_brightness


def fog_factor(total_depth: np.ndarray, hit: np.ndarray, scene: Scene, full_size: tuple[int, int]) -> np.ndarray:
    l = scene.lighting
    zword = depth_word(np.where(hit, total_depth, -1.0), scene, full_size)
    s_depth = l.depth_fog * 0.8e-6
    dtmp = np.where(hit, np.maximum(0.0, (zword - 28000.0) * s_depth + 1.0),
                    np.maximum(0.0, 1.0 - 28000.0 * s_depth))
    if l.far_fog:
        lt = dtmp < 1.0
        dtmp = np.where(lt, 1.0 - (1.0 - dtmp) ** 2, dtmp)
    return dtmp


# ----------------------------------------------------------------- shading
def shade(gbuf: np.ndarray, scene: Scene, opts: ShadeOptions | None = None) -> np.ndarray:
    """The image (H, W, 3) float32 0..1 of a G-buffer."""
    opts = opts or ShadeOptions()
    H, W = gbuf.shape[:2]
    full_w, full_h = opts.full_size or (W, H)
    ox, oy = opts.origin
    l = scene.lighting
    vg = scene.camera.vgrads
    unit = vg / np.linalg.norm(vg, axis=1, keepdims=True)
    fov = math.radians(scene.camera.fov_y)
    view = view_directions(full_w, full_h, fov)[oy:oy + H, ox:ox + W]
    view_world = view @ unit
    maps = load_scene_maps(scene)

    prim = primary_layer(gbuf, scene, opts, view_world)
    hit = prim.hit
    out, ispe, positions = shade_layer(prim, scene, opts, maps, gbuf, (full_w, full_h), (ox, oy))

    # background and depth fog
    ys = (np.arange(H, dtype=np.float64) + oy) / full_h
    dep_c = (np.asarray(l.depth_color_bottom) * ys[:, None] + np.asarray(l.depth_color_top) * (1.0 - ys)[:, None]) * 255.0
    dep_c = np.broadcast_to(dep_c[:, None, :], (H, W, 3))
    bg = background_of(view_world, maps, scene, dep_c, (full_w, full_h), (ox, oy), (W, H))
    dtmp = fog_factor(prim.total_depth, hit, scene, (full_w, full_h))
    out = np.where(hit[..., None], out, bg)

    # reflections: the mirrored layer, fogged by its own distance, blended
    # by the reflection amount and the specular colour
    if l.reflection > 0.0 and gbuf.shape[2] > PL.G_R_DEPTH:
        if positions is None:
            positions = object_positions(prim.depth, scene, (full_w, full_h), (ox, oy), (W, H))
        refl = reflection_layer(gbuf, scene, opts, prim, positions)
        r_out, _, _ = shade_layer(refl, scene, opts, maps, gbuf, (full_w, full_h), (ox, oy))
        r_bg = background_of(refl.view, maps, scene, dep_c, (full_w, full_h), (ox, oy), (W, H))
        r_fog = fog_factor(refl.total_depth, refl.hit, scene, (full_w, full_h))
        r_img = np.where(refl.hit[..., None], r_out, r_bg)
        r_img = r_img * r_fog[..., None] + dep_c * np.maximum(0.0, 1.0 - r_fog)[..., None]
        luma_s = ispe @ np.array([0.3, 0.59, 0.11])
        amt = np.where(hit, l.reflection * np.clip(luma_s, 0.0, 1.0), 0.0)[..., None]
        out = out * (1.0 - amt) + r_img * amt

    out = out * dtmp[..., None] + dep_c * np.maximum(0.0, 1.0 - dtmp)[..., None]

    # volumetric light: the lit path length along the view ray scatters the light's colour
    if 0 < l.vol_light <= PL.MAX_LIGHTS and l.lights[l.vol_light - 1].enabled:
        vl = l.lights[l.vol_light - 1]
        zend = max(1e-10, (scene.camera.z_end - scene.camera.z_start) / scene.camera.step_width(full_w))
        k = gbuf[..., PL.G_VOL].astype(np.float64) / zend * l.vol_light_strength
        amount = 0.6 * (1.0 - np.exp(-k))
        out = out + np.asarray(vl.color) * 255.0 * vl.amplitude * amount[..., None]

    # visible light sources: a glow where the view looks towards a light
    for light in l.lights[: PL.MAX_LIGHTS]:
        if not light.enabled or light.visible <= 0 or light.positional:
            continue
        ln = light_vectors_view(light, vg) @ unit
        cosang = np.clip(view_world @ ln, 0.0, 1.0)
        size = {1: 400.0, 2: 120.0, 3: 40.0}.get(light.visible, 120.0)
        glow = np.where(hit, 0.0, cosang ** size) * light.amplitude
        out = out + np.asarray(light.color) * 255.0 * glow[..., None]

    # dynamic fog on the step count
    if abs(l.dyn_fog - 53.0) > 1e-6:
        zsd = max(1e-4, scene.z_step_div)
        steps = gbuf[..., PL.G_STEPS].astype(np.float64)
        s_shad_gr = (l.dyn_fog - 53.0) * zsd * 0.00065
        dt = 2.2 / zsd
        s_shad = (137.0 - math.sqrt(max(l.fog_offset, 0.0)) * 11.313708) * dt * 0.28
        dfog = np.clip((steps - s_shad) * s_shad_gr, 0.0, 1.0)
        out = out * (1.0 - dfog)[..., None] + np.asarray(l.dyn_fog_color) * 255.0 * dfog[..., None]

    c = np.clip(out, 0.0, 255.0)
    gi = int(round(l.gamma))
    if gi != 32:
        if gi > 32:
            g = np.sqrt(c * 255.0)
            s_gamma = (gi - 32) / 31.0
        else:
            g = c * c / 255.0
            s_gamma = 1.0 - gi / 32.0
        c = g * s_gamma + c * (1.0 - s_gamma)
    return (c / 255.0).astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def to_uint16(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 65535.0), 0, 65535).astype(np.uint16)
