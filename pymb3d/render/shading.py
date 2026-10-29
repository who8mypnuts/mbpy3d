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
def diffuse_function(kind: int, d: np.ndarray) -> np.ndarray:
    """MB3D's diffuse light functions (`SetCosTabFunction`)."""
    if kind == 1:
        return np.maximum(d, 0.0) ** 2
    if kind == 2:
        return d * 0.5 + 0.5
    if kind == 3:
        return (d * 0.5 + 0.5) ** 2
    out = np.zeros_like(d)
    hi = d > 0.15
    out[hi] = (d[hi] - 0.08) * 1.0869565
    mid = (d > 0.0) & ~hi
    dm = d[mid]
    out[mid] = dm ** np.maximum(1.0, (0.505 - dm) * 3.8)
    return out


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


# ----------------------------------------------------------------- shading
def shade(gbuf: np.ndarray, scene: Scene, opts: ShadeOptions | None = None) -> np.ndarray:
    """The image (H, W, 3) float32 0..1 of a G-buffer."""
    opts = opts or ShadeOptions()
    H, W = gbuf.shape[:2]
    full_w, full_h = opts.full_size or (W, H)
    ox, oy = opts.origin
    l = scene.lighting
    depth = gbuf[..., PL.G_DEPTH].astype(np.float64)
    hit = depth >= 0.0
    vg = scene.camera.vgrads
    unit = vg / np.linalg.norm(vg, axis=1, keepdims=True)

    # normals in view space
    n_world = gbuf[..., PL.G_NX:PL.G_NZ + 1].astype(np.float64)
    n_view = n_world @ unit.T
    nn = np.linalg.norm(n_view, axis=-1, keepdims=True)
    n_view = n_view / np.where(nn > 0, nn, 1.0)

    # per pixel view direction (the ray direction in view space)
    fov = math.radians(scene.camera.fov_y)
    view_full = view_directions(full_w, full_h, fov)
    view = view_full[oy:oy + H, ox:ox + W]

    # ambient occlusion
    occ = 1.0 - gbuf[..., PL.G_AO].astype(np.float64) if opts.use_ao else np.zeros((H, W))
    s_amb_shad = l.amb_shadow_mul
    if s_amb_shad < 1.0:
        d_amb_sh = 1.0 - s_amb_shad * occ
    else:
        a = 1.0 - occ
        d_amb_sh = a + (s_amb_shad - 1.0) * (a * a - a)
    d_amb_sh = np.clip(d_amb_sh, 0.0, 1.0)

    # palette colours
    idif, ispe = surface_colors(gbuf, scene)
    idif = idif.astype(np.float64)
    ispe = ispe.astype(np.float64)

    s_diff = l.diffuse_mul
    s_spec = l.specular_mul
    li_dif = np.zeros((H, W, 3))
    li_spe = np.zeros((H, W, 3))
    li_amb = np.zeros((H, W, 3))
    # lighting in world space: view direction and normal per pixel
    view_world = view @ unit
    ndotv = np.sum(n_world * view_world, axis=-1, keepdims=True)
    refl = view_world - n_world * (2.0 * ndotv)
    obj_world = None
    for i, light in enumerate(l.lights[: PL.MAX_LIGHTS]):
        if not light.enabled:
            continue
        col = np.asarray(light.color, dtype=np.float64) * 255.0 * light.amplitude
        if light.positional:
            if obj_world is None:
                obj_world = object_positions(depth, scene, (full_w, full_h), (ox, oy), (W, H))
            lv = np.asarray(light.position, dtype=np.float64) - obj_world
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
        d = diffuse_function(light.diff_func & 3, dotp) * att
        li_amb += col * d[..., None]
        sh = gbuf[..., PL.G_SHADOW + i].astype(np.float64) if opts.use_shadows else np.ones((H, W))
        li_dif += col * (d * sh * d_amb_sh)[..., None]
        pf = float(2 << (light.spec_func & 7))
        pos = ds > 0.0
        t2 = s_spec
        li_amb += col * (t2 / pf)
        spec = np.where(pos, np.maximum(ds, 0.0) ** pf, 0.0) * t2 * sh * d_amb_sh * att
        li_spe += col * spec[..., None]

    amb_mul = l.ambient_mul
    amb_top = np.asarray(l.ambient_top) * 255.0 * amb_mul
    amb_bot = np.asarray(l.ambient_bottom) * 255.0 * amb_mul
    ny = n_view[..., 1]
    t2 = (ny * 0.5 + 0.5) * d_amb_sh
    t1 = d_amb_sh - t2
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

    # background and depth fog
    ys = (np.arange(H, dtype=np.float64) + oy) / full_h
    dep_c = (np.asarray(l.depth_color_bottom) * ys[:, None] + np.asarray(l.depth_color_top) * (1.0 - ys)[:, None]) * 255.0
    dep_c = np.broadcast_to(dep_c[:, None, :], (H, W, 3))
    zword = depth_word(depth, scene, (full_w, full_h))
    s_depth = l.depth_fog * 0.8e-6
    dtmp = np.where(hit, np.maximum(0.0, (zword - 28000.0) * s_depth + 1.0),
                    np.maximum(0.0, 1.0 - 28000.0 * s_depth))
    if l.far_fog:
        lt = dtmp < 1.0
        dtmp = np.where(lt, 1.0 - (1.0 - dtmp) ** 2, dtmp)
    out = np.where(hit[..., None], out, dep_c)
    out = out * dtmp[..., None] + dep_c * np.maximum(0.0, 1.0 - dtmp)[..., None]

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
