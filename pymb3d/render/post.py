"""Post processing: depth of field on the depth buffer, brightness /
contrast / saturation and sharpening (numpy)."""
from __future__ import annotations

import numpy as np

from ..core import params as PL
from ..core.scene import PostSettings, Scene


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur (radius in pixels) with edge replication."""
    if radius < 1:
        return img
    r = int(radius)
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode="edge")
    c = np.cumsum(pad, axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c], axis=0)
    v = (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)
    c = np.cumsum(v, axis=1)
    c = np.concatenate([np.zeros_like(c[:, :1]), c], axis=1)
    return (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)


def gaussian_like_blur(img: np.ndarray, radius: float) -> np.ndarray:
    """Three box blurs approximate a Gaussian of the given radius."""
    r = max(1, int(round(radius / 1.7)))
    out = img
    for _ in range(3):
        out = _box_blur(out, r)
    return out


def depth_of_field(img: np.ndarray, depth: np.ndarray, scene: Scene, full_size: tuple[int, int] | None = None) -> np.ndarray:
    """Variable blur by the distance from the focus plane: the image is
    blurred at a few radii and each pixel takes the level of its circle of
    confusion (a fast layered approximation of MB3D's DOF)."""
    p = scene.post
    W, H = full_size or (img.shape[1], img.shape[0])
    cam = scene.camera
    zend = max(1e-10, (cam.z_end - cam.z_start) / cam.step_width(W))
    d = np.where(depth >= 0, depth, zend) / zend          # 0..1 of the depth range
    defocus = d - p.dof_focus
    if p.dof_forward_only:
        defocus = np.maximum(defocus, 0.0)
    coc = np.abs(defocus) * p.dof_aperture * (W / 640.0)   # blur radius in pixels
    max_r = float(np.nanmax(coc)) if coc.size else 0.0
    if max_r < 0.5:
        return img
    radii = [0.0]
    r = 1.0
    while r < max_r:
        radii.append(r)
        r *= 2.0
    radii.append(max(max_r, r))
    levels = [img.astype(np.float32)] + [gaussian_like_blur(img.astype(np.float32), rr) for rr in radii[1:]]
    out = np.empty_like(levels[0])
    coc_c = np.clip(coc, 0.0, radii[-1])
    for k in range(len(radii) - 1):
        lo, hi = radii[k], radii[k + 1]
        sel = (coc_c >= lo) & (coc_c <= hi) if k == len(radii) - 2 else (coc_c >= lo) & (coc_c < hi)
        if not sel.any():
            continue
        t = ((coc_c - lo) / max(hi - lo, 1e-9))[..., None]
        mixed = levels[k] * (1 - t) + levels[k + 1] * t
        out[sel] = mixed[sel]
    return out


def color_adjust(img: np.ndarray, p: PostSettings) -> np.ndarray:
    """Brightness, contrast and saturation on a 0..1 float image."""
    out = img.astype(np.float32)
    if p.contrast:
        f = (1.0 + p.contrast) if p.contrast > 0 else 1.0 / (1.0 - p.contrast)
        out = (out - 0.5) * f + 0.5
    if p.brightness:
        out = out + p.brightness * 0.5
    if p.saturation:
        luma = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        out = luma[..., None] + (out - luma[..., None]) * (1.0 + p.saturation)
    return out


def sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    blurred = _box_blur(img.astype(np.float32), 1)
    return img + (img - blurred) * amount


def post_process(img: np.ndarray, gbuf: np.ndarray | None, scene: Scene,
                 full_size: tuple[int, int] | None = None) -> np.ndarray:
    """All post processing on a float 0..1 image; returns float 0..1."""
    p = scene.post
    out = img
    if p.dof and gbuf is not None:
        out = depth_of_field(out, gbuf[..., PL.G_DEPTH].astype(np.float64), scene, full_size)
    if p.brightness or p.contrast or p.saturation:
        out = color_adjust(out, p)
    if p.sharpen > 0:
        out = sharpen(out, p.sharpen)
    return np.clip(out, 0.0, 1.0)
