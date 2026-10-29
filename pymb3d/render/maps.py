"""Image maps: background pictures, light maps and diffuse colour maps.

Images are loaded once (cached by path) as float32 RGB in 0..1 and sampled
bilinearly, either by image coordinates or by a direction (equirectangular
sphere mapping, as MB3D maps its backgrounds and light maps).
"""
from __future__ import annotations

import math
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

_LOCK = threading.Lock()


def find_map(name: str | os.PathLike, search_dirs: list[str | os.PathLike] | None = None) -> Path | None:
    """The file of a map: the path as given, or the name in the search
    directories (a scene's folder, its ``M3Maps`` folder, the parent's)."""
    if not name:
        return None
    p = Path(name)
    if p.is_file():
        return p
    for d in search_dirs or []:
        d = Path(d)
        for cand in (d / p.name, d / "M3Maps" / p.name, d.parent / "M3Maps" / p.name):
            if cand.is_file():
                return cand
        # MB3D names maps by number: "123.jpg" may be "123 anything.jpg"
        stem = p.stem
        for folder in (d, d / "M3Maps", d.parent / "M3Maps"):
            if folder.is_dir():
                for f in folder.iterdir():
                    if f.stem.split(" ")[0] == stem and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                        return f
    return None


@lru_cache(maxsize=16)
def _load(path: str, mtime: float) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def load_map(path: str | os.PathLike) -> np.ndarray:
    p = Path(path)
    with _LOCK:
        return _load(str(p), p.stat().st_mtime)


def sample_xy(img: np.ndarray, u: np.ndarray, v: np.ndarray, wrap: bool = True) -> np.ndarray:
    """Bilinear samples at image coordinates u, v in 0..1 (v = 0 top)."""
    h, w = img.shape[:2]
    x = u * w - 0.5
    y = v * h - 0.5
    if wrap:
        x = np.mod(x, w)
        y = np.mod(y, h)
    else:
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x1 = (x0 + 1) % w if wrap else np.minimum(x0 + 1, w - 1)
    y1 = (y0 + 1) % h if wrap else np.minimum(y0 + 1, h - 1)
    x0 %= w
    y0 %= h
    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy


def rotation_matrix(angles_deg) -> np.ndarray:
    """Rotation about X, Y, Z (degrees) for the map orientation."""
    from ..core.vec import vgrads_from_angles

    ax, ay, az = (math.radians(a) for a in angles_deg)
    return vgrads_from_angles(ax, ay, az)


def sample_sphere(img: np.ndarray, dirs: np.ndarray, rot: np.ndarray | None = None) -> np.ndarray:
    """Equirectangular samples by unit directions (..., 3)."""
    d = dirs if rot is None else dirs @ rot.T
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    u = (np.arctan2(x, z) / (2 * math.pi)) + 0.5
    v = 0.5 - np.arcsin(np.clip(y, -1.0, 1.0)) / math.pi
    return sample_xy(img, u, v, wrap=True)


def small_blurred(img: np.ndarray, size: int = 32) -> np.ndarray:
    """A small, strongly blurred copy (the ambient light of a background)."""
    from PIL import Image, ImageFilter

    im = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).resize((size * 2, size))
    im = im.filter(ImageFilter.GaussianBlur(radius=size / 6))
    return np.asarray(im, dtype=np.float32) / 255.0
