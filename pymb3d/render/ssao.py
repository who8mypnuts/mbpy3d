"""Screen space ambient occlusion on the depth buffer (numpy).

Samples the depth at offsets on a few rings around every pixel; a sample in
front of the pixel occludes it, weighted by its distance and attenuated
when it is far in front (a different object).  Depth is in pixel units, so
one pixel of depth difference roughly matches one pixel of screen distance.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import params as PL


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``a`` moved by (dy, dx) with edge replication."""
    H, W = a.shape
    ys = np.clip(np.arange(H) + dy, 0, H - 1)
    xs = np.clip(np.arange(W) + dx, 0, W - 1)
    return a[ys[:, None], xs[None, :]]


def ssao(depth: np.ndarray, radius: float = 32.0, strength: float = 1.0, rings: int = 3,
         directions: int = 8) -> np.ndarray:
    """Ambient occlusion factor (1 = open) of every pixel."""
    H, W = depth.shape
    hit = depth >= 0.0
    far = depth[hit].max() * 4.0 + 1.0 if hit.any() else 1.0
    d = np.where(hit, depth, far)
    occ = np.zeros((H, W), dtype=np.float64)
    wsum = 0.0
    for r_i in range(1, rings + 1):
        r = radius * r_i / rings
        w = 1.0 / r_i
        for k in range(directions):
            a = 2.0 * math.pi * (k + 0.5 * (r_i % 2)) / directions
            dx, dy = int(round(math.cos(a) * r)), int(round(math.sin(a) * r))
            if dx == 0 and dy == 0:
                continue
            ds = _shift(d, dy, dx)
            diff = (d - ds) / max(r, 1.0)          # > 0: the sample is in front
            o = np.clip(diff * 1.5, 0.0, 1.0) / (1.0 + diff * diff * 0.05)
            occ += o * w
            wsum += w
    occ /= max(wsum, 1e-9)
    ao = np.clip(1.0 - strength * occ * 1.6, 0.0, 1.0)
    ao[~hit] = 1.0
    return ao


def apply_ssao(gbuf: np.ndarray, radius: float, strength: float) -> None:
    """Writes the SSAO result into the AO channel of the G-buffer."""
    gbuf[..., PL.G_AO] = ssao(gbuf[..., PL.G_DEPTH].astype(np.float64), radius, strength)
