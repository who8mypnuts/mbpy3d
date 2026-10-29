"""The precision probe: decides whether a scene can be marched in single
precision.

Consumer graphics cards run double precision at 1/16 to 1/64 of their
single precision rate, so a final render in double can take far longer
than one in single.  Most scenes look the same in single precision; the
probe marches a coarse version of the image (every k-th pixel) in both
precisions and compares the hits, depths and colour indices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import params as PL


@dataclass
class ProbeResult:
    single_ok: bool
    hit_mismatch: float      # fraction of pixels whose hit status differs
    depth_error: float       # mean |depth difference| in DE-stop units
    color_error: float       # mean colour index difference (0..32767)
    reason: str

    @property
    def summary(self) -> str:
        return (f"{'single' if self.single_ok else 'double'} precision: hits differ {self.hit_mismatch * 100:.2f} %, "
                f"depth error {self.depth_error:.3f} DE stops, colour error {self.color_error:.0f}")


def probe_params(P: np.ndarray, width: int, height: int, max_pixels: int = 6000) -> tuple[np.ndarray, int, int]:
    """A parameter block marching every k-th pixel of the rectangle."""
    k = 1
    while (width // k) * (height // k) > max_pixels:
        k += 1
    Pp = P.copy()
    Pp[PL.P_PIXEL_STEP] = float(k)
    w, h = max(1, width // k), max(1, height // k)
    Pp[PL.P_WIDTH], Pp[PL.P_HEIGHT] = w, h
    return Pp, w, h


def compare(single: np.ndarray, double: np.ndarray, de_stop: float, de_stop_factor: float,
            max_mismatch: float = 0.02, max_depth_error: float = 0.15, max_color_error: float = 400.0) -> ProbeResult:
    hit_s = single[..., PL.G_DEPTH] >= 0
    hit_d = double[..., PL.G_DEPTH] >= 0
    n = max(1, hit_s.size)
    mismatch = float((hit_s != hit_d).sum()) / n
    both = hit_s & hit_d
    if both.any():
        dd = double[..., PL.G_DEPTH][both].astype(np.float64)
        ds = single[..., PL.G_DEPTH][both].astype(np.float64)
        stop = de_stop * (1.0 + dd * de_stop_factor)
        depth_err = float(np.mean(np.abs(ds - dd) / np.maximum(stop, 1e-9)))
        color_err = float(np.mean(np.abs(single[..., PL.G_COLOR][both] - double[..., PL.G_COLOR][both])))
    else:
        depth_err = 0.0
        color_err = 0.0
    if mismatch > max_mismatch:
        return ProbeResult(False, mismatch, depth_err, color_err, "too many pixels hit differently")
    if depth_err > max_depth_error:
        return ProbeResult(False, mismatch, depth_err, color_err, "depths differ")
    if color_err > max_color_error:
        return ProbeResult(False, mismatch, depth_err, color_err, "colours differ")
    return ProbeResult(True, mismatch, depth_err, color_err, "single precision matches")
