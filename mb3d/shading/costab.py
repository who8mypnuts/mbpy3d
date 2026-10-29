"""The diffuse light functions (``SetCosTabFunction`` / ``GetCosTabVal``)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def cos_tabs() -> np.ndarray:
    """8 x 128 table: 4 diffuse functions and their 'rough' (blurred) versions."""
    diff = np.zeros((8, 128), dtype=np.float64)
    for i in range(128):
        d = 1.0 - (i - 2.0) / 60.0
        if d > 0.15:
            diff[0, i] = (d - 0.08) * 1.0869565
        elif d <= 0.0:
            diff[0, i] = 0.0
        else:
            diff[0, i] = d ** max(1.0, (0.505 - d) * 3.8)
        dc = max(d, 0.0)
        diff[1, i] = dc * dc
        diff[2, i] = d * 0.5 + 0.5
        diff[3, i] = (d * 0.5 + 0.5) ** 2
    for k in range(4):
        tmp = np.sqrt(np.maximum(diff[k], 0.0))
        for j in range(128):
            e = 0.0
            for i in range(61):
                l = abs(j + i - 30)
                if l < 128:
                    e += tmp[l]
            a = e * 0.011 + (e * 0.007) ** 2
            diff[k + 4, j] = a * a
    return diff.astype(np.float32)


def spline_weights(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``MakeSplineCoeff``: cubic B-spline weights for the fraction ``xs``."""
    r3 = xs * xs * xs / 6.0
    r0 = 1.0 / 6.0 + 0.5 * xs * (xs - 1.0) - r3
    r2 = xs + r0 - 2.0 * r3
    r1 = 1.0 - r0 - r2 - r3
    return r0, r1, r2, r3


def cos_tab_val(tnr: int, dotp: np.ndarray, rough: np.ndarray | float = 0.0, sqr: bool = False) -> np.ndarray:
    """``GetCosTabVal`` vectorised: diffuse function ``tnr`` of ``dotp`` = n . l."""
    tabs = cos_tabs()
    t = 62.0 - 60.0 * dotp
    ip = np.trunc(t).astype(np.int64) - 1
    frac = t - np.trunc(t)
    frac = np.where(ip < 0, 0.0, np.where(ip > 124, 1.0, frac))
    ip = np.clip(ip, 0, 124)
    w = spline_weights(frac.astype(np.float32))
    a = sum(tabs[tnr, ip + k] * w[k] for k in range(4))
    b = sum(tabs[tnr + 4, ip + k] * w[k] for k in range(4))
    if sqr:
        a = a * a
        b = b * b
    return a + rough * (b - a)
