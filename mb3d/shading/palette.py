"""The surface palette (``CalcColors``) and the interior colours (``CalcColorsInside``)."""

from __future__ import annotations

import numpy as np

from .light_values import LightVals


def _interpolate(ir: np.ndarray, positions: np.ndarray, divs: np.ndarray, cols: np.ndarray, wrap: bool) -> np.ndarray:
    """Linear interpolation in a palette of ``len(positions)`` colours at ``ir`` (0..32767)."""
    n = len(positions)
    # il2 = first index with positions[il2] >= ir (1..n), il1 = il2 - 1
    il2 = np.searchsorted(positions, ir, side="left")
    il2 = np.clip(il2, 1, n)
    il1 = il2 - 1
    il2w = np.where(il2 >= n, 0, il2) if wrap else np.minimum(il2, n - 1)
    t = (ir - positions[il1]).astype(np.float32) * divs[il1]
    t = np.clip(t, 0.0, 1.0)[..., None]
    c1 = cols[il1]
    c2 = cols[il2w]
    return c1 + t * (c2 - c1)


def surface_colors(lv: LightVals, si_gradient: np.ndarray, otrap: np.ndarray, idif0: np.ndarray | float = 0.0):
    """Diffuse and specular colour (0..1, or squared with internal gamma 2) per pixel."""
    ir = np.where(lv.col_on_otrap, otrap, si_gradient).astype(np.float32)
    ir = np.round(np.clip((((ir - lv.s_c_start) * lv.s_c_mul) + idif0) * 16384.0, -1e9, 1e9)).astype(np.int64)
    if lv.col_cycling:
        ir = ir & 32767
    if lv.no_col_ipol:
        il2 = np.clip(np.searchsorted(lv.col_pos, ir, side="left"), 1, 10) - 1
        dif, spe = lv.col_dif[il2], lv.col_spe[il2]
    else:
        dif = _interpolate(ir, lv.col_pos, lv.s_c_div, lv.col_dif, wrap=True)
        spe = _interpolate(ir, lv.col_pos, lv.s_c_div, lv.col_spe, wrap=True)
    if not lv.col_cycling:
        low = (ir < 0)[..., None]
        high = (ir >= lv.col_pos[9])[..., None]
        dif = np.where(low, lv.col_dif[0], np.where(high, lv.col_dif[9], dif))
        spe = np.where(low, lv.col_spe[0], np.where(high, lv.col_spe[9], spe))
    return dif.astype(np.float32), spe.astype(np.float32)


def interior_colors(lv: LightVals, si_gradient: np.ndarray, idif0: np.ndarray | float = 0.0):
    """Colours of pixels coloured 'inside' (si_gradient >= 32768)."""
    ir = np.round(((si_gradient - lv.s_ci_start) * lv.s_ci_mul + idif0) * 16384.0).astype(np.int64)
    if lv.col_cycling:
        ir = ir & 32767
    cols = np.concatenate([lv.col_int, lv.col_int_spec[:, None]], axis=1)
    if lv.no_col_ipol:
        il2 = np.clip(np.searchsorted(lv.icol_pos, ir, side="left"), 1, 4) - 1
        c = cols[il2]
    else:
        c = _interpolate(ir, lv.icol_pos, lv.s_ic_div, cols, wrap=True)
    if not lv.col_cycling:
        c = np.where((ir < 0)[..., None], cols[0], np.where((ir > lv.icol_pos[3])[..., None], cols[3], c))
    return c[..., :3].astype(np.float32), np.repeat(c[..., 3:4], 3, axis=-1).astype(np.float32)
