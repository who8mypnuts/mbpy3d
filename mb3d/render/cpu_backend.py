"""The CPU back end: numba, all cores, double precision."""

from __future__ import annotations

import os
import threading

import numba
import numpy as np
from numba import njit, prange

from mb3d.kernel.params import KernelParams
from mb3d.kernel.source.march import march_pixel

from .backend import Backend, DeviceInfo, check_cancel
from .gbuffer import GBuffer


@njit(parallel=True, nogil=True, cache=True)
def _render_rows(rows, width, sp, fp, ip, gbuf, cancel_flag):
    for i in prange(rows.shape[0]):
        y = rows[i]
        if cancel_flag[0] != 0:
            continue
        st = np.zeros(16)
        res = np.zeros(12)
        for x in range(width):
            march_pixel(x, y, sp, fp, ip, st, res)
            for k in range(12):
                gbuf[y, x, k] = res[k]


def warm_up() -> None:
    """Compiles (or loads from the cache) the CPU kernels."""
    from mb3d.kernel.params import RenderOptions, build_params
    from mb3d.params import Scene

    s = Scene()
    s.width, s.height = 4, 4
    p = build_params(s, RenderOptions(ao_samples=2, shadow=True))
    g = np.zeros((4, 4, 12))
    _render_rows(np.arange(4, dtype=np.int64), 4, p.sp, p.fp, p.ip, g, np.zeros(1, dtype=np.int32))


class CpuBackend(Backend):
    kind = "cpu"
    precision = "double"

    def __init__(self, threads: int | None = None, band_rows: int = 16):
        self.threads = threads or numba.config.NUMBA_NUM_THREADS
        self.band_rows = band_rows

    def info(self) -> DeviceInfo:
        return DeviceInfo("cpu", f"numba, {self.threads} threads ({os.cpu_count()} cores)", "double")

    def render(self, params: KernelParams, gbuf: GBuffer, rows: np.ndarray, cancel: threading.Event | None = None, progress=None) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        flag = np.zeros(1, dtype=np.int32)
        data = gbuf.data if gbuf.data.dtype == np.float64 else np.zeros(gbuf.data.shape, dtype=np.float64)
        done = 0
        # bands of interleaved rows keep all threads busy even when a few rows are slow
        band = max(self.band_rows, self.threads * 2)
        for start in range(0, len(rows), band):
            check_cancel(cancel)
            chunk = rows[start:start + band]
            numba.set_num_threads(max(1, min(self.threads, numba.config.NUMBA_NUM_THREADS)))
            _render_rows(chunk, params.width, params.sp, params.fp, params.ip, data, flag)
            done += len(chunk)
            if data is not gbuf.data:
                gbuf.data[chunk] = data[chunk]
            if progress is not None:
                progress(done, len(rows))
        gbuf.backend = "cpu"
        gbuf.precision = "double"
