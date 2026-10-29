"""The CPU backend: the numba kernel on all cores (``nogil`` kernels in a
thread pool, a few rows per task)."""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

from ..core import params as PL
from ..kernels.numba_loader import load_program_module
from ..kernels.program import Program
from .base import Backend, DeviceInfo, RenderCancelled


def cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except AttributeError:
        return max(1, os.cpu_count() or 1)


class CpuBackend(Backend):
    name = "cpu"

    def __init__(self, threads: int | None = None) -> None:
        super().__init__()
        self.threads = threads or cpu_count()
        self._pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="pymb3d-cpu")

    @property
    def info(self) -> DeviceInfo:
        return DeviceInfo("cpu", f"numba, {self.threads} threads", "float64", "cpu")

    def _compile(self, program: Program) -> object:
        mod = load_program_module(program)
        # trigger the numba compilation now (so compile time is accounted for)
        P = np.zeros(PL.N_PARAMS, dtype=np.float64)
        out = np.zeros((1, 1, PL.N_CHANNELS), dtype=np.float32)
        P[PL.P_NSLOTS] = 0  # no slots -> the orbit loop returns at once
        mod.render_rows(P, out, 0, 1, np.zeros(1, dtype=np.int32))
        return mod

    def band_rows(self, width: int) -> int:
        # small tasks keep all threads busy on images with very uneven rows
        return 2 if width > 256 else 4

    def _render_band(self, compiled, P, y0, rows, out, abort) -> None:  # pragma: no cover - see render()
        compiled.render_rows(P, out, y0, y0 + rows, np.zeros(1, dtype=np.int32))

    def finish(self, program, P, out, progress=None, abort=None):
        """Finishes the hits of a single precision G-buffer in double (see
        ``finish_rows`` of the numba core)."""
        return self.render(program, P, out.shape[1], out.shape[0], progress, abort, out, finish=True)

    def render(self, program, P, width, height, progress=None, abort=None, out=None, finish=False,
               done_rows=None):
        mod = self.compile(program)
        fn = mod.finish_rows if finish else mod.render_rows
        abort = abort or threading.Event()
        if out is None:
            out = np.zeros((height, width, PL.N_CHANNELS), dtype=np.float32)
        abort_flag = np.zeros(1, dtype=np.int32)
        band = self.band_rows(width)
        bands = [(y, min(band, height - y)) for y in range(0, height, band)]
        if done_rows is not None:
            bands = [(y, r) for y, r in bands if not done_rows[y:y + r].all()]
        # rows are interleaved over the threads like MB3D's CalcMandT so the
        # expensive parts of an image are shared
        futures = {}
        pending = list(bands)
        done_count = int(done_rows.sum()) if done_rows is not None else 0
        try:
            while pending or futures:
                while pending and len(futures) < self.threads * 2:
                    y, rows = pending.pop(0)
                    f = self._pool.submit(fn, P, out, y, y + rows, abort_flag)
                    futures[f] = (y, rows)
                finished, _ = wait(list(futures), timeout=0.05, return_when=FIRST_COMPLETED)
                for f in finished:
                    y, rows = futures.pop(f)
                    f.result()
                    if done_rows is not None:
                        done_rows[y:y + rows] = True
                        done_count = int(done_rows.sum())
                    else:
                        done_count += rows
                    if progress:
                        progress(y, rows, done_count, height)
                if abort.is_set():
                    abort_flag[0] = 1
                    pending.clear()
                    for f in futures:
                        f.result()
                    raise RenderCancelled()
        finally:
            abort_flag[0] = 1 if abort.is_set() else 0
        return out

    def close(self) -> None:
        self._pool.shutdown(wait=False)
