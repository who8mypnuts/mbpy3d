"""The OpenCL back end (pyopencl): double precision where the device has it.

Runs on NVIDIA, AMD, Intel and Apple devices (and on the CPU through
pocl).  The final render uses ``double`` when the device reports
``cl_khr_fp64``; otherwise the same kernel is built in single precision.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from mb3d.kernel import build
from mb3d.kernel.params import KernelParams
from mb3d.kernel.source import layout as L

from .backend import Backend, DeviceInfo, check_cancel
from .gbuffer import GBuffer


def list_devices() -> list[tuple[str, str, bool]]:
    """(platform, device, has_fp64) of every OpenCL device."""
    try:
        import pyopencl as cl
    except ImportError:
        return []
    out = []
    try:
        for p in cl.get_platforms():
            for d in p.get_devices():
                out.append((p.name.strip(), d.name.strip(), "cl_khr_fp64" in d.extensions or d.double_fp_config != 0))
    except Exception:
        pass
    return out


def _pick_device(prefer_gpu: bool = True):
    import pyopencl as cl

    devices = []
    for p in cl.get_platforms():
        for d in p.get_devices():
            devices.append(d)
    if not devices:
        raise RuntimeError("no OpenCL device")

    def score(d):
        gpu = d.type & cl.device_type.GPU
        fp64 = "cl_khr_fp64" in d.extensions or d.double_fp_config != 0
        return (int(bool(gpu)) if prefer_gpu else 0, int(fp64), d.max_compute_units)

    return max(devices, key=score)


class ClBackend(Backend):
    kind = "opencl"

    def __init__(self, device=None, prefer_gpu: bool = True, force_single: bool = False, band_rows: int = 16):
        import pyopencl as cl

        self.cl = cl
        self.device = device or _pick_device(prefer_gpu)
        has_fp64 = "cl_khr_fp64" in self.device.extensions or self.device.double_fp_config != 0
        self.precision = "single" if (force_single or not has_fp64) else "double"
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self.band_rows = band_rows
        self._programs: dict[tuple[int, ...], object] = {}

    def info(self) -> DeviceInfo:
        return DeviceInfo("opencl", f"{self.device.name.strip()} ({self.device.platform.name.strip()})", self.precision)

    @property
    def real_dtype(self):
        return np.float64 if self.precision == "double" else np.float32

    def program(self, fids: tuple[int, ...]):
        prog = self._programs.get(fids)
        if prog is None:
            target = "opencl_double" if self.precision == "double" else "opencl_float"
            src = build.kernel_source(target, fids)
            options = [] if self.precision == "double" else ["-cl-single-precision-constant"]
            prog = self.cl.Kernel(self.cl.Program(self.ctx, src).build(options=options), "render")
            self._programs[fids] = prog
        return prog

    def render(self, params: KernelParams, gbuf: GBuffer, rows: np.ndarray, cancel: threading.Event | None = None, progress=None) -> None:
        cl = self.cl
        t0 = time.time()
        kernel = self.program(build.formula_ids(params))
        rt = self.real_dtype
        w, h = params.width, params.height
        mf = cl.mem_flags
        sp = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(params.sp, dtype=rt))
        fp = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(params.fp, dtype=rt))
        ip = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(params.ip, dtype=np.int32))
        out = np.zeros((h, w, L.OUT_SIZE), dtype=rt)
        obuf = cl.Buffer(self.ctx, mf.WRITE_ONLY, out.nbytes)
        rows = np.asarray(rows, dtype=np.int32)
        done = 0
        for start in range(0, len(rows), self.band_rows):
            check_cancel(cancel)
            chunk = np.ascontiguousarray(rows[start:start + self.band_rows])
            rbuf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chunk)
            kernel(self.queue, (w, len(chunk)), None, obuf, sp, fp, ip, rbuf, np.int32(w))
            self.queue.finish()
            done += len(chunk)
            if progress is not None:
                progress(done, len(rows))
        cl.enqueue_copy(self.queue, out, obuf)
        self.queue.finish()
        gbuf.data[rows] = out[rows].astype(gbuf.data.dtype)
        gbuf.backend = "opencl"
        gbuf.precision = self.precision
        gbuf.seconds = time.time() - t0
