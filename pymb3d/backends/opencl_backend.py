"""The OpenCL backend: the C-like core in double precision (when the device
has ``cl_khr_fp64``), for the final renders.  Runs on NVIDIA, AMD, Intel
and Apple devices and on CPU OpenCL runtimes (pocl, Intel)."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np

from ..core import params as PL
from ..kernels.program import Program
from .base import Backend, BackendError, DeviceInfo


def _cl():
    try:
        import pyopencl as cl
    except ImportError as e:  # pragma: no cover
        raise BackendError("pyopencl is not installed") from e
    return cl


def list_opencl_devices() -> list[tuple[int, int, str, bool, str]]:
    """(platform index, device index, name, has fp64, type) of every device."""
    try:
        cl = _cl()
        out = []
        for pi, p in enumerate(cl.get_platforms()):
            try:
                devices = p.get_devices()
            except Exception:
                continue
            for di, d in enumerate(devices):
                fp64 = "cl_khr_fp64" in d.extensions or getattr(d, "double_fp_config", 0) != 0
                kind = "gpu" if d.type & cl.device_type.GPU else ("cpu" if d.type & cl.device_type.CPU else "other")
                out.append((pi, di, f"{d.name.strip()} [{p.name.strip()}]", bool(fp64), kind))
        return out
    except Exception:
        return []


@dataclass
class _Compiled:
    program: object
    kernel: object
    test_kernel: object


class OpenCLBackend(Backend):
    name = "opencl"

    def __init__(self, platform_index: int = 0, device_index: int = 0, prefer_double: bool = True) -> None:
        super().__init__()
        cl = _cl()
        platforms = cl.get_platforms()
        if platform_index >= len(platforms):
            raise BackendError(f"no OpenCL platform {platform_index}")
        devices = platforms[platform_index].get_devices()
        if device_index >= len(devices):
            raise BackendError(f"no OpenCL device {device_index} on platform {platform_index}")
        self.device = devices[device_index]
        self.platform = platforms[platform_index]
        self.double = prefer_double and ("cl_khr_fp64" in self.device.extensions
                                         or getattr(self.device, "double_fp_config", 0) != 0)
        self.ctx = cl.Context(devices=[self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self._thread = threading.get_ident()
        kind = "gpu" if self.device.type & cl.device_type.GPU else "cpu"
        self._info = DeviceInfo("opencl", f"{self.device.name.strip()}", "float64" if self.double else "float32",
                                kind, self.platform.name.strip())

    @property
    def info(self) -> DeviceInfo:
        return self._info

    # --------------------------------------------------------------- compile
    def _compile(self, program: Program) -> _Compiled:
        cl = _cl()
        src = program.opencl_source(self.double)
        opts = ["-cl-std=CL1.2"]
        if os.environ.get("PYMB3D_CL_FAST_MATH", "0") == "1":
            opts.append("-cl-fast-relaxed-math")
        try:
            prg = cl.Program(self.ctx, src).build(options=opts)
        except Exception as e:
            raise BackendError(f"OpenCL build failed: {e}") from e
        return _Compiled(prg, prg.march_kernel, prg.formula_test_kernel)

    def band_rows(self, width: int) -> int:
        # about 50k pixels per launch keeps display drivers responsive
        return max(1, min(64, 50_000 // max(1, width)))

    # ---------------------------------------------------------------- render
    def _render_band(self, compiled: _Compiled, P: np.ndarray, y0: int, rows: int, out: np.ndarray, abort) -> None:
        cl = _cl()
        mf = cl.mem_flags
        real = np.float64 if self.double else np.float32
        Pb = np.ascontiguousarray(P, dtype=real).copy()
        Pb[PL.P_RECT_Y] += y0
        Pb[PL.P_HEIGHT] = rows
        width = out.shape[1]
        p_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Pb)
        res = np.empty((rows, width, PL.N_CHANNELS), dtype=real)
        o_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, res.nbytes)
        local = (16, 4)
        gw = ((width + local[0] - 1) // local[0] * local[0], (rows + local[1] - 1) // local[1] * local[1])
        compiled.kernel(self.queue, gw, local, p_buf, o_buf, np.int32(width), np.int32(rows))
        cl.enqueue_copy(self.queue, res, o_buf)
        self.queue.finish()
        out[y0:y0 + rows] = res

    # ------------------------------------------------------------ test hook
    def run_formula(self, program: Program, P: np.ndarray, slot: int, states: np.ndarray) -> np.ndarray:
        """Applies formula ``slot`` once to ``states`` (n, 10): the
        transpiler equivalence tests."""
        cl = _cl()
        c = self.compile(program)
        mf = cl.mem_flags
        real = np.float64 if self.double else np.float32
        Pb = np.ascontiguousarray(P, dtype=real)
        io = np.ascontiguousarray(states, dtype=real).copy()
        p_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Pb)
        io_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=io)
        c.test_kernel(self.queue, (io.shape[0],), None, p_buf, np.int32(slot), io_buf, np.int32(io.shape[0]))
        cl.enqueue_copy(self.queue, io, io_buf)
        self.queue.finish()
        return io
