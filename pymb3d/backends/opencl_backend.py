"""The OpenCL backend: the C-like core for the final renders.

Precision: ``"auto"`` (default) marches in single precision when the
precision probe (:mod:`precision`) finds no difference to double on a
coarse version of the image, ``"mixed"`` always marches in single
precision and, when the probe finds differences, asks the renderer to
finish the hits in double on the CPU (``needs_finish``), ``"float32"`` /
``"float64"`` force one.
Runs on NVIDIA, AMD, Intel and Apple devices and on CPU OpenCL runtimes
(pocl, Intel).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np

from ..core import params as PL
from ..kernels.program import Program
from . import precision as prec
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
    double: bool


class OpenCLBackend(Backend):
    name = "opencl"
    band_target_seconds = 0.15

    def __init__(self, platform_index: int = 0, device_index: int = 0, precision: str = "auto") -> None:
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
        self.double_ok = ("cl_khr_fp64" in self.device.extensions
                          or getattr(self.device, "double_fp_config", 0) != 0)
        if precision not in ("auto", "mixed", "float32", "float64"):
            raise BackendError(f"unknown precision {precision}")
        if precision == "float64" and not self.double_ok:
            precision = "float32"
        if precision in ("auto", "mixed") and not self.double_ok:
            precision = "float32"
        #: set by the probe in "mixed" mode: the CPU should finish the hits in double
        self.needs_finish = False
        self.precision_mode = precision
        self.ctx = cl.Context(devices=[self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self._thread = threading.get_ident()
        self._active: _Compiled | None = None
        self.last_probe: prec.ProbeResult | None = None
        self.last_precision = "float64" if precision == "float64" else "float32"
        kind = "gpu" if self.device.type & cl.device_type.GPU else "cpu"
        label = {"auto": "auto precision", "mixed": "mixed precision", "float32": "float32", "float64": "float64"}[precision]
        self._info = DeviceInfo("opencl", self.device.name.strip(), label, kind, self.platform.name.strip())

    @property
    def info(self) -> DeviceInfo:
        return self._info

    @property
    def dtype(self):
        return np.float64

    # --------------------------------------------------------------- compile
    def _build(self, program: Program, double: bool) -> _Compiled:
        cl = _cl()
        src = program.opencl_source(double)
        opts = ["-cl-std=CL1.2"]
        if not double:
            opts.append("-cl-single-precision-constant")
        if os.environ.get("PYMB3D_CL_FAST_MATH", "0") == "1":
            opts.append("-cl-fast-relaxed-math")
        try:
            prg = cl.Program(self.ctx, src).build(options=opts)
        except Exception as e:
            raise BackendError(f"OpenCL build failed: {e}") from e
        return _Compiled(prg, prg.march_kernel, prg.formula_test_kernel, double)

    def _compile(self, program: Program) -> dict[str, _Compiled]:
        out = {}
        if self.precision_mode in ("auto", "mixed", "float32"):
            out["float32"] = self._build(program, False)
        if self.precision_mode in ("auto", "mixed", "float64"):
            out["float64"] = self._build(program, True)
        return out

    def _variant(self, program: Program, key: str) -> _Compiled:
        """The compiled program of one precision (built on demand)."""
        variants = self.compile(program)
        if key not in variants:
            variants[key] = self._build(program, key == "float64")
        return variants[key]

    def band_rows(self, width: int) -> int:
        # about 50k pixels per launch keeps display drivers responsive
        return max(1, min(64, 50_000 // max(1, width)))

    # ---------------------------------------------------------------- render
    def _launch(self, compiled: _Compiled, P: np.ndarray, width: int, rows: int) -> np.ndarray:
        """Runs the march kernel on a (rows, width) rectangle."""
        cl = _cl()
        mf = cl.mem_flags
        real = np.float64 if compiled.double else np.float32
        Pb = np.ascontiguousarray(P, dtype=real)
        p_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Pb)
        res = np.empty((rows, width, PL.N_CHANNELS), dtype=real)
        o_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, res.nbytes)
        local = (16, 4)
        gw = ((width + local[0] - 1) // local[0] * local[0], (rows + local[1] - 1) // local[1] * local[1])
        compiled.kernel(self.queue, gw, local, p_buf, o_buf, np.int32(width), np.int32(rows))
        cl.enqueue_copy(self.queue, res, o_buf)
        self.queue.finish()
        return res

    def choose_precision(self, program: Program, P: np.ndarray, width: int, height: int) -> _Compiled:
        """The compiled variant for this render (runs the probe in auto mode)."""
        self.needs_finish = False
        if self.precision_mode not in ("auto", "mixed"):
            self.last_precision = self.precision_mode
            self.last_probe = None
            return self._variant(program, self.precision_mode)
        Pp, w, h = prec.probe_params(P, width, height)
        single = self._launch(self._variant(program, "float32"), Pp, w, h)
        double = self._launch(self._variant(program, "float64"), Pp, w, h)
        self.last_probe = prec.compare(single, double, float(P[PL.P_DE_STOP]), float(P[PL.P_DE_STOP_FACTOR]))
        if self.precision_mode == "mixed":
            self.needs_finish = not self.last_probe.single_ok
            self.last_precision = "float32" + (" + CPU finish" if self.needs_finish else "")
            return self._variant(program, "float32")
        self.last_precision = "float32" if self.last_probe.single_ok else "float64"
        return self._variant(program, self.last_precision)

    def render(self, program, P, width, height, progress=None, abort=None, out=None, done_rows=None):
        self._active = self.choose_precision(program, P, width, height)
        try:
            return super().render(program, P, width, height, progress, abort, out, done_rows)
        finally:
            self._active = None

    def _render_band(self, compiled, P: np.ndarray, y0: int, rows: int, out: np.ndarray, abort) -> None:
        c = self._active or next(iter(compiled.values()))
        Pb = np.array(P, dtype=np.float64)
        Pb[PL.P_RECT_Y] += y0
        Pb[PL.P_HEIGHT] = rows
        out[y0:y0 + rows] = self._launch(c, Pb, out.shape[1], rows)

    # ------------------------------------------------------------ test hook
    def run_formula(self, program: Program, P: np.ndarray, slot: int, states: np.ndarray,
                    double: bool | None = None) -> np.ndarray:
        """Applies formula ``slot`` once to ``states`` (n, 14): the
        transpiler equivalence tests."""
        cl = _cl()
        if double is None:
            double = self.double_ok
        c = self._variant(program, "float64" if double else "float32")
        mf = cl.mem_flags
        real = np.float64 if c.double else np.float32
        Pb = np.ascontiguousarray(P, dtype=real)
        io = np.ascontiguousarray(states, dtype=real).copy()
        p_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Pb)
        io_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=io)
        c.test_kernel(self.queue, (io.shape[0],), None, p_buf, np.int32(slot), io_buf, np.int32(io.shape[0]))
        cl.enqueue_copy(self.queue, io, io_buf)
        self.queue.finish()
        return io.astype(np.float64)
