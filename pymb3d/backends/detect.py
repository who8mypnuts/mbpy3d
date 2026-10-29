"""Finds the compute devices of this machine and picks backends for the
preview (fast, single precision is fine) and the final render (precise).

Selection order for the final render: an OpenCL GPU with double precision,
then the CPU (numba, double), then an OpenCL GPU in single precision; the
preview prefers OpenGL on a real GPU, then OpenCL GPU, then CPU.  Software
OpenGL (llvmpipe) and CPU OpenCL runtimes are listed but ranked below the
numba CPU backend.  ``PYMB3D_BACKEND`` (``cpu``, ``opencl``, ``opengl``)
forces a backend for both.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .base import Backend, BackendError, DeviceInfo


@dataclass
class DeviceChoice:
    backend: str            # "cpu", "opencl", "opengl"
    name: str
    precision: str          # "float32", "float64" or "auto" (OpenCL: probe per render)
    kind: str
    index: tuple[int, int] = (0, 0)   # OpenCL platform/device

    @property
    def key(self) -> str:
        return f"{self.backend}:{self.index[0]}:{self.index[1]}:{self.precision}"

    @property
    def label(self) -> str:
        p = {"auto": "auto precision", "mixed": "single precision + CPU finish", "float32": "single precision",
             "float64": "double precision"}[self.precision]
        return f"{self.backend}: {self.name} ({p}, {self.kind})"


def available_devices(probe_opengl: bool = True) -> list[DeviceChoice]:
    """All usable devices (cheap to call; OpenGL is probed in a helper
    thread so the context never binds to the caller's thread)."""
    out = [DeviceChoice("cpu", "numba (all cores)", "float64", "cpu")]
    try:
        from .opencl_backend import list_opencl_devices

        for pi, di, name, fp64, kind in list_opencl_devices():
            if fp64:
                out.append(DeviceChoice("opencl", name, "mixed", kind, (pi, di)))
                out.append(DeviceChoice("opencl", name, "auto", kind, (pi, di)))
                out.append(DeviceChoice("opencl", name, "float64", kind, (pi, di)))
            out.append(DeviceChoice("opencl", name, "float32", kind, (pi, di)))
    except Exception:
        pass
    if probe_opengl:
        info = probe_opengl_info()
        if info is not None:
            out.append(DeviceChoice("opengl", info.name, "float32", info.kind))
    return out


def probe_opengl_info() -> DeviceInfo | None:
    import threading

    result: list[DeviceInfo | None] = [None]

    def run() -> None:
        try:
            from .opengl_backend import OpenGLBackend

            be = OpenGLBackend()
            be.ensure_context()
            result[0] = be.info
            be.close()
        except Exception:
            result[0] = None

    t = threading.Thread(target=run, name="pymb3d-gl-probe", daemon=True)
    t.start()
    t.join(timeout=20)
    return result[0]


def _rank_final(d: DeviceChoice) -> tuple:
    if d.backend == "opencl" and d.kind == "gpu":
        return (0, {"mixed": 0, "auto": 1, "float64": 2, "float32": 3}[d.precision])
    if d.backend == "cpu":
        return (1, 0)
    if d.backend == "opencl":
        return (3, {"mixed": 0, "auto": 1, "float64": 2, "float32": 3}[d.precision])
    return (9, 0)


def _rank_preview(d: DeviceChoice) -> tuple:
    if d.backend == "opengl" and d.kind == "gpu":
        return (0, 0)
    if d.backend == "opencl" and d.kind == "gpu":
        return (1, {"float32": 0, "auto": 1, "mixed": 2, "float64": 3}[d.precision])
    if d.backend == "cpu":
        return (2, 0)
    if d.backend == "opencl":
        return (3, {"float32": 0, "auto": 1, "mixed": 2, "float64": 3}[d.precision])
    return (4, 0)  # software OpenGL


def choose(devices: list[DeviceChoice], purpose: str) -> DeviceChoice:
    forced = os.environ.get("PYMB3D_BACKEND", "").strip().lower()
    if forced:
        for d in devices:
            if d.backend == forced:
                return d
    rank = _rank_final if purpose == "final" else _rank_preview
    return sorted(devices, key=rank)[0]


def create_backend(choice: DeviceChoice) -> Backend:
    if choice.backend == "cpu":
        from .cpu_backend import CpuBackend

        return CpuBackend()
    if choice.backend == "opencl":
        from .opencl_backend import OpenCLBackend

        return OpenCLBackend(choice.index[0], choice.index[1], precision=choice.precision)
    if choice.backend == "opengl":
        from .opengl_backend import OpenGLBackend

        return OpenGLBackend()
    raise BackendError(f"unknown backend {choice.backend}")
