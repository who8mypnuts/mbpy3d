"""Device discovery and back end selection.

Back ends are created lazily, once per (kind, thread): an OpenGL context
belongs to the thread that made it, so the render worker owns its own.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .backend import Backend, DeviceInfo

KINDS = ("cpu", "opengl", "opencl")


@dataclass
class DeviceReport:
    available: list[DeviceInfo] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def probe(allow_software_gl: bool = False) -> DeviceReport:
    """What this machine offers (probing creates and releases contexts)."""
    rep = DeviceReport()
    from .cpu_backend import CpuBackend

    rep.available.append(CpuBackend().info())
    try:
        from .gl_backend import GlBackend

        gl = GlBackend(allow_software=allow_software_gl)
        if gl.is_software() and not allow_software_gl:
            rep.errors["opengl"] = f"software renderer ({gl.renderer_name()})"
        else:
            rep.available.append(gl.info())
        gl.close()
    except Exception as e:
        rep.errors["opengl"] = str(e)
    try:
        from .cl_backend import ClBackend

        cl = ClBackend()
        rep.available.append(cl.info())
    except Exception as e:
        rep.errors["opencl"] = str(e)
    return rep


class BackendPool:
    """Creates back ends on demand for the calling thread and keeps them."""

    def __init__(self, allow_software_gl: bool = True, prefer_gpu_cl: bool = True):
        self.allow_software_gl = allow_software_gl
        self.prefer_gpu_cl = prefer_gpu_cl
        self._lock = threading.Lock()
        self._backends: dict[tuple[str, int], Backend] = {}
        self._failed: dict[str, str] = {}

    def get(self, kind: str) -> Backend:
        key = (kind, threading.get_ident())
        with self._lock:
            b = self._backends.get(key)
            if b is not None:
                return b
            if kind in self._failed:
                raise RuntimeError(self._failed[kind])
        try:
            b = self._create(kind)
        except Exception as e:
            with self._lock:
                self._failed[kind] = f"{kind}: {e}"
            raise
        with self._lock:
            self._backends[key] = b
        return b

    def _create(self, kind: str) -> Backend:
        if kind == "cpu":
            from .cpu_backend import CpuBackend

            return CpuBackend()
        if kind == "opengl":
            from .gl_backend import GlBackend

            b = GlBackend(allow_software=self.allow_software_gl)
            b.renderer_name()  # creates the context now, fails early
            if not self.allow_software_gl and b.is_software():
                raise RuntimeError(f"software OpenGL ({b.renderer_name()})")
            return b
        if kind == "opencl":
            from .cl_backend import ClBackend

            return ClBackend(prefer_gpu=self.prefer_gpu_cl)
        raise ValueError(f"unknown back end '{kind}'")

    def choose(self, preference: str, precise: bool) -> Backend:
        """The back end for ``preference`` ("auto" or a kind), falling back to the CPU.

        ``precise``: a final render wants double precision (OpenCL with fp64,
        else the CPU); a preview wants speed (OpenGL, else OpenCL, else CPU).
        """
        if preference != "auto":
            return self.get(preference)
        order = ["opencl", "cpu"] if precise else ["opengl", "opencl", "cpu"]
        last: Exception | None = None
        for kind in order:
            try:
                b = self.get(kind)
            except Exception as e:
                last = e
                continue
            if precise and kind == "opencl" and b.precision != "double":
                continue
            return b
        raise RuntimeError(f"no render back end available: {last}")

    def close_all(self) -> None:
        with self._lock:
            for b in self._backends.values():
                try:
                    b.close()
                except Exception:
                    pass
            self._backends.clear()
