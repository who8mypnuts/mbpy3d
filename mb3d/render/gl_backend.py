"""The OpenGL back end (moderngl): single precision, for previews.

A standalone context is created in the thread that renders (the worker
thread of the GUI), so no context is shared with Qt.  The marcher runs as a
fragment shader that writes three RGBA32F render targets (the 12 G-buffer
values); the image is rendered in bands of rows so that a long image
neither blocks the desktop nor trips the driver's time out.
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


def create_context():
    """A standalone moderngl context; EGL first on Linux (works headless)."""
    import moderngl

    errors = []
    import sys

    backends = ["egl", None] if sys.platform.startswith("linux") else [None]
    for b in backends:
        try:
            return moderngl.create_standalone_context(backend=b) if b else moderngl.create_standalone_context()
        except Exception as e:  # pragma: no cover - depends on the machine
            errors.append(f"{b or 'default'}: {e}")
    raise RuntimeError("no OpenGL context: " + "; ".join(errors))


class GlBackend(Backend):
    kind = "opengl"
    precision = "single"

    def __init__(self, band_rows: int = 32, allow_software: bool = True):
        self.band_rows = band_rows
        self.allow_software = allow_software
        self._ctx = None
        self._thread_id = None
        self._programs: dict[tuple[int, ...], object] = {}
        self._fbo = None
        self._fbo_size = (0, 0)
        self._vao = None

    # -------------------------------------------------------------- context
    def _context(self):
        tid = threading.get_ident()
        if self._ctx is None or self._thread_id != tid:
            if self._ctx is not None:
                self.close()
            self._ctx = create_context()
            self._thread_id = tid
            self._programs.clear()
            self._fbo = None
        return self._ctx

    def renderer_name(self) -> str:
        ctx = self._context()
        return f"{ctx.info.get('GL_RENDERER', '?')} / GL {ctx.info.get('GL_VERSION', '?')}"

    def is_software(self) -> bool:
        name = self.renderer_name().lower()
        return any(k in name for k in ("llvmpipe", "softpipe", "swrast", "software"))

    def info(self) -> DeviceInfo:
        try:
            name = self.renderer_name()
        except Exception as e:  # pragma: no cover
            name = f"unavailable ({e})"
        return DeviceInfo("opengl", name, "single", "moderngl fragment shader")

    def close(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.release()
            except Exception:
                pass
        self._ctx = None
        self._programs.clear()
        self._fbo = None
        self._vao = None

    # -------------------------------------------------------------- shaders
    def program(self, fids: tuple[int, ...]):
        ctx = self._context()
        prog = self._programs.get(fids)
        if prog is None:
            src = build.kernel_source("glsl", fids)
            prog = ctx.program(vertex_shader=build.GLSL_VERTEX, fragment_shader=src)
            self._programs[fids] = prog
        return prog

    def _framebuffer(self, width: int, height: int):
        ctx = self._context()
        if self._fbo is None or self._fbo_size != (width, height):
            if self._fbo is not None:
                self._fbo.release()
            tex = [ctx.texture((width, height), 4, dtype="f4") for _ in range(3)]
            self._fbo = ctx.framebuffer(color_attachments=tex)
            self._fbo_size = (width, height)
        return self._fbo

    def _upload(self, prog, params: KernelParams) -> None:
        for name, arr in (("sp", params.sp), ("fp", params.fp)):
            if name in prog:
                prog[name].write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        if "ip" in prog:
            prog["ip"].write(np.ascontiguousarray(params.ip, dtype=np.int32).tobytes())
        prog["u_height"].value = params.height

    # --------------------------------------------------------------- render
    def render(self, params: KernelParams, gbuf: GBuffer, rows: np.ndarray, cancel: threading.Event | None = None, progress=None) -> None:
        ctx = self._context()
        if not self.allow_software and self.is_software():
            raise RuntimeError("OpenGL is software rendered (llvmpipe); not using it")
        t0 = time.time()
        prog = self.program(build.formula_ids(params))
        self._upload(prog, params)
        w, h = params.width, params.height
        fbo = self._framebuffer(w, h)
        fbo.use()
        if self._vao is None:
            self._vao = ctx.vertex_array(prog, [])
        vao = ctx.vertex_array(prog, [])
        rows = np.asarray(rows, dtype=np.int64)
        # the rows are rendered in bands (contiguous runs of requested rows)
        bands = _contiguous_bands(rows)
        done = 0
        for y0, y1 in bands:
            check_cancel(cancel)
            for b0 in range(y0, y1, self.band_rows):
                b1 = min(y1, b0 + self.band_rows)
                # GL rows count from the bottom; image row iy is GL row h - 1 - iy
                ctx.viewport = (0, h - b1, w, b1 - b0)
                vao.render(mode=4, vertices=3)  # TRIANGLES
                ctx.finish()
                done += b1 - b0
                if progress is not None:
                    progress(done, len(rows))
                check_cancel(cancel)
        vao.release()
        ctx.viewport = (0, 0, w, h)
        planes = []
        for tex in fbo.color_attachments:
            raw = np.frombuffer(tex.read(), dtype=np.float32).reshape(h, w, 4)
            planes.append(raw[::-1])
        full = np.concatenate(planes, axis=2)[..., :L.OUT_SIZE]
        gbuf.data[rows] = full[rows].astype(gbuf.data.dtype)
        gbuf.backend = "opengl"
        gbuf.precision = "single"
        gbuf.seconds = time.time() - t0


def _contiguous_bands(rows: np.ndarray) -> list[tuple[int, int]]:
    if len(rows) == 0:
        return []
    rows = np.sort(rows)
    bands = []
    start = prev = int(rows[0])
    for r in rows[1:]:
        r = int(r)
        if r != prev + 1:
            bands.append((start, prev + 1))
            start = r
        prev = r
    bands.append((start, prev + 1))
    return bands
