"""The OpenGL backend: the C-like core as a GLSL 3.30 fragment shader
(single precision) rendering the G-buffer into four RGBA32F targets.

Uses a standalone moderngl context (EGL/X11 on Linux, WGL on Windows, CGL
on macOS), so it needs no window and runs in the render worker thread.
The context is created by the first thread that uses the backend and
belongs to it.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np

from ..core import params as PL
from ..kernels.program import Program
from .base import Backend, BackendError, DeviceInfo



def _load_vertex_source() -> str:
    from ..kernels.program import KERNEL_DIR

    return (KERNEL_DIR / "vertex_glsl.glsl").read_text()


@dataclass
class _Compiled:
    program: object
    vao: object


def _write_params(prg, Pb: np.ndarray) -> None:
    """Uploads the parameter block (drivers shrink a uniform array to the
    highest index a shader uses, so only that many values are written)."""
    u = prg["u_params"]
    n = min(int(u.array_length), Pb.shape[0])
    u.write(np.ascontiguousarray(Pb[:n], dtype=np.float32).tobytes())


def _create_context():
    import moderngl

    errors = []
    backends = [None]
    if os.name == "posix" and os.uname().sysname == "Linux":
        backends = ["egl", None]
    for b in backends:
        try:
            if b:
                return moderngl.create_context(standalone=True, backend=b, require=330)
            return moderngl.create_context(standalone=True, require=330)
        except Exception as e:  # pragma: no cover - depends on the machine
            errors.append(f"{b or 'default'}: {e}")
    raise BackendError("no OpenGL 3.3 context: " + "; ".join(errors))


class OpenGLBackend(Backend):
    name = "opengl"
    band_target_seconds = 0.1

    def __init__(self) -> None:
        super().__init__()
        self.ctx = None
        self._thread: int | None = None
        self._fbo = None
        self._textures: list = []
        self._fbo_size = (0, 0)
        self._info = DeviceInfo("opengl", "not initialised", "float32", "gpu")

    # ----------------------------------------------------------- context
    def ensure_context(self) -> None:
        tid = threading.get_ident()
        if self.ctx is not None and self._thread == tid:
            return
        if self.ctx is not None and self._thread != tid:
            raise BackendError("the OpenGL backend is bound to another thread")
        self.ctx = _create_context()
        self._thread = tid
        renderer = self.ctx.info.get("GL_RENDERER", "?")
        version = self.ctx.info.get("GL_VERSION", "?")
        soft = any(k in renderer.lower() for k in ("llvmpipe", "softpipe", "swiftshader", "software"))
        self._info = DeviceInfo("opengl", renderer, "float32", "software" if soft else "gpu", version)

    @property
    def info(self) -> DeviceInfo:
        if self.ctx is None:
            try:
                self.ensure_context()
            except BackendError:
                pass
        return self._info

    # ----------------------------------------------------------- compile
    def _compile(self, program: Program) -> _Compiled:
        self.ensure_context()
        try:
            prg = self.ctx.program(vertex_shader=_load_vertex_source(), fragment_shader=program.glsl_source())
        except Exception as e:
            raise BackendError(f"GLSL build failed: {e}") from e
        vao = self.ctx.vertex_array(prg, [])
        return _Compiled(prg, vao)

    def band_rows(self, width: int) -> int:
        return max(1, min(64, 60_000 // max(1, width)))

    def _framebuffer(self, width: int, rows: int):
        if self._fbo is None or self._fbo_size != (width, rows):
            if self._fbo is not None:
                self._fbo.release()
                for t in self._textures:
                    t.release()
            self._textures = [self.ctx.texture((width, rows), 4, dtype="f4") for _ in range(8)]
            self._fbo = self.ctx.framebuffer(color_attachments=self._textures)
            self._fbo_size = (width, rows)
        return self._fbo

    # ------------------------------------------------------------ render
    def _render_band(self, compiled: _Compiled, P: np.ndarray, y0: int, rows: int, out: np.ndarray, abort) -> None:
        self.ensure_context()
        width = out.shape[1]
        Pb = np.ascontiguousarray(P, dtype=np.float32).copy()
        Pb[PL.P_RECT_Y] += y0
        Pb[PL.P_HEIGHT] = rows
        fbo = self._framebuffer(width, rows)
        fbo.use()
        self.ctx.viewport = (0, 0, width, rows)
        _write_params(compiled.program, Pb)
        compiled.vao.render(mode=4, vertices=3)  # TRIANGLES
        self.ctx.finish()
        for k in range(8):
            raw = fbo.read(components=4, attachment=k, dtype="f4")
            arr = np.frombuffer(raw, dtype=np.float32).reshape(rows, width, 4)
            out[y0:y0 + rows, :, 4 * k:4 * k + 4] = arr

    # ---------------------------------------------------------- test hook
    def run_formula(self, program: Program, P: np.ndarray, slot: int, states: np.ndarray) -> np.ndarray:
        """Applies formula ``slot`` once to ``states`` (n, 13) through a small
        test shader (the transpiler equivalence tests); states are (n, 14)."""
        self.ensure_context()
        n = states.shape[0]
        src = "\n".join([
            program.glsl_source().split("/* Fragment shader entry")[0],
            FORMULA_TEST_FRAGMENT,
        ])
        prg = self.ctx.program(vertex_shader=_load_vertex_source(), fragment_shader=src)
        vao = self.ctx.vertex_array(prg, [])
        # 14 values per state in four RGBA texels
        packed = np.zeros((n, 16), dtype=np.float32)
        packed[:, :14] = states
        tex_in = self.ctx.texture((4, n), 4, dtype="f4", data=packed.reshape(n, 4, 4).tobytes())
        outs = [self.ctx.texture((4, n), 4, dtype="f4")]
        fbo = self.ctx.framebuffer(color_attachments=outs)
        fbo.use()
        self.ctx.viewport = (0, 0, 4, n)
        _write_params(prg, np.ascontiguousarray(P, dtype=np.float32))
        prg["u_slot"].value = int(slot)
        tex_in.use(location=0)
        prg["u_states"].value = 0
        vao.render(mode=4, vertices=3)
        self.ctx.finish()
        res = np.frombuffer(fbo.read(components=4, attachment=0, dtype="f4"), dtype=np.float32).reshape(n, 16)
        fbo.release()
        for t in outs:
            t.release()
        tex_in.release()
        vao.release()
        prg.release()
        return res[:, :14].astype(np.float64)

    def close(self) -> None:
        if self.ctx is not None:
            try:
                self.ctx.release()
            except Exception:
                pass
            self.ctx = None


FORMULA_TEST_FRAGMENT = """
uniform sampler2D u_states;
uniform int u_slot;
layout(location = 0) out vec4 o0;
void main() {
    int col = int(floor(gl_FragCoord.x));
    int row = int(floor(gl_FragCoord.y));
    vec4 a = texelFetch(u_states, ivec2(0, row), 0);
    vec4 b = texelFetch(u_states, ivec2(1, row), 0);
    vec4 c = texelFetch(u_states, ivec2(2, row), 0);
    vec4 d = texelFetch(u_states, ivec2(3, row), 0);
    State s;
    s.z = a.xyz;
    s.w = a.w;
    s.c = b.xyz;
    s.cw = b.w;
    s.dr = c.x;
    s.r2 = c.y;
    s.de = c.z;
    s.trap = c.w;
    s.aux = d.x;
    s.vary = d.y;
    s = run_slot(u_slot, s);
    if (col == 0) o0 = vec4(s.z, s.w);
    else if (col == 1) o0 = vec4(s.c, s.cw);
    else if (col == 2) o0 = vec4(s.dr, s.r2, s.de, s.trap);
    else o0 = vec4(s.aux, s.vary, 0.0, 0.0);
}
"""
