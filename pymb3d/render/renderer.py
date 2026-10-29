"""The render pipeline: program -> march (backend) -> AO -> shading.

Everything here is plain Python with callbacks; the GUI wraps it in a
worker thread and turns the callbacks into signals.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..backends.base import Backend, RenderCancelled
from ..core import params as PL
from ..core.scene import Scene
from ..kernels.program import Program
from .post import post_process
from .shading import ShadeOptions, shade, to_uint8
from .ssao import apply_ssao

#: progress callback: (fraction 0..1, message)
ProgressFn = Callable[[float, str], None]
#: a strip of the image is ready: (y0, rgb uint8 rows (h, W, 3))
BandFn = Callable[[int, np.ndarray], None]


@dataclass
class RenderResult:
    scene: Scene
    image: np.ndarray                 # (H, W, 3) uint8
    gbuffer: np.ndarray | None        # (H, W, N_CHANNELS) float32, None for huge images
    seconds: float
    compile_seconds: float
    backend: str
    stats: dict = field(default_factory=dict)
    #: the G-buffer is this many times larger than the image (anti-aliasing)
    supersample: int = 1

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


@dataclass
class PartialRender:
    """What a cancelled render calculated (resumable)."""

    gbuffer: np.ndarray
    done_rows: np.ndarray

    @property
    def fraction(self) -> float:
        return float(self.done_rows.mean()) if self.done_rows.size else 0.0


@dataclass
class RenderOptions:
    #: keep the G-buffer (for re-lighting) up to this many pixels
    gbuffer_limit: int = 6_000_000
    #: rows per strip for images beyond the limit
    strip_rows: int = 256
    #: report shaded bands while marching (the final render window)
    live_bands: bool = True
    #: render this many times larger and average down (anti-aliasing)
    supersample: int = 1


def relight_float(gbuf: np.ndarray, scene: Scene, supersample: int = 1) -> np.ndarray:
    """Shade a kept G-buffer again (float 0..1), averaged down by the
    supersampling factor."""
    g = gbuf
    big = scene.resized(gbuf.shape[1], gbuf.shape[0]) if supersample > 1 else scene
    if scene.ao.mode == "ssao":
        g = gbuf.copy()
        apply_ssao(g, scene.ao.radius * supersample, scene.ao.strength)
    img = post_process(shade(g, big), g, big)
    return downscale(img, supersample)


def relight(gbuf: np.ndarray, scene: Scene, supersample: int = 1) -> np.ndarray:
    """Shade a kept G-buffer again with the scene's current lighting and
    post processing."""
    return to_uint8(relight_float(gbuf, scene, supersample))


def downscale(img: np.ndarray, k: int) -> np.ndarray:
    """Averages k x k blocks (the anti-aliasing of a supersampled render)."""
    if k <= 1:
        return img
    H, W = img.shape[:2]
    h, w = H // k, W // k
    a = img[:h * k, :w * k].reshape(h, k, w, k, -1).mean(axis=(1, 3))
    return a.astype(img.dtype) if img.dtype.kind == "f" else a


def hit_alpha(gbuf: np.ndarray, k: int = 1) -> np.ndarray:
    """The coverage (0..1) of the object per image pixel: a transparent
    background, soft at the edges with supersampling."""
    hit = (gbuf[..., PL.G_DEPTH] >= 0).astype(np.float32)[..., None]
    return downscale(hit, k)[..., 0]


class Renderer:
    """Renders scenes on one backend (the backend belongs to the thread
    using the renderer)."""

    def __init__(self, backend: Backend, options: RenderOptions | None = None) -> None:
        self.backend = backend
        self.options = options or RenderOptions()
        self._cpu = None
        self._partial = None

    def cpu_helper(self):
        """A CPU backend for finishing single precision hits in double."""
        if self._cpu is None:
            from ..backends.cpu_backend import CpuBackend

            self._cpu = self.backend if isinstance(self.backend, CpuBackend) else CpuBackend()
        return self._cpu

    def _finish_if_needed(self, program: Program, P: np.ndarray, g: np.ndarray, progress, abort) -> None:
        if getattr(self.backend, "needs_finish", False):
            if progress:
                progress(0.96, "finishing the hits in double precision on the CPU ...")
            self.cpu_helper().finish(program, np.asarray(P, dtype=np.float64), g, abort=abort)

    # ------------------------------------------------------------------ api
    def render(self, scene: Scene, progress: ProgressFn | None = None, band: BandFn | None = None,
               abort: threading.Event | None = None, resume: "PartialRender | None" = None) -> RenderResult:
        """Renders ``scene``; ``resume`` continues a cancelled render of the
        same scene.  A cancellation raises :class:`RenderCancelled` whose
        ``partial`` attribute holds what was calculated."""
        t0 = time.perf_counter()
        abort = abort or threading.Event()
        self._partial = resume
        k = max(1, int(self.options.supersample))
        target = scene
        if k > 1:
            scene = scene.resized(scene.width * k, scene.height * k)
            scene.ao.radius *= k
        program = Program.from_scene(scene)
        compile_s = 0.0
        if not self.backend.is_compiled(program):
            if progress:
                progress(0.0, f"compiling {', '.join(program.formula_names)} for {self.backend.info.label} ...")
            self.backend.compile(program)
            compile_s = self.backend.last_compile_seconds
        W, H = scene.width, scene.height
        if W * H <= self.options.gbuffer_limit:
            try:
                gbuf = self._march_full(program, scene, progress, band, abort)
            except RenderCancelled as e:
                e.partial = self._partial
                raise
            if scene.ao.mode == "ssao":
                if progress:
                    progress(0.97, "ambient occlusion ...")
                apply_ssao(gbuf, scene.ao.radius, scene.ao.strength)
            if progress:
                progress(0.98, "shading ...")
            image = to_uint8(downscale(post_process(shade(gbuf, scene), gbuf, scene), k))
            kept = gbuf
        else:
            image, kept = self._march_strips(program, scene, progress, band, abort)
            image = to_uint8(downscale(image.astype(np.float32) / 255.0, k)) if k > 1 else image
        seconds = time.perf_counter() - t0
        if progress:
            progress(1.0, f"done in {seconds:.1f} s")
        stats = self._stats(kept) if kept is not None else {}
        return RenderResult(target, image, kept, seconds, compile_s, self.backend.info.label, stats, k)

    # ------------------------------------------------------------- internals
    def _march_full(self, program: Program, scene: Scene, progress, band, abort) -> np.ndarray:
        W, H = scene.width, scene.height
        k = max(1, int(self.options.supersample))
        P = program.pack(scene, dtype=self.backend.dtype)
        opts = ShadeOptions(use_ao=scene.ao.mode in ("de", "mc"), full_size=(W, H))
        partial = self._partial
        if partial is not None and partial.gbuffer.shape == (H, W, PL.N_CHANNELS):
            out, done_rows = partial.gbuffer, partial.done_rows
        else:
            out = np.zeros((H, W, PL.N_CHANNELS), dtype=np.float32)
            done_rows = np.zeros(H, dtype=bool)
        self._partial = PartialRender(out, done_rows)
        state = {"flushed": 0, "last": time.perf_counter()}

        def flush(force: bool = False) -> None:
            if not (band and self.options.live_bands):
                return
            now = time.perf_counter()
            if not force and now - state["last"] < 0.25:
                return
            y0 = state["flushed"]
            y1 = y0
            while y1 < H and done_rows[y1]:
                y1 += 1
            y1 -= (y1 - y0) % k if y1 < H else 0
            if y1 > y0:
                opts.origin = (0, y0)
                band(y0 // k, to_uint8(downscale(shade(out[y0:y1], scene, opts), k)))
                state["flushed"] = y1
            state["last"] = now

        def on_band(y0: int, rows: int, done: int, total: int) -> None:
            done_rows[y0:y0 + rows] = True
            if progress:
                progress(0.02 + 0.95 * done / max(1, total), f"marching rows {done}/{total}")
            flush()

        g = self.backend.render(program, P, W, H, progress=on_band, abort=abort, out=out, done_rows=done_rows)
        self._finish_if_needed(program, P, g, progress, abort)
        flush(force=True)
        passes = max(1, scene.ao.passes) if scene.ao.mode == "mc" else 1
        if passes > 1:
            # Monte Carlo: further passes with other seeds, the sky visibility averaged
            acc = g[..., PL.G_AO].astype(np.float64)
            for k in range(1, passes):
                if abort.is_set():
                    raise RenderCancelled()
                Pk = P.copy()
                Pk[PL.P_MC_SEED] = k + 1
                done_rows[:] = False
                state["flushed"] = 0

                def on_band_k(y0, rows, done, total, _k=k):
                    if progress:
                        progress(0.02 + 0.95 * (_k + done / max(1, total)) / passes, f"Monte Carlo pass {_k + 1}/{passes}")

                gk = self.backend.render(program, Pk, W, H, progress=on_band_k, abort=abort)
                acc += gk[..., PL.G_AO]
                g[..., PL.G_AO] = acc / (k + 1)
                if band and self.options.live_bands:
                    opts.origin = (0, 0)
                    band(0, to_uint8(downscale(shade(g, scene, opts), k)))
        return g

    def _march_strips(self, program: Program, scene: Scene, progress, band, abort):
        W, H = scene.width, scene.height
        rows = self.options.strip_rows
        margin = int(scene.ao.radius) + 2 if scene.ao.mode == "ssao" else 0
        image = np.zeros((H, W, 3), dtype=np.uint8)
        y = 0
        while y < H:
            if abort.is_set():
                raise RenderCancelled()
            y1 = min(H, y + rows)
            ya, yb = max(0, y - margin), min(H, y1 + margin)
            P = program.pack(scene, rect=(0, ya, W, yb - ya), dtype=self.backend.dtype)

            def on_band(b0, brows, done, total, _ya=ya):
                if progress:
                    progress(0.02 + 0.95 * (_ya + done) / H, f"marching rows {_ya + done}/{H}")

            g = self.backend.render(program, P, W, yb - ya, progress=on_band, abort=abort)
            self._finish_if_needed(program, P, g, progress, abort)
            if scene.ao.mode == "ssao":
                apply_ssao(g, scene.ao.radius, scene.ao.strength)
            opts = ShadeOptions(use_ao=scene.ao.mode != "none", full_size=(W, H), origin=(0, ya))
            rgb = to_uint8(post_process(shade(g, scene, opts), g, scene, (W, H)))
            image[y:y1] = rgb[y - ya:y1 - ya]
            if band:
                band(y, image[y:y1].copy())
            y = y1
        return image, None

    @staticmethod
    def _stats(gbuf: np.ndarray) -> dict:
        depth = gbuf[..., PL.G_DEPTH]
        hit = depth >= 0
        n = int(hit.sum())
        its = gbuf[..., PL.G_ITER][hit]
        return {
            "hit_fraction": n / depth.size if depth.size else 0.0,
            "avg_steps": float(gbuf[..., PL.G_STEPS].mean()) if depth.size else 0.0,
            "avg_iterations": float(its.mean()) if n else 0.0,
            "max_iterations": int(its.max()) if n else 0,
        }
