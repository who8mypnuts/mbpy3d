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

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


@dataclass
class RenderOptions:
    #: keep the G-buffer (for re-lighting) up to this many pixels
    gbuffer_limit: int = 6_000_000
    #: rows per strip for images beyond the limit
    strip_rows: int = 256
    #: report shaded bands while marching (the final render window)
    live_bands: bool = True


def relight(gbuf: np.ndarray, scene: Scene) -> np.ndarray:
    """Shade a kept G-buffer again with the scene's current lighting and
    post processing."""
    g = gbuf
    if scene.ao.mode == "ssao":
        g = gbuf.copy()
        apply_ssao(g, scene.ao.radius, scene.ao.strength)
    return to_uint8(post_process(shade(g, scene), g, scene))


class Renderer:
    """Renders scenes on one backend (the backend belongs to the thread
    using the renderer)."""

    def __init__(self, backend: Backend, options: RenderOptions | None = None) -> None:
        self.backend = backend
        self.options = options or RenderOptions()

    # ------------------------------------------------------------------ api
    def render(self, scene: Scene, progress: ProgressFn | None = None, band: BandFn | None = None,
               abort: threading.Event | None = None) -> RenderResult:
        t0 = time.perf_counter()
        abort = abort or threading.Event()
        program = Program.from_scene(scene)
        compile_s = 0.0
        if not self.backend.is_compiled(program):
            if progress:
                progress(0.0, f"compiling {', '.join(program.formula_names)} for {self.backend.info.label} ...")
            self.backend.compile(program)
            compile_s = self.backend.last_compile_seconds
        W, H = scene.width, scene.height
        if W * H <= self.options.gbuffer_limit:
            gbuf = self._march_full(program, scene, progress, band, abort)
            if scene.ao.mode == "ssao":
                if progress:
                    progress(0.97, "ambient occlusion ...")
                apply_ssao(gbuf, scene.ao.radius, scene.ao.strength)
            if progress:
                progress(0.98, "shading ...")
            image = to_uint8(post_process(shade(gbuf, scene), gbuf, scene))
            kept = gbuf
        else:
            image, kept = self._march_strips(program, scene, progress, band, abort)
        seconds = time.perf_counter() - t0
        if progress:
            progress(1.0, f"done in {seconds:.1f} s")
        stats = self._stats(kept) if kept is not None else {}
        return RenderResult(scene, image, kept, seconds, compile_s, self.backend.info.label, stats)

    # ------------------------------------------------------------- internals
    def _march_full(self, program: Program, scene: Scene, progress, band, abort) -> np.ndarray:
        W, H = scene.width, scene.height
        P = program.pack(scene, dtype=self.backend.dtype)
        opts = ShadeOptions(use_ao=scene.ao.mode in ("de", "mc"), full_size=(W, H))
        out = np.zeros((H, W, PL.N_CHANNELS), dtype=np.float32)
        done_rows = np.zeros(H, dtype=bool)
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
            if y1 > y0:
                opts.origin = (0, y0)
                band(y0, to_uint8(shade(out[y0:y1], scene, opts)))
                state["flushed"] = y1
            state["last"] = now

        def on_band(y0: int, rows: int, done: int, total: int) -> None:
            done_rows[y0:y0 + rows] = True
            if progress:
                progress(0.02 + 0.95 * done / max(1, total), f"marching rows {done}/{total}")
            flush()

        g = self.backend.render(program, P, W, H, progress=on_band, abort=abort, out=out)
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
                    band(0, to_uint8(shade(g, scene, opts)))
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
