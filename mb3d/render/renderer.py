"""The renderer: scene -> painted image, progressively, on the best device.

``Renderer.render`` runs in the calling thread (the GUI runs it in a worker)
and reports through callbacks; it never touches Qt.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from mb3d.kernel.params import KernelParams, RenderOptions, build_params
from mb3d.params.scene import Scene
from mb3d.shading import paint

from .backend import Backend, RenderCancelled, check_cancel
from .device import BackendPool
from .gbuffer import GBuffer


@dataclass
class RenderSettings:
    backend: str = "auto"  # "auto" | "cpu" | "opengl" | "opencl"
    precise: bool = False  # final render: double precision preferred
    ao_samples: int = 5
    ao_length: float = 6.0
    shadow: bool = True
    progressive_scales: tuple[float, ...] = (0.25, 0.5, 1.0)
    first_step_random: bool | None = None


@dataclass
class RenderResult:
    image: np.ndarray  # (h, w, 3) uint8
    gbuffer: GBuffer
    scene: Scene
    backend: str
    precision: str
    seconds: float
    march_seconds: float
    paint_seconds: float
    stats: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


def shadow_direction(scene: Scene) -> tuple[float, float, float]:
    """World space unit direction towards the first light that is on."""
    vg = np.array(scene.vgrads, dtype=np.float64)
    vg = vg / np.linalg.norm(vg, axis=1)[:, None]
    for li in scene.lighting.lights:
        if li.on and not li.positional:
            dtmp = -li.x_angle
            v = np.array([-math.sin(dtmp), math.sin(li.y_angle), math.cos(li.y_angle) * math.cos(dtmp)])
            ln = -v / np.linalg.norm(v)
            world = ln[0] * vg[0] + ln[1] * vg[1] + ln[2] * vg[2]
            return tuple(float(x) for x in world)
    return (0.0, 0.0, -1.0)


def kernel_options(scene: Scene, settings: RenderSettings) -> RenderOptions:
    return RenderOptions(ao_samples=settings.ao_samples, ao_length=settings.ao_length, shadow=settings.shadow,
                         shadow_dir=shadow_direction(scene), first_step_random=settings.first_step_random)


class Renderer:
    def __init__(self, pool: BackendPool | None = None):
        self.pool = pool or BackendPool()

    def backend_for(self, settings: RenderSettings) -> Backend:
        return self.pool.choose(settings.backend, settings.precise)

    def render_pass(self, scene: Scene, settings: RenderSettings, backend: Backend, cancel: threading.Event | None = None,
                    progress=None) -> RenderResult:
        """One full resolution pass of ``scene`` on ``backend``."""
        t0 = time.time()
        params: KernelParams = build_params(scene, kernel_options(scene, settings))
        gbuf = GBuffer.empty(scene.width, scene.height, backend.dtype)
        backend.render(params, gbuf, np.arange(scene.height), cancel, progress)
        t1 = time.time()
        check_cancel(cancel)
        image = paint(gbuf, scene)
        t2 = time.time()
        info = backend.info()
        return RenderResult(image, gbuf, scene, info.kind, info.precision, t2 - t0, t1 - t0, t2 - t1, gbuf.stats())

    def render(self, scene: Scene, settings: RenderSettings | None = None, cancel: threading.Event | None = None,
               on_progress=None, on_pass=None) -> RenderResult:
        """Progressive render: coarse passes first (``on_pass(result, scale)``), the full one last.

        ``on_progress(done_rows, total_rows, scale)`` reports within a pass.
        """
        settings = settings or RenderSettings()
        backend = self.backend_for(settings)
        scales = settings.progressive_scales if settings.progressive_scales else (1.0,)
        result = None
        for scale in scales:
            check_cancel(cancel)
            s = scene.copy()
            if scale != 1.0:
                s.scale_image(scale)
                if s.width == scene.width and s.height == scene.height:
                    continue

            def prog(done, total, _scale=scale):
                if on_progress is not None:
                    on_progress(done, total, _scale)

            result = self.render_pass(s, settings, backend, cancel, prog)
            if on_pass is not None:
                on_pass(result, scale)
        if result is None:
            raise RenderCancelled()
        return result
