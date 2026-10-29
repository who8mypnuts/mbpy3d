"""The render worker: a QObject living in its own QThread that owns one
backend and renders scenes it receives through a queued signal.

Everything crossing the thread boundary is a signal; cancellation uses a
``threading.Event`` the main thread may set at any time.
"""
from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from ...backends.base import RenderCancelled
from ...backends.detect import DeviceChoice, create_backend
from ...core.scene import Scene
from ...render import Renderer, RenderOptions
from ...stats import benchmark


class RenderWorker(QObject):
    #: (request id, fraction 0..1, message)
    progress = Signal(int, float, str)
    #: (request id, first image row, rgb rows as numpy uint8 (h, W, 3))
    band = Signal(int, int, object)
    #: (request id, RenderResult)
    finished = Signal(int, object)
    #: (request id, error message)
    failed = Signal(int, str)
    #: (request id, the partial render or None)
    cancelled = Signal(int, object)
    #: the backend was created: (device label)
    device_ready = Signal(str)

    def __init__(self, choice: DeviceChoice, options: RenderOptions | None = None) -> None:
        super().__init__()
        self.choice = choice
        self.options = options or RenderOptions()
        self._backend = None
        self._renderer: Renderer | None = None
        self._abort = threading.Event()
        self._busy = False

    # ----------------------------------------------------------- thread side
    def _ensure_backend(self) -> Renderer:
        if self._renderer is None:
            self._backend = create_backend(self.choice)
            self._renderer = Renderer(self._backend, self.options)
            self.device_ready.emit(self._backend.info.label)
        return self._renderer

    @Slot(int, object, object)
    def render(self, request_id: int, scene: Scene, resume=None) -> None:
        """Renders ``scene`` (a copy owned by the worker), continuing a
        cancelled render when ``resume`` is given."""
        self._abort.clear()
        self._busy = True
        try:
            renderer = self._ensure_backend()
            result = renderer.render(
                scene,
                progress=lambda f, m: self.progress.emit(request_id, f, m),
                band=lambda y0, rgb: self.band.emit(request_id, y0, rgb),
                abort=self._abort,
                resume=resume,
            )
            info = renderer.backend.info
            probe = getattr(renderer.backend, "last_probe", None)
            if probe is not None:
                result.stats["precision"] = probe.summary
            elif hasattr(renderer.backend, "last_precision"):
                result.stats["precision"] = renderer.backend.last_precision
            benchmark.record(info.name, info.backend, info.precision, "|".join(s.name for s in scene.active_slots),
                             scene.width * scene.height, result.seconds, result.compile_seconds, scene.shadows,
                             scene.ao.mode)
            self.finished.emit(request_id, result)
        except RenderCancelled as e:
            self.cancelled.emit(request_id, getattr(e, "partial", None))
        except Exception as e:  # noqa: BLE001 - reported to the user
            self.failed.emit(request_id, f"{e}\n{traceback.format_exc(limit=3)}")
        finally:
            self._busy = False

    @Slot()
    def shutdown(self) -> None:
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
            self._backend = None
            self._renderer = None

    # ------------------------------------------------------------ main side
    def cancel(self) -> None:
        """Thread safe: asks the running render to stop."""
        self._abort.set()

    @property
    def busy(self) -> bool:
        return self._busy
