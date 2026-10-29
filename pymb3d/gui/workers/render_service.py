"""A render service: one worker thread plus request bookkeeping.

The service accepts scenes at any rate; while a render runs, a newer
request cancels it ("latest wins") so the preview never lags behind the
user.  Results arrive as signals carrying numpy images / RenderResults.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ...backends.detect import DeviceChoice
from ...core.scene import Scene
from ...render import RenderOptions
from .render_worker import RenderWorker


class RenderService(QObject):
    #: (fraction 0..1, message) of the current request
    progress = Signal(float, str)
    #: (first row, rgb rows) of the current request
    band = Signal(int, object)
    #: RenderResult of the current request
    finished = Signal(object)
    failed = Signal(str)
    #: the partial render (resumable) or None
    cancelled = Signal(object)
    device_ready = Signal(str)
    #: the service became idle (no render running, nothing pending)
    idle = Signal()
    _request = Signal(int, object, object)

    def __init__(self, choice: DeviceChoice, options: RenderOptions | None = None, name: str = "render") -> None:
        super().__init__()
        self.choice = choice
        self._thread = QThread()
        self._thread.setObjectName(f"pymb3d-{name}")
        self._worker = RenderWorker(choice, options)
        #: the render options (shared with the worker's renderer; change in place)
        self.options = self._worker.options
        self._worker.moveToThread(self._thread)
        self._request.connect(self._worker.render)
        self._worker.progress.connect(self._on_progress)
        self._worker.band.connect(self._on_band)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.device_ready.connect(self.device_ready)
        self._thread.start()
        self._next_id = 0
        self._current: int | None = None
        self._pending: Scene | None = None

    # ------------------------------------------------------------------ api
    @property
    def busy(self) -> bool:
        return self._current is not None

    def submit(self, scene: Scene, resume=None) -> None:
        """Render ``scene`` as soon as possible, cancelling a running render;
        ``resume`` continues a cancelled render."""
        if self._current is not None:
            self._pending = (scene.copy(), resume)
            self._worker.cancel()
            return
        self._start(scene.copy(), resume)

    def cancel(self) -> None:
        self._pending = None
        if self._current is not None:
            self._worker.cancel()

    def shutdown(self) -> None:
        self.cancel()
        self._thread.quit()
        self._thread.wait(5000)

    # ------------------------------------------------------------- internals
    def _start(self, scene: Scene, resume=None) -> None:
        self._next_id += 1
        self._current = self._next_id
        self._request.emit(self._current, scene, resume)

    def _finish_request(self) -> None:
        self._current = None
        if self._pending is not None:
            (scene, resume), self._pending = self._pending, None
            self._start(scene, resume)
        else:
            self.idle.emit()

    @Slot(int, float, str)
    def _on_progress(self, rid: int, f: float, m: str) -> None:
        if rid == self._current:
            self.progress.emit(f, m)

    @Slot(int, int, object)
    def _on_band(self, rid: int, y0: int, rgb) -> None:
        if rid == self._current:
            self.band.emit(y0, rgb)

    @Slot(int, object)
    def _on_finished(self, rid: int, result) -> None:
        if rid == self._current:
            self.finished.emit(result)
        self._finish_request()

    @Slot(int, str)
    def _on_failed(self, rid: int, msg: str) -> None:
        if rid == self._current:
            self.failed.emit(msg)
        self._finish_request()

    @Slot(int, object)
    def _on_cancelled(self, rid: int, partial) -> None:
        if rid == self._current:
            self.cancelled.emit(partial)
        self._finish_request()
