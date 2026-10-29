"""Non-blocking rendering for the GUI: a worker thread, debounced requests.

``RenderController`` lives in the GUI thread.  It owns one ``QThread`` with a
``RenderWorker`` that renders one job at a time; the newest request cancels
the running one (through a ``threading.Event`` the kernels poll between
bands) and stale queued jobs are skipped.  Results come back as signals
carrying numpy images; the GUI thread never renders.
"""

from __future__ import annotations

import itertools
import threading
import traceback
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

from mb3d.params import Scene
from mb3d.render import BackendPool, RenderCancelled, Renderer, RenderResult, RenderSettings


@dataclass
class RenderJob:
    job_id: int
    scene: Scene
    settings: RenderSettings
    kind: str  # "preview" | "final"
    cancel: threading.Event = field(default_factory=threading.Event)


class RenderWorker(QObject):
    """Runs in the worker thread; owns the renderer and its device contexts."""

    pass_ready = Signal(int, object, float)  # job id, RenderResult, scale
    progress = Signal(int, int, int, float)  # job id, done, total, scale
    finished = Signal(int, object)  # job id, RenderResult
    failed = Signal(int, str)
    cancelled = Signal(int)
    device_report = Signal(object)
    message = Signal(str)

    def __init__(self, pool: BackendPool | None = None):
        super().__init__()
        self.renderer = Renderer(pool)
        self._latest = 0
        self._lock = threading.Lock()

    def note_latest(self, job_id: int) -> None:
        with self._lock:
            self._latest = max(self._latest, job_id)

    def _is_stale(self, job: RenderJob) -> bool:
        with self._lock:
            return job.kind == "preview" and job.job_id < self._latest

    @Slot(object)
    def run_job(self, job: RenderJob) -> None:
        if self._is_stale(job) or job.cancel.is_set():
            self.cancelled.emit(job.job_id)
            return
        try:
            result = self.renderer.render(
                job.scene, job.settings, job.cancel,
                on_progress=lambda d, t, s: self.progress.emit(job.job_id, d, t, s),
                on_pass=lambda r, s: self.pass_ready.emit(job.job_id, r, s),
            )
            self.finished.emit(job.job_id, result)
        except RenderCancelled:
            self.cancelled.emit(job.job_id)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(job.job_id, f"{e}\n{traceback.format_exc()}")

    @Slot()
    def probe_devices(self) -> None:
        from mb3d.render.device import probe

        try:
            self.device_report.emit(probe(allow_software_gl=True))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(0, str(e))

    @Slot()
    def warm_up(self) -> None:
        try:
            from mb3d.render.cpu_backend import warm_up

            self.message.emit("compiling the CPU kernels (cached for the next start)…")
            warm_up()
            self.message.emit("ready")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(0, f"warm up: {e}")


class RenderController(QObject):
    """GUI side: request previews and final renders, get signals back."""

    preview_pass = Signal(object, float)  # RenderResult, scale
    preview_finished = Signal(object)
    final_pass = Signal(object, float)
    final_finished = Signal(object)
    progress = Signal(int, int, float, str)  # done, total, scale, kind
    failed = Signal(str)
    busy_changed = Signal(bool)
    devices_found = Signal(object)
    message = Signal(str)

    _submit = Signal(object)
    _probe = Signal()
    _warm = Signal()

    def __init__(self, parent: QObject | None = None, debounce_ms: int = 120, pool: BackendPool | None = None):
        super().__init__(parent)
        self._ids = itertools.count(1)
        self._jobs: dict[int, RenderJob] = {}
        self._pending: RenderJob | None = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._flush_pending)
        self._thread = QThread(self)
        self._thread.setObjectName("mb3d-render")
        self._worker = RenderWorker(pool)
        self._worker.moveToThread(self._thread)
        self._submit.connect(self._worker.run_job, Qt.QueuedConnection)
        self._probe.connect(self._worker.probe_devices, Qt.QueuedConnection)
        self._warm.connect(self._worker.warm_up, Qt.QueuedConnection)
        self._worker.pass_ready.connect(self._on_pass)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.device_report.connect(self.devices_found)
        self._worker.message.connect(self.message)
        self._thread.start()
        self._running: set[int] = set()

    # --------------------------------------------------------------- API
    @property
    def busy(self) -> bool:
        return bool(self._running)

    def request_preview(self, scene: Scene, settings: RenderSettings) -> None:
        """Render ``scene`` soon; earlier previews are cancelled."""
        job = RenderJob(next(self._ids), scene.copy(), settings, "preview")
        self._pending = job
        self._worker.note_latest(job.job_id)
        self.cancel_previews()
        self._debounce.start()

    def request_final(self, scene: Scene, settings: RenderSettings) -> int:
        job = RenderJob(next(self._ids), scene.copy(), settings, "final")
        self._start(job)
        return job.job_id

    def cancel_previews(self) -> None:
        for job in list(self._jobs.values()):
            if job.kind == "preview":
                job.cancel.set()

    def cancel_all(self) -> None:
        self._pending = None
        for job in self._jobs.values():
            job.cancel.set()

    def probe_devices(self) -> None:
        self._probe.emit()

    def warm_up(self) -> None:
        self._warm.emit()

    def shutdown(self) -> None:
        self.cancel_all()
        self._thread.quit()
        self._thread.wait(5000)

    # --------------------------------------------------------- internals
    def _flush_pending(self) -> None:
        job, self._pending = self._pending, None
        if job is not None:
            self._start(job)

    def _start(self, job: RenderJob) -> None:
        self._jobs[job.job_id] = job
        self._running.add(job.job_id)
        self.busy_changed.emit(True)
        self._submit.emit(job)

    def _done(self, job_id: int) -> RenderJob | None:
        job = self._jobs.pop(job_id, None)
        self._running.discard(job_id)
        if not self._running:
            self.busy_changed.emit(False)
        return job

    def _on_pass(self, job_id: int, result: RenderResult, scale: float) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.cancel.is_set():
            return
        (self.preview_pass if job.kind == "preview" else self.final_pass).emit(result, scale)

    def _on_progress(self, job_id: int, done: int, total: int, scale: float) -> None:
        job = self._jobs.get(job_id)
        if job is not None and not job.cancel.is_set():
            self.progress.emit(done, total, scale, job.kind)

    def _on_finished(self, job_id: int, result: RenderResult) -> None:
        job = self._done(job_id)
        if job is None:
            return
        (self.preview_finished if job.kind == "preview" else self.final_finished).emit(result)

    def _on_failed(self, job_id: int, message: str) -> None:
        self._done(job_id)
        self.failed.emit(message)

    def _on_cancelled(self, job_id: int) -> None:
        self._done(job_id)
