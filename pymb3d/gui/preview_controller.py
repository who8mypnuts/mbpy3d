"""The preview controller: turns scene changes into progressive preview
renders (a coarse pass while the user interacts, then the full preview
size) and keeps the last G-buffer so that lighting changes are re-shaded
instantly without marching again."""
from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from ..backends.detect import DeviceChoice
from ..core.scene import Scene
from ..render import RenderOptions, RenderResult, relight
from .workers.render_service import RenderService


class PreviewController(QObject):
    #: rgb uint8 image (H, W, 3) to display
    image_ready = Signal(object)
    status = Signal(str)
    progress = Signal(float)
    #: the render finished: RenderResult
    result_ready = Signal(object)
    device_ready = Signal(str)

    def __init__(self, choice: DeviceChoice, preview_width: int = 480, coarse_factor: int = 3) -> None:
        super().__init__()
        self.preview_width = preview_width
        self.coarse_factor = coarse_factor
        self._service = RenderService(choice, RenderOptions(live_bands=False), "preview")
        self._service.finished.connect(self._on_finished)
        self._service.failed.connect(self._on_failed)
        self._service.progress.connect(self._on_progress)
        self._service.device_ready.connect(self.device_ready)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._flush)
        self._scene: Scene | None = None
        self._coarse_pending = False
        self._fine_pending = False
        self._last: RenderResult | None = None
        self._running_fine = False

    # ------------------------------------------------------------------ api
    @property
    def last_result(self) -> RenderResult | None:
        return self._last

    def preview_size(self, scene: Scene) -> tuple[int, int]:
        w = self.preview_width
        h = max(8, int(round(w * scene.height / max(1, scene.width))))
        return w, h

    def request(self, scene: Scene, interactive: bool = False) -> None:
        """Schedule a preview of ``scene``; ``interactive`` renders a coarse
        pass first (camera drags)."""
        self._scene = scene.copy()
        self._coarse_pending = interactive or self._last is None
        self._fine_pending = True
        self._timer.start()

    def relight(self, scene: Scene) -> bool:
        """Re-shade the kept G-buffer with the scene's lighting.  Returns
        False when a new march is needed."""
        if self._last is None or self._last.gbuffer is None:
            return False
        sized = self._sized(scene)
        if (sized.width, sized.height) != (self._last.width, self._last.height):
            return False
        self._last.scene = sized
        self._last.image = relight(self._last.gbuffer, sized)
        self.image_ready.emit(self._last.image)
        return True

    def change_device(self, choice: DeviceChoice) -> None:
        self._service.shutdown()
        self._service = RenderService(choice, RenderOptions(live_bands=False), "preview")
        self._service.finished.connect(self._on_finished)
        self._service.failed.connect(self._on_failed)
        self._service.progress.connect(self._on_progress)
        self._service.device_ready.connect(self.device_ready)
        self._last = None
        if self._scene is not None:
            self.request(self._scene)

    def shutdown(self) -> None:
        self._timer.stop()
        self._service.shutdown()

    # ------------------------------------------------------------- internals
    def _sized(self, scene: Scene, factor: int = 1) -> Scene:
        w, h = self.preview_size(scene)
        sc = scene.resized(max(8, w // factor), max(8, h // factor))
        if factor > 1:
            # the coarse pass while interacting: no shadows, no occlusion
            sc.shadows = False
            sc.ao.mode = "none"
        return sc

    @Slot()
    def _flush(self) -> None:
        if self._scene is None:
            return
        if self._coarse_pending:
            self._coarse_pending = False
            self._running_fine = False
            self._service.submit(self._sized(self._scene, self.coarse_factor))
        elif self._fine_pending:
            self._fine_pending = False
            self._running_fine = True
            self._service.submit(self._sized(self._scene))

    @Slot(object)
    def _on_finished(self, result: RenderResult) -> None:
        self.image_ready.emit(result.image)
        if self._running_fine:
            self._last = result
            self.result_ready.emit(result)
            self.status.emit(f"preview {result.width}x{result.height} in {result.seconds:.2f} s on {result.backend}")
        else:
            self.status.emit(f"coarse preview in {result.seconds:.2f} s")
        self.progress.emit(1.0)
        if self._fine_pending and not self._timer.isActive():
            self._timer.start()

    @Slot(str)
    def _on_failed(self, msg: str) -> None:
        self.status.emit("preview failed: " + msg.splitlines()[0])
        self.progress.emit(0.0)

    @Slot(float, str)
    def _on_progress(self, f: float, m: str) -> None:
        self.progress.emit(f)
        if m.startswith("compiling"):
            self.status.emit(m)
