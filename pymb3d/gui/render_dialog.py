"""The final render window: renders the scene at its full size on the
chosen (precise) device in its own worker thread, shows the image as the
rows arrive, and saves it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..backends.detect import DeviceChoice
from ..core.scene import Scene
from ..render import RenderResult
from .preview_widget import PreviewWidget
from .workers.render_service import RenderService


def save_image(path: str | Path, rgb: np.ndarray, scene: Scene | None = None) -> None:
    """Saves a PNG/JPEG; PNG files carry the scene as JSON metadata."""
    path = Path(path)
    img = Image.fromarray(np.ascontiguousarray(rgb))
    if path.suffix.lower() == ".png" and scene is not None:
        meta = PngImagePlugin.PngInfo()
        meta.add_text("pymb3d-scene", json.dumps(scene.to_dict()))
        img.save(path, pnginfo=meta)
    else:
        img.save(path, quality=95)


class RenderDialog(QDialog):
    def __init__(self, scene: Scene, choice: DeviceChoice, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Final render {scene.width} x {scene.height} on {choice.label}")
        self.scene = scene.copy()
        self.choice = choice
        self.result: RenderResult | None = None
        self._image = np.zeros((scene.height, scene.width, 3), dtype=np.uint8)
        self.preview = PreviewWidget()
        self.preview.setMinimumSize(480, 360)
        self.preview.set_image(self._image)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.status = QLabel("starting ...")
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn = QPushButton("Save image ...")
        self.save_btn.setEnabled(False)
        self.close_btn = QPushButton("Close")
        btns = QHBoxLayout()
        btns.addWidget(self.status, 1)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.save_btn)
        btns.addWidget(self.close_btn)
        lay = QVBoxLayout(self)
        lay.addWidget(self.preview, 1)
        lay.addWidget(self.progress)
        lay.addLayout(btns)
        self.resize(max(520, min(1400, scene.width + 40)), max(420, min(1000, scene.height + 120)))

        self.service = RenderService(choice, name="final")
        self.service.progress.connect(self._on_progress)
        self.service.band.connect(self._on_band)
        self.service.finished.connect(self._on_finished)
        self.service.failed.connect(self._on_failed)
        self.service.cancelled.connect(self._on_cancelled)
        self.cancel_btn.clicked.connect(self.service.cancel)
        self.save_btn.clicked.connect(self._save)
        self.close_btn.clicked.connect(self.close)
        self.service.submit(self.scene)

    # --------------------------------------------------------------- slots
    @Slot(float, str)
    def _on_progress(self, f: float, m: str) -> None:
        self.progress.setValue(int(f * 1000))
        self.status.setText(m)

    @Slot(int, object)
    def _on_band(self, y0: int, rgb) -> None:
        h = rgb.shape[0]
        self._image[y0:y0 + h] = rgb
        self.preview.set_image(self._image)

    @Slot(object)
    def _on_finished(self, result: RenderResult) -> None:
        self.result = result
        self._image = result.image
        self.preview.set_image(result.image)
        self.status.setText(f"done in {result.seconds:.1f} s on {result.backend}")
        self.cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(True)

    @Slot(str)
    def _on_failed(self, msg: str) -> None:
        self.status.setText("failed")
        self.cancel_btn.setEnabled(False)
        QMessageBox.critical(self, "Render failed", msg)

    @Slot()
    def _on_cancelled(self) -> None:
        self.status.setText("cancelled")
        self.cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(True)

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save image", "render.png", "PNG image (*.png);;JPEG image (*.jpg)")
        if path:
            save_image(path, self._image, self.scene)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.service.shutdown()
        super().closeEvent(event)
