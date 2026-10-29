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
from ..io.gbuffer_file import load_gbuffer, save_gbuffer
from ..render import RenderResult, relight
from .preview_widget import PreviewWidget
from .workers.render_service import RenderService


def save_image(path: str | Path, rgb: np.ndarray, scene: Scene | None = None, bits: int = 8) -> None:
    """Saves a PNG/JPEG; PNG files carry the scene as JSON metadata.  With
    ``bits=16`` a 16 bit PNG is written from a float 0..1 or uint16 image."""
    path = Path(path)
    meta_text = json.dumps(scene.to_dict()) if scene is not None else None
    if bits == 16 and path.suffix.lower() == ".png":
        if rgb.dtype.kind == "f":
            rgb16 = np.clip(np.rint(rgb.astype(np.float64) * 65535.0), 0, 65535).astype(np.uint16)
        elif rgb.dtype == np.uint8:
            rgb16 = rgb.astype(np.uint16) * 257
        else:
            rgb16 = rgb.astype(np.uint16)
        path.write_bytes(encode_png16(rgb16, meta_text))
        return
    img = Image.fromarray(np.ascontiguousarray(rgb))
    if path.suffix.lower() == ".png" and meta_text is not None:
        meta = PngImagePlugin.PngInfo()
        meta.add_text("pymb3d-scene", meta_text)
        img.save(path, pnginfo=meta)
    else:
        img.save(path, quality=95)


def encode_png16(rgb16: np.ndarray, text: str | None = None) -> bytes:
    """A 16 bit RGB PNG (PIL cannot write 16 bit RGB): a minimal encoder."""
    import struct
    import zlib

    h, w = rgb16.shape[:2]
    raw = np.ascontiguousarray(rgb16.astype(">u2"))
    rows = np.concatenate([np.zeros((h, 1), dtype=np.uint8), raw.view(np.uint8).reshape(h, w * 6)], axis=1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    out = [b"\x89PNG\r\n\x1a\n", chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))]
    if text:
        out.append(chunk(b"tEXt", b"pymb3d-scene\x00" + text.encode("latin-1", "replace")))
    out.append(chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)))
    out.append(chunk(b"IEND", b""))
    return b"".join(out)


class RenderDialog(QDialog):
    def __init__(self, scene: Scene, choice: DeviceChoice, parent=None, model=None, stereo: str = "off",
                 eye_distance: float = 30.0) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Final render {scene.width} x {scene.height} on {choice.label}"
                            + (f" ({stereo} stereo)" if stereo != "off" else ""))
        self.scene = scene.copy()
        self.choice = choice
        self.model = model
        self.stereo = stereo
        self.eye_distance = eye_distance
        self._thread = None
        self._worker = None
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
        self.save16_btn = QPushButton("Save 16 bit PNG ...")
        self.save16_btn.setEnabled(False)
        self.copy_btn = QPushButton("Copy")
        self.data_btn = QPushButton("Save render data ...")
        self.data_btn.setToolTip("The G-buffer with the scene (.npz): re-light it later without marching")
        self.data_btn.setEnabled(False)
        self.load_btn = QPushButton("Load render data ...")
        self.relight_btn = QPushButton("Re-light with the current lighting")
        self.relight_btn.setToolTip("Shades the render again with the lighting of the main window")
        self.relight_btn.setEnabled(False)
        self.close_btn = QPushButton("Close")
        btns = QHBoxLayout()
        btns.addWidget(self.status, 1)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.save_btn)
        btns.addWidget(self.save16_btn)
        btns.addWidget(self.copy_btn)
        btns.addWidget(self.data_btn)
        btns.addWidget(self.load_btn)
        btns.addWidget(self.relight_btn)
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
        self.cancel_btn.clicked.connect(self._cancel)
        self.save_btn.clicked.connect(self._save)
        self.save16_btn.clicked.connect(self._save16)
        self.copy_btn.clicked.connect(self._copy)
        self.data_btn.clicked.connect(self._save_data)
        self.load_btn.clicked.connect(self._load_data)
        self.relight_btn.clicked.connect(self._relight)
        self.close_btn.clicked.connect(self.close)
        if stereo != "off":
            self._start_stereo()
        else:
            self.service.submit(self.scene)

    def _start_stereo(self) -> None:
        from ..backends.detect import create_backend
        from ..render import Renderer, RenderOptions
        from ..render.stereo import render_stereo
        from .workers.function_worker import start_worker

        scene, mode, eyes, choice = self.scene, self.stereo, self.eye_distance, self.choice

        def job(progress, abort):
            backend = create_backend(choice)
            try:
                return render_stereo(Renderer(backend, RenderOptions(live_bands=False)), scene, mode, eyes,
                                     progress=progress, abort=abort)
            finally:
                backend.close()

        self._thread, self._worker = start_worker(job)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_stereo_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

    @Slot(object)
    def _on_stereo_done(self, image) -> None:
        self._image = image
        self.result = RenderResult(self.scene, image, None, 0.0, 0.0, self.choice.label)
        self.preview.set_image(image)
        self.status.setText("stereo pair done")
        self.cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(True)

    def _cancel(self) -> None:
        self.service.cancel()
        if self._worker is not None:
            self._worker.abort.set()

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
        prec = result.stats.get("precision")
        self.status.setText(f"done in {result.seconds:.1f} s on {result.backend}" + (f"; {prec}" if prec else ""))
        self.cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.save16_btn.setEnabled(result.gbuffer is not None)
        self.data_btn.setEnabled(result.gbuffer is not None)
        self.relight_btn.setEnabled(result.gbuffer is not None and self.model is not None)

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

    def _save16(self) -> None:
        if self.result is None or self.result.gbuffer is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save 16 bit PNG", "render16.png", "PNG image (*.png)")
        if path:
            from ..render.post import post_process
            from ..render.shading import shade
            from ..render.ssao import apply_ssao

            g = self.result.gbuffer
            sc = self.result.scene
            if sc.ao.mode == "ssao":
                g = g.copy()
                apply_ssao(g, sc.ao.radius, sc.ao.strength)
            img = post_process(shade(g, sc), g, sc)
            save_image(path, img, sc, bits=16)

    def _copy(self) -> None:
        from PySide6.QtGui import QGuiApplication

        from .widgets import numpy_to_qimage

        QGuiApplication.clipboard().setImage(numpy_to_qimage(self._image))
        self.status.setText("image copied to the clipboard")

    def _save_data(self) -> None:
        if self.result is None or self.result.gbuffer is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save render data", "render.pymb3d.npz", "Render data (*.npz)")
        if path:
            save_gbuffer(path, self.result.gbuffer, self.result.scene)

    def _load_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load render data", "", "Render data (*.npz)")
        if not path:
            return
        try:
            gbuf, scene = load_gbuffer(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self.service.cancel()
        self.scene = scene
        self.result = RenderResult(scene, relight(gbuf, scene), gbuf, 0.0, 0.0, "file")
        self._image = self.result.image
        self.preview.set_image(self._image)
        self.status.setText(f"loaded {Path(path).name}")
        self.save_btn.setEnabled(True)
        self.data_btn.setEnabled(True)
        self.relight_btn.setEnabled(self.model is not None)
        self.cancel_btn.setEnabled(False)

    def _relight(self) -> None:
        if self.result is None or self.result.gbuffer is None or self.model is None:
            return
        scene = self.result.scene.copy()
        scene.lighting = self.model.scene.lighting
        scene.ao = self.model.scene.ao
        scene.color_on_otrap = self.model.scene.color_on_otrap
        self.result.scene = scene
        self.result.image = relight(self.result.gbuffer, scene)
        self._image = self.result.image
        self.preview.set_image(self._image)
        self.status.setText("re-lit")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._cancel()
        self.service.shutdown()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        super().closeEvent(event)
