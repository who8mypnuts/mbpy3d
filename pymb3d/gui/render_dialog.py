"""The final render window: renders the scene at its full size on the
chosen (precise) device in its own worker thread, shows the image as the
rows arrive, and saves it."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from ..core import params as PL
from ..core.scene import Scene
from ..io.gbuffer_file import load_gbuffer, save_gbuffer
from ..render import RenderResult, downscale, hit_alpha, relight, relight_float
from .preview_widget import PreviewWidget
from .workers.render_service import RenderService


def save_image(path: str | Path, rgb: np.ndarray, scene: Scene | None = None, bits: int = 8,
               alpha: np.ndarray | None = None) -> None:
    """Saves a PNG/JPEG; PNG files carry the scene as JSON metadata.  With
    ``bits=16`` a 16 bit PNG is written from a float 0..1 or uint16 image;
    ``alpha`` (0..1) makes a transparent PNG."""
    path = Path(path)
    meta_text = json.dumps(scene.to_dict()) if scene is not None else None
    if bits == 16 and path.suffix.lower() == ".png":
        if rgb.dtype.kind == "f":
            rgb16 = np.clip(np.rint(rgb.astype(np.float64) * 65535.0), 0, 65535).astype(np.uint16)
        elif rgb.dtype == np.uint8:
            rgb16 = rgb.astype(np.uint16) * 257
        else:
            rgb16 = rgb.astype(np.uint16)
        a16 = None if alpha is None else np.clip(np.rint(alpha * 65535.0), 0, 65535).astype(np.uint16)
        path.write_bytes(encode_png16(rgb16, meta_text, a16))
        return
    if rgb.dtype.kind == "f":
        rgb = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    if alpha is not None and path.suffix.lower() == ".png":
        a8 = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
        img = Image.fromarray(np.ascontiguousarray(np.dstack([rgb, a8])), mode="RGBA")
    else:
        img = Image.fromarray(np.ascontiguousarray(rgb))
    if path.suffix.lower() == ".png" and meta_text is not None:
        meta = PngImagePlugin.PngInfo()
        meta.add_text("pymb3d-scene", meta_text)
        img.save(path, pnginfo=meta)
    else:
        img.save(path, quality=95)


def encode_png16(rgb16: np.ndarray, text: str | None = None, alpha16: np.ndarray | None = None) -> bytes:
    """A 16 bit RGB(A) PNG (PIL cannot write 16 bit RGB): a minimal encoder."""
    import struct
    import zlib

    h, w = rgb16.shape[:2]
    if alpha16 is not None:
        rgb16 = np.dstack([rgb16, alpha16])
    channels = rgb16.shape[2]
    raw = np.ascontiguousarray(rgb16.astype(">u2"))
    rows = np.concatenate([np.zeros((h, 1), dtype=np.uint8), raw.view(np.uint8).reshape(h, w * 2 * channels)], axis=1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    out = [b"\x89PNG\r\n\x1a\n", chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 6 if channels == 4 else 2, 0, 0, 0))]
    if text:
        out.append(chunk(b"tEXt", b"pymb3d-scene\x00" + text.encode("latin-1", "replace")))
    out.append(chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)))
    out.append(chunk(b"IEND", b""))
    return b"".join(out)


def save_depth(path: str | Path, gbuf: np.ndarray, scene: Scene, supersample: int = 1) -> None:
    """The Z buffer as a 16 bit grey PNG (near = bright, background = 0)."""
    depth = gbuf[..., PL.G_DEPTH].astype(np.float64)
    W = gbuf.shape[1] // supersample
    zend = max(1e-10, (scene.camera.z_end - scene.camera.z_start) / scene.camera.step_width(W))
    z = np.where(depth >= 0, 1.0 - np.clip(depth / zend, 0.0, 1.0), 0.0)
    z = downscale(z[..., None].astype(np.float32), supersample)[..., 0]
    z16 = np.clip(np.rint(z * 65535.0), 0, 65535).astype(np.uint16)
    Image.fromarray(np.ascontiguousarray(z16)).save(Path(path))


class RenderDialog(QDialog):
    """The final render window ("Calculate 3D"): renders in a worker thread
    on the chosen device with anti-aliasing, shows the rows as they arrive
    with the time left, and saves pictures (8/16 bit, transparent, Z buffer)
    and render data; the lighting can be changed afterwards."""

    def __init__(self, scene: Scene, choice: DeviceChoice, parent=None, model=None, stereo: str = "off",
                 eye_distance: float = 30.0, messages=None) -> None:
        super().__init__(parent)
        self.scene = scene.copy()
        self.choice = choice
        self.model = model
        self.stereo = stereo
        self.eye_distance = eye_distance
        self.messages = messages or (lambda m: None)
        self.result: RenderResult | None = None
        self._thread = None
        self._worker = None
        self._t0 = time.perf_counter()
        self._image = np.zeros((scene.height, scene.width, 3), dtype=np.uint8)
        self.setWindowTitle(f"Calculate 3D: {scene.width} x {scene.height} on {choice.label}"
                            + (f" ({stereo} stereo)" if stereo != "off" else ""))

        self.preview = PreviewWidget()
        self.preview.setMinimumSize(480, 360)
        self.preview.set_image(self._image)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.status = QLabel("starting ...")
        self.eta = QLabel("")
        self.aa = QComboBox()
        for k in (1, 2, 3):
            self.aa.addItem("no anti-aliasing" if k == 1 else f"anti-aliasing {k}x{k}", k)
        self.aa.setToolTip("Renders the picture this many times larger and averages it down")
        self.transparent = QCheckBox("transparent background")
        self.bits16 = QCheckBox("16 bit PNG")
        self.start_btn = QPushButton("Start")
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.setToolTip("Continue a cancelled render with the rows it had calculated")
        self.resume_btn.setEnabled(False)
        self._partial = None
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn = QPushButton("Save image ...")
        self.zbuf_btn = QPushButton("Save Z buffer ...")
        self.copy_btn = QPushButton("Copy")
        self.data_btn = QPushButton("Save render data ...")
        self.data_btn.setToolTip("The G-buffer with the scene (.npz): re-light it later without marching")
        self.load_btn = QPushButton("Load render data ...")
        self.relight_btn = QPushButton("Re-light with the current lighting")
        self.relight_btn.setToolTip("Shades the render again with the lighting of the main window")
        self.close_btn = QPushButton("Close")
        for b in (self.save_btn, self.zbuf_btn, self.data_btn, self.relight_btn):
            b.setEnabled(False)
        opts = QHBoxLayout()
        opts.addWidget(self.aa)
        opts.addWidget(self.transparent)
        opts.addWidget(self.bits16)
        opts.addStretch(1)
        opts.addWidget(self.start_btn)
        opts.addWidget(self.resume_btn)
        opts.addWidget(self.cancel_btn)
        btns = QHBoxLayout()
        for b in (self.save_btn, self.zbuf_btn, self.copy_btn, self.data_btn, self.load_btn, self.relight_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(self.close_btn)
        srow = QHBoxLayout()
        srow.addWidget(self.status, 1)
        srow.addWidget(self.eta)
        lay = QVBoxLayout(self)
        lay.addWidget(self.preview, 1)
        lay.addLayout(opts)
        lay.addWidget(self.progress)
        lay.addLayout(srow)
        lay.addLayout(btns)
        self.resize(max(560, min(1400, scene.width + 40)), max(460, min(1000, scene.height + 160)))

        self.service = RenderService(choice, name="final")
        self.service.progress.connect(self._on_progress)
        self.service.band.connect(self._on_band)
        self.service.finished.connect(self._on_finished)
        self.service.failed.connect(self._on_failed)
        self.service.cancelled.connect(self._on_cancelled)
        self.start_btn.clicked.connect(self.start)
        self.resume_btn.clicked.connect(lambda: self.start(resume=True))
        self.cancel_btn.clicked.connect(self._cancel)
        self.save_btn.clicked.connect(self._save)
        self.zbuf_btn.clicked.connect(self._save_zbuf)
        self.copy_btn.clicked.connect(self._copy)
        self.data_btn.clicked.connect(self._save_data)
        self.load_btn.clicked.connect(self._load_data)
        self.relight_btn.clicked.connect(self._relight)
        self.close_btn.clicked.connect(self.close)
        self.start()

    # ---------------------------------------------------------------- start
    def start(self, resume: bool = False) -> None:
        self._t0 = time.perf_counter()
        self.result = None
        self.progress.setValue(0)
        for b in (self.save_btn, self.zbuf_btn, self.data_btn, self.relight_btn, self.resume_btn):
            b.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        k = int(self.aa.currentData())
        partial = self._partial if resume else None
        self._partial = None
        self.messages(f"{'resume' if partial else 'render'} {self.scene.width} x {self.scene.height} on {self.choice.label}"
                      + (f", anti-aliasing {k}x{k}" if k > 1 else ""))
        if self.stereo != "off":
            self._start_stereo()
        else:
            self.service.options.supersample = k
            self.service.submit(self.scene, resume=partial)

    def _start_stereo(self) -> None:
        from ..backends.detect import create_backend
        from ..render import Renderer, RenderOptions
        from ..render.stereo import render_stereo
        from .workers.function_worker import start_worker

        scene, mode, eyes, choice = self.scene, self.stereo, self.eye_distance, self.choice
        k = int(self.aa.currentData())

        def job(progress, abort):
            backend = create_backend(choice)
            try:
                return render_stereo(Renderer(backend, RenderOptions(live_bands=False, supersample=k)), scene, mode,
                                     eyes, progress=progress, abort=abort)
            finally:
                backend.close()

        self._thread, self._worker = start_worker(job)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_stereo_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

    def _cancel(self) -> None:
        self.service.cancel()
        if self._worker is not None:
            self._worker.abort.set()

    # --------------------------------------------------------------- slots
    @Slot(float, str)
    def _on_progress(self, f: float, m: str) -> None:
        self.progress.setValue(int(f * 1000))
        self.status.setText(m)
        elapsed = time.perf_counter() - self._t0
        if 0.02 < f < 1.0:
            left = elapsed * (1.0 - f) / f
            self.eta.setText(f"{elapsed:.0f} s, about {left:.0f} s left")
        else:
            self.eta.setText(f"{elapsed:.0f} s")

    @Slot(int, object)
    def _on_band(self, y0: int, rgb) -> None:
        h = rgb.shape[0]
        if y0 + h <= self._image.shape[0] and rgb.shape[1] == self._image.shape[1]:
            self._image[y0:y0 + h] = rgb
            self.preview.set_image(self._image)

    @Slot(object)
    def _on_finished(self, result: RenderResult) -> None:
        self.result = result
        self._image = result.image
        self.preview.set_image(result.image)
        prec = result.stats.get("precision")
        msg = f"done in {result.seconds:.1f} s on {result.backend}" + (f"; {prec}" if prec else "")
        self.status.setText(msg)
        self.messages(msg)
        self._finished_ui(result.gbuffer is not None)

    @Slot(object)
    def _on_stereo_done(self, image) -> None:
        self._image = image
        self.result = RenderResult(self.scene, image, None, time.perf_counter() - self._t0, 0.0, self.choice.label)
        self.preview.set_image(image)
        self.status.setText("stereo pair done")
        self._finished_ui(False)

    def _finished_ui(self, has_gbuffer: bool) -> None:
        self.cancel_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        for b in (self.zbuf_btn, self.data_btn):
            b.setEnabled(has_gbuffer)
        self.relight_btn.setEnabled(has_gbuffer and self.model is not None)

    @Slot(str)
    def _on_failed(self, msg: str) -> None:
        self.status.setText("failed")
        self.messages("render failed: " + msg.splitlines()[0])
        self.cancel_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        QMessageBox.critical(self, "Render failed", msg)

    @Slot(object)
    def _on_cancelled(self, partial=None) -> None:
        self._partial = partial
        frac = f" ({partial.fraction * 100:.0f} % done, resumable)" if partial is not None else ""
        self.status.setText("cancelled" + frac)
        self.messages("render cancelled" + frac)
        self.cancel_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.resume_btn.setEnabled(partial is not None)

    # ---------------------------------------------------------------- saving
    def _float_image(self) -> np.ndarray | None:
        if self.result is None or self.result.gbuffer is None:
            return None
        return relight_float(self.result.gbuffer, self.result.scene, self.result.supersample)

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save image", "render.png", "PNG image (*.png);;JPEG image (*.jpg)")
        if not path:
            return
        alpha = None
        if self.transparent.isChecked() and self.result is not None and self.result.gbuffer is not None:
            alpha = hit_alpha(self.result.gbuffer, self.result.supersample)
        if self.bits16.isChecked():
            img = self._float_image()
            if img is not None:
                save_image(path, img, self.scene, bits=16, alpha=alpha)
                self.messages(f"saved {path} (16 bit)")
                return
        save_image(path, self._image, self.scene, alpha=alpha)
        self.messages(f"saved {path}")

    def _save_zbuf(self) -> None:
        if self.result is None or self.result.gbuffer is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Z buffer", "zbuffer.png", "PNG image (*.png)")
        if path:
            save_depth(path, self.result.gbuffer, self.result.scene, self.result.supersample)
            self.messages(f"saved {path}")

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
            save_gbuffer(path, self.result.gbuffer, self.result.scene, self.result.supersample)

    def _load_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load render data", "", "Render data (*.npz)")
        if not path:
            return
        try:
            gbuf, scene, k = load_gbuffer(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self._cancel()
        self.scene = scene
        self.result = RenderResult(scene, relight(gbuf, scene, k), gbuf, 0.0, 0.0, "file", supersample=k)
        self._image = self.result.image
        self.preview.set_image(self._image)
        self.status.setText(f"loaded {Path(path).name}")
        self._finished_ui(True)

    def _relight(self) -> None:
        if self.result is None or self.result.gbuffer is None or self.model is None:
            return
        scene = self.result.scene.copy()
        scene.lighting = self.model.scene.lighting
        scene.ao = self.model.scene.ao
        scene.post = self.model.scene.post
        scene.color_on_otrap = self.model.scene.color_on_otrap
        self.result.scene = scene
        self.result.image = relight(self.result.gbuffer, scene, self.result.supersample)
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
