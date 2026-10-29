"""The animation window: keyframes taken from the current scene, a scrub
slider that puts the interpolated scene into the main preview, and the
frame renderer running in a worker thread."""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from ..backends.base import RenderCancelled
from ..backends.detect import DeviceChoice, create_backend
from ..core.animation import Animation
from ..render.animation import encode_video, render_animation
from .scene_model import SceneModel
from .widgets import DoubleSpin, IntSpin


class FrameRenderWorker(QObject):
    progress = Signal(int, int, float, str)
    finished = Signal(int)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, anim: Animation, choice: DeviceChoice, out_dir: str, width: int, height: int,
                 video: str | None) -> None:
        super().__init__()
        self.anim = anim
        self.choice = choice
        self.out_dir = out_dir
        self.width, self.height = width, height
        self.video = video
        self.abort = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            backend = create_backend(self.choice)
            files = render_animation(self.anim, backend, self.out_dir, self.width, self.height,
                                     progress=lambda k, n, f, m: self.progress.emit(k, n, f, m), abort=self.abort)
            if self.video:
                encode_video(self.out_dir, self.video, self.anim.fps)
            backend.close()
            self.finished.emit(len(files))
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n{traceback.format_exc(limit=3)}")


class AnimationDialog(QDialog):
    def __init__(self, model: SceneModel, final_choice: DeviceChoice, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Animation")
        self.model = model
        self.choice = final_choice
        self.anim = Animation()
        self._thread: QThread | None = None
        self._worker: FrameRenderWorker | None = None

        self.list = QListWidget()
        self.frame_spin = IntSpin(0, 0, 1_000_000)
        add = QPushButton("Set keyframe at frame")
        add.setToolTip("Stores the current scene as the keyframe at this frame")
        goto = QPushButton("Apply selected keyframe to the scene")
        remove = QPushButton("Remove selected")
        self.fps = DoubleSpin(25.0, 1.0, 120.0, 2, 1.0)
        self.easing = QComboBox()
        self.easing.addItem("linear", "linear")
        self.easing.addItem("ease in / out", "ease")
        self.easing.addItem("spline through the keyframes", "spline")
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, 0)
        self.scrub_label = QLabel("frame 0 / 0")
        self.width = IntSpin(640, 8, 32767)
        self.height = IntSpin(360, 8, 32767)
        self.out_dir = QLineEdit(str(Path.home() / "pymb3d_frames"))
        browse = QPushButton("...")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.status = QLabel("")
        self.render_btn = QPushButton("Render frames")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        save = QPushButton("Save animation ...")
        load = QPushButton("Load animation ...")

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("<b>Keyframes</b> (frame numbers)"))
        lay.addWidget(self.list)
        row = QHBoxLayout()
        row.addWidget(self.frame_spin)
        row.addWidget(add)
        row.addWidget(goto)
        row.addWidget(remove)
        lay.addLayout(row)
        form = QFormLayout()
        form.addRow("frames per second", self.fps)
        form.addRow("interpolation", self.easing)
        lay.addLayout(form)
        lay.addWidget(QLabel("<b>Scrub</b> (shows the interpolated scene in the preview)"))
        srow = QHBoxLayout()
        srow.addWidget(self.scrub, 1)
        srow.addWidget(self.scrub_label)
        lay.addLayout(srow)
        lay.addWidget(QLabel("<b>Render</b>"))
        rform = QFormLayout()
        size = QHBoxLayout()
        size.addWidget(self.width)
        size.addWidget(QLabel("x"))
        size.addWidget(self.height)
        rform.addRow("frame size", size)
        orow = QHBoxLayout()
        orow.addWidget(self.out_dir, 1)
        orow.addWidget(browse)
        rform.addRow("output folder", orow)
        lay.addLayout(rform)
        lay.addWidget(self.progress)
        lay.addWidget(self.status)
        brow = QHBoxLayout()
        brow.addWidget(self.render_btn)
        brow.addWidget(self.cancel_btn)
        brow.addStretch(1)
        brow.addWidget(save)
        brow.addWidget(load)
        lay.addLayout(brow)
        self.resize(560, 640)

        add.clicked.connect(self._add)
        goto.clicked.connect(self._goto)
        remove.clicked.connect(self._remove)
        self.fps.valueChanged.connect(lambda v: setattr(self.anim, "fps", float(v)))
        self.easing.currentIndexChanged.connect(lambda i: setattr(self.anim, "easing", self.easing.currentData()))
        self.scrub.valueChanged.connect(self._scrub)
        browse.clicked.connect(self._browse)
        self.render_btn.clicked.connect(self._render)
        self.cancel_btn.clicked.connect(self._cancel)
        save.clicked.connect(self._save)
        load.clicked.connect(self._load)
        self.width.set_silently(model.scene.width)
        self.height.set_silently(model.scene.height)

    # ------------------------------------------------------------ keyframes
    def _refresh(self) -> None:
        self.list.clear()
        for k in self.anim.sorted():
            names = ", ".join(s["name"] for s in k.scene.get("slots", []) if s.get("iterations", 0) > 0)
            self.list.addItem(f"frame {k.frame}: zoom {k.scene['camera']['zoom']:.4g}, {names}")
        n = self.anim.frame_count
        self.scrub.setRange(0, max(0, n - 1))
        self.scrub_label.setText(f"frame {self.scrub.value()} / {max(0, n - 1)}")

    def _selected_frame(self) -> int | None:
        i = self.list.currentRow()
        keys = self.anim.sorted()
        return keys[i].frame if 0 <= i < len(keys) else None

    def _add(self) -> None:
        self.anim.set_keyframe(int(self.frame_spin.value()), self.model.scene)
        self._refresh()

    def _goto(self) -> None:
        f = self._selected_frame()
        if f is None:
            return
        self.model.replace(self.anim.scene_at(f), self.model.path)

    def _remove(self) -> None:
        f = self._selected_frame()
        if f is not None:
            self.anim.remove_keyframe(f)
            self._refresh()

    def _scrub(self, v: int) -> None:
        if not self.anim.keyframes:
            return
        self.scrub_label.setText(f"frame {v} / {max(0, self.anim.frame_count - 1)}")
        self.model.replace(self.anim.scene_at(int(v)), self.model.path, remember=False)

    # --------------------------------------------------------------- render
    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _render(self) -> None:
        if len(self.anim.keyframes) < 2:
            QMessageBox.information(self, "Animation", "Set at least two keyframes first.")
            return
        out = self.out_dir.text().strip()
        video = str(Path(out) / "animation.mp4")
        self._thread = QThread()
        self._worker = FrameRenderWorker(Animation.from_dict(self.anim.to_dict()), self.choice, out,
                                         int(self.width.value()), int(self.height.value()), video)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self.render_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker:
            self._worker.abort.set()

    def _done(self) -> None:
        self.render_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if self._thread:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
            self._worker = None

    @Slot(int, int, float, str)
    def _on_progress(self, k: int, n: int, f: float, m: str) -> None:
        self.progress.setValue(int((k + f) / max(1, n) * 1000))
        self.status.setText(f"frame {k + 1} / {n}: {m}")

    @Slot(int)
    def _on_finished(self, n: int) -> None:
        self.status.setText(f"{n} frames written to {self.out_dir.text()}")
        self._done()

    @Slot(str)
    def _on_failed(self, msg: str) -> None:
        self.status.setText("failed")
        self._done()
        QMessageBox.critical(self, "Animation render failed", msg)

    @Slot()
    def _on_cancelled(self) -> None:
        self.status.setText("cancelled")
        self._done()

    # ------------------------------------------------------------------- io
    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save animation", "animation.json", "Animation (*.json)")
        if path:
            self.anim.save(path)

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load animation", "", "Animation (*.json)")
        if path:
            self.anim = Animation.load(path)
            self.fps.set_silently(self.anim.fps)
            self.easing.setCurrentIndex(max(0, self.easing.findData(self.anim.easing)))
            self._refresh()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._cancel()
        self._done()
        super().closeEvent(event)
