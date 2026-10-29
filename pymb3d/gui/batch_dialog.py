"""Batch rendering: a list of scene / parameter files rendered one after
another to an output folder (a worker thread, progress per file)."""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
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
    QVBoxLayout,
)

from ..backends.base import RenderCancelled
from ..backends.detect import DeviceChoice, create_backend
from ..io import load_scene
from ..render import Renderer, RenderOptions
from .render_dialog import save_image
from .widgets import IntSpin


def render_files(files: list[str], out_dir: str | Path, choice: DeviceChoice, width: int = 0,
                 progress=None, abort: threading.Event | None = None) -> list[Path]:
    """Renders every file to ``out_dir/<name>.png``; shared by the CLI and the dialog."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    abort = abort or threading.Event()
    backend = create_backend(choice)
    renderer = Renderer(backend, RenderOptions(live_bands=False))
    done = []
    try:
        for k, f in enumerate(files):
            if abort.is_set():
                raise RenderCancelled()
            scene, _ = load_scene(f)
            if width:
                scene.height = max(1, int(round(width * scene.height / max(1, scene.width))))
                scene.width = width
            res = renderer.render(scene, progress=(lambda fr, m, _k=k: progress(_k, len(files), fr, m)) if progress else None,
                                  abort=abort)
            out = out_dir / (Path(f).stem + ".png")
            save_image(out, res.image, scene)
            done.append(out)
    finally:
        backend.close()
    return done


class BatchWorker(QObject):
    progress = Signal(int, int, float, str)
    finished = Signal(int)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, files, out_dir, choice, width) -> None:
        super().__init__()
        self.files, self.out_dir, self.choice, self.width = files, out_dir, choice, width
        self.abort = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            done = render_files(self.files, self.out_dir, self.choice, self.width,
                                progress=lambda k, n, f, m: self.progress.emit(k, n, f, m), abort=self.abort)
            self.finished.emit(len(done))
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n{traceback.format_exc(limit=3)}")


class BatchDialog(QDialog):
    def __init__(self, choice: DeviceChoice, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch render")
        self.choice = choice
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None
        self.list = QListWidget()
        add = QPushButton("Add files ...")
        remove = QPushButton("Remove selected")
        self.out_dir = QLineEdit(str(Path.home() / "pymb3d_batch"))
        browse = QPushButton("...")
        self.width = IntSpin(0, 0, 32767)
        self.width.setSpecialValueText("as in the file")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.status = QLabel("")
        self.start_btn = QPushButton("Render all")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"Device: {choice.label}"))
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(remove)
        lay.addLayout(row)
        form = QFormLayout()
        orow = QHBoxLayout()
        orow.addWidget(self.out_dir, 1)
        orow.addWidget(browse)
        form.addRow("output folder", orow)
        form.addRow("width (height keeps the aspect)", self.width)
        lay.addLayout(form)
        lay.addWidget(self.progress)
        lay.addWidget(self.status)
        brow = QHBoxLayout()
        brow.addWidget(self.start_btn)
        brow.addWidget(self.cancel_btn)
        lay.addLayout(brow)
        self.resize(600, 500)
        add.clicked.connect(self._add)
        remove.clicked.connect(lambda: [self.list.takeItem(self.list.row(i)) for i in self.list.selectedItems()])
        browse.clicked.connect(self._browse)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(lambda: self._worker and self._worker.abort.set())

    def files(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def _add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Scenes", "", "Scenes (*.json *.m3p *.m3i *.txt *.png)")
        for p in paths:
            self.list.addItem(p)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _start(self) -> None:
        files = self.files()
        if not files:
            QMessageBox.information(self, "Batch", "Add files first.")
            return
        self._thread = QThread()
        self._worker = BatchWorker(files, self.out_dir.text(), self.choice, int(self.width.value()))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # bound slots of this (main thread) object: the signals are queued
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._thread.start()

    @Slot(int, int, float, str)
    def _on_progress(self, k: int, n: int, f: float, m: str) -> None:
        self.progress.setValue(int((k + f) / max(1, n) * 1000))
        self.status.setText(f"{k + 1} / {n}: {m}")

    @Slot(int)
    def _on_finished(self, n: int) -> None:
        self._done(f"{n} pictures written to {self.out_dir.text()}")

    @Slot(str)
    def _on_failed(self, msg: str) -> None:
        self._done("failed")
        QMessageBox.critical(self, "Batch failed", msg)

    @Slot()
    def _on_cancelled(self) -> None:
        self._done("cancelled")

    def _done(self, msg: str) -> None:
        self.status.setText(msg)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if self._thread:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
            self._worker = None

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker:
            self._worker.abort.set()
        self._done("")
        super().closeEvent(event)
