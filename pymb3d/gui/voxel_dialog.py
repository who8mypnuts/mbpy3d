"""The voxel / mesh export window (MB3D's voxel export and BulbTracer):
the distance field is sampled in a box around a centre and written as PNG
slices or as an OBJ mesh."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..render.voxel import VoxelBox, export_mesh, sample_de_grid, save_slices
from .scene_model import SceneModel
from .widgets import DoubleSpin, IntSpin
from .workers.function_worker import start_worker


class VoxelDialog(QDialog):
    def __init__(self, model: SceneModel, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Voxel / mesh export")
        self.model = model
        self._thread = None
        self._worker = None
        cam = model.scene.camera
        self.center = [DoubleSpin(float(cam.mid[k]), -1e9, 1e9, 6, 0.1) for k in range(3)]
        self.size = DoubleSpin(3.0, 1e-9, 1e9, 6, 0.5)
        self.resolution = IntSpin(64, 8, 1024)
        form = QFormLayout()
        crow = QHBoxLayout()
        for sp in self.center:
            crow.addWidget(sp)
        form.addRow("box centre x y z", crow)
        form.addRow("box size (world units)", self.size)
        form.addRow("resolution (cells per edge)", self.resolution)
        use_mid = QPushButton("Centre on the camera midpoint")
        use_mid.clicked.connect(self._use_mid)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.status = QLabel("")
        mesh_btn = QPushButton("Export OBJ mesh ...")
        slices_btn = QPushButton("Export PNG slices ...")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(use_mid)
        lay.addWidget(self.progress)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        row.addWidget(mesh_btn)
        row.addWidget(slices_btn)
        row.addWidget(self.cancel_btn)
        lay.addLayout(row)
        mesh_btn.clicked.connect(self._mesh)
        slices_btn.clicked.connect(self._slices)
        self.cancel_btn.clicked.connect(lambda: self._worker and self._worker.abort.set())

    def _box(self) -> VoxelBox:
        return VoxelBox(tuple(sp.value() for sp in self.center), float(self.size.value()), int(self.resolution.value()))

    def _use_mid(self) -> None:
        for sp, v in zip(self.center, self.model.scene.camera.mid):
            sp.setValue(float(v))

    def _run(self, fn) -> None:
        self._thread, self._worker = start_worker(fn)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(lambda: self._done("cancelled"))
        self.cancel_btn.setEnabled(True)

    def _mesh(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export mesh", "fractal.obj", "Wavefront OBJ (*.obj)")
        if not path:
            return
        scene, box = self.model.scene.copy(), self._box()
        self._run(lambda progress, abort: f"{export_mesh(scene, box, path, progress, abort)} triangles written to {path}")

    def _slices(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder for the slices")
        if not d:
            return
        scene, box = self.model.scene.copy(), self._box()

        def job(progress, abort):
            field = sample_de_grid(scene, box, progress, abort)
            files = save_slices(field, box.step, d)
            return f"{len(files)} slices written to {d}"

        self._run(job)

    @Slot(float, str)
    def _on_progress(self, f: float, m: str) -> None:
        self.progress.setValue(int(f * 1000))
        self.status.setText(m)

    @Slot(object)
    def _on_finished(self, msg) -> None:
        self._done(str(msg))

    @Slot(str)
    def _on_failed(self, msg: str) -> None:
        self._done("failed")
        QMessageBox.critical(self, "Export failed", msg)

    def _done(self, msg: str) -> None:
        self.status.setText(msg)
        self.cancel_btn.setEnabled(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
            self._worker = None

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None:
            self._worker.abort.set()
        self._done("")
        super().closeEvent(event)
