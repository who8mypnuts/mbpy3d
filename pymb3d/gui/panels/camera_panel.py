"""The Camera panel: position, zoom, field of view, rotation angles, the
start / end planes and the cutting planes."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ...core.camera import Camera
from ..scene_model import SceneModel
from ..widgets import DoubleSpin, make_scroll


class CameraPanel(QWidget):
    def __init__(self, model: SceneModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self._loading = False
        inner = QWidget()
        lay = QVBoxLayout(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(make_scroll(inner))

        pos = QGroupBox("Position and view")
        form = QFormLayout(pos)
        self.mid = [DoubleSpin(0.0, -1e9, 1e9, 9, 0.01) for _ in range(3)]
        row = QHBoxLayout()
        for sp in self.mid:
            sp.setMinimumWidth(60)
            row.addWidget(sp)
        form.addRow("midpoint x y z", row)
        self.zoom = DoubleSpin(1.0, 1e-12, 1e15, 6, 0.1)
        self.fov = DoubleSpin(60.0, 0.1, 179.0, 2, 1.0)
        self.angles = [DoubleSpin(0.0, -3600, 3600, 3, 1.0) for _ in range(3)]
        arow = QHBoxLayout()
        for sp in self.angles:
            arow.addWidget(sp)
        self.z_start = DoubleSpin(-3.0, -1e9, 1e9, 6, 0.1)
        self.z_end = DoubleSpin(4.0, -1e9, 1e9, 6, 0.1)
        form.addRow("zoom", self.zoom)
        form.addRow("field of view (deg)", self.fov)
        form.addRow("rotation x y z (deg)", arow)
        form.addRow("Z start", self.z_start)
        form.addRow("Z end", self.z_end)
        btns = QHBoxLayout()
        reset = QPushButton("Reset camera")
        reset.clicked.connect(self._reset)
        btns.addWidget(reset)
        form.addRow(btns)
        lay.addWidget(pos)

        cut = QGroupBox("Cutting planes")
        cform = QFormLayout(cut)
        self.cut_checks = [QCheckBox(a) for a in ("x", "y", "z")]
        self.cut_pos = [DoubleSpin(0.0, -1e9, 1e9, 6, 0.01) for _ in range(3)]
        for c, sp in zip(self.cut_checks, self.cut_pos):
            r = QHBoxLayout()
            r.addWidget(c)
            r.addWidget(sp)
            cform.addRow(r)
        lay.addWidget(cut)

        r4 = QGroupBox("4D rotation (XW, YW, ZW planes, degrees)")
        rf = QHBoxLayout(r4)
        self.rot4d = [DoubleSpin(0.0, -3600, 3600, 3, 1.0) for _ in range(3)]
        for sp in self.rot4d:
            rf.addWidget(sp)
            sp.valueChanged.connect(self._rot4d_changed)
        lay.addWidget(r4)
        lay.addStretch(1)

        for sp in self.mid:
            sp.valueChanged.connect(self._mid_changed)
        self.zoom.valueChanged.connect(lambda v: self._cam_set("zoom", float(v)))
        self.fov.valueChanged.connect(lambda v: self._cam_set("fov_y", float(v)))
        self.z_start.valueChanged.connect(lambda v: self._cam_set("z_start", float(v)))
        self.z_end.valueChanged.connect(lambda v: self._cam_set("z_end", float(v)))
        for sp in self.angles:
            sp.valueChanged.connect(self._angles_changed)
        for c in self.cut_checks:
            c.toggled.connect(self._cut_changed)
        for sp in self.cut_pos:
            sp.valueChanged.connect(self._cut_changed)
        model.replaced.connect(self.load)
        model.changed.connect(self._on_changed)
        self.load()

    # ---------------------------------------------------------------- load
    def load(self) -> None:
        self._loading = True
        sc = self.model.scene
        cam = sc.camera
        for sp, v in zip(self.mid, cam.mid):
            sp.set_silently(float(v))
        self.zoom.set_silently(cam.zoom)
        self.fov.set_silently(cam.fov_y)
        for sp, v in zip(self.angles, cam.angles_deg):
            sp.set_silently(v)
        self.z_start.set_silently(cam.z_start)
        self.z_end.set_silently(cam.z_end)
        for k, (c, sp) in enumerate(zip(self.cut_checks, self.cut_pos)):
            c.blockSignals(True)
            c.setChecked(bool(sc.cut_options & (1 << k)))
            c.blockSignals(False)
            sp.set_silently(sc.cut_pos[k])
        for sp, v in zip(self.rot4d, sc.rot_4d):
            sp.set_silently(math.degrees(v))
        self._loading = False

    def _rot4d_changed(self, _v: float) -> None:
        if self._loading:
            return
        self.model.scene.rot_4d = tuple(math.radians(sp.value()) for sp in self.rot4d)  # type: ignore[assignment]
        self.model.touch("formulas")

    def _on_changed(self, kind: str) -> None:
        if kind == "camera":  # navigation changed the camera: show the values
            self.load()

    # --------------------------------------------------------------- edits
    def _cam_set(self, attr: str, value: float) -> None:
        if self._loading:
            return
        self.model.snapshot()
        setattr(self.model.scene.camera, attr, value)
        self.model.touch("camera")

    def _mid_changed(self, _v: float) -> None:
        if self._loading:
            return
        self.model.snapshot()
        self.model.scene.camera.mid = np.array([sp.value() for sp in self.mid])
        self.model.touch("camera")

    def _angles_changed(self, _v: float) -> None:
        if self._loading:
            return
        self.model.snapshot()
        self.model.scene.camera.set_angles_deg(*[sp.value() for sp in self.angles])
        self.model.touch("camera")

    def _cut_changed(self, _v=None) -> None:
        if self._loading:
            return
        sc = self.model.scene
        sc.cut_options = sum(1 << k for k, c in enumerate(self.cut_checks) if c.isChecked())
        sc.cut_pos = tuple(sp.value() for sp in self.cut_pos)  # type: ignore[assignment]
        self.model.touch("camera")

    def _reset(self) -> None:
        self.model.snapshot()
        c = Camera()
        c.set_angles_deg(-20.0, 30.0, 0.0)
        c.z_start, c.zoom = -2.0, 1.4
        self.model.scene.camera = c
        self.load()
        self.model.touch("camera")
