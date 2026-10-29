"""Camera: position, rotation, zoom, field of view, depth range."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QVBoxLayout, QWidget

from mb3d.params.scene import CameraOptic

from ..model import SceneModel
from ..widgets import Binder, attr_getter, attr_setter, tuple_setter


class CameraPanel(QWidget):
    def __init__(self, model: SceneModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.b = Binder(model, "camera")
        layout = QVBoxLayout(self)
        pos = QGroupBox("Position (scene middle)")
        f = QFormLayout(pos)
        for i, n in enumerate("XYZ"):
            self.b.float(f, f"{n} mid", lambda s, i=i: s.mid[i], tuple_setter("mid", i), step=0.01, decimals=6)
        layout.addWidget(pos)
        rot = QGroupBox("Rotation (degrees)")
        f = QFormLayout(rot)
        for i, n in enumerate("XYZ"):
            self.b.float(f, f"{n} angle", lambda s, i=i: self.model.angles()[i], self._angle_setter(i), -360, 360, 1.0, 2)
        layout.addWidget(rot)
        view = QGroupBox("View")
        f = QFormLayout(view)
        self.b.float(f, "Zoom", attr_getter("zoom"), attr_setter("zoom"), 1e-12, 1e12, 0.1, 6)
        self.b.float(f, "FOV y (°)", attr_getter("fov_y"), attr_setter("fov_y"), 0.1, 179.0, 1.0, 2)
        self.b.combo(f, "Optic", ["Common", "Planar", "Panorama"], lambda s: int(s.optic),
                     lambda s, v: setattr(s, "optic", CameraOptic(v)))
        self.b.float(f, "Z start", attr_getter("z_start"), attr_setter("z_start"), -1e6, 1e6, 0.1, 4)
        self.b.float(f, "Z end", attr_getter("z_end"), attr_setter("z_end"), -1e6, 1e6, 0.1, 4)
        layout.addWidget(view)
        layout.addStretch(1)
        self.b.refresh()

    def _angle_setter(self, index: int):
        def set_(s, v):
            a = list(self.model.angles())
            a[index] = v
            self.model.scene.vgrads = self._vgrads(*a)

        return set_

    @staticmethod
    def _vgrads(ax, ay, az):
        import math

        from mb3d.params.scene import vgrads_from_angles

        return vgrads_from_angles(math.radians(ax), math.radians(ay), math.radians(az))
