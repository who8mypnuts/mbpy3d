"""The Maps & Post panel: background picture, light map, diffuse map and
the post processing (depth of field, colour adjustments, sharpening)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..scene_model import SceneModel
from ..widgets import DoubleSpin, SliderSpin

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp)"


class MapField(QWidget):
    """A file name with a browse button."""

    def __init__(self, on_change, parent=None) -> None:
        super().__init__(parent)
        self.edit = QLineEdit()
        browse = QPushButton("...")
        browse.setFixedWidth(28)
        clear = QPushButton("x")
        clear.setFixedWidth(24)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.edit, 1)
        lay.addWidget(browse)
        lay.addWidget(clear)
        self.edit.editingFinished.connect(lambda: on_change(self.edit.text()))
        browse.clicked.connect(lambda: self._browse(on_change))
        clear.clicked.connect(lambda: (self.edit.setText(""), on_change("")))

    def _browse(self, on_change) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Image", "", IMAGE_FILTER)
        if path:
            self.edit.setText(path)
            on_change(path)

    def set_silently(self, text: str) -> None:
        self.edit.blockSignals(True)
        self.edit.setText(text)
        self.edit.blockSignals(False)


class PostPanel(QWidget):
    def __init__(self, model: SceneModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self._loading = False
        inner = QWidget()
        lay = QVBoxLayout(inner)

        bg = QGroupBox("Background picture")
        bf = QFormLayout(bg)
        self.bg_image = MapField(lambda t: self._lset("bg_image", t))
        self.bg_direct = QCheckBox("in image coordinates (instead of a sphere around the viewer)")
        self.bg_ambient = QCheckBox("a blurred copy is the ambient light")
        self.bg_brightness = SliderSpin(1.0, 0.0, 5.0, 2, 0.1)
        self.bg_rot = [DoubleSpin(0.0, -360, 360, 1, 5.0) for _ in range(3)]
        rot = QHBoxLayout()
        for sp in self.bg_rot:
            rot.addWidget(sp)
            sp.valueChanged.connect(self._bg_rot_changed)
        bf.addRow("file", self.bg_image)
        bf.addRow("brightness", self.bg_brightness)
        bf.addRow("rotation x y z", rot)
        bf.addRow(self.bg_direct)
        bf.addRow(self.bg_ambient)
        lay.addWidget(bg)

        lm = QGroupBox("Light map (lights the object from all directions)")
        lf = QFormLayout(lm)
        self.light_map = MapField(lambda t: self._lset("light_map", t))
        self.light_map_intensity = SliderSpin(1.0, 0.0, 5.0, 2, 0.1)
        lf.addRow("file", self.light_map)
        lf.addRow("intensity", self.light_map_intensity)
        lay.addWidget(lm)

        dm = QGroupBox("Diffuse colour map")
        df = QFormLayout(dm)
        self.diffuse_map = MapField(lambda t: self._lset("diffuse_map", t))
        self.diffuse_mode = QComboBox()
        for label, key in (("on the normals", "normals"), ("iterations + orbit trap", "iter_otrap"), ("wrapped around the object", "wrap")):
            self.diffuse_mode.addItem(label, key)
        self.diffuse_scale = SliderSpin(1.0, 0.01, 20.0, 2, 0.1)
        self.diffuse_combine = QCheckBox("combine the map's brightness with the palette colour")
        df.addRow("file", self.diffuse_map)
        df.addRow("mapping", self.diffuse_mode)
        df.addRow("scale", self.diffuse_scale)
        df.addRow(self.diffuse_combine)
        lay.addWidget(dm)

        rough = QGroupBox("Surface")
        rf = QFormLayout(rough)
        self.roughness = SliderSpin(0.0, 0.0, 255.0, 0, 5.0)
        self.roughness.setToolTip("How much the roughness of smooth normals softens the lights (needs smooth normals > 0)")
        rf.addRow("roughness", self.roughness)
        lay.addWidget(rough)

        post = QGroupBox("Post processing")
        pf = QFormLayout(post)
        self.brightness = SliderSpin(0.0, -1.0, 1.0, 2, 0.05)
        self.contrast = SliderSpin(0.0, -1.0, 1.0, 2, 0.05)
        self.saturation = SliderSpin(0.0, -1.0, 1.0, 2, 0.05)
        self.sharpen = SliderSpin(0.0, 0.0, 2.0, 2, 0.1)
        pf.addRow("brightness", self.brightness)
        pf.addRow("contrast", self.contrast)
        pf.addRow("saturation", self.saturation)
        pf.addRow("sharpen", self.sharpen)
        lay.addWidget(post)

        dof = QGroupBox("Depth of field (Shift + double click sets the focus)")
        ff = QFormLayout(dof)
        self.dof = QCheckBox("enabled")
        self.dof_focus = SliderSpin(0.3, 0.0, 1.0, 3, 0.01)
        self.dof_aperture = SliderSpin(8.0, 0.0, 100.0, 1, 1.0)
        self.dof_forward = QCheckBox("blur only behind the focus")
        ff.addRow(self.dof)
        ff.addRow("focus (0 = start plane, 1 = far end)", self.dof_focus)
        ff.addRow("aperture (pixels)", self.dof_aperture)
        ff.addRow(self.dof_forward)
        lay.addWidget(dof)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.bg_direct.toggled.connect(lambda v: self._lset("bg_direct", bool(v)))
        self.bg_ambient.toggled.connect(lambda v: self._lset("bg_ambient", bool(v)))
        self.bg_brightness.valueChanged.connect(lambda v: self._lset("bg_brightness", float(v)))
        self.light_map_intensity.valueChanged.connect(lambda v: self._lset("light_map_intensity", float(v)))
        self.diffuse_mode.currentIndexChanged.connect(lambda i: self._lset("diffuse_map_mode", self.diffuse_mode.currentData()))
        self.diffuse_scale.valueChanged.connect(lambda v: self._lset("diffuse_map_scale", float(v)))
        self.diffuse_combine.toggled.connect(lambda v: self._lset("diffuse_map_combine", bool(v)))
        self.roughness.valueChanged.connect(lambda v: self._lset("roughness", float(v)))
        for attr, w in (("brightness", self.brightness), ("contrast", self.contrast), ("saturation", self.saturation),
                        ("sharpen", self.sharpen), ("dof_focus", self.dof_focus), ("dof_aperture", self.dof_aperture)):
            w.valueChanged.connect(lambda v, a=attr: self._pset(a, float(v)))
        self.dof.toggled.connect(lambda v: self._pset("dof", bool(v)))
        self.dof_forward.toggled.connect(lambda v: self._pset("dof_forward_only", bool(v)))
        model.replaced.connect(self.load)
        self.load()

    def load(self) -> None:
        self._loading = True
        l = self.model.scene.lighting
        p = self.model.scene.post
        self.bg_image.set_silently(l.bg_image)
        self.bg_direct.setChecked(l.bg_direct)
        self.bg_ambient.setChecked(l.bg_ambient)
        self.bg_brightness.set_silently(l.bg_brightness)
        for sp, v in zip(self.bg_rot, l.bg_rotation):
            sp.set_silently(float(v))
        self.light_map.set_silently(l.light_map)
        self.light_map_intensity.set_silently(l.light_map_intensity)
        self.diffuse_map.set_silently(l.diffuse_map)
        self.diffuse_mode.setCurrentIndex(max(0, self.diffuse_mode.findData(l.diffuse_map_mode)))
        self.diffuse_scale.set_silently(l.diffuse_map_scale)
        self.diffuse_combine.setChecked(l.diffuse_map_combine)
        self.roughness.set_silently(l.roughness)
        self.brightness.set_silently(p.brightness)
        self.contrast.set_silently(p.contrast)
        self.saturation.set_silently(p.saturation)
        self.sharpen.set_silently(p.sharpen)
        self.dof.setChecked(p.dof)
        self.dof_focus.set_silently(p.dof_focus)
        self.dof_aperture.set_silently(p.dof_aperture)
        self.dof_forward.setChecked(p.dof_forward_only)
        self._loading = False

    def _lset(self, attr: str, value) -> None:
        if self._loading:
            return
        setattr(self.model.scene.lighting, attr, value)
        self.model.touch("lighting")

    def _pset(self, attr: str, value) -> None:
        if self._loading:
            return
        setattr(self.model.scene.post, attr, value)
        self.model.touch("lighting")

    def _bg_rot_changed(self, _v: float) -> None:
        if self._loading:
            return
        self.model.scene.lighting.bg_rotation = tuple(sp.value() for sp in self.bg_rot)  # type: ignore[assignment]
        self.model.touch("lighting")
