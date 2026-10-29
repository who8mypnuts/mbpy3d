"""The Render panel: image size, ray marching quality, ambient occlusion,
shadows and the devices used for the preview and the final render."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ...backends.detect import DeviceChoice
from ..scene_model import SceneModel
from ..widgets import DoubleSpin, IntSpin


class RenderPanel(QWidget):
    #: the preview device changed: DeviceChoice
    preview_device_changed = Signal(object)
    preview_width_changed = Signal(int)

    def __init__(self, model: SceneModel, devices: list[DeviceChoice], preview_choice: DeviceChoice,
                 final_choice: DeviceChoice, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.devices = devices
        self._loading = False
        lay = QVBoxLayout(self)

        size = QGroupBox("Image")
        sf = QFormLayout(size)
        self.width = IntSpin(640, 8, 32767)
        self.height = IntSpin(480, 8, 32767)
        r = QHBoxLayout()
        r.addWidget(self.width)
        r.addWidget(QLabel("x"))
        r.addWidget(self.height)
        sf.addRow("final size", r)
        self.preview_width = IntSpin(480, 64, 2048)
        self.preview_width.setToolTip("Width of the interactive preview (height follows the aspect ratio)")
        sf.addRow("preview width", self.preview_width)
        lay.addWidget(size)

        q = QGroupBox("Ray marching")
        qf = QFormLayout(q)
        self.de_stop = DoubleSpin(1.0, 0.001, 100.0, 4, 0.1)
        self.de_stop.setToolTip("DE stop in pixels: smaller = finer detail, slower")
        self.z_step_div = DoubleSpin(0.5, 0.001, 1.0, 3, 0.05)
        self.z_step_div.setToolTip("Raystep multiplier: fraction of the distance estimate stepped per ray step")
        self.raystep_limiter = DoubleSpin(1.0, 0.01, 100.0, 3, 0.1)
        self.bin_search = IntSpin(6, 0, 30)
        self.vary_de = QCheckBox("DE stop grows with the distance (perspective)")
        self.first_random = QCheckBox("random first step (hides banding)")
        qf.addRow("DE stop", self.de_stop)
        qf.addRow("raystep multiplier", self.z_step_div)
        qf.addRow("raystep limiter", self.raystep_limiter)
        qf.addRow("binary search steps", self.bin_search)
        qf.addRow(self.vary_de)
        qf.addRow(self.first_random)
        lay.addWidget(q)

        ao = QGroupBox("Ambient occlusion and shadows")
        af = QFormLayout(ao)
        self.ao_mode = QComboBox()
        self.ao_mode.addItem("none", "none")
        self.ao_mode.addItem("screen space (after the march)", "ssao")
        self.ao_mode.addItem("distance estimation (in the kernel)", "de")
        self.ao_strength = DoubleSpin(1.0, 0.0, 4.0, 2, 0.1)
        self.ao_radius = DoubleSpin(32.0, 1.0, 512.0, 1, 4.0)
        self.ao_samples = IntSpin(5, 1, 32)
        self.shadows = QCheckBox("hard / soft shadows of the lights")
        af.addRow("ambient occlusion", self.ao_mode)
        af.addRow("AO strength", self.ao_strength)
        af.addRow("AO radius (pixels)", self.ao_radius)
        af.addRow("DE AO samples", self.ao_samples)
        af.addRow(self.shadows)
        lay.addWidget(ao)

        dev = QGroupBox("Compute devices")
        df = QFormLayout(dev)
        self.preview_device = QComboBox()
        self.final_device = QComboBox()
        for d in devices:
            self.preview_device.addItem(d.label, d.key)
            self.final_device.addItem(d.label, d.key)
        self.preview_device.setCurrentIndex(max(0, self.preview_device.findData(preview_choice.key)))
        self.final_device.setCurrentIndex(max(0, self.final_device.findData(final_choice.key)))
        df.addRow("preview", self.preview_device)
        df.addRow("final render", self.final_device)
        lay.addWidget(dev)
        lay.addStretch(1)

        self.width.valueChanged.connect(lambda v: self._set("width", int(v), "size"))
        self.height.valueChanged.connect(lambda v: self._set("height", int(v), "size"))
        self.preview_width.valueChanged.connect(lambda v: self.preview_width_changed.emit(int(v)))
        self.de_stop.valueChanged.connect(lambda v: self._set("de_stop", float(v)))
        self.z_step_div.valueChanged.connect(lambda v: self._set("z_step_div", float(v)))
        self.raystep_limiter.valueChanged.connect(lambda v: self._set("raystep_limiter", float(v)))
        self.bin_search.valueChanged.connect(lambda v: self._set("bin_search_steps", int(v)))
        self.vary_de.toggled.connect(lambda v: self._set("vary_de_stop_on_fov", bool(v)))
        self.first_random.toggled.connect(lambda v: self._set("first_step_random", bool(v)))
        self.shadows.toggled.connect(lambda v: self._set("shadows", bool(v)))
        self.ao_mode.currentIndexChanged.connect(self._ao_changed)
        self.ao_strength.valueChanged.connect(self._ao_changed)
        self.ao_radius.valueChanged.connect(self._ao_changed)
        self.ao_samples.valueChanged.connect(self._ao_changed)
        self.preview_device.currentIndexChanged.connect(self._preview_device)
        model.replaced.connect(self.load)
        self.load()

    @property
    def final_choice(self) -> DeviceChoice:
        key = self.final_device.currentData()
        return next((d for d in self.devices if d.key == key), self.devices[0])

    def load(self) -> None:
        self._loading = True
        sc = self.model.scene
        self.width.set_silently(sc.width)
        self.height.set_silently(sc.height)
        self.de_stop.set_silently(sc.de_stop)
        self.z_step_div.set_silently(sc.z_step_div)
        self.raystep_limiter.set_silently(sc.raystep_limiter)
        self.bin_search.set_silently(sc.bin_search_steps)
        self.vary_de.setChecked(sc.vary_de_stop_on_fov)
        self.first_random.setChecked(sc.first_step_random)
        self.shadows.setChecked(sc.shadows)
        self.ao_mode.setCurrentIndex(max(0, self.ao_mode.findData(sc.ao.mode)))
        self.ao_strength.set_silently(sc.ao.strength)
        self.ao_radius.set_silently(sc.ao.radius)
        self.ao_samples.set_silently(sc.ao.samples)
        self._loading = False

    def _set(self, attr: str, value, kind: str = "render") -> None:
        if self._loading:
            return
        self.model.snapshot()
        setattr(self.model.scene, attr, value)
        self.model.touch(kind)

    def _ao_changed(self, _v=None) -> None:
        if self._loading:
            return
        ao = self.model.scene.ao
        old_mode = ao.mode
        ao.mode = self.ao_mode.currentData()
        ao.strength = self.ao_strength.value()
        ao.radius = self.ao_radius.value()
        ao.samples = self.ao_samples.value()
        # screen space AO is part of the shading: no new march needed
        self.model.touch("lighting" if (old_mode == ao.mode and ao.mode != "de") or
                         ({old_mode, ao.mode} <= {"none", "ssao"}) else "render")

    def _preview_device(self, _i: int) -> None:
        key = self.preview_device.currentData()
        for d in self.devices:
            if d.key == key:
                self.preview_device_changed.emit(d)
