"""Render: image size, preview / final render devices and quality."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mb3d.render import RenderSettings

from ..model import SceneModel

BACKENDS = [("Auto", "auto"), ("OpenGL (GLSL preview)", "opengl"), ("OpenCL", "opencl"), ("CPU (numba)", "cpu")]


class RenderPanel(QWidget):
    render_final = Signal()
    render_preview = Signal()
    cancel = Signal()
    settings_changed = Signal()
    save_image = Signal()

    def __init__(self, model: SceneModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._refreshing = False
        layout = QVBoxLayout(self)
        g = QGroupBox("Image size")
        f = QFormLayout(g)
        self.width = QSpinBox()
        self.width.setRange(16, 30000)
        self.height = QSpinBox()
        self.height.setRange(16, 30000)
        for w in (self.width, self.height):
            w.setKeyboardTracking(False)
            w.valueChanged.connect(self._size_changed)
        f.addRow("Width", self.width)
        f.addRow("Height", self.height)
        layout.addWidget(g)

        g = QGroupBox("Preview")
        f = QFormLayout(g)
        self.preview_backend = QComboBox()
        for label, key in BACKENDS:
            self.preview_backend.addItem(label, key)
        self.preview_backend.currentIndexChanged.connect(lambda _: self.settings_changed.emit())
        f.addRow("Device", self.preview_backend)
        self.auto_preview = QCheckBox("Render automatically on changes")
        self.auto_preview.setChecked(True)
        f.addRow(self.auto_preview)
        self.preview_ao = QSpinBox()
        self.preview_ao.setRange(0, 16)
        self.preview_ao.setValue(4)
        self.preview_ao.valueChanged.connect(lambda _: self.settings_changed.emit())
        f.addRow("Ambient occlusion samples", self.preview_ao)
        self.preview_shadow = QCheckBox("Soft shadow of light 1")
        self.preview_shadow.setChecked(True)
        self.preview_shadow.toggled.connect(lambda _: self.settings_changed.emit())
        f.addRow(self.preview_shadow)
        row = QHBoxLayout()
        self.preview_button = QPushButton("Preview now")
        self.preview_button.clicked.connect(self.render_preview.emit)
        row.addWidget(self.preview_button)
        layout.addWidget(g)

        g = QGroupBox("Final render (precise)")
        f = QFormLayout(g)
        self.final_backend = QComboBox()
        for label, key in BACKENDS:
            if key != "opengl":
                self.final_backend.addItem(label, key)
        f.addRow("Device", self.final_backend)
        self.final_ao = QSpinBox()
        self.final_ao.setRange(0, 32)
        self.final_ao.setValue(8)
        f.addRow("Ambient occlusion samples", self.final_ao)
        self.final_shadow = QCheckBox("Soft shadow of light 1")
        self.final_shadow.setChecked(True)
        f.addRow(self.final_shadow)
        self.final_button = QPushButton("Render final image")
        self.final_button.clicked.connect(self.render_final.emit)
        row.addWidget(self.final_button)
        layout.addWidget(g)
        layout.addLayout(row)
        row2 = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel.emit)
        self.save_button = QPushButton("Save image…")
        self.save_button.clicked.connect(self.save_image.emit)
        row2.addWidget(self.cancel_button)
        row2.addWidget(self.save_button)
        layout.addLayout(row2)
        self.device_label = QLabel("devices: probing…")
        self.device_label.setWordWrap(True)
        layout.addWidget(self.device_label)
        layout.addStretch(1)
        model.changed.connect(self._on_changed)
        self.refresh()

    def _size_changed(self, _):
        if self._refreshing:
            return
        self.model.set_size(self.width.value(), self.height.value())

    def _on_changed(self, what: str):
        if what in ("size", "scene"):
            self.refresh()

    def refresh(self):
        self._refreshing = True
        try:
            self.width.setValue(self.model.scene.width)
            self.height.setValue(self.model.scene.height)
        finally:
            self._refreshing = False

    def preview_settings(self) -> RenderSettings:
        return RenderSettings(backend=self.preview_backend.currentData(), precise=False,
                              ao_samples=self.preview_ao.value(), shadow=self.preview_shadow.isChecked(),
                              progressive_scales=(0.25, 0.5, 1.0))

    def final_settings(self) -> RenderSettings:
        return RenderSettings(backend=self.final_backend.currentData(), precise=True,
                              ao_samples=self.final_ao.value(), shadow=self.final_shadow.isChecked(),
                              progressive_scales=(1.0,))
