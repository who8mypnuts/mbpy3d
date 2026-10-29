"""MB3D's image size controls: width, height, aspect presets, keep aspect,
half / double buttons and the big "Calculate 3D" button, as a toolbar."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QToolBar, QWidget

from .scene_model import SceneModel
from .widgets import IntSpin

ASPECTS = [("user", None), ("4:3", 4 / 3), ("3:2", 3 / 2), ("16:9", 16 / 9), ("16:10", 16 / 10), ("1:1", 1.0),
           ("2:1", 2.0), ("3:4", 3 / 4), ("9:16", 9 / 16)]


class ImageSizeBar(QToolBar):
    render_requested = Signal()

    def __init__(self, model: SceneModel, parent=None) -> None:
        super().__init__("Image", parent)
        self.setObjectName("image_size_bar")
        self.model = model
        self._loading = False
        self.setMovable(False)
        self.calc = QPushButton("Calculate 3D")
        self.calc.setToolTip("Render the image at its full size on the final device (Ctrl+R)")
        self.calc.setStyleSheet("font-weight: bold; padding: 3px 14px;")
        self.calc.clicked.connect(self.render_requested)
        self.width = IntSpin(640, 8, 32767)
        self.height = IntSpin(480, 8, 32767)
        self.aspect = QComboBox()
        for name, _ in ASPECTS:
            self.aspect.addItem(name)
        self.keep = QCheckBox("keep aspect")
        self.keep.setChecked(True)
        half = QPushButton("½")
        double = QPushButton("×2")
        half.setFixedWidth(30)
        double.setFixedWidth(30)
        self.info = QLabel("")
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.addWidget(self.calc)
        lay.addSpacing(12)
        lay.addWidget(QLabel("Image"))
        lay.addWidget(self.width)
        lay.addWidget(QLabel("x"))
        lay.addWidget(self.height)
        lay.addWidget(self.aspect)
        lay.addWidget(self.keep)
        lay.addWidget(half)
        lay.addWidget(double)
        lay.addWidget(self.info)
        lay.addStretch(1)
        self.addWidget(w)
        self.width.valueChanged.connect(self._width_changed)
        self.height.valueChanged.connect(self._height_changed)
        self.aspect.currentIndexChanged.connect(self._aspect_changed)
        half.clicked.connect(lambda: self._scale(0.5))
        double.clicked.connect(lambda: self._scale(2.0))
        model.replaced.connect(self.load)
        model.changed.connect(lambda kind: self.load() if kind == "size" else None)
        self.load()

    def load(self) -> None:
        self._loading = True
        sc = self.model.scene
        self.width.set_silently(sc.width)
        self.height.set_silently(sc.height)
        self.info.setText(f"{sc.width * sc.height / 1e6:.2f} Mpixel")
        self._loading = False

    def _ratio(self) -> float | None:
        r = ASPECTS[self.aspect.currentIndex()][1]
        if r is None and self.keep.isChecked():
            sc = self.model.scene
            return sc.width / max(1, sc.height)
        return r

    def _set(self, w: int, h: int) -> None:
        self.model.snapshot()
        self.model.scene.width, self.model.scene.height = max(8, int(w)), max(8, int(h))
        self.model.touch("size")

    def _width_changed(self, v: int) -> None:
        if self._loading:
            return
        r = self._ratio()
        self._set(v, round(v / r) if r else self.model.scene.height)

    def _height_changed(self, v: int) -> None:
        if self._loading:
            return
        r = self._ratio()
        self._set(round(v * r) if r else self.model.scene.width, v)

    def _aspect_changed(self, _i: int) -> None:
        if self._loading:
            return
        r = self._ratio()
        if r:
            self._set(self.model.scene.width, round(self.model.scene.width / r))

    def _scale(self, f: float) -> None:
        sc = self.model.scene
        self._set(round(sc.width * f), round(sc.height * f))
