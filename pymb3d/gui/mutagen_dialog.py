"""The MutaGen window: a grid of random variations of the scene, rendered
as thumbnails one after another in a worker; click one to take it over (and
mutate again from it)."""
from __future__ import annotations

import random

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..backends.detect import DeviceChoice
from ..core.mutagen import mutate
from ..core.scene import Scene
from ..render import RenderOptions
from .scene_model import SceneModel
from .widgets import SliderSpin, numpy_to_pixmap
from .workers.render_service import RenderService

GRID = 3
THUMB = 160


class Thumb(QLabel):
    def __init__(self, index: int, on_click, parent=None) -> None:
        super().__init__(parent)
        self.index = index
        self.on_click = on_click
        self.setFixedSize(THUMB, THUMB * 3 // 4)
        self.setStyleSheet("background: #222; border: 1px solid #555;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("...")

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self.on_click(self.index)


class MutagenDialog(QDialog):
    def __init__(self, model: SceneModel, choice: DeviceChoice, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MutaGen - random variations")
        self.model = model
        self.base: Scene = model.scene.copy()
        self.scenes: list[Scene] = []
        self.rng = random.Random()
        self.service = RenderService(choice, RenderOptions(live_bands=False), "mutagen")
        self.service.finished.connect(self._on_finished)
        self.service.failed.connect(lambda m: self._next())
        self._queue: list[int] = []
        self._current: int | None = None

        self.strength = SliderSpin(0.25, 0.02, 1.0, 2, 0.05)
        self.f_formulas = QCheckBox("formulas")
        self.f_lights = QCheckBox("lights")
        self.f_palette = QCheckBox("palette")
        for c in (self.f_formulas, self.f_lights, self.f_palette):
            c.setChecked(True)
        again = QPushButton("Mutate again")
        again.clicked.connect(self.generate)
        top = QHBoxLayout()
        top.addWidget(QLabel("strength"))
        top.addWidget(self.strength, 1)
        top.addWidget(self.f_formulas)
        top.addWidget(self.f_lights)
        top.addWidget(self.f_palette)
        top.addWidget(again)
        grid = QGridLayout()
        self.thumbs = []
        for i in range(GRID * GRID):
            t = Thumb(i, self._pick)
            grid.addWidget(t, i // GRID, i % GRID)
            self.thumbs.append(t)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(grid)
        lay.addWidget(QLabel("The middle is the current scene. Click a variation to use it in the main window."))
        self.generate()

    def generate(self) -> None:
        what = tuple(n for n, c in (("formulas", self.f_formulas), ("lights", self.f_lights), ("palette", self.f_palette))
                     if c.isChecked())
        self.scenes = []
        center = GRID * GRID // 2
        for i in range(GRID * GRID):
            sc = self.base.copy() if i == center else mutate(self.base, self.strength.value(), what, self.rng)
            sc.width, sc.height = THUMB, THUMB * 3 // 4
            sc.shadows = False
            sc.ao.mode = "none"
            self.scenes.append(sc)
            self.thumbs[i].setText("...")
            self.thumbs[i].setPixmap(numpy_to_pixmap(__import__("numpy").zeros((THUMB * 3 // 4, THUMB, 3), dtype="uint8")))
        self._queue = list(range(GRID * GRID))
        self._current = None
        self._next()

    def _next(self) -> None:
        if not self._queue:
            self._current = None
            return
        self._current = self._queue.pop(0)
        self.service.submit(self.scenes[self._current])

    @Slot(object)
    def _on_finished(self, result) -> None:
        if self._current is not None:
            self.thumbs[self._current].setPixmap(numpy_to_pixmap(result.image))
        self._next()

    def _pick(self, index: int) -> None:
        if index >= len(self.scenes):
            return
        chosen = self.scenes[index].copy()
        chosen.width, chosen.height = self.model.scene.width, self.model.scene.height
        chosen.shadows, chosen.ao = self.model.scene.shadows, self.model.scene.ao
        self.model.replace(chosen, self.model.path)
        self.base = chosen.copy()
        self.generate()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.service.shutdown()
        super().closeEvent(event)
