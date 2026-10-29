"""The Lighting panel: lights, global light settings, colours and the
palette.  Every change here is re-shaded from the kept G-buffer, so it
is immediate."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.lights import Lighting
from ...render.shading import palette_lookup
from ..scene_model import SceneModel
from ..widgets import ColorButton, DoubleSpin, IntSpin, SliderSpin, make_scroll, numpy_to_pixmap


class LightEditor(QWidget):
    """Editor of one of the six lights."""

    def __init__(self, model: SceneModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.index = 0
        self._loading = False
        form = QFormLayout(self)
        self.selector = QComboBox()
        for i in range(6):
            self.selector.addItem(f"Light {i + 1}")
        self.enabled = QCheckBox("enabled")
        self.color = ColorButton()
        self.amplitude = SliderSpin(1.0, 0.0, 10.0, 2, 0.1)
        self.x_angle = SliderSpin(0.0, -180.0, 180.0, 1, 1.0)
        self.y_angle = SliderSpin(0.0, -180.0, 180.0, 1, 1.0)
        self.spec = QComboBox()
        for i in range(8):
            self.spec.addItem(f"^{2 << i}")
        self.diff = QComboBox()
        for name in ("standard", "squared", "half", "half squared"):
            self.diff.addItem(name)
        self.relative = QCheckBox("relative to the object (world space)")
        self.shadows = QCheckBox("casts shadows")
        self.visible = QComboBox()
        for name in ("invisible", "large glow", "medium glow", "small glow"):
            self.visible.addItem(name)
        self.positional = QCheckBox("positional light at")
        self.position = [DoubleSpin(0.0, -1e9, 1e9, 4, 0.1) for _ in range(3)]
        prow = QHBoxLayout()
        prow.addWidget(self.positional)
        for sp in self.position:
            prow.addWidget(sp)
        row = QHBoxLayout()
        row.addWidget(self.enabled)
        row.addWidget(QLabel("colour"))
        row.addWidget(self.color)
        row.addStretch(1)
        form.addRow(self.selector)
        form.addRow(row)
        form.addRow("amplitude", self.amplitude)
        form.addRow("horizontal angle", self.x_angle)
        form.addRow("vertical angle", self.y_angle)
        form.addRow("specular", self.spec)
        form.addRow("diffuse function", self.diff)
        form.addRow(self.relative)
        form.addRow(self.shadows)
        form.addRow("visible light source", self.visible)
        form.addRow(prow)
        self.visible.currentIndexChanged.connect(lambda i: self._set("visible", int(i)))
        self.positional.toggled.connect(lambda v: self._set("positional", bool(v), "render"))
        for sp in self.position:
            sp.valueChanged.connect(self._position_changed)
        self.selector.currentIndexChanged.connect(self._select)
        self.enabled.toggled.connect(lambda v: self._set("enabled", bool(v)))
        self.color.colorChanged.connect(lambda c: self._set("color", tuple(c)))
        self.amplitude.valueChanged.connect(lambda v: self._set("amplitude", float(v)))
        self.x_angle.valueChanged.connect(lambda v: self._set("x_angle", math.radians(v)))
        self.y_angle.valueChanged.connect(lambda v: self._set("y_angle", math.radians(v)))
        self.spec.currentIndexChanged.connect(lambda i: self._set("spec_func", int(i)))
        self.diff.currentIndexChanged.connect(lambda i: self._set("diff_func", int(i)))
        self.relative.toggled.connect(lambda v: self._set("relative_to_object", bool(v)))
        self.shadows.toggled.connect(lambda v: self._set("shadows", bool(v), "render"))

    def light(self):
        return self.model.scene.lighting.lights[self.index]

    def load(self) -> None:
        self._loading = True
        li = self.light()
        self.enabled.setChecked(li.enabled)
        self.color.set_rgb(li.color)
        self.amplitude.set_silently(li.amplitude)
        self.x_angle.set_silently(math.degrees(li.x_angle))
        self.y_angle.set_silently(math.degrees(li.y_angle))
        self.spec.setCurrentIndex(li.spec_func & 7)
        self.diff.setCurrentIndex(li.diff_func & 3)
        self.relative.setChecked(li.relative_to_object)
        self.shadows.setChecked(li.shadows)
        self.positional.setChecked(li.positional)
        self.visible.setCurrentIndex(min(3, max(0, li.visible)))
        for sp, v in zip(self.position, li.position):
            sp.set_silently(float(v))
        self._loading = False

    def _position_changed(self, _v: float) -> None:
        if self._loading:
            return
        self.light().position = tuple(sp.value() for sp in self.position)  # type: ignore[assignment]
        self.model.touch("render" if (self.model.scene.shadows and self.light().positional) else "lighting")

    def _select(self, i: int) -> None:
        self.index = i
        self.load()

    def _set(self, attr: str, value, kind: str = "lighting") -> None:
        if self._loading:
            return
        setattr(self.light(), attr, value)
        # shadows are marched in the kernel, everything else is shading
        if attr == "enabled" and self.model.scene.shadows:
            kind = "render"
        self.model.touch(kind)


class GlobalLightEditor(QWidget):
    SLIDERS = [
        ("diffuse", "diffuse", 0, 300, "50 = 1.0"),
        ("specular", "specular", 0, 300, "50 = 1.0"),
        ("ambient", "ambient", 0, 300, "90 = 1.0"),
        ("amb. occlusion", "amb_shadow", 0, 160, "53 = 1.0"),
        ("colour start", "color_start", 0, 120, ""),
        ("colour end", "color_end", 0, 120, ""),
        ("depth fog", "depth_fog", 0, 200, "0 = none"),
        ("fog offset", "fog_offset", 0, 255, "128 = 0"),
        ("dynamic fog", "dyn_fog", 0, 160, "53 = off"),
        ("gamma", "gamma", 0, 63, "32 = neutral"),
        ("volumetric light (0 = off, 1..6 = light)", "vol_light", 0, 6, "needs a new march"),
        ("volumetric strength", "vol_light_strength", 0, 10, ""),
        ("reflections (0 = off)", "reflection", 0, 1, "needs a new march; amount of the mirrored picture"),
    ]
    COLORS = [
        ("ambient top", "ambient_top"), ("ambient bottom", "ambient_bottom"),
        ("background top", "depth_color_top"), ("background bottom", "depth_color_bottom"),
        ("dynamic fog", "dyn_fog_color"),
    ]
    FLAGS = [
        ("colour cycling", "color_cycling"), ("colour on orbit trap", "color_on_otrap"),
        ("far fog", "far_fog"), ("no colour interpolation", "no_color_interpolation"),
    ]

    def __init__(self, model: SceneModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self._loading = False
        form = QFormLayout(self)
        self.sliders: dict[str, SliderSpin] = {}
        for label, attr, lo, hi, tip in self.SLIDERS:
            s = SliderSpin(0.0, lo, hi, 2 if attr == "reflection" else 0, 0.05 if attr == "reflection" else 1.0)
            s.setToolTip(tip)
            s.valueChanged.connect(lambda v, a=attr: self._set(a, float(v)))
            form.addRow(label, s)
            self.sliders[attr] = s
        self.colors: dict[str, ColorButton] = {}
        grid = QGridLayout()
        for k, (label, attr) in enumerate(self.COLORS):
            b = ColorButton()
            b.colorChanged.connect(lambda c, a=attr: self._set(a, tuple(c)))
            grid.addWidget(QLabel(label), k // 2, (k % 2) * 2)
            grid.addWidget(b, k // 2, (k % 2) * 2 + 1)
            self.colors[attr] = b
        form.addRow(grid)
        self.flags: dict[str, QCheckBox] = {}
        for label, attr in self.FLAGS:
            c = QCheckBox(label)
            c.toggled.connect(lambda v, a=attr: self._set(a, bool(v)))
            form.addRow(c)
            self.flags[attr] = c

    def load(self) -> None:
        self._loading = True
        l = self.model.scene.lighting
        for attr, s in self.sliders.items():
            s.set_silently(float(getattr(l, attr)))
        for attr, b in self.colors.items():
            b.set_rgb(getattr(l, attr))
        for attr, c in self.flags.items():
            c.setChecked(bool(getattr(l, attr)))
        self._loading = False

    def _set(self, attr: str, value) -> None:
        if self._loading:
            return
        if attr == "vol_light":
            value = int(value)
        old = getattr(self.model.scene.lighting, attr)
        setattr(self.model.scene.lighting, attr, value)
        needs_march = attr == "vol_light" or (attr == "reflection" and ((old > 0) != (value > 0)))
        self.model.touch("render" if needs_march else "lighting")


class PaletteEditor(QWidget):
    """The ten palette stops with a gradient preview."""

    def __init__(self, model: SceneModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self._loading = False
        lay = QVBoxLayout(self)
        self.preview = QLabel()
        self.preview.setFixedHeight(24)
        self.preview.setScaledContents(True)
        lay.addWidget(self.preview)
        grid = QGridLayout()
        grid.addWidget(QLabel("position"), 0, 1)
        grid.addWidget(QLabel("diffuse"), 0, 2)
        grid.addWidget(QLabel("specular"), 0, 3)
        self.rows = []
        for i in range(10):
            pos = IntSpin(0, 0, 32767)
            dif = ColorButton()
            spe = ColorButton()
            pos.valueChanged.connect(lambda v, k=i: self._stop_changed(k))
            dif.colorChanged.connect(lambda c, k=i: self._stop_changed(k))
            spe.colorChanged.connect(lambda c, k=i: self._stop_changed(k))
            grid.addWidget(QLabel(str(i + 1)), i + 1, 0)
            grid.addWidget(pos, i + 1, 1)
            grid.addWidget(dif, i + 1, 2)
            grid.addWidget(spe, i + 1, 3)
            self.rows.append((pos, dif, spe))
        lay.addLayout(grid)
        tools = QHBoxLayout()
        for text, fn in (("Random", self._random), ("From image...", self._from_image), ("Reverse", self._reverse),
                         ("Spread evenly", self._spread)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            tools.addWidget(b)
        lay.addLayout(tools)
        lay.addStretch(1)

    def _apply_stops(self, stops) -> None:
        self.model.snapshot()
        pal = self.model.scene.lighting.palette
        for st, (pos, rgb) in zip(pal, stops):
            st.position = int(pos)
            st.diffuse = tuple(float(v) for v in rgb)
        self.load()
        self.model.touch("lighting")

    def _random(self) -> None:
        from ...core.mutagen import random_palette

        self._apply_stops(random_palette())

    def _from_image(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        from ...render.maps import load_map

        path, _ = QFileDialog.getOpenFileName(self, "Gradient image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = load_map(path)
        h, w = img.shape[:2]
        row = img[h // 2]
        stops = [(int(i * 32767 / 9), tuple(row[min(w - 1, int(i * (w - 1) / 9))])) for i in range(10)]
        self._apply_stops(stops)

    def _reverse(self) -> None:
        pal = self.model.scene.lighting.palette
        cols = [st.diffuse for st in pal][::-1]
        self._apply_stops([(st.position, c) for st, c in zip(pal, cols)])

    def _spread(self) -> None:
        pal = self.model.scene.lighting.palette
        self._apply_stops([(int(i * 32767 / 9), st.diffuse) for i, st in enumerate(pal)])

    def load(self) -> None:
        self._loading = True
        for (pos, dif, spe), stop in zip(self.rows, self.model.scene.lighting.palette):
            pos.set_silently(stop.position)
            dif.set_rgb(stop.diffuse)
            spe.set_rgb(stop.specular)
        self._loading = False
        self._draw()

    def _draw(self) -> None:
        l: Lighting = self.model.scene.lighting
        idx = np.linspace(0, 32767, 256)
        dif, _ = palette_lookup(idx, l)
        rgb = np.clip(dif * 255, 0, 255).astype(np.uint8)[None, :, :]
        self.preview.setPixmap(numpy_to_pixmap(np.repeat(rgb, 8, axis=0)))

    def _stop_changed(self, k: int) -> None:
        if self._loading:
            return
        pos, dif, spe = self.rows[k]
        stop = self.model.scene.lighting.palette[k]
        stop.position = int(pos.value())
        stop.diffuse = dif.rgb()
        stop.specular = spe.rgb()
        self._draw()
        self.model.touch("lighting")


class LightPanel(QWidget):
    def __init__(self, model: SceneModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        tabs = QTabWidget()
        self.light_editor = LightEditor(model)
        self.global_editor = GlobalLightEditor(model)
        self.palette_editor = PaletteEditor(model)
        for w, name in ((self.light_editor, "Lights"), (self.global_editor, "Global"), (self.palette_editor, "Palette")):
            tabs.addTab(make_scroll(w), name)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(tabs)
        model.replaced.connect(self.load)
        self.load()

    def load(self) -> None:
        self.light_editor.load()
        self.global_editor.load()
        self.palette_editor.load()
