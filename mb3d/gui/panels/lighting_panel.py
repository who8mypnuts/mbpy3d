"""Lighting: the six lights, ambient and fog colours, the palette."""

from __future__ import annotations

import math

from PySide6.QtWidgets import QFormLayout, QGroupBox, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from ..model import SceneModel
from ..widgets import Binder


def _light_setter(i: int, attr: str, convert=lambda v: v):
    def set_(s, v):
        setattr(s.lighting.lights[i], attr, convert(v))

    return set_


def _light_getter(i: int, attr: str, convert=lambda v: v):
    return lambda s: convert(getattr(s.lighting.lights[i], attr))


def _l_setter(attr: str):
    return lambda s, v: setattr(s.lighting, attr, v)


def _l_getter(attr: str):
    return lambda s: getattr(s.lighting, attr)


class LightsTab(QWidget):
    def __init__(self, model: SceneModel, binder: Binder, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        for i in range(6):
            w = QWidget()
            f = QFormLayout(w)
            binder.check(f, "On", _light_getter(i, "on"), _light_setter(i, "on"))
            binder.color(f, "Colour", _light_getter(i, "color"), _light_setter(i, "color"))
            binder.float(f, "Amplitude", _light_getter(i, "amplitude"), _light_setter(i, "amplitude"), 0.0, 1000.0, 0.1, 3)
            binder.float(f, "X angle (°)", _light_getter(i, "x_angle", math.degrees), _light_setter(i, "x_angle", math.radians), -360, 360, 5.0, 1)
            binder.float(f, "Y angle (°)", _light_getter(i, "y_angle", math.degrees), _light_setter(i, "y_angle", math.radians), -360, 360, 5.0, 1)
            binder.combo(f, "Specular", [f"^{2 << k}" for k in range(8)], _light_getter(i, "spec_func"), _light_setter(i, "spec_func"))
            binder.combo(f, "Diffuse function", ["Rounded", "Squared", "Half", "Half squared"], _light_getter(i, "diff_func"),
                         _light_setter(i, "diff_func"))
            tabs.addTab(w, f"Light {i + 1}")
        layout.addWidget(tabs)


class AmbientTab(QWidget):
    def __init__(self, model: SceneModel, binder: Binder, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        g = QGroupBox("Light amounts")
        f = QFormLayout(g)
        binder.float(f, "Diffuse (50 = 1)", _l_getter("diffuse"), _l_setter("diffuse"), 0, 500, 5, 0)
        binder.float(f, "Specular (50 = 1)", _l_getter("specular"), _l_setter("specular"), 0, 500, 5, 0)
        binder.float(f, "Ambient (90 = 1)", _l_getter("ambient"), _l_setter("ambient"), 0, 500, 5, 0)
        binder.float(f, "Ambient shadow (53 = 1)", _l_getter("amb_shadow"), _l_setter("amb_shadow"), 0, 255, 5, 0)
        binder.float(f, "Indirect light", _l_getter("ind_light"), _l_setter("ind_light"), 0, 180, 5, 0)
        binder.float(f, "Gamma (32 = off)", _l_getter("gamma"), _l_setter("gamma"), 0, 63, 1, 0)
        binder.check(f, "Internal gamma 2", _l_getter("internal_gamma2"), _l_setter("internal_gamma2"))
        layout.addWidget(g)
        g = QGroupBox("Ambient & background")
        f = QFormLayout(g)
        binder.color(f, "Ambient top", _l_getter("amb_top"), _l_setter("amb_top"))
        binder.color(f, "Ambient bottom", _l_getter("amb_bottom"), _l_setter("amb_bottom"))
        binder.color(f, "Background top", _l_getter("depth_col"), _l_setter("depth_col"))
        binder.color(f, "Background bottom", _l_getter("depth_col2"), _l_setter("depth_col2"))
        binder.combo(f, "Background gradient", ["Linear", "Squared", "Square root"], _l_getter("depth_func"), _l_setter("depth_func"))
        layout.addWidget(g)
        g = QGroupBox("Fog")
        f = QFormLayout(g)
        binder.float(f, "Depth fog", _l_getter("depth_fog"), _l_setter("depth_fog"), 0, 255, 5, 0)
        binder.check(f, "Far fog", _l_getter("far_fog"), _l_setter("far_fog"))
        binder.float(f, "Dynamic fog (53 = off)", _l_getter("dyn_fog"), _l_setter("dyn_fog"), 0, 200, 5, 0)
        binder.float(f, "Fog offset", _l_getter("fog_offset"), _l_setter("fog_offset"), 0, 255, 5, 0)
        binder.color(f, "Dynamic fog colour", _l_getter("dyn_fog_col"), _l_setter("dyn_fog_col"))
        binder.color(f, "Dynamic fog colour 2", _l_getter("dyn_fog_col2"), _l_setter("dyn_fog_col2"))
        layout.addWidget(g)
        layout.addStretch(1)


class PaletteTab(QWidget):
    def __init__(self, model: SceneModel, binder: Binder, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        g = QGroupBox("Colour mapping")
        f = QFormLayout(g)
        binder.float(f, "Colour start", _l_getter("color_start"), _l_setter("color_start"), 0, 120, 1, 0)
        binder.float(f, "Colour end", _l_getter("color_end"), _l_setter("color_end"), 0, 120, 1, 0)
        binder.check(f, "Colour cycling", _l_getter("color_cycling"), _l_setter("color_cycling"))
        binder.check(f, "Colour on orbit trap", _l_getter("color_on_otrap"), _l_setter("color_on_otrap"))
        binder.check(f, "No interpolation", _l_getter("no_col_ipol"), _l_setter("no_col_ipol"))
        layout.addWidget(g)
        g = QGroupBox("Surface palette (diffuse / specular / position)")
        f = QFormLayout(g)
        for i in range(10):
            row = QWidget()
            from PySide6.QtWidgets import QHBoxLayout

            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            ff = QFormLayout()
            ff.setContentsMargins(0, 0, 0, 0)
            binder.color(ff, "", lambda s, i=i: s.lighting.palette[i].diffuse, lambda s, v, i=i: setattr(s.lighting.palette[i], "diffuse", v))
            ff2 = QFormLayout()
            ff2.setContentsMargins(0, 0, 0, 0)
            binder.color(ff2, "", lambda s, i=i: s.lighting.palette[i].specular, lambda s, v, i=i: setattr(s.lighting.palette[i], "specular", v))
            ff3 = QFormLayout()
            ff3.setContentsMargins(0, 0, 0, 0)
            binder.int(ff3, "", lambda s, i=i: s.lighting.palette[i].position, lambda s, v, i=i: setattr(s.lighting.palette[i], "position", v), 0, 32767)
            h.addLayout(ff)
            h.addLayout(ff2)
            h.addLayout(ff3)
            f.addRow(f"Colour {i + 1}", row)
        layout.addWidget(g)
        layout.addStretch(1)


class LightingPanel(QWidget):
    def __init__(self, model: SceneModel, parent=None):
        super().__init__(parent)
        self.b = Binder(model, "lighting")
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        for title, cls in (("Lights", LightsTab), ("Ambient & fog", AmbientTab), ("Palette", PaletteTab)):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(cls(model, self.b))
            tabs.addTab(scroll, title)
        layout.addWidget(tabs)
        self.b.refresh()
