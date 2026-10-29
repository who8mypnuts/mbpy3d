"""Two-way bindings between widgets and the scene (one small class, reused by every panel)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QWidget,
)

from mb3d.params import Scene

from ..model import SceneModel

Getter = Callable[[Scene], object]
Setter = Callable[[Scene, object], None]


@dataclass
class Binding:
    widget: QWidget
    get: Getter
    apply: Callable[[QWidget, object], None]


class ColorButton(QPushButton):
    """A button showing a colour; click opens the colour dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = (0, 0, 0)
        self.on_changed: Callable[[tuple[int, int, int]], None] | None = None
        self.clicked.connect(self._pick)
        self.setFixedWidth(64)

    def set_color(self, rgb) -> None:
        self._color = tuple(int(c) for c in rgb)
        self.setStyleSheet(f"background-color: rgb({self._color[0]}, {self._color[1]}, {self._color[2]});")

    def color(self) -> tuple[int, int, int]:
        return self._color

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor(*self._color), self, "Colour")
        if c.isValid():
            self.set_color((c.red(), c.green(), c.blue()))
            if self.on_changed is not None:
                self.on_changed(self._color)


class Binder:
    """Builds form rows bound to the model; ``refresh`` pulls the values back."""

    def __init__(self, model: SceneModel, what: str):
        self.model = model
        self.what = what
        self.bindings: list[Binding] = []
        self._refreshing = False
        model.changed.connect(self._on_changed)

    # ------------------------------------------------------------ rows
    def float(self, form: QFormLayout, label: str, get: Getter, set_: Setter, minimum=-1e9, maximum=1e9,
              step=0.1, decimals=4, what: str | None = None) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(minimum, maximum)
        w.setDecimals(decimals)
        w.setSingleStep(step)
        w.setKeyboardTracking(False)
        w.valueChanged.connect(lambda v: self._set(set_, float(v), what))
        self.bindings.append(Binding(w, get, lambda wd, v: wd.setValue(float(v))))
        form.addRow(label, w)
        return w

    def int(self, form: QFormLayout, label: str, get: Getter, set_: Setter, minimum=0, maximum=100000,
            what: str | None = None) -> QSpinBox:
        w = QSpinBox()
        w.setRange(minimum, maximum)
        w.setKeyboardTracking(False)
        w.valueChanged.connect(lambda v: self._set(set_, int(v), what))
        self.bindings.append(Binding(w, get, lambda wd, v: wd.setValue(int(v))))
        form.addRow(label, w)
        return w

    def check(self, form: QFormLayout, label: str, get: Getter, set_: Setter, what: str | None = None) -> QCheckBox:
        w = QCheckBox()
        w.toggled.connect(lambda v: self._set(set_, bool(v), what))
        self.bindings.append(Binding(w, get, lambda wd, v: wd.setChecked(bool(v))))
        form.addRow(label, w)
        return w

    def combo(self, form: QFormLayout, label: str, items: list[str], get: Getter, set_: Setter,
              what: str | None = None) -> QComboBox:
        w = QComboBox()
        w.addItems(items)
        w.currentIndexChanged.connect(lambda i: self._set(set_, int(i), what))
        self.bindings.append(Binding(w, get, lambda wd, v: wd.setCurrentIndex(int(v))))
        form.addRow(label, w)
        return w

    def color(self, form: QFormLayout, label: str, get: Getter, set_: Setter, what: str | None = None) -> ColorButton:
        w = ColorButton()
        w.on_changed = lambda rgb: self._set(set_, rgb, what)
        self.bindings.append(Binding(w, get, lambda wd, v: wd.set_color(v)))
        form.addRow(label, w)
        return w

    # --------------------------------------------------------- plumbing
    def _set(self, set_: Setter, value, what: str | None) -> None:
        if self._refreshing:
            return
        set_(self.model.scene, value)
        self.model.touch(what or self.what)

    def _on_changed(self, what: str) -> None:
        if self._refreshing:
            return
        self.refresh()

    def refresh(self) -> None:
        self._refreshing = True
        try:
            for b in self.bindings:
                b.widget.blockSignals(True)
                try:
                    b.apply(b.widget, b.get(self.model.scene))
                finally:
                    b.widget.blockSignals(False)
        finally:
            self._refreshing = False


def tuple_setter(attr: str, index: int) -> Setter:
    """A setter replacing element ``index`` of the tuple attribute ``attr``."""

    def set_(s: Scene, v) -> None:
        t = list(getattr(s, attr))
        t[index] = v
        setattr(s, attr, tuple(t))

    return set_


def attr_setter(attr: str) -> Setter:
    return lambda s, v: setattr(s, attr, v)


def attr_getter(attr: str) -> Getter:
    return lambda s: getattr(s, attr)


__all__ = ["Binder", "ColorButton", "attr_getter", "attr_setter", "tuple_setter", "Qt"]
