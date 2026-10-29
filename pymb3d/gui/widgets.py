"""Small reusable widgets: labelled spin boxes, slider+spin pairs, colour
buttons and a numpy -> QImage helper."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)


def numpy_to_qimage(rgb: np.ndarray) -> QImage:
    """A QImage (copy) of an (H, W, 3) uint8 array."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def numpy_to_pixmap(rgb: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(numpy_to_qimage(rgb))


class DoubleSpin(QDoubleSpinBox):
    """A double spin box with a sensible range and keyboard tracking off
    (one signal per committed value)."""

    def __init__(self, value: float = 0.0, minimum: float = -1e12, maximum: float = 1e12, decimals: int = 6,
                 step: float = 0.1, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRange(minimum, maximum)
        self.setDecimals(decimals)
        self.setSingleStep(step)
        self.setKeyboardTracking(False)
        self.setValue(value)
        self.setMinimumWidth(90)

    def set_silently(self, value: float) -> None:
        self.blockSignals(True)
        self.setValue(value)
        self.blockSignals(False)


class IntSpin(QSpinBox):
    def __init__(self, value: int = 0, minimum: int = 0, maximum: int = 100000, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRange(minimum, maximum)
        self.setKeyboardTracking(False)
        self.setValue(value)

    def set_silently(self, value: int) -> None:
        self.blockSignals(True)
        self.setValue(value)
        self.blockSignals(False)


class SliderSpin(QWidget):
    """A horizontal slider coupled with a spin box (MB3D's trackbars)."""

    valueChanged = Signal(float)

    def __init__(self, value: float, minimum: float, maximum: float, decimals: int = 0, step: float = 1.0,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scale = 10 ** decimals
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(minimum * self._scale), int(maximum * self._scale))
        self.spin = DoubleSpin(value, minimum, maximum, decimals, step)
        self.spin.setMinimumWidth(70)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.set_silently(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> float:
        return self.spin.value()

    def set_silently(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(value * self._scale)))
        self.slider.blockSignals(False)
        self.spin.set_silently(value)

    def _from_slider(self, v: int) -> None:
        self.spin.set_silently(v / self._scale)
        self.valueChanged.emit(v / self._scale)

    def _from_spin(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * self._scale)))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)


class ColorButton(QPushButton):
    """A button showing a colour; clicking opens a colour dialog."""

    colorChanged = Signal(object)  # (r, g, b) in 0..1

    def __init__(self, rgb=(1.0, 1.0, 1.0), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rgb = tuple(rgb)
        self.setFixedSize(34, 22)
        self.clicked.connect(self._pick)
        self._paint()

    def rgb(self):
        return self._rgb

    def set_rgb(self, rgb) -> None:
        self._rgb = tuple(float(v) for v in rgb)
        self._paint()

    def _paint(self) -> None:
        c = QColor.fromRgbF(*[max(0.0, min(1.0, v)) for v in self._rgb])
        self.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #555;")

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor.fromRgbF(*self._rgb), self, "Colour")
        if c.isValid():
            self.set_rgb((c.redF(), c.greenF(), c.blueF()))
            self.colorChanged.emit(self._rgb)


def form_row(form: QFormLayout, label: str, widget: QWidget) -> QWidget:
    form.addRow(label, widget)
    return widget


def make_scroll(inner: QWidget):
    """A vertical scroll area that never scrolls sideways: the inner widget
    follows the width of the panel."""
    from PySide6.QtWidgets import QScrollArea, QSizePolicy

    inner.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(inner)
    return scroll
