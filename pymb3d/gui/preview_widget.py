"""The preview: shows the latest image and turns mouse / keyboard input
into navigation signals (the Navigator of MB3D).

Left drag: rotate (look around).  Right drag / middle drag: pan.
Wheel: move forward / back.  Shift+wheel: zoom.  Keys: W/S forward/back,
A/D strafe, Q/E up/down, arrows rotate, +/- zoom.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget

from .widgets import numpy_to_pixmap


class PreviewWidget(QWidget):
    #: (dx, dy) in preview pixels: rotate the camera
    rotate = Signal(float, float)
    #: (dx, dy) in preview pixels: pan the camera
    pan = Signal(float, float)
    #: steps forward (negative = back), in units of the wheel notch
    move_forward = Signal(float)
    #: zoom factor multiplier (e.g. 1.1)
    zoom = Signal(float)
    #: (dx, dy, dz) camera space movement in preview pixels (keys)
    move = Signal(float, float, float)
    #: the user released the mouse / finished a gesture
    gesture_end = Signal()
    #: a pixel was double clicked: (x, y) in image pixels of the shown image
    pixel_picked = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(False)
        self._pixmap: QPixmap | None = None
        self._image_size = (0, 0)
        self._last: QPoint | None = None
        self._button = Qt.MouseButton.NoButton
        self._target = QRect()

    # ---------------------------------------------------------------- image
    def set_image(self, rgb: np.ndarray) -> None:
        self._pixmap = numpy_to_pixmap(rgb)
        self._image_size = (rgb.shape[1], rgb.shape[0])
        self.update()

    def set_pixmap(self, pm: QPixmap) -> None:
        self._pixmap = pm
        self._image_size = (pm.width(), pm.height())
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._pixmap is None:
            p.setPen(Qt.GlobalColor.gray)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "rendering ...")
            return
        w, h = self._image_size
        if w == 0 or h == 0:
            return
        scale = min(self.width() / w, self.height() / h)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        x, y = (self.width() - tw) // 2, (self.height() - th) // 2
        self._target = QRect(x, y, tw, th)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, scale < 1.0)
        p.drawPixmap(self._target, self._pixmap)

    # ---------------------------------------------------------------- mouse
    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self._last = e.position().toPoint()
        self._button = e.button()
        self.setFocus()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._last is None:
            return
        pos = e.position().toPoint()
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        if self._button == Qt.MouseButton.LeftButton:
            self.rotate.emit(dx, dy)
        elif self._button in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self.pan.emit(dx, dy)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self._last = None
        self._button = Qt.MouseButton.NoButton
        self.gesture_end.emit()

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._target.isEmpty() or self._image_size[0] == 0:
            return
        pos = e.position().toPoint()
        if not self._target.contains(pos):
            return
        fx = (pos.x() - self._target.x()) / self._target.width()
        fy = (pos.y() - self._target.y()) / self._target.height()
        self.pixel_picked.emit(int(fx * self._image_size[0]), int(fy * self._image_size[1]))

    def wheelEvent(self, e: QWheelEvent) -> None:  # noqa: N802
        notches = e.angleDelta().y() / 120.0
        if notches == 0:
            return
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.zoom.emit(1.15 ** notches)
        else:
            self.move_forward.emit(notches)
        self.gesture_end.emit()

    # ------------------------------------------------------------- keyboard
    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        k = e.key()
        step = 20.0 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 5.0
        moves = {
            Qt.Key.Key_W: (0, 0, step), Qt.Key.Key_S: (0, 0, -step),
            Qt.Key.Key_A: (-step, 0, 0), Qt.Key.Key_D: (step, 0, 0),
            Qt.Key.Key_Q: (0, step, 0), Qt.Key.Key_E: (0, -step, 0),
        }
        rots = {Qt.Key.Key_Left: (-step, 0), Qt.Key.Key_Right: (step, 0),
                Qt.Key.Key_Up: (0, -step), Qt.Key.Key_Down: (0, step)}
        if k in moves:
            self.move.emit(*moves[k])
        elif k in rots:
            self.rotate.emit(*rots[k])
        elif k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom.emit(1.25)
        elif k in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.zoom.emit(0.8)
        else:
            super().keyPressEvent(e)
            return
        self.gesture_end.emit()
