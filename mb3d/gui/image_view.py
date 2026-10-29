"""The image view: shows the render, navigates the camera with the mouse."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def numpy_to_qimage(img: np.ndarray) -> QImage:
    """(h, w, 3) uint8 -> QImage (a deep copy, safe to keep)."""
    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    return QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageView(QGraphicsView):
    """Zoomable image display that turns mouse gestures into camera signals."""

    rotated = Signal(float, float)  # d_yaw, d_pitch in degrees
    zoomed = Signal(float)  # factor
    moved = Signal(float, float, float)  # right, down, forward (fractions of the view width)
    clicked = Signal(int, int)  # image pixel

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setRenderHints(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.darkGray)
        self.setFocusPolicy(Qt.StrongFocus)
        self._last: QPointF | None = None
        self._image_size = (0, 0)
        self._fit = True
        self.navigation_enabled = True

    # --------------------------------------------------------------- image
    def set_image(self, img: np.ndarray, display_size: tuple[int, int] | None = None) -> None:
        """Show ``img``; a coarse pass is scaled up to ``display_size``."""
        pix = QPixmap.fromImage(numpy_to_qimage(img))
        target = display_size or (img.shape[1], img.shape[0])
        if (pix.width(), pix.height()) != target:
            pix = pix.scaled(target[0], target[1], Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self._item.setPixmap(pix)
        size_changed = self._image_size != target
        self._image_size = target
        self._scene.setSceneRect(0, 0, target[0], target[1])
        if self._fit or size_changed:
            self.fit()

    def fit(self) -> None:
        self._fit = True
        if self._image_size[0] > 0:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def actual_size(self) -> None:
        self._fit = False
        self.resetTransform()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self.fit()

    # --------------------------------------------------------------- mouse
    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier or not self.navigation_enabled:
            self._fit = False
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            steps = event.angleDelta().y() / 120.0
            self.zoomed.emit(1.2 ** steps)
        event.accept()

    def mousePressEvent(self, event):
        self._last = event.position()
        self.setFocus()
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier:
            p = self.mapToScene(event.position().toPoint())
            self.clicked.emit(int(p.x()), int(p.y()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._last is not None and self.navigation_enabled and event.buttons():
            d = event.position() - self._last
            self._last = event.position()
            w = max(1.0, float(self.viewport().width()))
            if event.buttons() & Qt.LeftButton:
                self.rotated.emit(-d.x() / w * 90.0, d.y() / w * 90.0)
            elif event.buttons() & Qt.RightButton or event.buttons() & Qt.MiddleButton:
                self.moved.emit(-d.x() / w, -d.y() / w, 0.0)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._last = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if not self.navigation_enabled:
            return super().keyPressEvent(event)
        step = 0.1
        k = event.key()
        if k in (Qt.Key_W, Qt.Key_Up):
            self.moved.emit(0.0, 0.0, step)
        elif k in (Qt.Key_S, Qt.Key_Down):
            self.moved.emit(0.0, 0.0, -step)
        elif k in (Qt.Key_A, Qt.Key_Left):
            self.moved.emit(-step, 0.0, 0.0)
        elif k in (Qt.Key_D, Qt.Key_Right):
            self.moved.emit(step, 0.0, 0.0)
        elif k == Qt.Key_Q:
            self.moved.emit(0.0, -step, 0.0)
        elif k == Qt.Key_E:
            self.moved.emit(0.0, step, 0.0)
        elif k == Qt.Key_Plus:
            self.zoomed.emit(1.2)
        elif k == Qt.Key_Minus:
            self.zoomed.emit(1 / 1.2)
        else:
            return super().keyPressEvent(event)
        event.accept()
