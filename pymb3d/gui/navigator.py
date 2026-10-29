"""The Navigator window (MB3D's Navigator): a fast small preview of its own
to fly through the fractal with buttons, keys, the mouse and a continuous
flight mode; the camera is transferred to the main scene (live or on
demand)."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..backends.detect import DeviceChoice
from ..core.scene import Scene
from .navigation import CameraNavigator
from .preview_controller import PreviewController
from .preview_widget import PreviewWidget
from .scene_model import SceneModel
from .widgets import DoubleSpin, IntSpin, SliderSpin


class NavigatorDialog(QDialog):
    def __init__(self, model: SceneModel, choice: DeviceChoice, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Navigator")
        self.model = model
        self.scene: Scene = model.scene.copy()
        self.nav = CameraNavigator(self.scene)
        self.controller = PreviewController(choice, preview_width=320, coarse_factor=2)
        self.preview = PreviewWidget()
        self.preview.setMinimumSize(320, 240)
        self.controller.image_ready.connect(self.preview.set_image)
        self.controller.status.connect(lambda m: self.status.setText(m))

        # ------------------------------------------------------------ settings
        settings = QGroupBox("Navigator settings")
        sf = QFormLayout(settings)
        self.width_spin = IntSpin(320, 64, 1024)
        self.de_mul = DoubleSpin(1.5, 0.1, 20.0, 2, 0.1)
        self.de_mul.setToolTip("DE stop multiplier for the navigator (coarser = faster)")
        self.max_it = IntSpin(0, 0, 100000)
        self.max_it.setSpecialValueText("scene")
        self.max_it.setToolTip("Iteration limit of the navigator (0: the scene's)")
        self.shadows = QCheckBox("shadows")
        self.ao = QCheckBox("ambient occlusion")
        self.live = QCheckBox("transfer the camera to the main window live")
        self.live.setChecked(True)
        sf.addRow("preview width", self.width_spin)
        sf.addRow("DE stop x", self.de_mul)
        sf.addRow("max iterations", self.max_it)
        sf.addRow(self.shadows)
        sf.addRow(self.ao)
        sf.addRow(self.live)

        # ------------------------------------------------------------ movement
        moves = QGroupBox("Move (step in % of the image width)")
        mg = QGridLayout(moves)
        self.step = SliderSpin(5.0, 0.1, 100.0, 1, 1.0)
        mg.addWidget(QLabel("step"), 0, 0)
        mg.addWidget(self.step, 0, 1, 1, 3)
        for text, (x, y, z), row, col in (("forward", (0, 0, 1), 1, 1), ("back", (0, 0, -1), 3, 1),
                                          ("left", (-1, 0, 0), 2, 0), ("right", (1, 0, 0), 2, 2),
                                          ("up", (0, -1, 0), 1, 3), ("down", (0, 1, 0), 3, 3)):
            b = QPushButton(text)
            b.setAutoRepeat(True)
            b.clicked.connect(lambda _=False, v=(x, y, z): self._move(*v))
            mg.addWidget(b, row, col)

        turns = QGroupBox("Turn (degrees)")
        tg = QGridLayout(turns)
        self.angle = SliderSpin(10.0, 0.1, 90.0, 1, 1.0)
        tg.addWidget(QLabel("angle"), 0, 0)
        tg.addWidget(self.angle, 0, 1, 1, 3)
        for text, (yaw, pitch, roll), row, col in (("pitch up", (0, -1, 0), 1, 1), ("pitch down", (0, 1, 0), 3, 1),
                                                   ("yaw left", (-1, 0, 0), 2, 0), ("yaw right", (1, 0, 0), 2, 2),
                                                   ("roll left", (0, 0, -1), 1, 3), ("roll right", (0, 0, 1), 3, 3)):
            b = QPushButton(text)
            b.setAutoRepeat(True)
            b.clicked.connect(lambda _=False, v=(yaw, pitch, roll): self._turn(*v))
            tg.addWidget(b, row, col)

        zooms = QGroupBox("Zoom and flight")
        zg = QGridLayout(zooms)
        zin = QPushButton("zoom in")
        zout = QPushButton("zoom out")
        zin.clicked.connect(lambda: self._zoom(1.25))
        zout.clicked.connect(lambda: self._zoom(0.8))
        self.fly = QCheckBox("fly (continuous forward motion)")
        self.fly_speed = SliderSpin(2.0, -20.0, 20.0, 1, 0.5)
        zg.addWidget(zin, 0, 0)
        zg.addWidget(zout, 0, 1)
        zg.addWidget(self.fly, 1, 0, 1, 2)
        zg.addWidget(QLabel("speed (% width / s)"), 2, 0)
        zg.addWidget(self.fly_speed, 2, 1)

        transfer = QHBoxLayout()
        to_main = QPushButton("Apply camera to main")
        from_main = QPushButton("Get scene from main")
        reset = QPushButton("Reset view")
        to_main.clicked.connect(self._apply_to_main)
        from_main.clicked.connect(self._from_main)
        reset.clicked.connect(self._reset)
        transfer.addWidget(to_main)
        transfer.addWidget(from_main)
        transfer.addWidget(reset)
        self.info = QLabel()
        self.info.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.status = QLabel()

        right = QVBoxLayout()
        right.addWidget(settings)
        right.addWidget(moves)
        right.addWidget(turns)
        right.addWidget(zooms)
        right.addLayout(transfer)
        right.addWidget(self.info)
        right.addStretch(1)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(self.preview, 1)
        left.addWidget(QLabel("Left drag: look around. Right drag: pan. Wheel: forward/back. "
                              "Shift+wheel: zoom. Double click: centre the point. Keys: WASD QE, arrows, +/-"))
        left.addWidget(self.status)
        lay.addLayout(left, 1)
        lay.addLayout(right)
        self.resize(900, 600)

        # ------------------------------------------------------------ wiring
        p = self.preview
        p.rotate.connect(self._drag_rotate)
        p.pan.connect(self._drag_pan)
        p.move_forward.connect(lambda n: self._move(0, 0, n))
        p.zoom.connect(self._zoom)
        p.move.connect(lambda dx, dy, dz: self._move(dx / 5.0, dy / 5.0, dz / 5.0))
        p.gesture_end.connect(self._changed)
        p.pixel_picked.connect(self._pick)
        for w in (self.width_spin, self.de_mul, self.max_it):
            w.valueChanged.connect(lambda _v: self._changed())
        self.shadows.toggled.connect(lambda _v: self._changed())
        self.ao.toggled.connect(lambda _v: self._changed())
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._fly_tick)
        self.fly.toggled.connect(lambda on: self._timer.start() if on else self._timer.stop())
        self._update_info()
        self._changed()

    # ------------------------------------------------------------- scene
    def _navi_scene(self) -> Scene:
        sc = self.scene.copy()
        sc.de_stop = self.scene.de_stop * float(self.de_mul.value())
        if self.max_it.value() > 0:
            sc.iterations = int(self.max_it.value())
        sc.shadows = self.shadows.isChecked()
        sc.ao.mode = "de" if self.ao.isChecked() else "none"
        sc.post.dof = False
        return sc

    def _changed(self, interactive: bool = True) -> None:
        self.controller.preview_width = int(self.width_spin.value())
        self.controller.request(self._navi_scene(), interactive=interactive)
        self._update_info()
        if self.live.isChecked():
            self._apply_to_main()

    def _update_info(self) -> None:
        cam = self.scene.camera
        ax, ay, az = cam.angles_deg
        self.info.setText(f"mid  {cam.mid[0]: .6g} {cam.mid[1]: .6g} {cam.mid[2]: .6g}\n"
                          f"zoom {cam.zoom:.6g}   fov {cam.fov_y:.1f}\n"
                          f"rot  {ax:.2f} {ay:.2f} {az:.2f}\nz    {cam.z_start:.4g} .. {cam.z_end:.4g}")

    # ---------------------------------------------------------- gestures
    def _view_size(self) -> tuple[int, int]:
        return self.controller.preview_size(self.scene)

    @Slot(float, float)
    def _drag_rotate(self, dx: float, dy: float) -> None:
        w, h = self._view_size()
        self.nav.rotate(dx, dy, w, h)
        self._changed()

    @Slot(float, float)
    def _drag_pan(self, dx: float, dy: float) -> None:
        w, _ = self._view_size()
        self.nav.pan(dx, dy, w)
        self._changed()

    def _move(self, dx: float, dy: float, dz: float) -> None:
        self.nav.move(dx, dy, dz, self.step.value() / 100.0)
        self._changed()

    def _turn(self, yaw: float, pitch: float, roll: float) -> None:
        a = self.angle.value()
        self.nav.turn(yaw * a, pitch * a, roll * a)
        self._changed()

    def _zoom(self, factor: float) -> None:
        self.nav.zoom(factor)
        self._changed()

    def _fly_tick(self) -> None:
        self.nav.forward(1.0, self.fly_speed.value() / 100.0 * 0.1)
        self._changed()

    @Slot(int, int)
    def _pick(self, x: int, y: int) -> None:
        res = self.controller.last_result
        if res is None:
            return
        point = self.nav.surface_point(res, x, y)
        if point is None:
            return
        self.nav.center_on(point)
        self._changed(interactive=False)

    # ---------------------------------------------------------- transfer
    def _apply_to_main(self) -> None:
        self.model.scene.camera = self.scene.camera.copy()
        self.model.touch("camera")

    def _from_main(self) -> None:
        self.scene = self.model.scene.copy()
        self.nav = CameraNavigator(self.scene)
        self._changed(interactive=False)

    def _reset(self) -> None:
        from ..core.camera import Camera

        c = Camera()
        c.set_angles_deg(-20.0, 30.0, 0.0)
        c.z_start, c.zoom = -2.0, 1.4
        self.scene.camera = c
        self._changed(interactive=False)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        self.controller.shutdown()
        super().closeEvent(event)
