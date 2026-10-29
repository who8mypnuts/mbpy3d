"""The main window: the interactive preview, the parameter panels and the
actions (open / save / import / final render)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings, Qt, Slot
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QTabWidget,
)

from ..backends.detect import DeviceChoice, available_devices, choose
from ..core.scene import Scene, default_scene
from ..io import load_scene, save_scene, scene_from_clipboard_text
from ..io.m3p import raw_to_text
from .panels.camera_panel import CameraPanel
from .panels.formula_panel import FormulaPanel
from .panels.info_panel import InfoPanel
from .panels.light_panel import LightPanel
from .panels.render_panel import RenderPanel
from .preview_controller import PreviewController
from .preview_widget import PreviewWidget
from .render_dialog import RenderDialog, save_image
from .scene_model import SceneModel

FILE_FILTER = ("Scenes (*.json *.m3p *.m3i *.txt *.png);;pymb3d scene (*.json);;"
               "Mandelbulb3D parameters (*.m3p *.m3i *.txt);;rendered PNG with scene (*.png)")


class MainWindow(QMainWindow):
    def __init__(self, devices: list[DeviceChoice] | None = None, scene: Scene | None = None) -> None:
        super().__init__()
        self.setWindowTitle("pymb3d - Mandelbulb3D in Python")
        self.settings = QSettings("pymb3d", "pymb3d")
        self.devices = devices or available_devices()
        self.preview_choice = choose(self.devices, "preview")
        self.final_choice = choose(self.devices, "final")
        self.model = SceneModel(scene or default_scene())
        if scene is None:
            self.model.scene.camera.z_start = -2.0
            self.model.scene.camera.zoom = 1.4

        self.preview = PreviewWidget()
        self.setCentralWidget(self.preview)
        self.controller = PreviewController(self.preview_choice, preview_width=480)
        self.controller.image_ready.connect(self.preview.set_image)
        self.controller.status.connect(self._status)
        self.controller.progress.connect(self._progress)
        self.controller.result_ready.connect(self._result_ready)
        self.controller.device_ready.connect(lambda s: self._status(f"preview device: {s}"))

        self._build_panels()
        self._build_actions()
        self._connect_navigation()
        self.model.changed.connect(self._scene_changed)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setRange(0, 1000)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.resize(1280, 800)
        self.controller.request(self.model.scene)

    # ------------------------------------------------------------------ ui
    def _build_panels(self) -> None:
        tabs = QTabWidget()
        self.formula_panel = FormulaPanel(self.model)
        self.camera_panel = CameraPanel(self.model)
        self.render_panel = RenderPanel(self.model, self.devices, self.preview_choice, self.final_choice)
        self.light_panel = LightPanel(self.model)
        self.info_panel = InfoPanel(self.devices)
        tabs.addTab(self.formula_panel, "Formulas")
        tabs.addTab(self.camera_panel, "Camera")
        tabs.addTab(self.render_panel, "Render")
        tabs.addTab(self.light_panel, "Lighting")
        tabs.addTab(self.info_panel, "Info")
        dock = QDockWidget("Parameters", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dock.setWidget(tabs)
        dock.setMinimumWidth(380)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.render_panel.preview_device_changed.connect(self._change_preview_device)
        self.render_panel.preview_width_changed.connect(self._change_preview_width)

    def _build_actions(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        menu_file = self.menuBar().addMenu("&File")
        menu_edit = self.menuBar().addMenu("&Edit")
        menu_render = self.menuBar().addMenu("&Render")

        def act(text, slot, shortcut=None, menu=None, toolbar=True):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            (menu or menu_file).addAction(a)
            if toolbar:
                tb.addAction(a)
            return a

        act("New", self._new, "Ctrl+N")
        act("Open...", self._open, "Ctrl+O")
        act("Save", self._save, "Ctrl+S")
        act("Save as...", self._save_as, "Ctrl+Shift+S", toolbar=False)
        act("Save preview image...", self._save_preview, toolbar=False)
        menu_file.addSeparator()
        act("Quit", self.close, "Ctrl+Q", toolbar=False)
        act("Undo", self.model.undo, "Ctrl+Z", menu_edit, toolbar=False)
        act("Redo", self.model.redo, "Ctrl+Y", menu_edit, toolbar=False)
        menu_edit.addSeparator()
        act("Copy parameters (MB3D text)", self._copy_params, "Ctrl+C", menu_edit, toolbar=False)
        act("Paste parameters", self._paste_params, "Ctrl+V", menu_edit)
        tb.addSeparator()
        act("Refresh preview", self._refresh, "F5", menu_render)
        act("Final render...", self._final_render, "Ctrl+R", menu_render)

    def _connect_navigation(self) -> None:
        p = self.preview
        p.rotate.connect(self._nav_rotate)
        p.pan.connect(self._nav_pan)
        p.move_forward.connect(self._nav_forward)
        p.zoom.connect(self._nav_zoom)
        p.move.connect(self._nav_move)
        p.gesture_end.connect(self._nav_end)
        p.pixel_picked.connect(self._pixel_picked)

    # ------------------------------------------------------------ navigation
    def _cam(self):
        return self.model.scene.camera

    @Slot(float, float)
    def _nav_rotate(self, dx: float, dy: float) -> None:
        cam = self._cam()
        fov = np.radians(cam.fov_y)
        pw, ph = self.controller.preview_size(self.model.scene)
        cam.rotate(np.array([0.0, 1.0, 0.0]), dx / max(1, pw) * fov * pw / ph)
        cam.rotate(np.array([1.0, 0.0, 0.0]), -dy / max(1, ph) * fov)
        self.model.touch("camera")

    @Slot(float, float)
    def _nav_pan(self, dx: float, dy: float) -> None:
        sc = self.model.scene
        pw, _ = self.controller.preview_size(sc)
        scale = sc.width / max(1, pw)
        self._cam().move_pixels(-dx * scale, -dy * scale, 0.0, sc.width)
        self.model.touch("camera")

    @Slot(float)
    def _nav_forward(self, notches: float) -> None:
        sc = self.model.scene
        self._cam().move_pixels(0.0, 0.0, notches * sc.width * 0.05, sc.width)
        self.model.touch("camera")

    @Slot(float)
    def _nav_zoom(self, factor: float) -> None:
        self._cam().zoom *= factor
        self.model.touch("camera")

    @Slot(float, float, float)
    def _nav_move(self, dx: float, dy: float, dz: float) -> None:
        sc = self.model.scene
        scale = sc.width / 100.0
        self._cam().move_pixels(dx * scale, dy * scale, dz * scale, sc.width)
        self.model.touch("camera")

    @Slot()
    def _nav_end(self) -> None:
        self.controller.request(self.model.scene, interactive=False)

    @Slot(int, int)
    def _pixel_picked(self, x: int, y: int) -> None:
        """Double click: move the midpoint onto the surface under the cursor."""
        res = self.controller.last_result
        if res is None or res.gbuffer is None:
            return
        from ..core import params as PL

        H, W = res.gbuffer.shape[:2]
        if not (0 <= x < W and 0 <= y < H):
            return
        depth = float(res.gbuffer[y, x, PL.G_DEPTH])
        if depth < 0:
            return
        setup = res.scene.camera.kernel_setup(W, H, res.scene.raystep_limiter, res.scene.z_step_div,
                                              res.scene.vary_de_stop_on_fov)
        start, d = setup.ray(x, y)
        point = start + d * depth
        cam = self._cam()
        self.model.snapshot()
        # keep the camera where it is: move mid and the start plane together
        shift = point - cam.mid
        cam.mid = point
        cam.z_start -= float(shift @ cam.forward)
        cam.z_end -= float(shift @ cam.forward)
        self.model.touch("camera")
        self._nav_end()

    # ---------------------------------------------------------------- scene
    @Slot(str)
    def _scene_changed(self, kind: str) -> None:
        if kind == "lighting" and self.controller.relight(self.model.scene):
            return
        if kind == "size":
            return
        self.controller.request(self.model.scene, interactive=(kind == "camera"))
        self._update_title()

    def _update_title(self) -> None:
        name = Path(self.model.path).name if self.model.path else "untitled"
        self.setWindowTitle(f"pymb3d - {name}{' *' if self.model.dirty else ''}")

    @Slot(object)
    def _result_ready(self, result) -> None:
        self.info_panel.show_result(result)

    @Slot(str)
    def _status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    @Slot(float)
    def _progress(self, f: float) -> None:
        self.progress_bar.setValue(int(f * 1000))

    # -------------------------------------------------------------- actions
    def _new(self) -> None:
        sc = default_scene()
        sc.camera.z_start, sc.camera.zoom = -2.0, 1.4
        self.model.replace(sc)
        self._update_title()

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open scene", "", FILE_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        try:
            scene, warnings = load_scene(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.model.replace(scene, path)
        self._update_title()
        self._show_notes(warnings)

    def _show_notes(self, warnings: list[str]) -> None:
        """Import notes in a non-modal box (the preview keeps rendering)."""
        if not warnings:
            return
        self._status("; ".join(warnings[:3]))
        box = QMessageBox(QMessageBox.Icon.Information, "Imported with notes", "\n".join(warnings), parent=self)
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.show()
        self._notes_box = box

    def _save(self) -> None:
        if self.model.path and self.model.path.lower().endswith(".json"):
            save_scene(self.model.scene, self.model.path)
            self.model.dirty = False
            self._update_title()
        else:
            self._save_as()

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save scene", "scene.json", "pymb3d scene (*.json)")
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            save_scene(self.model.scene, path)
            self.model.path = path
            self.model.dirty = False
            self._update_title()

    def _save_preview(self) -> None:
        res = self.controller.last_result
        if res is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save preview", "preview.png", "PNG image (*.png)")
        if path:
            save_image(path, res.image, res.scene)

    def _copy_params(self) -> None:
        QGuiApplication.clipboard().setText(json.dumps(self.model.scene.to_dict(), indent=1))
        self._status("scene JSON copied to the clipboard")

    def _paste_params(self) -> None:
        text = QGuiApplication.clipboard().text()
        if not text.strip():
            return
        try:
            scene, warnings = scene_from_clipboard_text(text)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Paste failed", f"The clipboard holds no scene: {e}")
            return
        self.model.replace(scene)
        self._update_title()
        self._show_notes(warnings)

    def _refresh(self) -> None:
        self.controller.request(self.model.scene)

    def _final_render(self) -> None:
        dlg = RenderDialog(self.model.scene, self.render_panel.final_choice, self)
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.show()
        self._dialogs = getattr(self, "_dialogs", [])
        self._dialogs.append(dlg)

    @Slot(object)
    def _change_preview_device(self, choice: DeviceChoice) -> None:
        self.preview_choice = choice
        self.controller.change_device(choice)

    @Slot(int)
    def _change_preview_width(self, w: int) -> None:
        self.controller.preview_width = int(w)
        self.controller.request(self.model.scene)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.controller.shutdown()
        for d in getattr(self, "_dialogs", []):
            d.close()
        super().closeEvent(event)
