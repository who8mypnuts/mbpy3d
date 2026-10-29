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
from ..io import load_lighting, load_scene, save_lighting, save_scene, scene_from_clipboard_text
from ..io.m3p_write import to_text
from .panels.camera_panel import CameraPanel
from .panels.formula_panel import FormulaPanel
from .navigation import CameraNavigator
from .panels.info_panel import InfoPanel
from .panels.post_panel import PostPanel
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
        self.post_panel = PostPanel(self.model)
        self.info_panel = InfoPanel(self.devices)
        tabs.addTab(self.formula_panel, "Formulas")
        tabs.addTab(self.camera_panel, "Camera")
        tabs.addTab(self.render_panel, "Render")
        tabs.addTab(self.light_panel, "Lighting")
        tabs.addTab(self.post_panel, "Maps && Post")
        tabs.addTab(self.info_panel, "Info")
        dock = QDockWidget("Parameters", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dock.setWidget(tabs)
        dock.setMinimumWidth(380)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.render_panel.preview_device_changed.connect(self._change_preview_device)
        self.render_panel.preview_width_changed.connect(self._change_preview_width)
        self.render_panel.preview_quality_changed.connect(self._change_preview_quality)

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
        act("Export Mandelbulb3D parameters (.m3p)...", self._export_m3p, toolbar=False)
        act("Save preview image...", self._save_preview, toolbar=False)
        menu_file.addSeparator()
        act("Load lighting preset...", self._load_lighting, toolbar=False)
        act("Save lighting preset...", self._save_lighting, toolbar=False)
        menu_file.addSeparator()
        act("Quit", self.close, "Ctrl+Q", toolbar=False)
        act("Undo", self.model.undo, "Ctrl+Z", menu_edit, toolbar=False)
        act("Redo", self.model.redo, "Ctrl+Y", menu_edit, toolbar=False)
        menu_edit.addSeparator()
        act("Copy parameters (MB3D text)", self._copy_params, "Ctrl+C", menu_edit, toolbar=False)
        act("Copy scene as JSON", self._copy_json, toolbar=False, menu=menu_edit)
        act("Copy preview image", self._copy_image, "Ctrl+Shift+C", menu_edit, toolbar=False)
        act("Paste parameters", self._paste_params, "Ctrl+V", menu_edit)
        tb.addSeparator()
        menu_help = self.menuBar().addMenu("&Help")
        act("About pymb3d", self._about, menu=menu_help, toolbar=False)
        act("Refresh preview", self._refresh, "F5", menu_render)
        act("Final render...", self._final_render, "Ctrl+R", menu_render)
        act("Navigator...", self._navigator, "Ctrl+G", menu_render)
        act("Animation...", self._animation, "Ctrl+A", menu_render)
        act("MutaGen (random variations)...", self._mutagen, "Ctrl+M", menu_render)
        act("Batch render...", self._batch, toolbar=False, menu=menu_render)
        act("Voxel / mesh export...", self._voxel, toolbar=False, menu=menu_render)
        act("Stereo render...", self._stereo, toolbar=False, menu=menu_render)
        self.recent_menu = menu_file.addMenu("Open recent")
        self._fill_recent()

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
    @property
    def nav(self) -> CameraNavigator:
        return CameraNavigator(self.model.scene)

    @Slot(float, float)
    def _nav_rotate(self, dx: float, dy: float) -> None:
        pw, ph = self.controller.preview_size(self.model.scene)
        self.nav.rotate(dx, dy, pw, ph)
        self.model.touch("camera")

    @Slot(float, float)
    def _nav_pan(self, dx: float, dy: float) -> None:
        pw, _ = self.controller.preview_size(self.model.scene)
        self.nav.pan(dx, dy, pw)
        self.model.touch("camera")

    @Slot(float)
    def _nav_forward(self, notches: float) -> None:
        self.nav.forward(notches)
        self.model.touch("camera")

    @Slot(float)
    def _nav_zoom(self, factor: float) -> None:
        self.nav.zoom(factor)
        self.model.touch("camera")

    @Slot(float, float, float)
    def _nav_move(self, dx: float, dy: float, dz: float) -> None:
        self.nav.move(dx / 5.0, dy / 5.0, dz / 5.0)
        self.model.touch("camera")

    @Slot()
    def _nav_end(self) -> None:
        self.controller.request(self.model.scene, interactive=False)

    @Slot(int, int)
    def _pixel_picked(self, x: int, y: int) -> None:
        """Double click: move the midpoint onto the surface under the cursor
        (Shift: set the depth of field focus there, Ctrl: Julia seed)."""
        res = self.controller.last_result
        if res is None:
            return
        point = self.nav.surface_point(res, x, y)
        if point is None:
            return
        mods = QGuiApplication.keyboardModifiers()
        self.model.snapshot()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            depth = float(res.gbuffer[y, x, 0])
            sc = res.scene
            zend = max(1e-10, (sc.camera.z_end - sc.camera.z_start) / sc.camera.step_width(res.width))
            self.model.scene.post.dof_focus = min(1.0, max(0.0, depth / zend))
            self.model.scene.post.dof = True
            self.post_panel.load()
            self.model.touch("lighting")
            return
        if mods & Qt.KeyboardModifier.ControlModifier:
            self.model.scene.julia = True
            self.model.scene.julia_c = (float(point[0]), float(point[1]), float(point[2]), 0.0)
            self.formula_panel.load()
            self.model.touch("formulas")
            return
        self.nav.center_on(point)
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
        self._remember(path)
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
        QGuiApplication.clipboard().setText(to_text(self.model.scene, self.model.scene.title or "pymb3d"))
        self._status("Mandelbulb3D text parameters copied to the clipboard")

    def _copy_json(self) -> None:
        QGuiApplication.clipboard().setText(json.dumps(self.model.scene.to_dict(), indent=1))
        self._status("scene JSON copied to the clipboard")

    def _copy_image(self) -> None:
        res = self.controller.last_result
        if res is not None:
            from .widgets import numpy_to_qimage

            QGuiApplication.clipboard().setImage(numpy_to_qimage(res.image))
            self._status("preview image copied to the clipboard")

    def _about(self) -> None:
        from .. import __version__
        from ..formulas import registry

        QMessageBox.about(self, "About pymb3d",
                          f"<b>pymb3d {__version__}</b><br>Mandelbulb3D re-implemented in Python.<br><br>"
                          f"{len(registry.names())} formulas, devices:<br>" +
                          "<br>".join(d.label for d in self.devices) +
                          "<br><br>Navigation: left drag look, right drag pan, wheel forward, Shift+wheel zoom, "
                          "double click centre, Shift+double click focus, Ctrl+double click Julia seed.")

    def _export_m3p(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Mandelbulb3D parameters", "scene.m3p",
                                              "Mandelbulb3D parameters (*.m3p)")
        if path:
            if not path.lower().endswith(".m3p"):
                path += ".m3p"
            save_scene(self.model.scene, path)
            self._status(f"exported {path}")

    def _load_lighting(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load lighting preset", "", "Lighting (*.json *.m3l)")
        if path:
            try:
                self.model.snapshot()
                self.model.scene.lighting = load_lighting(path)
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(self, "Load failed", str(e))
                return
            self.light_panel.load()
            self.model.touch("lighting")

    def _save_lighting(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save lighting preset", "lighting.json", "Lighting (*.json)")
        if path:
            save_lighting(self.model.scene.lighting, path)

    def _navigator(self) -> None:
        from .navigator import NavigatorDialog

        dlg = NavigatorDialog(self.model, self.preview_choice, self)
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.show()
        self._dialogs = getattr(self, "_dialogs", [])
        self._dialogs.append(dlg)

    def _mutagen(self) -> None:
        from .mutagen_dialog import MutagenDialog

        dlg = MutagenDialog(self.model, self.preview_choice, self)
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.show()
        self._dialogs = getattr(self, "_dialogs", [])
        self._dialogs.append(dlg)

    def _batch(self) -> None:
        from .batch_dialog import BatchDialog

        dlg = BatchDialog(self.render_panel.final_choice, self)
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.show()
        self._dialogs = getattr(self, "_dialogs", [])
        self._dialogs.append(dlg)

    def _voxel(self) -> None:
        from .voxel_dialog import VoxelDialog

        dlg = VoxelDialog(self.model, self)
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.show()
        self._dialogs = getattr(self, "_dialogs", [])
        self._dialogs.append(dlg)

    def _stereo(self) -> None:
        from .render_dialog import RenderDialog

        dlg = RenderDialog(self.model.scene, self.render_panel.final_choice, self, model=self.model,
                           stereo=self.render_panel.stereo_mode(), eye_distance=self.render_panel.eye_distance())
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.show()
        self._dialogs = getattr(self, "_dialogs", [])
        self._dialogs.append(dlg)

    # ---------------------------------------------------------- recent files
    def _recent(self) -> list[str]:
        v = self.settings.value("recent_files", [])
        return [str(x) for x in (v if isinstance(v, list) else [v])] if v else []

    def _remember(self, path: str) -> None:
        files = [p for p in self._recent() if p != path]
        files.insert(0, path)
        self.settings.setValue("recent_files", files[:12])
        self._fill_recent()

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for p in self._recent():
            a = QAction(Path(p).name, self)
            a.setToolTip(p)
            a.triggered.connect(lambda _=False, pp=p: self.open_path(pp))
            self.recent_menu.addAction(a)

    def _animation(self) -> None:
        from .animation_dialog import AnimationDialog

        dlg = AnimationDialog(self.model, self.render_panel.final_choice, self)
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.show()
        self._dialogs = getattr(self, "_dialogs", [])
        self._dialogs.append(dlg)

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
        dlg = RenderDialog(self.model.scene, self.render_panel.final_choice, self, model=self.model)
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

    @Slot(float)
    def _change_preview_quality(self, mul: float) -> None:
        self.controller.de_stop_mul = float(mul)
        self.controller.request(self.model.scene)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.controller.shutdown()
        for d in getattr(self, "_dialogs", []):
            d.close()
        super().closeEvent(event)
