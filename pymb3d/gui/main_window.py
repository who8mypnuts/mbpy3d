"""The main window: the interactive preview, the parameter panels, the
image size bar, the messages dock and the actions (files, tool windows,
final render)."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Slot
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence
from PySide6.QtWidgets import QApplication, QDockWidget, QFileDialog, QMainWindow, QMessageBox, QProgressBar, QTabWidget

from ..backends.detect import DeviceChoice, available_devices, choose
from ..core.scene import Scene, default_scene
from ..io import load_lighting, load_scene, save_lighting, save_scene, scene_from_clipboard_text
from ..io.m3p_write import to_text
from .image_size import ImageSizeBar
from .messages import MessagesDock
from .navigation import CameraNavigator
from .panels.camera_panel import CameraPanel
from .panels.formula_panel import FormulaPanel
from .panels.info_panel import InfoPanel
from .panels.light_panel import LightPanel
from .panels.post_panel import PostPanel
from .panels.render_panel import RenderPanel
from .preview_controller import PreviewController
from .preview_widget import PreviewWidget
from .render_dialog import RenderDialog, save_image
from .scene_model import SceneModel
from .theme import apply_theme, theme_names
from .tool_windows import HOTKEYS, TITLES, ToolWindows
from .widgets import numpy_to_qimage

FILE_FILTER = ("Scenes (*.json *.m3p *.m3i *.txt *.png);;pymb3d scene (*.json);;"
               "Mandelbulb3D parameters (*.m3p *.m3i *.txt);;rendered PNG with scene (*.png)")
PANEL_HOTKEYS = {"Formulas": "Ctrl+7", "Lighting": "Ctrl+8", "Maps && Post": "Ctrl+9"}


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
        self.tools = ToolWindows(self)

        self.preview = PreviewWidget()
        self.setCentralWidget(self.preview)
        self.controller = PreviewController(self.preview_choice, preview_width=480)
        self.controller.image_ready.connect(self.preview.set_image)
        self.controller.status.connect(self._status)
        self.controller.progress.connect(self._progress)
        self.controller.result_ready.connect(self._result_ready)
        self.controller.device_ready.connect(lambda s: self._status(f"preview device: {s}"))

        self.messages = MessagesDock(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.messages)
        self._build_panels()
        self.size_bar = ImageSizeBar(self.model, self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.size_bar)
        self.size_bar.render_requested.connect(self._final_render)
        self._build_actions()
        self._connect_navigation()
        self.model.changed.connect(self._scene_changed)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setRange(0, 1000)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.resize(1320, 860)
        self._restore_layout()
        self.messages.log("devices: " + "; ".join(d.label for d in self.devices))
        self.messages.log(f"preview on {self.preview_choice.label}, final render on {self.final_choice.label}")
        self.controller.request(self.model.scene)

    # ------------------------------------------------------------------ ui
    def _build_panels(self) -> None:
        self.tabs = QTabWidget()
        self.formula_panel = FormulaPanel(self.model)
        self.camera_panel = CameraPanel(self.model)
        self.render_panel = RenderPanel(self.model, self.devices, self.preview_choice, self.final_choice)
        self.light_panel = LightPanel(self.model)
        self.post_panel = PostPanel(self.model)
        self.info_panel = InfoPanel(self.devices, model=self.model)
        for panel, name in ((self.formula_panel, "Formulas"), (self.camera_panel, "Camera"),
                            (self.render_panel, "Render"), (self.light_panel, "Lighting"),
                            (self.post_panel, "Maps && Post"), (self.info_panel, "Info")):
            self.tabs.addTab(panel, name)
        dock = QDockWidget("Parameters", self)
        dock.setObjectName("parameters")
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(380)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.messages.setObjectName("messages")
        self.render_panel.preview_device_changed.connect(self._change_preview_device)
        self.render_panel.preview_width_changed.connect(self._change_preview_width)
        self.render_panel.preview_quality_changed.connect(self._change_preview_quality)

    def _build_actions(self) -> None:
        tb = self.addToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        self.insertToolBar(self.size_bar, tb)
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        mb = self.menuBar()
        menu_file = mb.addMenu("&File")
        menu_edit = mb.addMenu("&Edit")
        menu_view = mb.addMenu("&View")
        menu_render = mb.addMenu("&Render")
        menu_tools = mb.addMenu("&Tools")
        menu_prefs = mb.addMenu("&Prefs")
        menu_help = mb.addMenu("&Help")

        def act(text, slot, shortcut=None, menu=None, toolbar=False):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            (menu or menu_file).addAction(a)
            if toolbar:
                tb.addAction(a)
            return a

        act("New", self._new, "Ctrl+N", toolbar=True)
        act("Open...", self._open, "Ctrl+O", toolbar=True)
        self.recent_menu = menu_file.addMenu("Open recent")
        act("Save", self._save, "Ctrl+S", toolbar=True)
        act("Save as...", self._save_as, "Ctrl+Shift+S")
        act("Export Mandelbulb3D parameters (.m3p)...", self._export_m3p)
        act("Save preview image...", self._save_preview)
        menu_file.addSeparator()
        act("Load lighting preset...", self._load_lighting)
        act("Save lighting preset...", self._save_lighting)
        menu_file.addSeparator()
        act("Quit", self.close, "Ctrl+Q")
        act("Undo", self.model.undo, "Ctrl+Z", menu_edit)
        act("Redo", self.model.redo, "Ctrl+Y", menu_edit)
        menu_edit.addSeparator()
        act("Copy parameters (MB3D text)", self._copy_params, "Ctrl+C", menu_edit)
        act("Copy scene as JSON", self._copy_json, menu=menu_edit)
        act("Copy preview image", self._copy_image, "Ctrl+Shift+C", menu_edit)
        act("Paste parameters", self._paste_params, "Ctrl+V", menu_edit, toolbar=True)
        tb.addSeparator()
        act("Refresh preview", self._refresh, "F5", menu_render, toolbar=True)
        act("Final render (Calculate 3D)...", self._final_render, "Ctrl+R", menu_render)
        act("Stereo render...", self._stereo, menu=menu_render)
        menu_render.addSeparator()
        for name in ("navigator", "animation", "mutagen", "batch", "voxel"):
            act(f"{TITLES[name]}...", lambda _=False, n=name: self.tools.open(n), HOTKEYS[name], menu_tools,
                toolbar=name in ("navigator", "animation", "mutagen"))
        act("Main window", lambda: (self.raise_(), self.activateWindow()), "Ctrl+1", menu_view)
        for tab_name, key in PANEL_HOTKEYS.items():
            act(f"{tab_name.replace('&&', '&')} panel", lambda _=False, t=tab_name: self._show_tab(t), key, menu_view)
        menu_view.addSeparator()
        menu_view.addAction(self.messages.toggleViewAction())
        act("Reset window layout", self._reset_layout, menu=menu_view)
        self.theme_menu = menu_prefs.addMenu("Visual theme")
        group = QActionGroup(self)
        current = str(self.settings.value("theme", "glossy (dark)"))
        for name in theme_names():
            a = QAction(name, self, checkable=True)
            a.setChecked(name == current)
            a.triggered.connect(lambda _=False, n=name: self._set_theme(n))
            group.addAction(a)
            self.theme_menu.addAction(a)
        act("About pymb3d", self._about, menu=menu_help)
        act("Navigation help", self._nav_help, "F1", menu_help)
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

    def _show_tab(self, name: str) -> None:
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == name.replace("&&", "&"):
                self.tabs.setCurrentIndex(i)
        self.raise_()
        self.activateWindow()

    # --------------------------------------------------------------- layout
    def _restore_layout(self) -> None:
        geo = self.settings.value("geometry")
        state = self.settings.value("window_state")
        if geo:
            self.restoreGeometry(geo)
        if state:
            self.restoreState(state)

    def _reset_layout(self) -> None:
        self.settings.remove("geometry")
        self.settings.remove("window_state")
        self.messages.show()
        self.resize(1320, 860)

    def _set_theme(self, name: str) -> None:
        apply_theme(QApplication.instance(), name)
        self.settings.setValue("theme", name)

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
            self.messages.log(f"depth of field focus set to {self.model.scene.post.dof_focus:.3f}")
            return
        if mods & Qt.KeyboardModifier.ControlModifier:
            self.model.scene.julia = True
            self.model.scene.julia_c = (float(point[0]), float(point[1]), float(point[2]), 0.0)
            self.formula_panel.load()
            self.model.touch("formulas")
            self.messages.log(f"Julia seed set to {point}")
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
            self._update_title()
            return
        self.controller.request(self.model.scene, interactive=(kind == "camera"))
        self._update_title()

    def _update_title(self) -> None:
        name = Path(self.model.path).name if self.model.path else "untitled"
        sc = self.model.scene
        self.setWindowTitle(f"pymb3d - {name}{' *' if self.model.dirty else ''}  [{sc.width} x {sc.height}]")

    @Slot(object)
    def _result_ready(self, result) -> None:
        self.info_panel.show_result(result)

    @Slot(str)
    def _status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)
        if msg.startswith(("preview ", "compiling", "preview failed")):
            self.messages.log(msg)

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
        from .open_dialog import SceneOpenDialog

        recent = self._recent()
        dlg = SceneOpenDialog(self, str(Path(recent[0]).parent) if recent else "")
        if dlg.exec() and dlg.selectedFiles():
            self.open_path(dlg.selectedFiles()[0])

    def open_path(self, path: str) -> None:
        try:
            scene, warnings = load_scene(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.model.replace(scene, path)
        self._update_title()
        self._remember(path)
        self.messages.log(f"opened {path}: {', '.join(s.name + ' x' + str(s.iterations) for s in scene.active_slots)}")
        self._show_notes(warnings)

    def _show_notes(self, warnings: list[str]) -> None:
        """Import notes go to the messages dock (no modal box: the preview keeps rendering)."""
        if not warnings:
            return
        self.messages.log_lines("  note: " + w for w in warnings)
        self._status("; ".join(warnings[:2]))
        self.messages.show()

    def _save(self) -> None:
        if self.model.path and self.model.path.lower().endswith(".json"):
            save_scene(self.model.scene, self.model.path)
            self.model.dirty = False
            self._update_title()
            self.messages.log(f"saved {self.model.path}")
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
            self._remember(path)
            self.messages.log(f"saved {path}")

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
            QGuiApplication.clipboard().setImage(numpy_to_qimage(res.image))
            self._status("preview image copied to the clipboard")

    def _export_m3p(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Mandelbulb3D parameters", "scene.m3p",
                                              "Mandelbulb3D parameters (*.m3p)")
        if path:
            if not path.lower().endswith(".m3p"):
                path += ".m3p"
            save_scene(self.model.scene, path)
            self.messages.log(f"exported {path}")

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
            self.post_panel.load()
            self.model.touch("lighting")

    def _save_lighting(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save lighting preset", "lighting.json", "Lighting (*.json)")
        if path:
            save_lighting(self.model.scene.lighting, path)

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
        self._open_render_dialog()

    def _stereo(self) -> None:
        self._open_render_dialog(stereo=self.render_panel.stereo_mode(), eye_distance=self.render_panel.eye_distance())

    def _open_render_dialog(self, **kwargs) -> None:
        dlg = RenderDialog(self.model.scene, self.render_panel.final_choice, self, model=self.model,
                           messages=self.messages.log, **kwargs)
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

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

    # ----------------------------------------------------------------- help
    def _about(self) -> None:
        from .. import __version__
        from ..formulas import registry

        QMessageBox.about(self, "About pymb3d",
                          f"<b>pymb3d {__version__}</b><br>Mandelbulb3D re-implemented in Python.<br><br>"
                          f"{len(registry.names())} formulas, devices:<br>" + "<br>".join(d.label for d in self.devices))

    def _nav_help(self) -> None:
        QMessageBox.information(self, "Navigation", (
            "Preview: left drag looks around, right drag pans, the wheel moves forward / back, Shift + wheel zooms.\n"
            "Keys: W/S forward/back, A/D sideways, Q/E up/down, arrows turn, +/- zoom (Shift: bigger steps).\n"
            "Double click centres the midpoint on the surface; Shift + double click sets the depth of field focus;\n"
            "Ctrl + double click sets the Julia seed.\n\n"
            "Windows: Ctrl+1 main, Ctrl+2 Navigator, Ctrl+3 Animation, Ctrl+4 MutaGen, Ctrl+5 Batch, Ctrl+6 Voxel,\n"
            "Ctrl+7/8/9 the Formulas / Lighting / Maps panels, Ctrl+R final render, F5 refresh."))

    # ------------------------------------------------------------- devices
    @Slot(object)
    def _change_preview_device(self, choice: DeviceChoice) -> None:
        self.preview_choice = choice
        self.controller.change_device(choice)
        self.messages.log(f"preview device: {choice.label}")

    @Slot(int)
    def _change_preview_width(self, w: int) -> None:
        self.controller.preview_width = int(w)
        self.controller.request(self.model.scene)

    @Slot(float)
    def _change_preview_quality(self, mul: float) -> None:
        self.controller.de_stop_mul = float(mul)
        self.controller.request(self.model.scene)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("window_state", self.saveState())
        self.controller.shutdown()
        self.tools.close_all()
        super().closeEvent(event)
