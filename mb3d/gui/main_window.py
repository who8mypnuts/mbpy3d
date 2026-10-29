"""The main window: image view, parameter docks, menus, render plumbing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QTabWidget,
)

from mb3d import __version__
from mb3d.params import Scene
from mb3d.render import RenderResult

from .image_view import ImageView
from .model import SceneModel
from .panels.calc_panel import CalcPanel
from .panels.camera_panel import CameraPanel
from .panels.formula_panel import FormulaPanel
from .panels.lighting_panel import LightingPanel
from .panels.render_panel import RenderPanel
from .render_controller import RenderController


class MainWindow(QMainWindow):
    def __init__(self, scene: Scene | None = None, auto_preview: bool = True):
        super().__init__()
        self.setWindowTitle(f"Mandelbulb3D (Python) {__version__}")
        self.resize(1400, 900)
        self.model = SceneModel(scene, self)
        self.controller = RenderController(self)
        self.last_result: RenderResult | None = None
        self.view = ImageView()
        self.setCentralWidget(self.view)
        self._build_docks()
        self._build_menus()
        self._build_statusbar()
        self._connect()
        self.render_panel.auto_preview.setChecked(auto_preview)
        self.controller.probe_devices()
        self.controller.warm_up()
        QTimer.singleShot(0, self._initial_preview)

    # ------------------------------------------------------------ layout
    def _build_docks(self) -> None:
        self.tabs = QTabWidget()
        self.camera_panel = CameraPanel(self.model)
        self.calc_panel = CalcPanel(self.model)
        self.formula_panel = FormulaPanel(self.model)
        self.lighting_panel = LightingPanel(self.model)
        self.render_panel = RenderPanel(self.model)
        for w, t in ((self.render_panel, "Render"), (self.camera_panel, "Camera"), (self.formula_panel, "Formulas"),
                     (self.calc_panel, "Calculation"), (self.lighting_panel, "Lighting")):
            self.tabs.addTab(w, t)
        dock = QDockWidget("Parameters", self)
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(380)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _build_menus(self) -> None:
        m = self.menuBar().addMenu("&File")
        self._action(m, "&New", self.new_scene, QKeySequence.New)
        self._action(m, "&Open parameters…", self.open_file, QKeySequence.Open)
        self._action(m, "&Save parameters", self.save_file, QKeySequence.Save)
        self._action(m, "Save parameters &as…", self.save_file_as, QKeySequence.SaveAs)
        self._action(m, "Copy parameters as text", self.copy_text)
        self._action(m, "Paste parameters from text", self.paste_text)
        m.addSeparator()
        self._action(m, "Save &image…", self.save_image, "Ctrl+E")
        m.addSeparator()
        self._action(m, "&Quit", self.close, QKeySequence.Quit)
        r = self.menuBar().addMenu("&Render")
        self._action(r, "&Preview", self.preview_now, "F5")
        self._action(r, "&Final render", self.final_render, "F9")
        self._action(r, "&Cancel", self.controller.cancel_all, "Escape")
        v = self.menuBar().addMenu("&View")
        self._action(v, "&Fit", self.view.fit, "Ctrl+0")
        self._action(v, "Actual &size", self.view.actual_size, "Ctrl+1")
        h = self.menuBar().addMenu("&Help")
        self._action(h, "&Devices", self.show_devices)
        self._action(h, "&About", self.about)

    def _action(self, menu, text, slot, shortcut=None) -> QAction:
        a = QAction(text, self)
        if shortcut is not None:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    def _build_statusbar(self) -> None:
        self.status = QLabel("ready")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(250)
        self.progress.setVisible(False)
        self.device_label = QLabel("")
        sb = self.statusBar()
        sb.addWidget(self.status, 1)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.device_label)

    def _connect(self) -> None:
        c = self.controller
        c.preview_pass.connect(self._show_pass)
        c.final_pass.connect(self._show_pass)
        c.preview_finished.connect(self._finished)
        c.final_finished.connect(self._finished)
        c.progress.connect(self._progress)
        c.failed.connect(self._failed)
        c.busy_changed.connect(self._busy)
        c.devices_found.connect(self._devices)
        c.message.connect(self.status.setText)
        self.model.changed.connect(self._scene_changed)
        self.model.file_changed.connect(self._title)
        self.view.rotated.connect(self.model.rotate)
        self.view.zoomed.connect(self.model.zoom_by)
        self.view.moved.connect(self.model.move)
        rp = self.render_panel
        rp.render_final.connect(self.final_render)
        rp.render_preview.connect(self.preview_now)
        rp.cancel.connect(c.cancel_all)
        rp.settings_changed.connect(self._scene_changed)
        rp.save_image.connect(self.save_image)

    # ------------------------------------------------------------ render
    def _initial_preview(self) -> None:
        if self.render_panel.auto_preview.isChecked():
            self.preview_now()

    def preview_now(self) -> None:
        self.controller.request_preview(self.model.scene, self.render_panel.preview_settings())

    def final_render(self) -> None:
        self.controller.cancel_previews()
        self.status.setText("final render…")
        self.controller.request_final(self.model.scene, self.render_panel.final_settings())

    def _scene_changed(self, what: str = "") -> None:
        if self.render_panel.auto_preview.isChecked():
            self.preview_now()

    def _show_pass(self, result: RenderResult, scale: float) -> None:
        self.last_result = result
        self.view.set_image(result.image, (self.model.scene.width, self.model.scene.height))
        st = result.stats
        self.status.setText(f"{result.backend} ({result.precision}) {result.width}x{result.height} "
                            f"march {result.march_seconds:.2f}s paint {result.paint_seconds:.2f}s "
                            f"steps {st.get('avg_steps', 0):.1f} its {st.get('avg_iterations', 0):.1f}")

    def _finished(self, result: RenderResult) -> None:
        self._show_pass(result, 1.0)

    def _progress(self, done: int, total: int, scale: float, kind: str) -> None:
        self.progress.setVisible(True)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.progress.setFormat(f"{kind} {scale:g}x  %p%")

    def _busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        self.render_panel.cancel_button.setEnabled(busy)

    def _failed(self, message: str) -> None:
        first = message.splitlines()[0] if message else "render failed"
        self.status.setText(f"error: {first}")
        print(message)

    def _devices(self, report) -> None:
        names = "; ".join(str(d) for d in report.available)
        errs = "; ".join(f"{k}: {v}" for k, v in report.errors.items())
        self.device_label.setText(" | ".join(d.kind for d in report.available))
        self.render_panel.device_label.setText(f"available: {names}" + (f"\nunavailable: {errs}" if errs else ""))
        self._device_report = report

    # -------------------------------------------------------------- files
    def _title(self, path: str = "") -> None:
        name = Path(path).name if path else "untitled"
        self.setWindowTitle(f"{name} - Mandelbulb3D (Python) {__version__}")

    def new_scene(self) -> None:
        self.model.set_scene(Scene())

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open parameters", "", "MB3D parameters (*.m3p *.m3i *.txt);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path: str) -> None:
        try:
            self.model.open(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(e))

    def save_file(self) -> None:
        if self.model.path is None:
            return self.save_file_as()
        self.model.save()

    def save_file_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save parameters", "", "MB3D parameters (*.m3p);;Text parameters (*.txt)")
        if path:
            self.model.save(path)

    def copy_text(self) -> None:
        from mb3d.params import to_text

        QApplication.clipboard().setText(to_text(self.model.scene))
        self.status.setText("parameters copied as text")

    def paste_text(self) -> None:
        from mb3d.params.m3p_reader import parse
        from mb3d.params.m3p_text import raw_from_text

        text = QApplication.clipboard().text()
        try:
            raw, title = raw_from_text(text)
            scene = parse(raw)
            scene.title = title
            self.model.set_scene(scene)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Paste failed", str(e))

    def save_image(self) -> None:
        if self.last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save image", "", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            from mb3d.io.image_io import save_image

            save_image(self.last_result.image, path)
            self.status.setText(f"saved {path}")

    def show_devices(self) -> None:
        rep = getattr(self, "_device_report", None)
        text = "probing…" if rep is None else "\n".join(str(d) for d in rep.available) + "\n\n" + "\n".join(f"{k}: {v}" for k, v in rep.errors.items())
        QMessageBox.information(self, "Render devices", text)

    def about(self) -> None:
        QMessageBox.about(self, "About", f"Mandelbulb3D in Python {__version__}\n\nnumba (CPU), OpenGL (preview), OpenCL (final) back ends "
                                         "from one kernel source; PySide6 GUI.")

    def closeEvent(self, event) -> None:
        self.controller.shutdown()
        super().closeEvent(event)
