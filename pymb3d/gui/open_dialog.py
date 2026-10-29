"""The Open dialog with a preview: a picture for PNG files, and for
parameter files their size, iterations and formulas (MB3D-rs shows a
small picture; a parameter file is not rendered here to keep the dialog
quick)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QGridLayout, QLabel, QVBoxLayout, QWidget

FILTER = ("Scenes (*.json *.m3p *.m3i *.txt *.png);;pymb3d scene (*.json);;"
          "Mandelbulb3D parameters (*.m3p *.m3i *.txt);;rendered PNG with scene (*.png)")


def describe(path: str) -> str:
    """A short description of a scene file."""
    try:
        from ..io import load_scene

        sc, warnings = load_scene(path)
    except Exception as e:  # noqa: BLE001
        return f"not a scene file\n{e}"
    slots = ", ".join(f"{s.name} x{s.iterations}" for s in sc.active_slots)
    lines = [f"{sc.width} x {sc.height}", f"{sc.iterations} iterations", f"zoom {sc.camera.zoom:.4g}", slots]
    if sc.decomb.kind:
        lines.append("DE combination")
    if sc.title:
        lines.insert(0, sc.title)
    missing = [w for w in warnings if "not available" in w]
    if missing:
        lines.append(f"{len(missing)} formula(s) missing")
    return "\n".join(lines)


class SceneOpenDialog(QFileDialog):
    def __init__(self, parent=None, directory: str = "") -> None:
        super().__init__(parent, "Open scene", directory, FILTER)
        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.preview = QLabel()
        self.preview.setFixedSize(240, 180)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background: #202020; border: 1px solid #555;")
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setFixedWidth(240)
        self.info.setAlignment(Qt.AlignmentFlag.AlignTop)
        side = QWidget()
        lay = QVBoxLayout(side)
        lay.addWidget(self.preview)
        lay.addWidget(self.info)
        lay.addStretch(1)
        grid = self.layout()
        if isinstance(grid, QGridLayout):
            grid.addWidget(side, 0, grid.columnCount(), grid.rowCount(), 1)
        self.currentChanged.connect(self.show_preview)

    def show_preview(self, path: str) -> None:
        p = Path(path)
        self.preview.clear()
        self.info.clear()
        if not p.is_file():
            return
        pic = p if p.suffix.lower() in (".png", ".jpg", ".jpeg") else None
        if pic is None:
            for ext in (".png", ".jpg", ".jpeg"):
                cand = p.with_suffix(ext)
                if cand.is_file():
                    pic = cand
                    break
        if pic is not None:
            pm = QPixmap(str(pic))
            if not pm.isNull():
                self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation))
        if p.suffix.lower() in (".json", ".m3p", ".m3i", ".txt", ".png"):
            self.info.setText(describe(path))
