"""The tool windows (Navigator, Animation, MutaGen, Batch, Voxel export):
one instance each, opened or brought to the front by the main window and
its hotkeys (Ctrl+2 .. Ctrl+6, as MB3D-rs)."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QDialog

HOTKEYS = {"navigator": "Ctrl+2", "animation": "Ctrl+3", "mutagen": "Ctrl+4", "batch": "Ctrl+5", "voxel": "Ctrl+6"}
TITLES = {"navigator": "Navigator", "animation": "Animation", "mutagen": "MutaGen (random variations)",
          "batch": "Batch render", "voxel": "Voxel / mesh export"}


class ToolWindows(QObject):
    def __init__(self, main) -> None:
        super().__init__(main)
        self.main = main
        self._open: dict[str, QDialog] = {}

    def factory(self, name: str) -> Callable[[], QDialog]:
        m = self.main
        if name == "navigator":
            from .navigator import NavigatorDialog

            return lambda: NavigatorDialog(m.model, m.preview_choice, m)
        if name == "animation":
            from .animation_dialog import AnimationDialog

            return lambda: AnimationDialog(m.model, m.render_panel.final_choice, m)
        if name == "mutagen":
            from .mutagen_dialog import MutagenDialog

            return lambda: MutagenDialog(m.model, m.preview_choice, m)
        if name == "batch":
            from .batch_dialog import BatchDialog

            return lambda: BatchDialog(m.render_panel.final_choice, m)
        if name == "voxel":
            from .voxel_dialog import VoxelDialog

            return lambda: VoxelDialog(m.model, m)
        raise KeyError(name)

    def open(self, name: str) -> QDialog:
        dlg = self._open.get(name)
        if dlg is not None and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return dlg
        dlg = self.factory(name)()
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.destroyed.connect(lambda *_: self._open.pop(name, None))
        self._open[name] = dlg
        dlg.show()
        return dlg

    def close_all(self) -> None:
        for dlg in list(self._open.values()):
            try:
                dlg.close()
            except RuntimeError:
                pass
        self._open.clear()
