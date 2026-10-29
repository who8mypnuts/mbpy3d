"""The scene as a Qt model: panels edit ``model.scene`` and call
``model.touch(kind)``; the main window listens to ``changed``.

``kind`` tells listeners how much work a change needs:
``"lighting"`` is re-shaded from the kept G-buffer, ``"camera"`` marks an
interactive change (coarse pass first), everything else re-marches.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from ..core.scene import Scene


class SceneModel(QObject):
    changed = Signal(str)
    #: the whole scene was replaced (load, reset): panels reload their widgets
    replaced = Signal()

    def __init__(self, scene: Scene) -> None:
        super().__init__()
        self.scene = scene
        self.path: str | None = None
        self.dirty = False
        self._undo: list[dict] = []
        self._redo: list[dict] = []

    def touch(self, kind: str = "all") -> None:
        self.dirty = True
        self.changed.emit(kind)

    def snapshot(self) -> None:
        """Remember the state for undo (called before a discrete edit)."""
        self._undo.append(self.scene.to_dict())
        if len(self._undo) > 100:
            self._undo.pop(0)
        self._redo.clear()

    def replace(self, scene: Scene, path: str | None = None, remember: bool = True) -> None:
        if remember:
            self.snapshot()
        self.scene = scene
        self.path = path
        self.dirty = False
        self.replaced.emit()
        self.changed.emit("all")

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.scene.to_dict())
        self.scene = Scene.from_dict(self._undo.pop())
        self.replaced.emit()
        self.changed.emit("all")
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.scene.to_dict())
        self.scene = Scene.from_dict(self._redo.pop())
        self.replaced.emit()
        self.changed.emit("all")
        return True
