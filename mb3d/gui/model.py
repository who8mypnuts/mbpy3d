"""``SceneModel``: the scene of the window with change signals.

Panels edit the scene through this object and listen to ``changed`` so
that every view stays in step; the render controller listens too.
"""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from mb3d.params import Scene, load, save, save_text
from mb3d.params.scene import angles_from_vgrads, vgrads_from_angles


class SceneModel(QObject):
    changed = Signal(str)  # what changed: "camera", "calc", "formulas", "lighting", "size", "scene"
    file_changed = Signal(str)

    def __init__(self, scene: Scene | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._scene = scene or Scene()
        self.path: Path | None = None
        self.dirty = False

    # ------------------------------------------------------------ access
    @property
    def scene(self) -> Scene:
        return self._scene

    def snapshot(self) -> Scene:
        return self._scene.copy()

    def set_scene(self, scene: Scene, path: Path | None = None) -> None:
        self._scene = scene
        self.path = path
        self.dirty = False
        self.changed.emit("scene")
        self.file_changed.emit(str(path) if path else "")

    def touch(self, what: str) -> None:
        """Announce an edit made directly on ``scene``."""
        self.dirty = True
        self.changed.emit(what)

    # ------------------------------------------------------------- files
    def open(self, path: str | Path) -> None:
        path = Path(path)
        self.set_scene(load(path), path)

    def save(self, path: str | Path | None = None, text: bool = False) -> None:
        path = Path(path or self.path)
        if text or path.suffix.lower() == ".txt":
            save_text(self._scene, path)
        else:
            save(self._scene, path)
        self.path = path
        self.dirty = False
        self.file_changed.emit(str(path))

    # ------------------------------------------------------------ camera
    def angles(self) -> tuple[float, float, float]:
        """Camera rotation in degrees (X, Y, Z)."""
        return tuple(math.degrees(a) for a in angles_from_vgrads(self._scene.vgrads))  # type: ignore[return-value]

    def set_angles(self, ax: float, ay: float, az: float) -> None:
        self._scene.vgrads = vgrads_from_angles(math.radians(ax), math.radians(ay), math.radians(az))
        self.touch("camera")

    def rotate(self, d_yaw: float, d_pitch: float, d_roll: float = 0.0) -> None:
        ax, ay, az = self.angles()
        self.set_angles(ax + d_pitch, ay + d_yaw, az + d_roll)

    def zoom_by(self, factor: float) -> None:
        self._scene.zoom = max(1e-12, self._scene.zoom * factor)
        self.touch("camera")

    def move(self, right: float, down: float, forward: float) -> None:
        """Move the camera by fractions of the view width along its axes."""
        s = self._scene
        import numpy as np

        vg = np.array(s.vgrads, dtype=np.float64)
        vg /= np.linalg.norm(vg, axis=1)[:, None]
        width_world = s.step_width() * s.width
        delta = (vg[0] * right + vg[1] * down + vg[2] * forward) * width_world
        s.mid = tuple(float(v) for v in (np.array(s.mid) + delta))
        self.touch("camera")

    def set_size(self, width: int, height: int, keep_look: bool = True) -> None:
        s = self._scene
        if keep_look and width != s.width:
            s.scale_image(width / s.width)
            s.height = max(16, height)
        else:
            s.width, s.height = max(16, width), max(16, height)
        self.touch("size")
