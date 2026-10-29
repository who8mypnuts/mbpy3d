"""Camera navigation shared by the main preview and the Navigator window:
turns screen-space gestures into camera changes with MB3D's conventions
(movement in pixel units of the image, rotation by the field of view)."""
from __future__ import annotations

import math

import numpy as np

from ..core import params as PL
from ..core.scene import Scene
from ..render import RenderResult


class CameraNavigator:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene

    @property
    def cam(self):
        return self.scene.camera

    # ---------------------------------------------------------- rotations
    def rotate(self, dx: float, dy: float, view_w: int, view_h: int) -> None:
        """Turn by a mouse drag of (dx, dy) pixels over a view of the given size."""
        fov = math.radians(self.cam.fov_y)
        self.cam.rotate(np.array([0.0, 1.0, 0.0]), dx / max(1, view_w) * fov * view_w / max(1, view_h))
        self.cam.rotate(np.array([1.0, 0.0, 0.0]), -dy / max(1, view_h) * fov)

    def turn(self, yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> None:
        if yaw_deg:
            self.cam.rotate(np.array([0.0, 1.0, 0.0]), math.radians(yaw_deg))
        if pitch_deg:
            self.cam.rotate(np.array([1.0, 0.0, 0.0]), math.radians(pitch_deg))
        if roll_deg:
            self.cam.rotate(np.array([0.0, 0.0, 1.0]), math.radians(roll_deg))

    # ---------------------------------------------------------- movements
    def pan(self, dx: float, dy: float, view_w: int) -> None:
        """Move sideways by a drag of (dx, dy) view pixels."""
        scale = self.scene.width / max(1, view_w)
        self.cam.move_pixels(-dx * scale, -dy * scale, 0.0, self.scene.width)

    def move(self, dx: float, dy: float, dz: float, fraction: float = 0.05) -> None:
        """Move by (dx, dy, dz) steps of ``fraction`` of the image width."""
        step = self.scene.width * fraction
        self.cam.move_pixels(dx * step, dy * step, dz * step, self.scene.width)

    def forward(self, notches: float, fraction: float = 0.05) -> None:
        self.move(0.0, 0.0, notches, fraction)

    def zoom(self, factor: float) -> None:
        self.cam.zoom *= factor

    def zoom_towards(self, factor: float) -> None:
        """Zoom in keeping the start plane where it is: the camera moves
        closer to the middle (MB3D's zoom)."""
        self.cam.zoom *= factor

    # ------------------------------------------------------------ picking
    def surface_point(self, result: RenderResult, x: int, y: int) -> np.ndarray | None:
        """The world position under image pixel (x, y) of a render result."""
        if result.gbuffer is None:
            return None
        H, W = result.gbuffer.shape[:2]
        if not (0 <= x < W and 0 <= y < H):
            return None
        depth = float(result.gbuffer[y, x, PL.G_DEPTH])
        if depth < 0:
            return None
        sc = result.scene
        setup = sc.camera.kernel_setup(W, H, sc.raystep_limiter, sc.z_step_div, sc.vary_de_stop_on_fov)
        start, d = setup.ray(x, y)
        return start + d * depth

    def center_on(self, point: np.ndarray) -> None:
        """Put the midpoint on ``point`` keeping the camera in place."""
        cam = self.cam
        shift = point - cam.mid
        along = float(shift @ cam.forward)
        cam.mid = np.asarray(point, dtype=float)
        cam.z_start -= along
        cam.z_end -= along

    def look_at(self, point: np.ndarray) -> None:
        """Turn the camera so that ``point`` is in the middle of the view."""
        cam = self.cam
        start = cam.mid + (cam.z_start - cam.mid[2]) * cam.forward
        d = np.asarray(point, dtype=float) - start
        n = np.linalg.norm(d)
        if n < 1e-12:
            return
        f = d / n
        up = cam.up
        r = np.cross(up, f)
        if np.linalg.norm(r) < 1e-9:
            r = np.cross(np.array([1.0, 0.0, 0.0]), f)
        r /= np.linalg.norm(r)
        u = np.cross(f, r)
        cam.vgrads = np.array([r, u, f])
