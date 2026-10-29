"""The camera, with Mandelbulb3D's conventions.

MB3D's rays start on a plane (the *start plane* at ``z_start``, which lies
``(z_start - mid.z)`` world units in front of the midpoint along the view axis)
and fan out with the field of view.  Distances along a ray are counted in
*pixel units*: the world size of one pixel on the start plane is
``step_width = 2.1345 / (zoom * width)``.

The camera matrix ``vgrads`` has the world directions of screen X, screen Y
and the view axis as rows (orthonormal here; the kernel gets them scaled to
``step_width``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .vec import (
    Mat3,
    angles_from_vgrads,
    normalise_matrix_to,
    normalize,
    rotation_about_axis,
    vgrads_from_angles,
)


@dataclass
class Camera:
    mid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    zoom: float = 1.0
    fov_y: float = 60.0  # degrees
    z_start: float = -3.0
    z_end: float = 4.0
    vgrads: np.ndarray = field(default_factory=lambda: np.eye(3))

    # ------------------------------------------------------------------ angles
    @property
    def angles(self) -> tuple[float, float, float]:
        """Rotation angles (radians) reproducing ``vgrads``."""
        return angles_from_vgrads(self.vgrads)

    def set_angles(self, ax: float, ay: float, az: float) -> None:
        self.vgrads = vgrads_from_angles(ax, ay, az)

    @property
    def angles_deg(self) -> tuple[float, float, float]:
        return tuple(math.degrees(a) for a in self.angles)  # type: ignore[return-value]

    def set_angles_deg(self, ax: float, ay: float, az: float) -> None:
        self.set_angles(math.radians(ax), math.radians(ay), math.radians(az))

    # --------------------------------------------------------------- geometry
    @property
    def right(self) -> np.ndarray:
        return normalize(self.vgrads[0])

    @property
    def up(self) -> np.ndarray:
        return normalize(self.vgrads[1])

    @property
    def forward(self) -> np.ndarray:
        return normalize(self.vgrads[2])

    def step_width(self, width: int) -> float:
        return 2.1345 / (self.zoom * max(1, width))

    def view_depth(self) -> float:
        """World distance from the start plane to the far end."""
        return max(1e-12, self.z_end - self.z_start)

    # ------------------------------------------------------------- navigation
    def rotate(self, axis_local: np.ndarray, angle: float) -> None:
        """Rotate the camera about one of its own axes (``axis_local`` is
        expressed in camera space, e.g. ``[0, 1, 0]`` = turn left/right)."""
        axis_world = np.asarray(axis_local, dtype=float) @ self.vgrads
        r = rotation_about_axis(axis_world, angle)
        self.vgrads = normalise_matrix_to(1.0, (r @ self.vgrads.T).T)

    def move(self, delta_local: np.ndarray) -> None:
        """Move the midpoint by ``delta_local`` camera-space world units."""
        self.mid = self.mid + np.asarray(delta_local, dtype=float) @ self.vgrads

    def move_pixels(self, dx: float, dy: float, dz: float, width: int) -> None:
        """Move in screen pixel units (as the Navigator does)."""
        sw = self.step_width(width)
        self.move(np.array([dx * sw, dy * sw, dz * sw]))

    def copy(self) -> "Camera":
        return Camera(self.mid.copy(), self.zoom, self.fov_y, self.z_start, self.z_end, self.vgrads.copy())

    # ------------------------------------------------------------ kernel values
    def kernel_setup(self, width: int, height: int, raystep_limiter: float = 1.0, z_step_div: float = 0.5,
                     vary_de_stop_on_fov: bool = True) -> "CameraSetup":
        return CameraSetup.build(self, width, height, raystep_limiter, z_step_div, vary_de_stop_on_fov)

    def to_dict(self) -> dict:
        return {
            "mid": [float(v) for v in self.mid],
            "zoom": self.zoom,
            "fov_y": self.fov_y,
            "z_start": self.z_start,
            "z_end": self.z_end,
            "vgrads": [[float(v) for v in row] for row in self.vgrads],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        c = cls()
        c.mid = np.asarray(d.get("mid", [0, 0, 0]), dtype=float)
        c.zoom = float(d.get("zoom", 1.0))
        c.fov_y = float(d.get("fov_y", 60.0))
        c.z_start = float(d.get("z_start", -3.0))
        c.z_end = float(d.get("z_end", 4.0))
        if "vgrads" in d:
            c.vgrads = normalise_matrix_to(1.0, np.asarray(d["vgrads"], dtype=float))
        elif "angles_deg" in d:
            c.set_angles_deg(*d["angles_deg"])
        return c


@dataclass
class CameraSetup:
    """Everything the march kernel needs about the camera (in the units MB3D
    uses: positions in world units, ray distances in pixels)."""

    width: int
    height: int
    step_width: float
    ystart: np.ndarray      # world position of pixel (0, 0) on the start plane
    vgrads: np.ndarray      # rows scaled to step_width
    fov_y: float            # radians
    fovx_mul: float
    fovx_off: float
    zend: float             # far end in pixel units
    de_stop_factor: float   # growth of the DE stop per pixel of depth
    mh04zsd: float          # step length limiter

    @classmethod
    def build(cls, cam: Camera, width: int, height: int, raystep_limiter: float, z_step_div: float,
              vary_de_stop_on_fov: bool) -> "CameraSetup":
        sw = cam.step_width(width)
        fov_y = math.radians(cam.fov_y)
        vg = normalise_matrix_to(sw, cam.vgrads)
        z1 = (cam.z_start - cam.mid[2]) / sw
        xh, yh = width * 0.5, height * 0.5
        ystart = cam.mid + z1 * vg[2] - yh * vg[1] - xh * vg[0]
        zend = max(1e-10, (cam.z_end - cam.z_start) / sw)
        if vary_de_stop_on_fov:
            x1 = 0.01 * height / (math.sin(0.01) * max(1.0 / 65535.0, fov_y))
            x2 = sw * (x1 + zend) / x1
            de_stop_factor = (x2 - sw) / (sw * max(1e-6, zend))
        else:
            de_stop_factor = 0.0
        mh04zsd = max(width, height) * 0.5 * math.sqrt(max(1e-4, z_step_div) + 0.001) * max(0.01, raystep_limiter)
        return cls(
            width=width,
            height=height,
            step_width=sw,
            ystart=ystart,
            vgrads=vg,
            fov_y=fov_y,
            fovx_mul=fov_y / height,
            fovx_off=0.5 * width,
            zend=zend,
            de_stop_factor=de_stop_factor,
            mh04zsd=mh04zsd,
        )

    def ray(self, ix: int, iy: int) -> tuple[np.ndarray, np.ndarray]:
        """Start position and per-pixel-unit direction of pixel (ix, iy)."""
        cafy = (iy / self.height - 0.5) * self.fov_y
        cafx = (self.fovx_off - (ix + 1)) * self.fovx_mul
        v = np.array([-math.sin(cafx), math.sin(cafy), math.cos(cafy) * math.cos(cafx)])
        v = v / np.linalg.norm(v)
        start = self.ystart + self.vgrads[0] * ix + self.vgrads[1] * iy
        return start, v @ self.vgrads
