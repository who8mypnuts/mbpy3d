"""The scene: MB3D's ``TMandHeader10`` + formula addon as dataclasses."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import IntEnum

from .lighting import Lighting

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

IDENTITY3: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
HEADER_SIZE = 840
ADDON_SIZE = 8 + 6 * 188
MAND_ID = 44


class CameraOptic(IntEnum):
    COMMON = 0
    PLANAR = 1
    PANORAMA = 2


class InsideMode(IntEnum):
    OUTSIDE = 0
    INSIDE = 1
    BOTH = 2


@dataclass
class FormulaSlot:
    """One of the six hybrid slots (``THAformula``)."""

    name: str = "Integer Power"
    iterations: int = 0  # 0: unused, < 0: not counted
    fnr: int = 0  # internal number, >= 20 custom
    options: list[float] = field(default_factory=lambda: [0.0] * 16)
    option_count: int = 0

    @property
    def used(self) -> bool:
        return self.iterations != 0


@dataclass
class SsaoParams:
    threshold: float = 0.5
    border_mirror: float = 0.0
    t0: bool = False
    random: int = 0
    bits15: bool = False


@dataclass
class DeaoParams:
    quality: int = 0
    dither: int = 0
    max_len: float = 1.0
    first_step_random: bool = False


@dataclass
class ShadowParams:
    lights: int = 1  # bit mask of the lights
    soft: bool = False
    soft_radius: float = 1.0
    max_len_mul: float = 1.0
    set_cos: bool = False


@dataclass
class DofParams:
    z_sharp: float = 0.5
    z_sharp2: float = 0.5
    clip_r: float = 25.0
    aperture: float = 0.02
    passes: int = 1
    forward: bool = False


@dataclass
class DeCombParams:
    kind: int = 1
    end1: int = 0
    start2: int = 1
    end2: int = 1
    repeat2: int = 1
    iterations2: int = 60
    smooth: float = 0.5
    mix_pow: float = 2.0
    mix_color: int = 0


@dataclass
class CalcStats:
    avg_de_steps10: int = 0
    avg_its10: int = 0
    max_its: int = 0


@dataclass
class Scene:
    width: int = 800
    height: int = 600
    iterations: int = 60
    min_iterations: int = 1
    color_on_it: int = 0
    slice_2d: int = 0
    stereo_mode: int = 0
    stereo_screen: tuple[float, float, float] = (1.0, 2.0, 0.5)
    rstop: float | None = None
    zoom: float = 1.0
    mid: Vec3 = (0.0, 0.0, 0.0)
    z_start: float = -2.0
    z_end: float = 30.0
    vgrads: Mat3 = IDENTITY3
    fov_y: float = 30.0
    optic: CameraOptic = CameraOptic.COMMON
    de_stop: float = 1.0
    z_step_div: float = 0.5
    raystep_limiter: float = 1.0
    smooth_normals: int = 0
    bin_search_steps: int = 8
    first_step_random: bool = True
    step_sub_de_stop: bool = False
    vary_de_stop_on_fov: bool = True
    normals_on_de: bool = True
    normals_on_zbuf: bool = False
    disable_analytic_de: bool = False
    julia: bool = False
    julia_c: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    rot_4d: Vec3 = (0.0, 0.0, 0.0)
    color_option: int = 0
    color_mul: float = 1.0
    dfog_on_it: int = 0
    formulas: list[FormulaSlot] = field(default_factory=lambda: [FormulaSlot("Integer Power", 1, 0, [8.0, -1.0] + [0.0] * 14, 2)])
    repeat_from: int = 0
    lighting: Lighting = field(default_factory=Lighting.start_preset)
    ao: SsaoParams | None = field(default_factory=SsaoParams)
    deao: DeaoParams | None = None
    shadows: ShadowParams | None = None
    cut_options: int = 0
    cut_pos: Vec3 = (0.0, 0.0, 0.0)
    inside: InsideMode = InsideMode.OUTSIDE
    decomb: DeCombParams | None = None
    interpolation: tuple[float, float] | None = None
    dof: DofParams | None = None
    stats: CalcStats = field(default_factory=CalcStats)
    mand_id: int = MAND_ID
    title: str = ""
    # the bytes of the file the scene came from (unknown fields survive a save)
    raw: bytes | None = field(default=None, repr=False)

    # ----------------------------------------------------------- helpers
    def copy(self) -> "Scene":
        return copy.deepcopy(self)

    def step_width(self) -> float:
        """``CalcStepWidth``: world units per pixel at the start plane."""
        return 2.1345 / (self.zoom * self.width)

    def used_formulas(self) -> list[FormulaSlot]:
        return [f for f in self.formulas if f.used]

    def effective_rstop(self, default_rstop_of) -> float:
        if self.rstop is not None and self.rstop > 1.0001:
            return self.rstop
        r = 0.0
        for f in self.used_formulas():
            r = max(r, default_rstop_of(f))
        return max(r, 1.0001)

    def scale_image(self, f: float) -> None:
        """Resize and keep the look (the DE stop is in pixels)."""
        f = max(f, 1e-4)
        w = max(16, int(round(self.width * f)))
        h = max(16, int(round(self.height * f)))
        real = w / self.width
        self.width, self.height = w, h
        self.de_stop = max(0.001, self.de_stop * real)
        if self.dof is not None:
            self.dof.clip_r = max(0.1, self.dof.clip_r * real)

    def camera_position(self) -> Vec3:
        """World position of the camera (the middle of the start plane)."""
        vz = normalize(self.vgrads[2])
        d = self.z_start - self.mid[2]
        return tuple(self.mid[k] + vz[k] * d for k in range(3))  # type: ignore[return-value]


def normalize(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        return v
    return (v[0] / n, v[1] / n, v[2] / n)


def vgrads_from_angles(ax: float, ay: float, az: float) -> Mat3:
    """``CalcVGradsFromHeader8rots``: rotate around X, then Y, then Z (radians)."""
    s1, c1 = math.sin(ax), math.cos(ax)
    s2, c2 = math.sin(ay), math.cos(ay)
    s3, c3 = math.sin(az), math.cos(az)
    rows = []
    for i in range(3):
        x1, y1, z1 = (1.0, 0.0, 0.0) if i == 0 else (0.0, 1.0, 0.0) if i == 1 else (0.0, 0.0, 1.0)
        x2 = x1
        y2 = c1 * y1 + s1 * z1
        z2 = c1 * z1 - s1 * y1
        x1 = c2 * x2 + s2 * z2
        y1 = y2
        z1 = c2 * z2 - s2 * x2
        rows.append((c3 * x1 + s3 * y1, c3 * y1 - s3 * x1, z1))
    return (rows[0], rows[1], rows[2])


def angles_from_vgrads(m: Mat3) -> Vec3:
    """The X, Y, Z rotation angles (radians) with ``vgrads_from_angles(*angles) == m``.

    From the construction: row0 = (c3 c2, -s3 c2, -s2), row1.z = -c2 s1,
    row2.z = c2 c1.
    """
    r0, r1, r2 = (normalize(row) for row in m)
    ay = math.asin(max(-1.0, min(1.0, -r0[2])))
    if abs(math.cos(ay)) < 1e-9:
        return (math.atan2(r1[0], r1[1]), ay, 0.0)
    az = math.atan2(-r0[1], r0[0])
    ax = math.atan2(-r1[2], r2[2])
    return (ax, ay, az)
