"""The scene: camera, hybrid formula slots, calculation settings, lighting."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .camera import Camera
from .lights import Lighting

MAX_SLOTS = 6


@dataclass
class FormulaSlot:
    """One slot of the alternating hybrid: ``iterations`` consecutive
    iterations of the formula ``name`` with the option values ``params``
    (missing options take the formula's defaults).  0 iterations disables
    the slot."""

    name: str = "Integer Power"
    iterations: int = 1
    params: dict[str, float] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.iterations > 0

    def to_dict(self) -> dict:
        return {"name": self.name, "iterations": self.iterations, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict) -> "FormulaSlot":
        return cls(str(d.get("name", "Integer Power")), int(d.get("iterations", 1)),
                   {k: float(v) for k, v in d.get("params", {}).items()})


@dataclass
class AOSettings:
    #: "none", "ssao" (screen space, after the march) or "de" (distance
    #: estimation AO, in the kernel)
    mode: str = "ssao"
    #: strength 0..2
    strength: float = 1.0
    #: ssao radius in pixels of the final image
    radius: float = 32.0
    #: DE AO: number of samples along the normal
    samples: int = 5


@dataclass
class Scene:
    width: int = 640
    height: int = 480
    camera: Camera = field(default_factory=Camera)
    slots: list[FormulaSlot] = field(default_factory=lambda: [FormulaSlot("Integer Power", 1, {"power": 8})])
    #: alternating hybrid: after the last active slot continue from this slot
    repeat_from: int = 0
    #: max iterations
    iterations: int = 12
    min_iterations: int = 0
    #: bailout radius (not squared); None = the formulas' default
    rstop: float | None = None
    #: DE stop in pixels (`sDEstop`)
    de_stop: float = 1.0
    #: raystep multiplier (`mZstepDiv`, 0.001..1)
    z_step_div: float = 0.5
    raystep_limiter: float = 1.0
    #: binary search steps after the DE stop
    bin_search_steps: int = 6
    #: grow the DE stop with the distance from the camera
    vary_de_stop_on_fov: bool = True
    first_step_random: bool = False
    #: use the analytic DE of the formulas when all slots provide one
    analytic_de: bool = True
    julia: bool = False
    julia_c: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    #: 4D rotation angles in radians (XW, YW, ZW planes)
    rot_4d: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: cutting planes: bit 0..2 = x, y, z plane on
    cut_options: int = 0
    cut_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: colour from the orbit trap (min |z|) instead of the smooth iteration count
    color_on_otrap: bool = False
    ao: AOSettings = field(default_factory=AOSettings)
    shadows: bool = False
    lighting: Lighting = field(default_factory=Lighting)
    title: str = ""

    # ----------------------------------------------------------- derived
    @property
    def active_slots(self) -> list[FormulaSlot]:
        return [s for s in self.slots if s.active]

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def copy(self) -> "Scene":
        return copy.deepcopy(self)

    def resized(self, width: int, height: int) -> "Scene":
        s = self.copy()
        s.width, s.height = int(width), int(height)
        return s

    # ------------------------------------------------------------ (de)serialise
    def to_dict(self) -> dict:
        return {
            "format": "pymb3d-scene-1",
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "camera": self.camera.to_dict(),
            "slots": [s.to_dict() for s in self.slots],
            "repeat_from": self.repeat_from,
            "iterations": self.iterations,
            "min_iterations": self.min_iterations,
            "rstop": self.rstop,
            "de_stop": self.de_stop,
            "z_step_div": self.z_step_div,
            "raystep_limiter": self.raystep_limiter,
            "bin_search_steps": self.bin_search_steps,
            "vary_de_stop_on_fov": self.vary_de_stop_on_fov,
            "first_step_random": self.first_step_random,
            "analytic_de": self.analytic_de,
            "julia": self.julia,
            "julia_c": list(self.julia_c),
            "rot_4d": list(self.rot_4d),
            "cut_options": self.cut_options,
            "cut_pos": list(self.cut_pos),
            "color_on_otrap": self.color_on_otrap,
            "ao": {"mode": self.ao.mode, "strength": self.ao.strength, "radius": self.ao.radius, "samples": self.ao.samples},
            "shadows": self.shadows,
            "lighting": self.lighting.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        s = cls()
        s.title = str(d.get("title", ""))
        s.width = int(d.get("width", s.width))
        s.height = int(d.get("height", s.height))
        s.camera = Camera.from_dict(d.get("camera", {}))
        s.slots = [FormulaSlot.from_dict(x) for x in d.get("slots", [])] or s.slots
        for key in ("repeat_from", "iterations", "min_iterations", "bin_search_steps", "cut_options"):
            if key in d:
                setattr(s, key, int(d[key]))
        for key in ("de_stop", "z_step_div", "raystep_limiter"):
            if key in d:
                setattr(s, key, float(d[key]))
        for key in ("vary_de_stop_on_fov", "first_step_random", "analytic_de", "julia", "color_on_otrap", "shadows"):
            if key in d:
                setattr(s, key, bool(d[key]))
        s.rstop = None if d.get("rstop") is None else float(d["rstop"])
        s.julia_c = tuple(float(v) for v in d.get("julia_c", s.julia_c))  # type: ignore[assignment]
        s.rot_4d = tuple(float(v) for v in d.get("rot_4d", s.rot_4d))  # type: ignore[assignment]
        s.cut_pos = tuple(float(v) for v in d.get("cut_pos", s.cut_pos))  # type: ignore[assignment]
        ao = d.get("ao", {})
        s.ao = AOSettings(str(ao.get("mode", "ssao")), float(ao.get("strength", 1.0)), float(ao.get("radius", 32.0)),
                          int(ao.get("samples", 5)))
        s.lighting = Lighting.from_dict(d.get("lighting", {}))
        return s


def default_scene() -> Scene:
    """A Mandelbulb power 8 with a camera looking at it."""
    s = Scene()
    s.camera = Camera(mid=np.array([0.0, 0.0, 0.0]), zoom=1.0, fov_y=60.0, z_start=-3.0, z_end=4.0)
    s.camera.set_angles_deg(-20.0, 30.0, 0.0)
    s.slots = [FormulaSlot("Integer Power", 1, {"power": 8.0, "z_mul": -1.0})]
    s.iterations = 12
    s.de_stop = 1.0
    s.z_step_div = 0.5
    return s
