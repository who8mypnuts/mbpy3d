"""The scene: camera, hybrid formula slots, calculation settings, lighting."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

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
    #: "none", "ssao" (screen space, after the march), "de" (distance
    #: estimation AO, in the kernel) or "mc" (Monte Carlo sky visibility rays)
    mode: str = "ssao"
    #: strength 0..2
    strength: float = 1.0
    #: ssao radius in pixels of the final image; the reach of the MC rays
    radius: float = 32.0
    #: DE AO: samples along the normal; MC: rays per pixel and pass
    samples: int = 5
    #: MC: passes averaged by the final render (progressive refinement)
    passes: int = 1


@dataclass
class DeCombination:
    """Two hybrid parts combined by their distance estimates (`DE
    combination`): slots ``0..start2-1`` form part 1, ``start2..`` part 2."""

    #: 0 off, 1 minimum, 2 maximum, 3 inverted maximum, 4 smooth minimum, 5 smooth minimum 2
    kind: int = 0
    #: first slot of the second part
    start2: int = 1
    #: repeat slot of the second part (index into the scene slots)
    repeat_from2: int = 1
    #: max iterations of the second part
    iterations2: int = 12
    #: smoothing distance (world units) of the smooth minima
    smooth: float = 0.5
    #: dIFS mix: power of the distance scaling, colour source (0 both, 1 part 1, 2 part 2)
    mix_pow: float = 2.0
    mix_color: int = 0

    KINDS = ("off", "minimum", "maximum", "inverted maximum", "smooth minimum", "smooth minimum 2", "dIFS mix")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "start2": self.start2, "repeat_from2": self.repeat_from2,
                "iterations2": self.iterations2, "smooth": self.smooth, "mix_pow": self.mix_pow,
                "mix_color": self.mix_color}

    @classmethod
    def from_dict(cls, d: dict) -> "DeCombination":
        return cls(int(d.get("kind", 0)), int(d.get("start2", 1)), int(d.get("repeat_from2", 1)),
                   int(d.get("iterations2", 12)), float(d.get("smooth", 0.5)), float(d.get("mix_pow", 2.0)),
                   int(d.get("mix_color", 0)))


@dataclass
class PostSettings:
    """Image post processing (applied after the shading)."""

    brightness: float = 0.0     # -1..1
    contrast: float = 0.0       # -1..1
    saturation: float = 0.0     # -1..1
    sharpen: float = 0.0        # 0..2
    denoise: float = 0.0        # 0 = off, 1..20 (DCT noise filter strength)
    dof: bool = False
    #: focus as a fraction of the depth range 0 (start plane) .. 1 (far end)
    dof_focus: float = 0.3
    #: blur radius in pixels per unit of defocus (fraction of the depth range)
    dof_aperture: float = 8.0
    #: blur only behind the focus
    dof_forward_only: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PostSettings":
        p = cls()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p


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
    #: "step sub DE stop" (`iOptions` bit 2): steps shortened near the surface
    step_sub_de_stop: bool = False
    #: inside rendering: 0 outside, 1 inside, 2 in and outside (per pixel)
    inside_mode: int = 0
    #: interpolation hybrid: weights of the first two formulas, None = alternating hybrid
    interpolation: tuple[float, float] | None = None
    #: smooth normals 0..8 (8 = 5x5x5 cube), with the surface roughness
    smooth_normals: int = 0
    #: 2D calculation of a plane instead of the 3D march: 0 off, 1 start plane, 2 middle, 3 far end
    slice_2d: int = 0
    #: cutting planes: bit 0..2 = x, y, z plane on
    cut_options: int = 0
    cut_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: colour from the orbit trap (min |z|) instead of the smooth iteration count
    color_on_otrap: bool = False
    #: MB3D's 2nd colour choice: 0 orbit trap, 1 log(r/rold), 2..4 angles, 5 spherical
    color_option: int = 0
    color_mul: float = 1.0
    decomb: DeCombination = field(default_factory=DeCombination)
    ao: AOSettings = field(default_factory=AOSettings)
    shadows: bool = False
    lighting: Lighting = field(default_factory=Lighting)
    post: PostSettings = field(default_factory=PostSettings)
    title: str = ""
    author: str = ""
    #: folder of the file the scene came from (for maps with relative names), not stored
    base_dir: str = ""

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
            "author": self.author,
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
            "color_option": self.color_option,
            "color_mul": self.color_mul,
            "decomb": self.decomb.to_dict(),
            "ao": {"mode": self.ao.mode, "strength": self.ao.strength, "radius": self.ao.radius,
                   "samples": self.ao.samples, "passes": self.ao.passes},
            "step_sub_de_stop": self.step_sub_de_stop,
            "inside_mode": self.inside_mode,
            "interpolation": None if self.interpolation is None else list(self.interpolation),
            "smooth_normals": self.smooth_normals,
            "slice_2d": self.slice_2d,
            "shadows": self.shadows,
            "lighting": self.lighting.to_dict(),
            "post": self.post.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        s = cls()
        s.title = str(d.get("title", ""))
        s.author = str(d.get("author", ""))
        s.width = int(d.get("width", s.width))
        s.height = int(d.get("height", s.height))
        s.camera = Camera.from_dict(d.get("camera", {}))
        s.slots = [FormulaSlot.from_dict(x) for x in d.get("slots", [])] or s.slots
        for key in ("repeat_from", "iterations", "min_iterations", "bin_search_steps", "cut_options", "color_option"):
            if key in d:
                setattr(s, key, int(d[key]))
        for key in ("de_stop", "z_step_div", "raystep_limiter", "color_mul"):
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
                          int(ao.get("samples", 5)), int(ao.get("passes", 1)))
        s.step_sub_de_stop = bool(d.get("step_sub_de_stop", False))
        s.inside_mode = int(d.get("inside_mode", 0))
        ip = d.get("interpolation")
        s.interpolation = None if ip is None else (float(ip[0]), float(ip[1]))
        s.smooth_normals = int(d.get("smooth_normals", 0))
        s.slice_2d = int(d.get("slice_2d", 0))
        s.lighting = Lighting.from_dict(d.get("lighting", {}))
        s.decomb = DeCombination.from_dict(d.get("decomb", {}))
        s.post = PostSettings.from_dict(d.get("post", {}))
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
