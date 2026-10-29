"""Lights, the surface palette and the global lighting settings.

The scalar settings keep Mandelbulb3D's slider units (documented per field)
so that `.m3p` files map onto them directly; :mod:`pymb3d.render.shading`
converts them to the multipliers it needs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

RGB = tuple[float, float, float]


def rgb_from_tcolor(c: int) -> RGB:
    """Delphi ``TColor`` ($BBGGRR) to (r, g, b) in 0..1."""
    return ((c & 0xFF) / 255.0, ((c >> 8) & 0xFF) / 255.0, ((c >> 16) & 0xFF) / 255.0)


def rgb_from_bytes(b) -> RGB:
    return (b[0] / 255.0, b[1] / 255.0, b[2] / 255.0)


@dataclass
class Light:
    enabled: bool = False
    color: RGB = (1.0, 1.0, 1.0)
    #: horizontal / vertical angle in radians (`LXpos`, `LYpos`)
    x_angle: float = 0.0
    y_angle: float = 0.0
    amplitude: float = 1.0
    #: specular exponent is ``2 << spec_func`` (0..7)
    spec_func: int = 4
    #: diffuse function 0..3 (see shading.diffuse_function)
    diff_func: int = 0
    #: angles relative to the object instead of the viewer
    relative_to_object: bool = False
    #: cast hard shadows
    shadows: bool = True
    positional: bool = False
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def direction_view(self) -> tuple[float, float, float]:
        """Unit vector *towards* the light in view space (`BuildViewVectorFOV`
        with MB3D's sign conventions)."""
        dtmp = -self.x_angle
        dtmp2 = self.y_angle
        v = (-math.sin(dtmp), math.sin(dtmp2), math.cos(dtmp2) * math.cos(dtmp))
        n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1.0
        return (-v[0] / n, -v[1] / n, -v[2] / n)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Light":
        l = cls()
        for k, v in d.items():
            if hasattr(l, k):
                setattr(l, k, tuple(v) if isinstance(v, list) else v)
        return l


@dataclass
class PaletteStop:
    #: 0..32767 along the colour gradient
    position: int = 0
    diffuse: RGB = (1.0, 1.0, 1.0)
    specular: RGB = (1.0, 1.0, 1.0)


def default_palette() -> list[PaletteStop]:
    """MB3D's start-up palette (`StartPreset`)."""
    cols = [5873889, 8837614, 0x8B491D, 2988346, 12248958, 0xFFC49F, 11287584, 14579248, 7481121]
    stops = []
    for i in range(3):
        stops.append(PaletteStop(i * 10922, rgb_from_tcolor(cols[i * 3]), rgb_from_tcolor(cols[i * 3 + 1])))
    for i in range(3, 10):
        stops.append(PaletteStop(32200 + (i - 3) * 84, stops[0].diffuse, stops[0].specular))
    return stops


def default_lights() -> list[Light]:
    lights = [Light() for _ in range(6)]
    lights[0] = Light(
        enabled=True,
        color=rgb_from_tcolor(0xA0E8FF),
        x_angle=1500.0 * math.pi / 16384.0,
        y_angle=5200.0 * math.pi / 16384.0,
        amplitude=1.0,
        spec_func=53 & 7,
        diff_func=(53 >> 4) & 3,
    )
    lights[1] = Light(
        enabled=False,
        color=rgb_from_tcolor(2307911),
        x_angle=-2822.0 * math.pi / 16384.0,
        y_angle=-7737.0 * math.pi / 16384.0,
        spec_func=16 & 7,
        diff_func=(16 >> 4) & 3,
    )
    return lights


@dataclass
class Lighting:
    lights: list[Light] = field(default_factory=default_lights)
    palette: list[PaletteStop] = field(default_factory=default_palette)
    ambient_top: RGB = rgb_from_tcolor(0x8B491D)
    ambient_bottom: RGB = rgb_from_tcolor(0xFFC49F)
    depth_color_top: RGB = rgb_from_tcolor(0x8B491D)
    depth_color_bottom: RGB = rgb_from_tcolor(0xFFC49F)
    dyn_fog_color: RGB = (1.0, 1.0, 1.0)
    #: TBpos[3]: fog offset (128 = 0)
    fog_offset: float = 128.0
    #: TBpos[4]: depth fog amount (0 = none)
    depth_fog: float = 27.0
    #: TBpos[5]: diffuse multiplier (50 = 1.0)
    diffuse: float = 120.0
    #: TBpos[6]: dynamic fog amount (53 = off)
    dyn_fog: float = 53.0
    #: TBpos[7]: specular multiplier (50 = 1.0)
    specular: float = 50.0
    #: TBpos[8]: ambient multiplier (90 = 1.0)
    ambient: float = 90.0
    #: TBpos[9] / TBpos[10]: colour mapping start / end
    color_start: float = 68.0
    color_end: float = 72.0
    #: TBpos[11]: ambient shadow (AO) amplitude (53 = 1.0)
    amb_shadow: float = 53.0
    #: gamma slider (32 = neutral, < 32 darker, > 32 brighter)
    gamma: float = 32.0
    color_cycling: bool = False
    color_on_otrap: bool = False
    far_fog: bool = False
    no_color_interpolation: bool = False
    #: `VarColZpos`: colour shift with depth
    var_col_z: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Lighting":
        l = cls()
        for k, v in d.items():
            if k == "lights":
                l.lights = [Light.from_dict(x) for x in v]
                while len(l.lights) < 6:
                    l.lights.append(Light())
            elif k == "palette":
                l.palette = [PaletteStop(int(x["position"]), tuple(x["diffuse"]), tuple(x["specular"])) for x in v]
            elif hasattr(l, k):
                setattr(l, k, tuple(v) if isinstance(v, list) else v)
        return l

    # convenient multipliers (see shading)
    @property
    def diffuse_mul(self) -> float:
        return self.diffuse * 0.02

    @property
    def specular_mul(self) -> float:
        return max(0.004, self.specular * 0.02)

    @property
    def ambient_mul(self) -> float:
        return self.ambient / 90.0

    @property
    def amb_shadow_mul(self) -> float:
        return self.amb_shadow / 53.0
