"""The lighting record (``TLightingParas9``) as plain dataclasses.

Scalar values keep MB3D's trackbar units so that they map one-to-one to the
bytes of a parameter file; the shading code converts them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

RGB = tuple[int, int, int]


def _rgb(c: int) -> RGB:
    """Delphi TColor ($BBGGRR) -> (r, g, b)."""
    return (c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF)


@dataclass
class Light:
    on: bool = False
    color: RGB = (255, 255, 255)
    x_angle: float = 0.0  # radians (LXpos)
    y_angle: float = 0.0  # radians (LYpos)
    amplitude: float = 1.0
    spec_func: int = 4  # specular exponent = 2 << spec_func
    diff_func: int = 0  # diffuse function 0..3
    relative_to_object: bool = False
    hs_enabled: bool = True
    positional: bool = False
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    visible: int = 0
    map: int = 0
    map_rot: tuple[int, int, int] = (0, 0, 0)


@dataclass
class PaletteColor:
    position: int = 0
    diffuse: RGB = (0, 0, 0)
    specular: RGB = (0, 0, 0)


@dataclass
class Lighting:
    lights: list[Light] = field(default_factory=lambda: [Light() for _ in range(6)])
    amb_top: RGB = _rgb(0x8B491D)
    amb_bottom: RGB = _rgb(0xFFC49F)
    depth_col: RGB = _rgb(0x8B491D)
    depth_col2: RGB = _rgb(0xFFC49F)
    dyn_fog_col: RGB = (255, 255, 255)
    dyn_fog_col2: RGB = (255, 255, 255)
    fog_offset: float = 128.0  # TBpos[3]
    depth_fog: float = 27.0  # TBpos[4]
    diffuse: float = 120.0  # TBpos[5] (50 = 1.0)
    dyn_fog: float = 53.0  # TBpos[6] (53 = off)
    specular: float = 50.0  # TBpos[7]
    ambient: float = 90.0  # TBpos[8] (90 = 1.0)
    color_start: float = 68.0  # TBpos[9]
    color_end: float = 72.0  # TBpos[10]
    amb_shadow: float = 53.0  # TBpos[11] low byte
    ind_light: float = 53.0
    gamma: float = 32.0
    roughness: float = 0.0
    color_cycling: bool = False
    var_col_z: float = 0.0
    color_on_otrap: bool = False
    far_fog: bool = False
    depth_func: int = 0
    fine_col_adj: tuple[int, int] | None = None
    interior_start: float = 60.0
    interior_end: float = 60.0
    diffuse_shadowing: float = 0.0
    hs_soft: bool = False
    hs_set_cos: bool = False
    palette: list[PaletteColor] = field(default_factory=lambda: [PaletteColor() for _ in range(10)])
    interior: list[tuple[int, RGB]] = field(default_factory=lambda: [(0, (0, 0, 0)) for _ in range(4)])
    interior_spec: list[int] = field(default_factory=lambda: [160] * 4)
    palette_alpha: list[int] | None = None
    bg_image: str = ""
    bg_direct: bool = False
    bg_rot: tuple[int, int, int] = (0, 0, 0)
    bg_brightness: int = 40
    bg_add_light: bool = False
    internal_gamma2: bool = False
    no_col_ipol: bool = False
    amb_rel_obj: bool = False
    dfog_options: int = 0
    ex_mode: int = 0
    diff_map: int = 0
    diff_map_mode: int = 0
    diff_map_offset: tuple[int, int] = (0, 0)
    diff_map_rot: int = 128
    diff_map_scale: int = 30
    yc_comb: bool = False
    bg_ambient: bool = False

    @staticmethod
    def start_preset() -> "Lighting":
        """MB3D's start-up light preset (``SetStartPreset``)."""
        cols = [5873889, 8837614, 0x8B491D, 2988346, 12248958, 0xFFC49F, 11287584, 14579248, 7481121]
        palette = []
        for i in range(3):
            palette.append(PaletteColor(i * 10922, _rgb(cols[i * 3]), _rgb(cols[i * 3 + 1])))
        for i in range(3, 10):
            palette.append(PaletteColor(32200 + (i - 3) * 84, palette[0].diffuse, palette[0].specular))
        interior = [(i * 10922, _rgb(cols[i * 3])) for i in range(3)]
        interior.append((32700, interior[0][1]))
        lights = [Light() for _ in range(6)]
        k = math.pi / 16384.0
        lights[0] = Light(on=True, color=_rgb(0xA0E8FF), x_angle=1500.0 * k, y_angle=5200.0 * k,
                          amplitude=1.0, spec_func=53 & 7, diff_func=(53 >> 4) & 3)
        lights[1] = Light(on=False, color=_rgb(2307911), x_angle=-2822.0 * k, y_angle=-7737.0 * k,
                          spec_func=16 & 7, diff_func=(16 >> 4) & 3)
        return Lighting(lights=lights, palette=palette, interior=interior)

    def palette_alpha_of(self, i: int) -> int:
        if self.palette_alpha is not None:
            return self.palette_alpha[i]
        return max(self.palette[i].specular)
