"""Parameter files and the Scene model."""

from .lighting import Light, Lighting, PaletteColor
from .m3p_reader import M3pError, load, parse
from .m3p_writer import save, save_text, to_bytes, to_text
from .scene import (
    CameraOptic,
    DeCombParams,
    FormulaSlot,
    InsideMode,
    Scene,
    ShadowParams,
    SsaoParams,
    angles_from_vgrads,
    vgrads_from_angles,
)

__all__ = [
    "Scene", "FormulaSlot", "Lighting", "Light", "PaletteColor", "CameraOptic", "InsideMode",
    "SsaoParams", "ShadowParams", "DeCombParams", "load", "parse", "save", "save_text", "to_bytes",
    "to_text", "M3pError", "vgrads_from_angles", "angles_from_vgrads",
]
