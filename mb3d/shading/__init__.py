"""G-buffer -> RGB: MB3D's painter (lights, palette, fogs) in numpy."""

from .painter import paint
from .light_values import LightVals, PaintCamera

__all__ = ["paint", "LightVals", "PaintCamera"]
