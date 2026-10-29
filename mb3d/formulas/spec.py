"""Formula descriptions: options, DE meta data and the kernel function."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class OptionSpec:
    name: str
    default: float
    kind: str = "real"  # "real" | "int" | "bool" | "angle" (degrees)
    minimum: float | None = None
    maximum: float | None = None


@dataclass
class FormulaSpec:
    """What the renderer and the GUI need to know about a formula."""

    fid: int  # the id used in the kernel dispatcher (stable, unique)
    name: str  # MB3D name (the one found in .m3p files)
    kernel: object  # the @kfunc (numba dispatcher) with signature (st, fp, base, ade)
    options: list[OptionSpec] = field(default_factory=list)
    fnr: int = 20  # MB3D internal number (< 20) or 20 for custom
    de_option: int = 0  # 0 numerical 3D, 4 numerical 4D, 2/11 analytic (w)
    de_scale: float = 1.0
    ade_scale: float = 1.0
    rstop: float = 16.0
    category: str = "3D"
    description: str = ""
    aliases: tuple[str, ...] = ()
    si_pow: Callable[[list[float]], float] | None = None

    @property
    def analytic(self) -> bool:
        return self.de_option in (2, 5, 6, 11)

    @property
    def four_d(self) -> bool:
        return self.de_option in (4, 5, 6)

    def defaults(self) -> list[float]:
        v = [o.default for o in self.options]
        return v + [0.0] * (16 - len(v))

    def smooth_it_power(self, options: list[float]) -> float:
        if self.si_pow is not None:
            p = self.si_pow(options)
            return p if p != 0.0 else 1e-40
        return 2.0

    def option_type_first(self) -> str:
        return self.options[0].kind if self.options else "real"
