"""The formula registry: every formula the kernels know, by id and name."""

from __future__ import annotations

from .builtin import boxes, bulbs, ifs, quaternions, transforms
from .spec import FormulaSpec, OptionSpec

R = "real"
I = "int"
B = "bool"
A = "angle"


def _angles(prefix: str) -> list[OptionSpec]:
    return [OptionSpec(f"{prefix} {c}", 0.0, A) for c in "XYZ"]


_SPECS: list[FormulaSpec] = [
    FormulaSpec(
        1, "Integer Power", bulbs.f_integer_power,
        [OptionSpec("Integer power", 8, I, 2, 8), OptionSpec("Z multiplier", -1.0)],
        fnr=0, de_option=0, de_scale=1.0, rstop=16.0, category="3D",
        description="The Mandelbulb: triplex power 2..8 (White/Nylander).",
        aliases=("intpow", "mandelbulb", "bulb"),
        si_pow=lambda o: float(max(2, min(8, int(round(o[0]))))),
    ),
    FormulaSpec(
        2, "Real Power", bulbs.f_real_power,
        [OptionSpec("Float power", 8.0), OptionSpec("Z multiplier", -1.0)],
        fnr=1, de_option=0, de_scale=1.0, rstop=16.0, category="3D",
        description="Triplex power with any real exponent.",
        aliases=("realpower", "floatpow"),
        si_pow=lambda o: o[0],
    ),
    FormulaSpec(
        3, "Quaternion", quaternions.f_quaternion,
        [OptionSpec("YW multiplier", -1.0), OptionSpec("W add", 0.0)],
        fnr=2, de_option=4, de_scale=1.0, rstop=16.0, category="4D",
        description="Quaternion Julia / Mandelbrot set (4D).",
        aliases=("quat",),
    ),
    FormulaSpec(
        4, "Tricorn", quaternions.f_tricorn,
        [OptionSpec("Z multiplier", -2.0), OptionSpec("CZ multiplier", 1.0)],
        fnr=3, de_option=0, de_scale=1.0, rstop=16.0, category="3D",
    ),
    FormulaSpec(
        5, "Amazing Box", boxes.f_amazing_box,
        [OptionSpec("Scale", 2.0), OptionSpec("Min R", 0.5), OptionSpec("Fold", 1.0)],
        fnr=4, de_option=11, de_scale=0.2, rstop=1024.0, category="3Da",
        description="The Mandelbox (Tglad): box fold, sphere fold, scale.",
        aliases=("amazingbox", "mandelbox", "box"),
    ),
    FormulaSpec(
        6, "Bulbox", boxes.f_bulbox,
        [OptionSpec("Box Scale", 2.0), OptionSpec("Box Min R", 0.5), OptionSpec("Box fold", 1.0),
         OptionSpec("Bulb scaling", 1.0), OptionSpec("Box/Bulb R threshold", 2.0), OptionSpec("Box/Bulb R threshold 2", 2.0)],
        fnr=5, de_option=0, de_scale=0.2, rstop=1024.0, category="3D",
    ),
    FormulaSpec(
        7, "Folding Int Pow", bulbs.f_folding_int_pow,
        [OptionSpec("Integer power", 2, I, 2, 8), OptionSpec("Z multiplier", -1.0), OptionSpec("R fold", 2.0)],
        fnr=6, de_option=0, de_scale=0.5, rstop=1024.0, category="3D",
        aliases=("foldingintpow", "foldintpow"),
        si_pow=lambda o: float(max(2, min(8, int(round(o[0]))))),
    ),
    FormulaSpec(
        20, "Amazing Surf", boxes.f_amazing_surf,
        [OptionSpec("Scale", 1.5), OptionSpec("Min R", 0.5), OptionSpec("Fold", 1.0)] + _angles("Rotation1")
        + [OptionSpec("Scale vary", 0.0), OptionSpec("Sphere or Cylinder", 1, I, 0, 1)],
        fnr=20, de_option=11, de_scale=0.2, rstop=20.0, category="3Da",
        description="Kali's Amazing Surf: the Mandelbox folded in x and y only.",
    ),
    FormulaSpec(
        21, "ABoxMod1", boxes.f_abox_mod1,
        [OptionSpec("Scale", 2.0), OptionSpec("Min R", 0.0), OptionSpec("Fold", 2.0), OptionSpec("Scale vary", 0.0),
         OptionSpec("FoldXMod", 0.0), OptionSpec("FoldYMod", 0.0), OptionSpec("FoldZMod", 0.0)],
        fnr=20, de_option=2, de_scale=0.2, rstop=1024.0, category="3Da",
    ),
    FormulaSpec(
        22, "Menger3", ifs.f_menger3,
        [OptionSpec("Scale", 3.0), OptionSpec("CScale X", 1.0), OptionSpec("CScale Y", 1.0), OptionSpec("CScale Z", 1.0)]
        + _angles("Rotation1") + _angles("Rotation2"),
        fnr=20, de_option=2, de_scale=0.2, rstop=1024.0, category="3Da",
        description="The Menger sponge as a kaleidoscopic IFS.",
    ),
    FormulaSpec(
        23, "Sierpinski3", ifs.f_sierpinski3,
        [OptionSpec("Scale", 2.0), OptionSpec("CScale X", 1.0), OptionSpec("CScale Y", 1.0), OptionSpec("CScale Z", 1.0)]
        + _angles("Rotation1") + _angles("Rotation2"),
        fnr=20, de_option=2, de_scale=0.2, rstop=1024.0, category="3Da",
    ),
    FormulaSpec(
        24, "ABoxModKali", ifs.f_abox_mod_kali,
        [OptionSpec("Scale", 1.5), OptionSpec("Min R", 0.6), OptionSpec("Z add", 0.5), OptionSpec("Y add", 0.5), OptionSpec("X add", 0.5)],
        fnr=20, de_option=2, de_scale=0.15, ade_scale=0.5, rstop=1024.0, category="3Da",
    ),
    FormulaSpec(40, "_Rotate", transforms.f_rotate, _angles("Rotation1"), fnr=20, de_option=-1, category="Ads",
                description="Rotates the vector (a transformation for hybrids)."),
    FormulaSpec(41, "_Translate", transforms.f_translate,
                [OptionSpec("X add", 0.0), OptionSpec("Y add", 0.0), OptionSpec("Z add", 0.0)], fnr=20, de_option=-1, category="Ads"),
    FormulaSpec(42, "_Scaling", transforms.f_scaling, [OptionSpec("Scale", 1.0)], fnr=20, de_option=2, de_scale=0.2, category="Ads"),
    FormulaSpec(43, "_AbsY", transforms.f_abs_y, [], fnr=20, de_option=-1, category="Ads"),
]

_BY_ID = {s.fid: s for s in _SPECS}
_BY_NAME: dict[str, FormulaSpec] = {}
for _s in _SPECS:
    _BY_NAME[_s.name.lower()] = _s
    for _a in _s.aliases:
        _BY_NAME[_a.lower()] = _s


def _norm(name: str) -> str:
    return name.strip().lower()


def all_specs() -> list[FormulaSpec]:
    return list(_SPECS)


def by_id(fid: int) -> FormulaSpec:
    return _BY_ID[fid]


def find(name: str) -> FormulaSpec | None:
    n = _norm(name)
    s = _BY_NAME.get(n)
    if s is None:
        s = _BY_NAME.get(n.replace("_", " ").replace("-", " "))
    return s


def lookup(name: str) -> FormulaSpec:
    s = find(name)
    if s is None:
        raise KeyError(f"formula '{name}' is not available in this build")
    return s


def kernel_name(spec: FormulaSpec) -> str:
    return spec.kernel._kernel_function.name
