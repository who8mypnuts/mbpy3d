"""The ``[OPTIONS]`` block of MB3D ``.m3f`` formula files (meta data only).

The code of those files is x86 machine code, which this port does not run;
their options, DE settings and descriptions feed the formula catalog so
that parameter files using them can be inspected and edited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# MB3D's option type table (CustomFormulas.pas); the index is the type code
_TYPES = [
    "DOUBLE", "SINGLE", "INTEGER", "DOUBLEANGLE", "SINGLEANGLE", "3DOUBLEANGLES", "3SINGLEANGLES", "BOXSCALE",
    "FOLDING", "DSQUARE", "NOVARIABLE", "FOLDING16", "6SINGLEANGLES", "DRECIPRO", "2DOUBLES", "DSQRRECI",
    "2SINGLES", "4SINGLES", "3SCALESANGLES", "SCALESROT", "2INTEGER", "SRECI2", "DRECI2",
]
_TYPE_INDEX = {t: i for i, t in enumerate(_TYPES)}


@dataclass
class M3fOption:
    name: str
    default: float
    kind: int  # MB3D type code


@dataclass
class M3fMeta:
    name: str
    version: int = 0
    de_option: int = 0
    de_scale: float = 1.0
    ade_scale: float = 1.0
    si_pow: float = 0.0
    rstop: float = 16.0
    options: list[M3fOption] = field(default_factory=list)
    description: str = ""
    path: Path | None = None

    @property
    def category(self) -> str:
        d = self.de_option
        if d in (2, 11):
            return "3Da"
        if d == 4:
            return "4D"
        if d in (5, 6):
            return "4Da"
        if d < 0:
            return "Ads"
        if d == 20:
            return "dIFS"
        if d in (21, 22):
            return "dIFS-T"
        return "3D"


def _num(s: str) -> float:
    try:
        return float(s.strip().replace(",", "."))
    except ValueError:
        return 0.0


def parse_m3f(text: str, name: str) -> M3fMeta:
    m = M3fMeta(name)
    lines = [l.strip() for l in text.splitlines()]
    i = 0
    while i < len(lines) and lines[i].upper() != "[OPTIONS]":
        i += 1
    i += 1
    while i < len(lines):
        s = lines[i]
        i += 1
        if not s:
            continue
        if s.startswith("["):
            break
        if not s.startswith("."):
            continue
        body = s[1:]
        if "=" in body:
            key, _, value = body.partition("=")
        else:
            key, value = body, ""
        key = key.strip()
        kw = key.split()[0].upper() if key.split() else ""
        if kw == "VERSION":
            m.version = int(_num(value))
        elif kw == "DEOPTION":
            m.de_option = int(_num(value))
        elif kw == "DESCALE":
            m.de_scale = _num(value)
        elif kw == "ADESCALE":
            m.ade_scale = _num(value)
        elif kw == "SIPOWER":
            m.si_pow = _num(value)
        elif kw == "RSTOP":
            m.rstop = _num(value)
        elif kw in _TYPE_INDEX:
            ty = _TYPE_INDEX[kw]
            oname = key[len(kw):].strip() or kw.title()
            val = _num(value)
            if ty in (5, 6):
                for sfx in (" X", " Y", " Z"):
                    m.options.append(M3fOption(oname + sfx, val, ty))
            elif ty == 12:
                for sfx in (" YZ", " XZ", " XY", " XW", " YW", " ZW"):
                    m.options.append(M3fOption(oname + sfx, val, ty))
            elif ty == 18:
                m.options.append(M3fOption(oname, val, ty))
                for c in "ABC":
                    m.options.append(M3fOption(f"Rotation {c}", 0.0, 6))
            elif ty == 19:
                m.options.append(M3fOption(f"Scale {oname}", 1.0, ty))
                m.options.append(M3fOption(f"Rotation {oname}", val, 6))
            else:
                m.options.append(M3fOption(oname, val, ty))
    m.options = m.options[:16]
    # the description follows [END]
    try:
        end = next(k for k, l in enumerate(lines) if l.upper() == "[END]")
        desc = [l for l in lines[end + 1:] if l]
        if desc and desc[0].lower().startswith("description:"):
            desc = desc[1:]
        m.description = "\n".join(desc[:40])
    except StopIteration:
        pass
    return m


def load_m3f(path: str | Path) -> M3fMeta:
    path = Path(path)
    m = parse_m3f(path.read_text(encoding="latin-1"), path.stem)
    m.path = path
    return m


def scan_formula_dir(directory: str | Path) -> list[M3fMeta]:
    directory = Path(directory)
    out = []
    if not directory.is_dir():
        return out
    for f in sorted(directory.glob("*.m3f"), key=lambda p: p.name.lower()):
        try:
            out.append(load_m3f(f))
        except Exception:
            continue
    return out
