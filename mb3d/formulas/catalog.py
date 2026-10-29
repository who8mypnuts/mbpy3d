"""The formula catalog: a polars table of every formula this program knows.

Native formulas (kernel DSL) and the meta data of MB3D's ``.m3f`` files are
merged into one frame, so the GUI can search, filter by category and look
up option names and defaults with vectorised queries.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import polars as pl

from mb3d.params.m3f_meta import M3fMeta, scan_formula_dir

from . import registry

SCHEMA = {
    "name": pl.Utf8,
    "category": pl.Utf8,
    "source": pl.Utf8,  # "native" | "m3f"
    "available": pl.Boolean,
    "de_option": pl.Int32,
    "de_scale": pl.Float64,
    "rstop": pl.Float64,
    "n_options": pl.Int32,
    "options": pl.List(pl.Utf8),
    "defaults": pl.List(pl.Float64),
    "description": pl.Utf8,
}


def formula_dirs() -> list[Path]:
    """Folders searched for ``.m3f`` files: $MB3D_FORMULAS, ./M3Formulas, the MB3D sources."""
    dirs: list[Path] = []
    env = os.environ.get("MB3D_FORMULAS")
    if env:
        dirs += [Path(p) for p in env.split(os.pathsep) if p]
    dirs.append(Path.cwd() / "M3Formulas")
    here = Path(__file__).resolve()
    for up in here.parents:
        cand = up / "mb3d-code" / "mb3d" / "M3Formulas"
        if cand.is_dir():
            dirs.append(cand)
            break
    dirs.append(Path.home() / "mb3d-rs" / "M3Formulas")
    seen, out = set(), []
    for d in dirs:
        if d.is_dir() and d.resolve() not in seen:
            seen.add(d.resolve())
            out.append(d)
    return out


def _native_rows() -> list[dict]:
    rows = []
    for s in registry.all_specs():
        rows.append({
            "name": s.name, "category": s.category, "source": "native", "available": True,
            "de_option": s.de_option, "de_scale": s.de_scale, "rstop": s.rstop, "n_options": len(s.options),
            "options": [o.name for o in s.options], "defaults": [float(o.default) for o in s.options],
            "description": s.description,
        })
    return rows


def _m3f_rows(metas: list[M3fMeta], native_names: set[str]) -> list[dict]:
    rows = []
    for m in metas:
        if m.name.lower() in native_names:
            continue
        rows.append({
            "name": m.name, "category": m.category, "source": "m3f", "available": False,
            "de_option": m.de_option, "de_scale": m.de_scale, "rstop": m.rstop, "n_options": len(m.options),
            "options": [o.name for o in m.options], "defaults": [o.default for o in m.options],
            "description": m.description,
        })
    return rows


@lru_cache(maxsize=4)
def catalog(extra_dirs: tuple[str, ...] = ()) -> pl.DataFrame:
    native = _native_rows()
    names = {r["name"].lower() for r in native}
    metas: list[M3fMeta] = []
    for d in list(formula_dirs()) + [Path(p) for p in extra_dirs]:
        metas += scan_formula_dir(d)
    rows = native + _m3f_rows(metas, names)
    if not rows:
        return pl.DataFrame(schema=SCHEMA)
    df = pl.DataFrame(rows, schema=SCHEMA)
    return df.unique(subset=["name"], keep="first").sort(["available", "category", "name"], descending=[True, False, False])


def search(text: str = "", category: str | None = None, available_only: bool = False) -> pl.DataFrame:
    df = catalog()
    if text:
        df = df.filter(pl.col("name").str.to_lowercase().str.contains(text.lower(), literal=True))
    if category:
        df = df.filter(pl.col("category") == category)
    if available_only:
        df = df.filter(pl.col("available"))
    return df


def option_names(name: str) -> list[str]:
    df = catalog().filter(pl.col("name").str.to_lowercase() == name.lower())
    if df.height == 0:
        return []
    return list(df["options"][0])


def defaults(name: str) -> list[float]:
    df = catalog().filter(pl.col("name").str.to_lowercase() == name.lower())
    if df.height == 0:
        return []
    return list(df["defaults"][0])


def categories() -> list[str]:
    return sorted(catalog()["category"].unique().to_list())
