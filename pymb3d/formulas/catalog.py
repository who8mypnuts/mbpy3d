"""The formula catalog as a polars DataFrame (the GUI's formula browser
filters and sorts it)."""
from __future__ import annotations

import polars as pl

from . import registry


def catalog() -> pl.DataFrame:
    rows = []
    for s in registry.all_specs():
        rows.append({
            "name": s.name,
            "category": s.category,
            "de": s.de,
            "params": len(s.params),
            "param_names": ", ".join(p.name for p in s.params),
            "four_d": s.four_d,
            "counted": s.counted,
            "mb3d_id": s.mb3d_id,
            "description": s.description,
        })
    return pl.DataFrame(rows).sort(["category", "name"])


def names_in_category(cat: str | None = None) -> list[str]:
    df = catalog()
    if cat:
        df = df.filter(pl.col("category") == cat)
    return df["name"].to_list()


def categories() -> list[str]:
    return catalog()["category"].unique().sort().to_list()
