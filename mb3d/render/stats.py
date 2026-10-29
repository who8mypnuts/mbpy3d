"""Render statistics as a polars frame (benchmarks, the Infos panel)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import polars as pl

from .renderer import RenderResult

SCHEMA = {
    "when": pl.Float64,
    "scene": pl.Utf8,
    "backend": pl.Utf8,
    "precision": pl.Utf8,
    "width": pl.Int32,
    "height": pl.Int32,
    "march_s": pl.Float64,
    "paint_s": pl.Float64,
    "total_s": pl.Float64,
    "surface_pixels": pl.Int64,
    "avg_steps": pl.Float64,
    "avg_iterations": pl.Float64,
    "max_iterations": pl.Float64,
}


@dataclass
class RenderStats:
    rows: list[dict] = field(default_factory=list)

    def add(self, result: RenderResult, scene_name: str = "") -> None:
        st = result.stats
        self.rows.append({
            "when": time.time(), "scene": scene_name or result.scene.title, "backend": result.backend,
            "precision": result.precision, "width": result.width, "height": result.height,
            "march_s": result.march_seconds, "paint_s": result.paint_seconds, "total_s": result.seconds,
            "surface_pixels": int(st.get("surface_pixels", 0)), "avg_steps": float(st.get("avg_steps", 0.0)),
            "avg_iterations": float(st.get("avg_iterations", 0.0)), "max_iterations": float(st.get("max_iterations", 0.0)),
        })

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows, schema=SCHEMA) if self.rows else pl.DataFrame(schema=SCHEMA)

    def summary(self) -> pl.DataFrame:
        """Mean timings and megapixels per second per back end."""
        df = self.frame()
        if df.height == 0:
            return df
        return (
            df.with_columns(((pl.col("width") * pl.col("height")) / 1e6 / pl.col("march_s")).alias("mpix_per_s"))
            .group_by(["backend", "precision"])
            .agg(pl.len().alias("renders"), pl.col("march_s").mean(), pl.col("paint_s").mean(), pl.col("mpix_per_s").mean())
            .sort("mpix_per_s", descending=True)
        )
