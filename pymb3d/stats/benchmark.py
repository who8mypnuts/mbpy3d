"""Timing history of the backends (a polars DataFrame kept as parquet).

Every render appends a row; :func:`throughput` aggregates pixels per second
per device, which the GUI shows and *Auto* uses to rank devices that have
already been measured on this machine.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import polars as pl

SCHEMA = {
    "time": pl.Float64,
    "device": pl.Utf8,
    "backend": pl.Utf8,
    "precision": pl.Utf8,
    "formulas": pl.Utf8,
    "pixels": pl.Int64,
    "seconds": pl.Float64,
    "compile_seconds": pl.Float64,
    "shadows": pl.Boolean,
    "ao": pl.Utf8,
}

_LOCK = threading.Lock()


def history_path() -> Path:
    root = os.environ.get("PYMB3D_CACHE")
    base = Path(root) if root else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pymb3d"
    base.mkdir(parents=True, exist_ok=True)
    return base / "render_history.parquet"


def load() -> pl.DataFrame:
    p = history_path()
    if p.exists():
        try:
            return pl.read_parquet(p)
        except Exception:
            pass
    return pl.DataFrame(schema=SCHEMA)


def record(device: str, backend: str, precision: str, formulas: str, pixels: int, seconds: float,
           compile_seconds: float = 0.0, shadows: bool = False, ao: str = "none") -> None:
    row = pl.DataFrame({
        "time": [time.time()], "device": [device], "backend": [backend], "precision": [precision],
        "formulas": [formulas], "pixels": [int(pixels)], "seconds": [float(seconds)],
        "compile_seconds": [float(compile_seconds)], "shadows": [bool(shadows)], "ao": [ao],
    }, schema=SCHEMA)
    with _LOCK:
        df = pl.concat([load(), row], how="vertical")
        if df.height > 5000:
            df = df.tail(5000)
        try:
            df.write_parquet(history_path())
        except Exception:
            pass


def throughput(df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Pixels per second per device (mean over the recorded renders)."""
    df = load() if df is None else df
    if df.height == 0:
        return pl.DataFrame(schema={"device": pl.Utf8, "backend": pl.Utf8, "renders": pl.Int64,
                                    "pixels_per_second": pl.Float64, "mean_seconds": pl.Float64})
    return (
        df.filter(pl.col("seconds") > 0)
        .group_by(["device", "backend"])
        .agg([
            pl.len().alias("renders"),
            (pl.col("pixels") / pl.col("seconds")).mean().alias("pixels_per_second"),
            pl.col("seconds").mean().alias("mean_seconds"),
        ])
        .sort("pixels_per_second", descending=True)
    )


def fastest_device(purpose_filter: str | None = None) -> str | None:
    t = throughput()
    if purpose_filter:
        t = t.filter(pl.col("backend") == purpose_filter)
    if t.height == 0:
        return None
    return str(t["device"][0])
