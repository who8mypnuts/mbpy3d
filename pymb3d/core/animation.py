"""Keyframe animation: scenes at key frames, interpolated in between.

Every numeric setting of the scene (camera, formula options, iteration
counts, lights, palette ...) is a column of a polars DataFrame with one row
per keyframe; the frames in between are filled by polars' linear
interpolation (optionally eased).  The camera rotation is interpolated as a
quaternion (slerp) and the zoom logarithmically, so that a zoom-in looks
uniform.  Non-numeric settings (formula names, modes) step at the keyframe.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .scene import Scene
from .vec import normalise_matrix_to

SPECIAL = ("camera.vgrads", "camera.zoom")


# ------------------------------------------------------------------ flatten
def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def unflatten(flat: dict[str, Any], template: Any) -> Any:
    """Rebuilds the nested structure of ``template`` from flat keys."""

    def build(t: Any, prefix: str) -> Any:
        if isinstance(t, dict):
            return {k: build(v, f"{prefix}{k}.") for k, v in t.items()}
        if isinstance(t, (list, tuple)):
            return [build(v, f"{prefix}{i}.") for i, v in enumerate(t)]
        return flat.get(prefix[:-1], t)

    return build(template, "")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# -------------------------------------------------------------- quaternions
def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    m = normalise_matrix_to(1.0, np.asarray(m, dtype=float))
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1.0 + m[i, i] - m[j, j] - m[k, k])) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q / np.linalg.norm(q)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return (q0 + (q1 - q0) * t) / np.linalg.norm(q0 + (q1 - q0) * t)
    th = math.acos(min(1.0, d))
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


# ---------------------------------------------------------------- animation
@dataclass
class Keyframe:
    frame: int
    scene: dict

    def to_dict(self) -> dict:
        return {"frame": self.frame, "scene": self.scene}

    @classmethod
    def from_dict(cls, d: dict) -> "Keyframe":
        return cls(int(d["frame"]), dict(d["scene"]))


@dataclass
class Animation:
    keyframes: list[Keyframe] = field(default_factory=list)
    fps: float = 25.0
    #: "linear" or "ease" (smooth start and end between keyframes)
    easing: str = "linear"

    # ----------------------------------------------------------- keyframes
    def sorted(self) -> list[Keyframe]:
        return sorted(self.keyframes, key=lambda k: k.frame)

    @property
    def frame_count(self) -> int:
        return (self.sorted()[-1].frame + 1) if self.keyframes else 0

    def set_keyframe(self, frame: int, scene: Scene) -> Keyframe:
        """Adds or replaces the keyframe at ``frame``."""
        d = scene.to_dict()
        for k in self.keyframes:
            if k.frame == frame:
                k.scene = d
                return k
        k = Keyframe(frame, d)
        self.keyframes.append(k)
        return k

    def remove_keyframe(self, frame: int) -> None:
        self.keyframes = [k for k in self.keyframes if k.frame != frame]

    # --------------------------------------------------------------- table
    def table(self) -> pl.DataFrame:
        """The keyframes as a DataFrame: one row per keyframe, one column
        per numeric setting (plus ``frame``)."""
        rows = []
        for k in self.sorted():
            flat = {p: v for p, v in flatten(k.scene).items() if _is_number(v) and not p.startswith(SPECIAL)}
            flat["frame"] = k.frame
            rows.append(flat)
        if not rows:
            return pl.DataFrame({"frame": []}, schema={"frame": pl.Int64})
        cols = sorted({c for r in rows for c in r})
        data = {c: [float(r[c]) if c in r else None for r in rows] for c in cols}
        data["frame"] = [int(r["frame"]) for r in rows]
        return pl.DataFrame(data)

    def interpolated(self) -> pl.DataFrame:
        """One row per frame 0..frame_count-1 with every numeric setting
        interpolated between the keyframes."""
        keys = self.table()
        n = self.frame_count
        frames = pl.DataFrame({"frame": list(range(n))}, schema={"frame": pl.Int64})
        df = frames.join(keys, on="frame", how="left").sort("frame")
        value_cols = [c for c in df.columns if c != "frame"]
        if not value_cols:
            return df
        if self.easing == "ease":
            # ease by re-parametrising the frame position between keyframes
            kf = keys["frame"].to_list()
            eased = np.array([_eased_position(f, kf) for f in range(n)])
            df = df.with_columns(pl.Series("frame_eased", eased))
            filled = pl.DataFrame({"frame": kf}).join(keys, on="frame", how="left")
            out = {"frame": list(range(n))}
            for c in value_cols:
                xs = np.array(kf, dtype=float)
                ys = np.array(filled[c].fill_null(strategy="forward").fill_null(strategy="backward").to_list(),
                              dtype=float)
                out[c] = np.interp(eased, xs, ys).tolist()
            return pl.DataFrame(out)
        df = df.with_columns([pl.col(c).interpolate().forward_fill().backward_fill() for c in value_cols])
        return df

    # --------------------------------------------------------------- scenes
    def scene_at(self, frame: int) -> Scene:
        keys = self.sorted()
        if not keys:
            raise ValueError("no keyframes")
        frame = max(0, min(frame, keys[-1].frame))
        prev = max((k for k in keys if k.frame <= frame), key=lambda k: k.frame, default=keys[0])
        nxt = min((k for k in keys if k.frame >= frame), key=lambda k: k.frame, default=keys[-1])
        if nxt.frame == prev.frame:
            t = 0.0
        else:
            t = (frame - prev.frame) / (nxt.frame - prev.frame)
            if self.easing == "ease":
                t = t * t * (3 - 2 * t)
        table = self.interpolated()
        row = table.filter(pl.col("frame") == frame)
        flat = flatten(prev.scene)
        if row.height:
            r = row.row(0, named=True)
            for c, v in r.items():
                if c != "frame" and v is not None and c in flat:
                    flat[c] = int(round(v)) if isinstance(flat[c], int) else float(v)
        # zoom logarithmically, rotation by slerp
        z0, z1 = float(prev.scene["camera"]["zoom"]), float(nxt.scene["camera"]["zoom"])
        flat["camera.zoom"] = math.exp(math.log(max(z0, 1e-300)) * (1 - t) + math.log(max(z1, 1e-300)) * t)
        q = slerp(quat_from_matrix(np.array(prev.scene["camera"]["vgrads"])),
                  quat_from_matrix(np.array(nxt.scene["camera"]["vgrads"])), t)
        m = matrix_from_quat(q)
        for i in range(3):
            for j in range(3):
                flat[f"camera.vgrads.{i}.{j}"] = float(m[i, j])
        return Scene.from_dict(unflatten(flat, prev.scene))

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {"format": "pymb3d-animation-1", "fps": self.fps, "easing": self.easing,
                "keyframes": [k.to_dict() for k in self.sorted()]}

    @classmethod
    def from_dict(cls, d: dict) -> "Animation":
        return cls([Keyframe.from_dict(k) for k in d.get("keyframes", [])], float(d.get("fps", 25.0)),
                   str(d.get("easing", "linear")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Animation":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _eased_position(frame: int, key_frames: list[int]) -> float:
    """The frame position re-parametrised with a smoothstep between keys."""
    prev = max((k for k in key_frames if k <= frame), default=key_frames[0])
    nxt = min((k for k in key_frames if k >= frame), default=key_frames[-1])
    if nxt == prev:
        return float(frame)
    t = (frame - prev) / (nxt - prev)
    t = t * t * (3 - 2 * t)
    return prev + t * (nxt - prev)
