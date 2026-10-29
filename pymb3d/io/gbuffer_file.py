"""Render data files (``.pymb3d.npz``): the G-buffer of a render with its
scene, so that the lighting can be changed later without marching again
(MB3D's ``.m3i`` files)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.scene import Scene


def save_gbuffer(path: str | Path, gbuf: np.ndarray, scene: Scene) -> None:
    np.savez_compressed(Path(path), gbuffer=gbuf.astype(np.float32), scene=json.dumps(scene.to_dict()))


def load_gbuffer(path: str | Path) -> tuple[np.ndarray, Scene]:
    with np.load(Path(path), allow_pickle=False) as z:
        gbuf = np.asarray(z["gbuffer"], dtype=np.float32)
        scene = Scene.from_dict(json.loads(str(z["scene"])))
    return gbuf, scene
