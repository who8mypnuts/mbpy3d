"""Stereo pairs: the scene rendered from two eyes shifted along the camera's
right vector (MB3D's stereo modes), combined side by side, cross-eyed or
as a red/cyan anaglyph."""
from __future__ import annotations

import threading

import numpy as np

from ..core.scene import Scene
from .renderer import Renderer

MODES = ("off", "side by side", "cross-eyed", "anaglyph")


def eye_scene(scene: Scene, eye: float, eye_distance_px: float) -> Scene:
    """The scene seen from an eye (-1 left, +1 right) ``eye_distance_px``
    pixels apart (in pixels of the image at the start plane)."""
    sc = scene.copy()
    shift = sc.camera.right * (eye * 0.5 * eye_distance_px * sc.camera.step_width(sc.width))
    sc.camera.mid = sc.camera.mid + shift
    return sc


def render_stereo(renderer: Renderer, scene: Scene, mode: str, eye_distance_px: float = 30.0,
                  progress=None, abort: threading.Event | None = None) -> np.ndarray:
    left = renderer.render(eye_scene(scene, -1.0, eye_distance_px),
                           progress=(lambda f, m: progress(f * 0.5, "left eye: " + m)) if progress else None,
                           abort=abort).image
    right = renderer.render(eye_scene(scene, 1.0, eye_distance_px),
                            progress=(lambda f, m: progress(0.5 + f * 0.5, "right eye: " + m)) if progress else None,
                            abort=abort).image
    if mode == "cross-eyed":
        return np.concatenate([right, left], axis=1)
    if mode == "anaglyph":
        out = right.copy()
        out[..., 0] = left[..., 0]
        return out
    return np.concatenate([left, right], axis=1)
