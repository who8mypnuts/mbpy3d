"""MutaGen: random variations of a scene (MB3D's mutation generator).

Formula options, lights and the palette are perturbed with a strength; the
camera is kept so that variations stay comparable.
"""
from __future__ import annotations

import colorsys
import random

from ..core.scene import Scene
from ..formulas import registry


def mutate(scene: Scene, strength: float = 0.2, what: tuple[str, ...] = ("formulas", "lights", "palette"),
           rng: random.Random | None = None) -> Scene:
    rng = rng or random.Random()
    sc = scene.copy()
    if "formulas" in what:
        for slot in sc.active_slots:
            spec = registry.lookup(slot.name)
            values = spec.values(slot.params)
            for p in spec.editable_params:
                if p.kind == "bool":
                    if rng.random() < strength * 0.3:
                        slot.params[p.name] = 1.0 - values[p.name]
                    continue
                if p.kind == "int":
                    if rng.random() < strength:
                        v = values[p.name] + rng.choice((-1, 1))
                        lo = p.minimum if p.minimum is not None else -1e9
                        hi = p.maximum if p.maximum is not None else 1e9
                        slot.params[p.name] = float(min(hi, max(lo, v)))
                    continue
                v = values[p.name]
                scale = abs(v) if abs(v) > 1e-6 else 0.5
                v = v + rng.gauss(0.0, strength * scale * 0.5)
                if p.minimum is not None:
                    v = max(p.minimum, v)
                if p.maximum is not None:
                    v = min(p.maximum, v)
                slot.params[p.name] = float(v)
        if rng.random() < strength * 0.5:
            sc.iterations = max(1, sc.iterations + rng.choice((-2, -1, 1, 2)))
    if "lights" in what:
        for li in sc.lighting.lights:
            if not li.enabled:
                continue
            li.x_angle += rng.gauss(0.0, strength)
            li.y_angle += rng.gauss(0.0, strength)
            li.color = _jitter_color(li.color, strength, rng)
        sc.lighting.ambient_top = _jitter_color(sc.lighting.ambient_top, strength, rng)
        sc.lighting.ambient_bottom = _jitter_color(sc.lighting.ambient_bottom, strength, rng)
    if "palette" in what:
        for stop in sc.lighting.palette:
            stop.diffuse = _jitter_color(stop.diffuse, strength, rng)
        if rng.random() < strength:
            sc.lighting.color_start = min(120.0, max(0.0, sc.lighting.color_start + rng.gauss(0.0, 5.0 * strength)))
            sc.lighting.color_end = min(120.0, max(0.0, sc.lighting.color_end + rng.gauss(0.0, 5.0 * strength)))
    return sc


def _jitter_color(c, strength: float, rng: random.Random):
    h, s, v = colorsys.rgb_to_hsv(*[min(1.0, max(0.0, x)) for x in c])
    h = (h + rng.gauss(0.0, 0.15 * strength)) % 1.0
    s = min(1.0, max(0.0, s + rng.gauss(0.0, 0.2 * strength)))
    v = min(1.0, max(0.0, v + rng.gauss(0.0, 0.15 * strength)))
    return tuple(colorsys.hsv_to_rgb(h, s, v))


def random_palette(rng: random.Random | None = None, n: int = 10):
    """A random but harmonious palette (hue walk) as (position, rgb) stops."""
    rng = rng or random.Random()
    h0 = rng.random()
    step = rng.uniform(0.03, 0.12) * rng.choice((-1, 1))
    out = []
    for i in range(n):
        h = (h0 + i * step) % 1.0
        s = rng.uniform(0.3, 0.9)
        v = rng.uniform(0.35, 1.0)
        out.append((int(i * 32767 / max(1, n - 1)), colorsys.hsv_to_rgb(h, s, v)))
    return out
