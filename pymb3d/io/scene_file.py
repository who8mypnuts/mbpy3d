"""Scene files: JSON (``.pymb3d.json`` / ``.json``) and MB3D ``.m3p``."""
from __future__ import annotations

import json
from pathlib import Path

from ..core.scene import Scene


def save_scene(scene: Scene, path: str | Path) -> None:
    """JSON, or a MB3D parameter file when the name ends in .m3p."""
    p = Path(path)
    if p.suffix.lower() == ".m3p":
        from .m3p_write import save_m3p

        save_m3p(scene, p)
        return
    p.write_text(json.dumps(scene.to_dict(), indent=2))


def load_scene(path: str | Path) -> tuple[Scene, list[str]]:
    """Loads a JSON scene or a MB3D parameter file; returns (scene, warnings)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".png":
        from PIL import Image

        with Image.open(p) as img:
            text = (img.text or {}).get("pymb3d-scene") if hasattr(img, "text") else None
        if not text:
            raise ValueError(f"{p.name} carries no pymb3d scene")
        return Scene.from_dict(json.loads(text)), []
    if suffix in (".m3p", ".m3i", ".txt"):
        from .m3p import load

        res = load(p)
        res.scene.base_dir = str(p.parent)
        return res.scene, res.warnings
    data = json.loads(p.read_text())
    sc = Scene.from_dict(data)
    sc.base_dir = str(p.parent)
    return sc, []


def scene_from_clipboard_text(text: str) -> tuple[Scene, list[str]]:
    """A scene from text: JSON or MB3D text parameters."""
    t = text.strip()
    if t.startswith("{"):
        return Scene.from_dict(json.loads(t)), []
    from .m3p import load_text

    res = load_text(t)
    return res.scene, res.warnings
