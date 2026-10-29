"""Lighting presets (`.json`, the lighting record of a scene) and
MB3D's `.m3l` light files (the `TLightingParas9` record)."""
from __future__ import annotations

import json
from pathlib import Path

from ..core.lights import Lighting


def save_lighting(lighting: Lighting, path: str | Path) -> None:
    Path(path).write_text(json.dumps({"format": "pymb3d-lighting-1", "lighting": lighting.to_dict()}, indent=2))


def load_lighting(path: str | Path) -> Lighting:
    p = Path(path)
    if p.suffix.lower() == ".m3l":
        from .m3p import HEADER_SIZE, _Rd, _parse_light, _upgrade_light

        rec = p.read_bytes()
        if len(rec) < 408:
            raise ValueError(f"{p.name}: too short for a MB3D light file")
        d = bytearray(HEADER_SIZE)
        d[432:840] = rec[:408]
        _upgrade_light(d)
        warnings: list[str] = []
        return _parse_light(_Rd(bytes(d)), warnings)
    data = json.loads(p.read_text())
    return Lighting.from_dict(data.get("lighting", data))
