"""Image files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_image(img: np.ndarray, path: str | Path, quality: int = 95) -> None:
    im = Image.fromarray(np.ascontiguousarray(img))
    path = Path(path)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        im.save(path, quality=quality)
    else:
        im.save(path)
