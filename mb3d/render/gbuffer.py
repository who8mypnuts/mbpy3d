"""The G-buffer: what the marcher produces for every pixel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mb3d.kernel.source import layout as L


@dataclass
class GBuffer:
    """Per pixel results as a (height, width, 12) float array (see ``layout.O_*``)."""

    data: np.ndarray
    backend: str = ""
    precision: str = "double"
    seconds: float = 0.0

    @classmethod
    def empty(cls, width: int, height: int, dtype=np.float64) -> "GBuffer":
        d = np.zeros((height, width, L.OUT_SIZE), dtype=dtype)
        d[..., L.O_AO] = 1.0
        d[..., L.O_SHADOW] = 1.0
        d[..., L.O_NZ] = -1.0
        return cls(d)

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def hit(self) -> np.ndarray:
        return self.data[..., L.O_HIT]

    @property
    def depth(self) -> np.ndarray:
        return self.data[..., L.O_DEPTH]

    @property
    def normal(self) -> np.ndarray:
        return self.data[..., L.O_NX:L.O_NZ + 1]

    @property
    def si_gradient(self) -> np.ndarray:
        return self.data[..., L.O_SIG]

    @property
    def otrap(self) -> np.ndarray:
        return self.data[..., L.O_OTRAP]

    @property
    def steps(self) -> np.ndarray:
        return self.data[..., L.O_STEPS]

    @property
    def iterations(self) -> np.ndarray:
        return self.data[..., L.O_ITS]

    @property
    def ao(self) -> np.ndarray:
        return self.data[..., L.O_AO]

    @property
    def shadow(self) -> np.ndarray:
        return self.data[..., L.O_SHADOW]

    def surface_mask(self) -> np.ndarray:
        return self.hit > 0.5

    def stats(self) -> dict:
        m = self.surface_mask()
        return {
            "surface_pixels": int(m.sum()),
            "avg_steps": float(self.steps.mean()),
            "avg_iterations": float(self.iterations[m].mean()) if m.any() else 0.0,
            "max_iterations": float(self.iterations.max()),
        }
