"""The backend interface: compile a program, march a rectangle of pixels
band by band (progress, cancellation), return the G-buffer."""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core import params as PL
from ..kernels.program import Program

#: called after every band: (first row, number of rows, rows done so far, rows total)
ProgressFn = Callable[[int, int, int, int], None]


class RenderCancelled(Exception):
    pass


class BackendError(Exception):
    pass


@dataclass
class DeviceInfo:
    backend: str
    name: str
    precision: str          # "float32" or "float64"
    kind: str               # "gpu", "cpu", "software"
    details: str = ""

    @property
    def label(self) -> str:
        return f"{self.backend}: {self.name} ({self.precision})"


class Backend(ABC):
    """One compute device.  Instances are not thread safe: a backend is
    used from the thread that created its context (the render worker)."""

    name: str = "base"

    def __init__(self) -> None:
        self._compiled: dict[str, object] = {}
        self.last_compile_seconds = 0.0

    @property
    @abstractmethod
    def info(self) -> DeviceInfo: ...

    @property
    def precision(self) -> str:
        return self.info.precision

    @property
    def dtype(self):
        return np.float64 if self.precision == "float64" else np.float32

    # --------------------------------------------------------------- compile
    @abstractmethod
    def _compile(self, program: Program) -> object: ...

    def compile(self, program: Program) -> object:
        key = program.key
        if key not in self._compiled:
            t0 = time.perf_counter()
            self._compiled[key] = self._compile(program)
            self.last_compile_seconds = time.perf_counter() - t0
        return self._compiled[key]

    def is_compiled(self, program: Program) -> bool:
        return program.key in self._compiled

    # ---------------------------------------------------------------- render
    @abstractmethod
    def _render_band(self, compiled: object, P: np.ndarray, y0: int, rows: int, out: np.ndarray,
                     abort: threading.Event) -> None:
        """Marches rows y0..y0+rows of the rectangle into out[y0:y0+rows]."""

    def band_rows(self, width: int) -> int:
        return 16

    #: seconds a band should take (0 = fixed band size); GPU backends keep
    #: launches short so that the display driver stays responsive
    band_target_seconds: float = 0.0

    def render(self, program: Program, P: np.ndarray, width: int, height: int,
               progress: ProgressFn | None = None, abort: threading.Event | None = None,
               out: np.ndarray | None = None) -> np.ndarray:
        """The G-buffer (height, width, N_CHANNELS) float32 of the rectangle
        the parameter block describes (written into ``out`` if given)."""
        compiled = self.compile(program)
        abort = abort or threading.Event()
        if out is None:
            out = np.zeros((height, width, PL.N_CHANNELS), dtype=np.float32)
        band = max(1, self.band_rows(width))
        y = 0
        while y < height:
            if abort.is_set():
                raise RenderCancelled()
            rows = min(band, height - y)
            t0 = time.perf_counter()
            self._render_band(compiled, P, y, rows, out, abort)
            dt = time.perf_counter() - t0
            y += rows
            if progress:
                progress(y - rows, rows, y, height)
            if self.band_target_seconds > 0 and dt > 0:
                # adapt the band to the measured speed (bounded growth)
                want = rows * self.band_target_seconds / dt
                band = int(max(1, min(want, rows * 4, 4096)))
        return out

    def close(self) -> None:
        pass
