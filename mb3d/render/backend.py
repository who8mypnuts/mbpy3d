"""The back end interface: render rows of a scene into a G-buffer."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from mb3d.kernel.params import KernelParams

from .gbuffer import GBuffer


class RenderCancelled(Exception):
    pass


@dataclass
class DeviceInfo:
    kind: str  # "cpu" | "opengl" | "opencl"
    name: str
    precision: str  # "double" | "single"
    details: str = ""

    def __str__(self) -> str:
        return f"{self.kind}: {self.name} ({self.precision})"


class Backend(ABC):
    """A device that marches pixels.

    ``render`` fills ``gbuf`` row band by row band, calling ``progress``
    with the number of rows done after each band and raising
    ``RenderCancelled`` when ``cancel`` is set.
    """

    kind: str = ""
    precision: str = "double"

    @abstractmethod
    def info(self) -> DeviceInfo: ...

    @abstractmethod
    def render(self, params: KernelParams, gbuf: GBuffer, rows: np.ndarray, cancel: threading.Event | None = None, progress=None) -> None: ...

    def close(self) -> None:
        pass

    @property
    def dtype(self):
        return np.float64 if self.precision == "double" else np.float32


def check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise RenderCancelled()
