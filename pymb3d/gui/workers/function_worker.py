"""A worker that runs one callable ``fn(progress, abort)`` in a QThread and
reports through signals (used for stereo renders and other one-off jobs)."""
from __future__ import annotations

import threading
import traceback
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ...backends.base import RenderCancelled


class FunctionWorker(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn: Callable) -> None:
        super().__init__()
        self.fn = fn
        self.abort = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(self.fn(lambda f, m: self.progress.emit(f, m), self.abort))
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n{traceback.format_exc(limit=3)}")


def start_worker(fn: Callable) -> tuple[QThread, FunctionWorker]:
    """Runs ``fn`` in a new thread; the caller connects the worker's signals
    and keeps both objects alive."""
    thread = QThread()
    worker = FunctionWorker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    thread.start()
    return thread, worker
