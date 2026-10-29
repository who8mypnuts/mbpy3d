"""Application bootstrap."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication


def run(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("pymb3d")
    from PySide6.QtCore import QSettings

    from .main_window import MainWindow
    from .theme import apply_saved

    apply_saved(app, QSettings("pymb3d", "pymb3d"))

    win = MainWindow()
    for a in argv[1:]:
        if not a.startswith("-"):
            win.open_path(a)
            break
    win.show()
    return app.exec()
