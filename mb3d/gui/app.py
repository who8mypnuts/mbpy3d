"""Entry point of the desktop program: ``python -m mb3d`` or ``mb3d-gui``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mandelbulb3D in Python")
    parser.add_argument("file", nargs="?", help="a .m3p / .m3i / text parameter file")
    parser.add_argument("--no-auto", action="store_true", help="do not render previews automatically")
    args = parser.parse_args(argv)
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Mandelbulb3D")
    win = MainWindow(auto_preview=not args.no_auto)
    if args.file:
        win.load_path(args.file)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
