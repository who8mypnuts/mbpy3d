"""The Messages dock: a log of what the program did (status messages,
import notes, render times), as MB3D's message box."""
from __future__ import annotations

import time

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QDockWidget, QPlainTextEdit


class MessagesDock(QDockWidget):
    def __init__(self, parent=None) -> None:
        super().__init__("Messages", parent)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(2000)
        self.setWidget(self.text)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetClosable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable)

    @Slot(str)
    def log(self, message: str) -> None:
        if not message:
            return
        self.text.appendPlainText(f"{time.strftime('%H:%M:%S')}  {message}")
        self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())

    def log_lines(self, lines) -> None:
        for line in lines:
            self.log(line)
