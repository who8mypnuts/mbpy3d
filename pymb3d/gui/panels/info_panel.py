"""The Info panel: devices, the statistics of the last preview and the
timing history (polars)."""
from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from ...backends.detect import DeviceChoice
from ...render import RenderResult
from ...stats import benchmark


class InfoPanel(QWidget):
    def __init__(self, devices: list[DeviceChoice], parent: QWidget | None = None, model=None) -> None:
        super().__init__(parent)
        self.model = model
        lay = QVBoxLayout(self)
        if model is not None:
            from PySide6.QtWidgets import QFormLayout, QLineEdit

            form = QFormLayout()
            self.title = QLineEdit()
            self.author = QLineEdit()
            form.addRow("title", self.title)
            form.addRow("author", self.author)
            lay.addLayout(form)
            self.title.editingFinished.connect(lambda: setattr(model.scene, "title", self.title.text()))
            self.author.editingFinished.connect(lambda: setattr(model.scene, "author", self.author.text()))
            model.replaced.connect(self.load)
            self.load()
        lay.addWidget(QLabel("<b>Compute devices</b>"))
        self.devices = QPlainTextEdit()
        self.devices.setReadOnly(True)
        self.devices.setPlainText("\n".join(d.label for d in devices))
        self.devices.setMaximumHeight(90)
        lay.addWidget(self.devices)
        lay.addWidget(QLabel("<b>Last render</b>"))
        self.stats = QPlainTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setMaximumHeight(140)
        lay.addWidget(self.stats)
        lay.addWidget(QLabel("<b>Throughput history (pixels / s)</b>"))
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        lay.addWidget(self.history)
        refresh = QPushButton("Refresh history")
        refresh.clicked.connect(self.refresh_history)
        lay.addWidget(refresh)
        self.refresh_history()

    def load(self) -> None:
        if self.model is not None:
            self.title.setText(self.model.scene.title)
            self.author.setText(self.model.scene.author)

    def show_result(self, res: RenderResult) -> None:
        lines = [f"{res.width} x {res.height} on {res.backend}", f"{res.seconds:.2f} s"]
        if res.compile_seconds:
            lines.append(f"compile {res.compile_seconds:.1f} s")
        for k, v in res.stats.items():
            lines.append(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
        self.stats.setPlainText("\n".join(lines))

    def refresh_history(self) -> None:
        try:
            t = benchmark.throughput()
            self.history.setPlainText(str(t) if t.height else "no renders recorded yet")
        except Exception as e:  # pragma: no cover
            self.history.setPlainText(f"history unavailable: {e}")
