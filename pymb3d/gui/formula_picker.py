"""The formula picker: all formulas by category with a search box and
the description (MB3D's formula window lists them in a grid)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout

from ..formulas import registry


class FormulaPickerDialog(QDialog):
    def __init__(self, current: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose a formula")
        self.chosen: str | None = None
        self.search = QLineEdit()
        self.search.setPlaceholderText("search name, category or description ...")
        self.list = QListWidget()
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setMinimumHeight(60)
        ok = QPushButton("Use")
        cancel = QPushButton("Cancel")
        lay = QVBoxLayout(self)
        lay.addWidget(self.search)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.desc)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(ok)
        row.addWidget(cancel)
        lay.addLayout(row)
        self.resize(520, 560)
        self.search.textChanged.connect(self.refill)
        self.list.currentItemChanged.connect(self._show)
        self.list.itemDoubleClicked.connect(lambda _i: self.accept())
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        self.refill("")
        if current:
            for i in range(self.list.count()):
                if self.list.item(i).data(Qt.ItemDataRole.UserRole) == current:
                    self.list.setCurrentRow(i)
                    break

    def refill(self, text: str) -> None:
        text = text.lower().strip()
        self.list.clear()
        last_cat = None
        for spec in sorted(registry.all_specs(), key=lambda s: (s.category, s.name.lower())):
            hay = f"{spec.name} {spec.category} {spec.description} {' '.join(spec.aliases)}".lower()
            if text and text not in hay:
                continue
            if spec.category != last_cat:
                head = QListWidgetItem(f"— {spec.category} —")
                head.setFlags(Qt.ItemFlag.NoItemFlags)
                self.list.addItem(head)
                last_cat = spec.category
            item = QListWidgetItem(f"    {spec.name}   ({len(spec.editable_params)} options, DE {spec.de})")
            item.setData(Qt.ItemDataRole.UserRole, spec.name)
            self.list.addItem(item)

    def _show(self, item, _prev=None) -> None:
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if name:
            spec = registry.lookup(name)
            self.desc.setText(f"<b>{spec.name}</b>: {spec.description}<br>options: "
                              + ", ".join(p.name for p in spec.editable_params))

    def accept(self) -> None:
        item = self.list.currentItem()
        if item is not None and item.data(Qt.ItemDataRole.UserRole):
            self.chosen = item.data(Qt.ItemDataRole.UserRole)
        super().accept()
