"""Formulas: the six hybrid slots with their iteration counts and options."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mb3d.formulas import catalog, registry
from mb3d.params.scene import FormulaSlot

from ..model import SceneModel


class SlotWidget(QWidget):
    """One hybrid slot: formula picker, iterations, options."""

    def __init__(self, model: SceneModel, index: int, parent=None):
        super().__init__(parent)
        self.model = model
        self.index = index
        self._refreshing = False
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("search formulas")
        self.search.textChanged.connect(self._fill_combo)
        self.combo = QComboBox()
        self.combo.setMinimumWidth(180)
        self.combo.currentIndexChanged.connect(self._formula_chosen)
        row.addWidget(self.search)
        row.addWidget(self.combo, 1)
        layout.addLayout(row)
        f = QFormLayout()
        self.iterations = QSpinBox()
        self.iterations.setRange(-1000, 100000)
        self.iterations.setToolTip("0: slot unused; negative: iterated but not counted")
        self.iterations.setKeyboardTracking(False)
        self.iterations.valueChanged.connect(self._iterations_changed)
        f.addRow("Iterations", self.iterations)
        layout.addLayout(f)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.options_box = QGroupBox("Options")
        self.options_form = QFormLayout(self.options_box)
        layout.addWidget(self.options_box)
        layout.addStretch(1)
        self.option_spins: list[QDoubleSpinBox] = []
        self._fill_combo("")

    # ------------------------------------------------------------ slot
    def slot(self) -> FormulaSlot | None:
        fs = self.model.scene.formulas
        return fs[self.index] if self.index < len(fs) else None

    def _ensure_slot(self) -> FormulaSlot:
        fs = self.model.scene.formulas
        while len(fs) <= self.index:
            fs.append(FormulaSlot("Integer Power", 0, 0, [8.0, -1.0] + [0.0] * 14, 2))
        return fs[self.index]

    def _fill_combo(self, text: str) -> None:
        self._refreshing = True
        current = self.combo.currentData()
        self.combo.clear()
        df = catalog.search(text).select(["name", "available", "category"])
        for name, available, cat in df.iter_rows():
            label = f"{name}  [{cat}]" if available else f"{name}  [{cat}] (not available)"
            self.combo.addItem(label, name)
        if current is not None:
            i = self.combo.findData(current)
            if i >= 0:
                self.combo.setCurrentIndex(i)
        self._refreshing = False

    def _formula_chosen(self, i: int) -> None:
        if self._refreshing or i < 0:
            return
        name = self.combo.itemData(i)
        slot = self._ensure_slot()
        if slot.name.lower() == name.lower():
            return
        spec = registry.find(name)
        slot.name = spec.name if spec else name
        slot.fnr = spec.fnr if spec else 20
        defaults = (spec.defaults() if spec else catalog.defaults(name))
        slot.options = list(defaults) + [0.0] * (16 - len(defaults))
        slot.option_count = len(spec.options) if spec else len(defaults)
        if slot.iterations == 0:
            slot.iterations = 1
        self.model.touch("formulas")

    def _iterations_changed(self, v: int) -> None:
        if self._refreshing:
            return
        self._ensure_slot().iterations = int(v)
        self.model.touch("formulas")

    def _option_changed(self, k: int, v: float) -> None:
        if self._refreshing:
            return
        self._ensure_slot().options[k] = float(v)
        self.model.touch("formulas")

    # --------------------------------------------------------- refresh
    def refresh(self) -> None:
        self._refreshing = True
        try:
            slot = self.slot()
            name = slot.name if slot else "Integer Power"
            i = self.combo.findData(name)
            if i < 0:
                i = self.combo.findData(next((d for d in (self.combo.itemData(k) for k in range(self.combo.count())) if d and d.lower() == name.lower()), None))
            if i < 0:
                self.combo.addItem(f"{name} (unknown)", name)
                i = self.combo.count() - 1
            self.combo.setCurrentIndex(i)
            self.iterations.setValue(slot.iterations if slot else 0)
            spec = registry.find(name)
            if spec is None:
                self.info.setStyleSheet("color: #a33;")
                self.info.setText("This formula is an MB3D machine code formula (.m3f) and has no native version yet; "
                                  "the scene cannot be rendered until it is replaced.")
            else:
                self.info.setStyleSheet("color: #555;")
                self.info.setText(spec.description)
            self._rebuild_options(slot, spec)
        finally:
            self._refreshing = False

    def _rebuild_options(self, slot: FormulaSlot | None, spec) -> None:
        while self.options_form.rowCount():
            self.options_form.removeRow(0)
        self.option_spins = []
        if slot is None:
            return
        names = [o.name for o in spec.options] if spec else catalog.option_names(slot.name)
        if not names:
            names = [f"Option {k + 1}" for k in range(slot.option_count)]
        for k, n in enumerate(names[:16]):
            sp = QDoubleSpinBox()
            sp.setRange(-1e9, 1e9)
            sp.setDecimals(6)
            sp.setSingleStep(0.1)
            sp.setKeyboardTracking(False)
            sp.setValue(slot.options[k] if k < len(slot.options) else 0.0)
            sp.valueChanged.connect(lambda v, k=k: self._option_changed(k, v))
            self.options_form.addRow(n, sp)
            self.option_spins.append(sp)


class FormulaPanel(QWidget):
    def __init__(self, model: SceneModel, parent=None):
        super().__init__(parent)
        self.model = model
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.slots = [SlotWidget(model, i) for i in range(6)]
        for i, w in enumerate(self.slots):
            self.tabs.addTab(w, f"F{i + 1}")
        layout.addWidget(self.tabs)
        f = QFormLayout()
        self.repeat = QSpinBox()
        self.repeat.setRange(0, 5)
        self.repeat.setToolTip("After the last slot the hybrid continues from this slot")
        self.repeat.valueChanged.connect(self._repeat_changed)
        f.addRow("Repeat from slot", self.repeat)
        layout.addLayout(f)
        self._refreshing = False
        model.changed.connect(self._on_changed)
        self.refresh()

    def _repeat_changed(self, v: int) -> None:
        if not self._refreshing:
            self.model.scene.repeat_from = int(v)
            self.model.touch("formulas")

    def _on_changed(self, what: str) -> None:
        if what in ("formulas", "scene"):
            self.refresh()

    def refresh(self) -> None:
        self._refreshing = True
        try:
            for i, w in enumerate(self.slots):
                w.refresh()
                slot = w.slot()
                self.tabs.setTabText(i, f"F{i + 1}" + (" •" if slot and slot.used else ""))
            self.repeat.setValue(self.model.scene.repeat_from)
        finally:
            self._refreshing = False
