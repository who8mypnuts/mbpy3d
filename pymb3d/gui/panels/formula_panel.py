"""The Formulas panel: the six hybrid slots, their options and the
iteration settings."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...core.scene import MAX_SLOTS, FormulaSlot
from ...formulas import registry
from ...formulas.catalog import catalog
from ..scene_model import SceneModel
from ..widgets import DoubleSpin, IntSpin

NONE = "(none)"


class SlotEditor(QGroupBox):
    """One hybrid slot: formula, iteration count and the option values."""

    def __init__(self, index: int, model: SceneModel, parent: QWidget | None = None) -> None:
        super().__init__(f"Slot {index + 1}", parent)
        self.index = index
        self.model = model
        self._loading = False
        self.combo = QComboBox()
        self.combo.addItem(NONE)
        df = catalog()
        for cat in df["category"].unique().sort().to_list():
            for name in df.filter(df["category"] == cat)["name"].to_list():
                self.combo.addItem(f"{name}", name)
        self.iters = IntSpin(1, 0, 10000)
        self.iters.setToolTip("Consecutive iterations of this formula (0 disables the slot)")
        top = QHBoxLayout()
        top.addWidget(self.combo, 1)
        top.addWidget(QLabel("x"))
        top.addWidget(self.iters)
        self.params_box = QWidget()
        self.params_form = QFormLayout(self.params_box)
        self.params_form.setContentsMargins(0, 0, 0, 0)
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color: #888; font-size: 11px;")
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.params_box)
        lay.addWidget(self.desc)
        self._spins: dict[str, DoubleSpin] = {}
        self.combo.currentIndexChanged.connect(self._formula_changed)
        self.iters.valueChanged.connect(self._iters_changed)

    # --------------------------------------------------------------- model
    def slot(self) -> FormulaSlot | None:
        slots = self.model.scene.slots
        return slots[self.index] if self.index < len(slots) else None

    def load(self) -> None:
        self._loading = True
        s = self.slot()
        if s is None or not registry.has(s.name):
            self.combo.setCurrentIndex(0)
            self.iters.set_silently(0)
            self._build_params(None)
        else:
            spec = registry.lookup(s.name)
            i = self.combo.findData(spec.name)
            self.combo.setCurrentIndex(max(0, i))
            self.iters.set_silently(s.iterations)
            self._build_params(spec, s.params)
        self._loading = False

    def _build_params(self, spec, values: dict | None = None) -> None:
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._spins.clear()
        if spec is None:
            self.desc.setText("")
            return
        vals = spec.values(values)
        for p in spec.editable_params:
            step = 1.0 if p.kind in ("int", "bool") else 0.05
            dec = 0 if p.kind in ("int", "bool") else 6
            spin = DoubleSpin(vals[p.name], p.minimum if p.minimum is not None else -1e9,
                              p.maximum if p.maximum is not None else 1e9, dec, step)
            spin.setToolTip(p.description or p.name)
            spin.valueChanged.connect(lambda v, name=p.name: self._param_changed(name, v))
            self.params_form.addRow(p.name, spin)
            self._spins[p.name] = spin
        self.desc.setText(f"{spec.description}  [DE: {spec.de}, bailout {spec.rstop:g}]")

    # ------------------------------------------------------------- slots
    def _ensure_slot(self) -> FormulaSlot:
        slots = self.model.scene.slots
        while len(slots) <= self.index:
            slots.append(FormulaSlot("Integer Power", 0, {}))
        return slots[self.index]

    def _formula_changed(self, _i: int) -> None:
        if self._loading:
            return
        self.model.snapshot()
        name = self.combo.currentData()
        s = self._ensure_slot()
        if name is None:
            s.iterations = 0
            self.iters.set_silently(0)
            self._build_params(None)
        else:
            spec = registry.lookup(name)
            s.name = spec.name
            s.params = spec.defaults()
            if s.iterations == 0:
                s.iterations = 1
                self.iters.set_silently(1)
            self._build_params(spec, s.params)
        self.model.touch("formulas")

    def _iters_changed(self, v: int) -> None:
        if self._loading:
            return
        self.model.snapshot()
        s = self._ensure_slot()
        s.iterations = int(v)
        if v > 0 and self.combo.currentIndex() == 0:
            self.combo.setCurrentIndex(1)
            return
        self.model.touch("formulas")

    def _param_changed(self, name: str, v: float) -> None:
        if self._loading:
            return
        s = self._ensure_slot()
        s.params[name] = float(v)
        self.model.touch("params")


class FormulaPanel(QWidget):
    def __init__(self, model: SceneModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self._loading = False
        inner = QWidget()
        lay = QVBoxLayout(inner)
        general = QGroupBox("Iteration")
        form = QFormLayout(general)
        self.iterations = IntSpin(12, 1, 100000)
        self.min_iterations = IntSpin(0, 0, 100000)
        self.repeat_from = IntSpin(0, 0, MAX_SLOTS - 1)
        self.repeat_from.setToolTip("After the last active slot the hybrid continues from this slot (0 based)")
        self.rstop = DoubleSpin(0.0, 0.0, 1e9, 3, 1.0)
        self.rstop.setSpecialValueText("formula default")
        self.analytic = QCheckBox("analytic DE when every formula provides one")
        self.analytic.setChecked(True)
        self.julia = QCheckBox("Julia mode")
        jl = QHBoxLayout()
        self.julia_c = [DoubleSpin(0.0, -100, 100, 6, 0.01) for _ in range(4)]
        for sp in self.julia_c:
            jl.addWidget(sp)
        form.addRow("max iterations", self.iterations)
        form.addRow("min iterations", self.min_iterations)
        form.addRow("repeat from slot", self.repeat_from)
        form.addRow("bailout radius", self.rstop)
        form.addRow(self.analytic)
        form.addRow(self.julia)
        form.addRow("Julia seed x y z w", jl)
        lay.addWidget(general)
        self.editors = [SlotEditor(i, model) for i in range(MAX_SLOTS)]
        for e in self.editors:
            lay.addWidget(e)
        lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.iterations.valueChanged.connect(lambda v: self._set("iterations", int(v)))
        self.min_iterations.valueChanged.connect(lambda v: self._set("min_iterations", int(v)))
        self.repeat_from.valueChanged.connect(lambda v: self._set("repeat_from", int(v)))
        self.rstop.valueChanged.connect(lambda v: self._set("rstop", None if v <= 0 else float(v)))
        self.analytic.toggled.connect(lambda v: self._set("analytic_de", bool(v)))
        self.julia.toggled.connect(lambda v: self._set("julia", bool(v)))
        for sp in self.julia_c:
            sp.valueChanged.connect(self._julia_changed)
        model.replaced.connect(self.load)
        self.load()

    def load(self) -> None:
        self._loading = True
        sc = self.model.scene
        self.iterations.set_silently(sc.iterations)
        self.min_iterations.set_silently(sc.min_iterations)
        self.repeat_from.set_silently(sc.repeat_from)
        self.rstop.set_silently(sc.rstop or 0.0)
        self.analytic.setChecked(sc.analytic_de)
        self.julia.setChecked(sc.julia)
        for sp, v in zip(self.julia_c, sc.julia_c):
            sp.set_silently(v)
        for e in self.editors:
            e.load()
        self._loading = False

    def _set(self, attr: str, value) -> None:
        if self._loading:
            return
        self.model.snapshot()
        setattr(self.model.scene, attr, value)
        self.model.touch("formulas")

    def _julia_changed(self, _v: float) -> None:
        if self._loading:
            return
        self.model.scene.julia_c = tuple(sp.value() for sp in self.julia_c)  # type: ignore[assignment]
        self.model.touch("formulas")
