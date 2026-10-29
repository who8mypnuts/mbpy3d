"""Calculation: iterations, distance estimation, ray stepping, Julia mode."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QVBoxLayout, QWidget

from ..model import SceneModel
from ..widgets import Binder, attr_getter, attr_setter, tuple_setter


class CalcPanel(QWidget):
    def __init__(self, model: SceneModel, parent=None):
        super().__init__(parent)
        self.b = Binder(model, "calc")
        layout = QVBoxLayout(self)
        g = QGroupBox("Iterations")
        f = QFormLayout(g)
        self.b.int(f, "Max iterations", attr_getter("iterations"), attr_setter("iterations"), 1, 100000)
        self.b.int(f, "Min iterations", attr_getter("min_iterations"), attr_setter("min_iterations"), 0, 100000)
        self.b.float(f, "Bailout R (0 = default)", lambda s: s.rstop or 0.0,
                     lambda s, v: setattr(s, "rstop", v if v > 1.0001 else None), 0.0, 1e9, 1.0, 3)
        layout.addWidget(g)
        g = QGroupBox("Ray marching")
        f = QFormLayout(g)
        self.b.float(f, "DE stop (pixels)", attr_getter("de_stop"), attr_setter("de_stop"), 0.001, 100.0, 0.1, 3)
        self.b.float(f, "Raystep multiplier", attr_getter("z_step_div"), attr_setter("z_step_div"), 0.001, 1.0, 0.05, 3)
        self.b.float(f, "Raystep limiter", attr_getter("raystep_limiter"), attr_setter("raystep_limiter"), 0.01, 100.0, 0.1, 2)
        self.b.int(f, "Binary search steps", attr_getter("bin_search_steps"), attr_setter("bin_search_steps"), 0, 64)
        self.b.check(f, "Vary DE stop on FOV", attr_getter("vary_de_stop_on_fov"), attr_setter("vary_de_stop_on_fov"))
        self.b.check(f, "First step random", attr_getter("first_step_random"), attr_setter("first_step_random"))
        self.b.check(f, "Step sub DE stop", attr_getter("step_sub_de_stop"), attr_setter("step_sub_de_stop"))
        self.b.check(f, "Disable analytic DE", attr_getter("disable_analytic_de"), attr_setter("disable_analytic_de"))
        layout.addWidget(g)
        g = QGroupBox("Julia / 4D")
        f = QFormLayout(g)
        self.b.check(f, "Julia mode", attr_getter("julia"), attr_setter("julia"))
        for i, n in enumerate("XYZW"):
            self.b.float(f, f"Julia {n}", lambda s, i=i: s.julia_c[i], tuple_setter("julia_c", i), -100, 100, 0.01, 6)
        for i, n in enumerate(("XW", "YW", "ZW")):
            self.b.float(f, f"4D rotation {n} (rad)", lambda s, i=i: s.rot_4d[i], tuple_setter("rot_4d", i), -10, 10, 0.01, 4)
        layout.addWidget(g)
        g = QGroupBox("Colouring")
        f = QFormLayout(g)
        self.b.combo(f, "2nd colour choice", ["Orbit trap", "Last length increase", "XY angle", "XZ angle", "YZ angle"],
                     lambda s: min(4, s.color_option), attr_setter("color_option"))
        self.b.float(f, "Colour multiplier", attr_getter("color_mul"), attr_setter("color_mul"), -1000, 1000, 0.1, 3)
        layout.addWidget(g)
        layout.addStretch(1)
        self.b.refresh()
