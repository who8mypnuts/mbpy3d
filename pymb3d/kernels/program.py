"""A *program*: the formula combination of a scene, compiled once per backend.

The program owns the generated formula code (C and Python), the DE mode
decision and the packing of a scene into the kernel parameter block.  Two
scenes with the same formula names share a program (their option values are
runtime parameters), so changing a slider never recompiles a kernel.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core import params as PL
from ..core.camera import CameraSetup
from ..core.scene import Scene
from ..formulas import registry
from ..formulas.dsl import FormulaSpec, helpers
from ..formulas.transpile import PY_PRELUDE, CEmitter, PyEmitter, helpers_used

KERNEL_DIR = Path(__file__).parent / "clike"

DE_NUMERIC, DE_LINEAR, DE_LOG = 0, 1, 2


@dataclass
class ProgramSlot:
    spec: FormulaSpec
    values: dict[str, float]
    iterations: int

    @property
    def counted(self) -> bool:
        return self.spec.counted


@dataclass
class Program:
    slots: list[ProgramSlot]
    analytic_de: bool = True
    _c_cache: str | None = field(default=None, repr=False)
    _py_cache: str | None = field(default=None, repr=False)

    # ------------------------------------------------------------- factory
    @classmethod
    def from_scene(cls, scene: Scene) -> "Program":
        slots = []
        for s in scene.slots[: PL.MAX_SLOTS]:
            if not s.active:
                continue
            spec = registry.lookup(s.name)
            slots.append(ProgramSlot(spec, spec.values(s.params), int(s.iterations)))
        if not slots:
            raise ValueError("no formula slot with iterations > 0")
        return cls(slots, analytic_de=scene.analytic_de)

    # ---------------------------------------------------------------- keys
    @property
    def formula_names(self) -> tuple[str, ...]:
        return tuple(s.spec.name for s in self.slots)

    @property
    def key(self) -> str:
        """Identifies the generated code (formula names in order)."""
        h = hashlib.sha1("|".join(self.formula_names).encode()).hexdigest()[:12]
        return h

    @property
    def four_d(self) -> bool:
        return any(s.spec.four_d for s in self.slots)

    @property
    def de_mode(self) -> int:
        if not self.analytic_de:
            return DE_NUMERIC
        modes = {s.spec.de for s in self.slots if s.counted}
        if modes == {"linear"}:
            return DE_LINEAR
        if modes == {"log"}:
            return DE_LOG
        return DE_NUMERIC

    # ------------------------------------------------------------- sources
    def _helper_order(self) -> list[str]:
        seen: set[str] = set()
        order: list[str] = []
        for s in self.slots:
            for h in helpers_used(s.spec.func, seen):
                if h not in order:
                    order.append(h)
        return order

    def c_source(self) -> str:
        """Helpers, formulas and ``run_slot`` in the C-like dialect."""
        if self._c_cache is None:
            parts = []
            for h in self._helper_order():
                parts.append(CEmitter(helpers()[h].func).emit())
            for i, s in enumerate(self.slots):
                pidx = {p.name: k for k, p in enumerate(s.spec.params)}
                parts.append(CEmitter(s.spec.func, slot=i, param_index=pidx).emit())
            lines = ["State run_slot(FN_ARGS int n, State s) {"]
            for i in range(len(self.slots)):
                kw = "if" if i == 0 else "} else if"
                lines.append(f"    {kw} (n == {i}) {{")
                lines.append(f"        return formula_{i}(FN_PASS s);")
            lines.append("    }")
            lines.append("    return s;")
            lines.append("}")
            parts.append("\n".join(lines))
            self._c_cache = "\n\n".join(parts) + "\n"
        return self._c_cache

    def py_source(self) -> str:
        """The complete numba module: prelude, helpers, formulas, run_slot
        and the marching core."""
        if self._py_cache is None:
            from .numba_core import NUMBA_CORE

            parts = [PY_PRELUDE]
            parts.append("\n".join(f"{k} = {v}" for k, v in PL.py_constants().items()) + "\n")
            for h in self._helper_order():
                parts.append(PyEmitter(helpers()[h].func).emit())
            for i, s in enumerate(self.slots):
                pidx = {p.name: k for k, p in enumerate(s.spec.params)}
                parts.append(PyEmitter(s.spec.func, slot=i, param_index=pidx,
                                       param_base=PL.P_FORMULA + i * PL.FSTRIDE).emit())
            lines = ["@njit(cache=True, nogil=True, fastmath=False, error_model=\"numpy\")",
                     "def run_slot(n, s_zx, s_zy, s_zz, s_w, s_cx, s_cy, s_cz, s_cw, s_dr, s_r2, P):"]
            for i in range(len(self.slots)):
                kw = "if" if i == 0 else "elif"
                lines.append(f"    {kw} n == {i}:")
                lines.append(f"        return formula_{i}(s_zx, s_zy, s_zz, s_w, s_cx, s_cy, s_cz, s_cw, s_dr, s_r2, P)")
            lines.append("    return s_zx, s_zy, s_zz, s_w, s_cx, s_cy, s_cz, s_dr")
            parts.append("\n".join(lines))
            parts.append(NUMBA_CORE)
            self._py_cache = "\n\n".join(parts) + "\n"
        return self._py_cache

    def glsl_source(self) -> str:
        return "\n".join([
            (KERNEL_DIR / "prelude_glsl.glsl").read_text(),
            PL.c_defines(),
            f"uniform float u_params[{PL.N_PARAMS}];",
            (KERNEL_DIR / "types.c").read_text(),
            self.c_source(),
            (KERNEL_DIR / "march_core.c").read_text(),
            (KERNEL_DIR / "wrapper_glsl.glsl").read_text(),
        ])

    def opencl_source(self, double: bool) -> str:
        return "\n".join([
            "#define USE_DOUBLE 1" if double else "",
            (KERNEL_DIR / "prelude_opencl.cl").read_text(),
            PL.c_defines(),
            (KERNEL_DIR / "types.c").read_text(),
            self.c_source(),
            (KERNEL_DIR / "march_core.c").read_text(),
            (KERNEL_DIR / "wrapper_opencl.cl").read_text(),
        ])

    # -------------------------------------------------------------- params
    def effective_rstop(self, scene: Scene) -> float:
        if scene.rstop is not None:
            return max(1.0001, float(scene.rstop))
        return max([s.spec.rstop for s in self.slots] + [1.0001])

    def de_heuristics(self, scene: Scene) -> tuple[float, float, float]:
        """(combined DE scale, colour offset, colour/DE-stop coupling), the
        heuristics of `GetMCTparasFromHeader`."""
        analytic = self.de_mode != DE_NUMERIC
        n = x1 = y1 = i = 0.0
        cy1 = cz1 = 0.0
        for s in self.slots:
            if not s.counted:
                continue
            cnt = float(s.iterations)
            scale_name = s.spec.scale_param
            scale = s.values.get(scale_name) if scale_name else None
            if analytic and scale is not None and s.spec.de == "linear":
                n += cnt
                i += cnt
                x2 = scale if scale != 0 else 1e-40
                x1 += (0.65 * cnt) if x2 < 0 else (x2 * 1.2 + 1.0) / x2 * cnt
                y1 += abs(x2) * cnt
            else:
                n += cnt
                x1 += (s.spec.ade_scale if analytic else s.spec.de_scale) * cnt
            # colour / DE stop coupling
            if scale is None:
                cy1 += cnt * 0.6
                cz1 += cnt
            else:
                v = float(scale)
                if v > 1.0:
                    if v < 1.2:
                        v = 1.1 + (v - 1.0) ** 2 * 2.5
                    cy1 += cnt * 0.5 / (math.log(v) + 0.03)
                    cz1 += cnt
                elif v < -1.0:
                    if v > -1.2:
                        v = -1.1 - (v + 1.0) ** 2 * 2.5
                    cy1 += cnt * 0.5 / (math.log(-v) + 0.03)
                    cz1 += cnt
        de_scale = x1 / n if n > 0 else 1.0
        col_plus = (abs(y1 / i) ** 0.25 * 40.0 - 49.0) if (n > 0 and i > 0) else 0.0
        col_plus += scene.iterations * 0.1
        col_var = cy1 / cz1 if cz1 > 1e-3 else 0.6
        return de_scale, col_plus, col_var

    def pack(self, scene: Scene, rect: tuple[int, int, int, int] | None = None,
             dtype=np.float64) -> np.ndarray:
        """The parameter block for ``scene`` (``rect`` = x, y, w, h of the
        part calculated; the camera is set up for the full image)."""
        W, H = scene.width, scene.height
        rx, ry, rw, rh = rect or (0, 0, W, H)
        cam = CameraSetup.build(scene.camera, W, H, scene.raystep_limiter, scene.z_step_div,
                                scene.vary_de_stop_on_fov)
        P = np.zeros(PL.N_PARAMS, dtype=np.float64)
        P[PL.P_WIDTH], P[PL.P_HEIGHT] = rw, rh
        P[PL.P_RECT_X], P[PL.P_RECT_Y] = rx, ry
        P[PL.P_FULL_WIDTH], P[PL.P_FULL_HEIGHT] = W, H
        P[PL.P_YSTART:PL.P_YSTART + 3] = cam.ystart
        P[PL.P_VGRADS:PL.P_VGRADS + 9] = cam.vgrads.reshape(-1)
        P[PL.P_FOV_Y] = cam.fov_y
        P[PL.P_FOVX_MUL] = cam.fovx_mul
        P[PL.P_FOVX_OFF] = cam.fovx_off
        P[PL.P_ZEND] = cam.zend
        P[PL.P_DE_STOP_FACTOR] = cam.de_stop_factor
        P[PL.P_MH04ZSD] = cam.mh04zsd
        P[PL.P_STEP_WIDTH] = cam.step_width
        P[PL.P_MAX_IT] = max(1, scene.iterations)
        P[PL.P_MIN_IT] = max(0, min(scene.min_iterations, scene.iterations))
        rstop = self.effective_rstop(scene)
        P[PL.P_RSTOP2] = rstop * rstop
        P[PL.P_LN_RSTOP] = math.log(math.log(rstop))
        P[PL.P_DE_STOP] = max(0.001, scene.de_stop)
        P[PL.P_Z_STEP_DIV] = min(1.0, max(0.0001, scene.z_step_div))
        P[PL.P_BIN_SEARCH] = max(0, scene.bin_search_steps)
        P[PL.P_FIRST_STEP_RANDOM] = 1.0 if scene.first_step_random else 0.0
        P[PL.P_DE_MODE] = self.de_mode
        de_scale, col_plus, col_var = self.de_heuristics(scene)
        P[PL.P_DE_SCALE] = de_scale
        P[PL.P_DE_OFFSET_PX] = min(P[PL.P_DE_STOP] * 0.1, 0.004)
        P[PL.P_JULIA] = 1.0 if scene.julia else 0.0
        P[PL.P_JULIA_C:PL.P_JULIA_C + 4] = scene.julia_c
        P[PL.P_NSLOTS] = len(self.slots)
        P[PL.P_REPEAT_FROM] = min(max(0, scene.repeat_from), len(self.slots) - 1)
        P[PL.P_COL_PLUS] = col_plus
        P[PL.P_COL_VAR] = col_var
        P[PL.P_FOUR_D] = 1.0 if self.four_d else 0.0
        P[PL.P_AO_MODE] = {"none": 0, "ssao": 1, "de": 2}.get(scene.ao.mode, 1)
        P[PL.P_AO_SAMPLES] = max(1, scene.ao.samples)
        P[PL.P_AO_STRENGTH] = scene.ao.strength
        P[PL.P_AO_RADIUS] = max(0.5, scene.ao.radius * 0.25)
        P[PL.P_AO_FALLOFF] = 0.7
        P[PL.P_SHADOW_SOFTNESS] = 8.0
        P[PL.P_SHADOWS] = 1.0 if scene.shadows else 0.0
        self._pack_lights(scene, P)
        P[PL.P_CUT_OPTIONS] = scene.cut_options & 7
        P[PL.P_CUT_POS:PL.P_CUT_POS + 3] = scene.cut_pos
        P[PL.P_ROT4D:PL.P_ROT4D + 16] = rot4d_matrix(*scene.rot_4d).reshape(-1)
        P[PL.P_MS_DE_SUB] = 0.0
        for i, s in enumerate(self.slots):
            o = PL.P_SLOTS + i * PL.SLOT_STRIDE
            P[o] = s.iterations
            P[o + 1] = 0.0 if s.counted else 1.0
            sip = s.spec.smooth_pow(s.values)
            P[o + 2] = 1.0 / math.log(max(2.0, sip))
            fo = PL.P_FORMULA + i * PL.FSTRIDE
            for k, p in enumerate(s.spec.params[: PL.FSTRIDE]):
                P[fo + k] = s.values[p.name]
        return P.astype(dtype)

    @staticmethod
    def _pack_lights(scene: Scene, P: np.ndarray) -> None:
        vg = scene.camera.vgrads
        n = 0
        for i, light in enumerate(scene.lighting.lights[: PL.MAX_LIGHTS]):
            d = np.asarray(light.direction_view(), dtype=float)
            world = d if light.relative_to_object else d @ vg
            P[PL.P_LIGHT_DIRS + 3 * i:PL.P_LIGHT_DIRS + 3 * i + 3] = world / (np.linalg.norm(world) or 1.0)
            P[PL.P_LIGHT_FLAGS + i] = 1.0 if (light.enabled and light.shadows) else 0.0
            n = i + 1
        P[PL.P_N_LIGHTS] = n


def rot4d_matrix(xw: float, yw: float, zw: float) -> np.ndarray:
    """`BuildSMatrix4`: 4D rotation from the XW, YW and ZW plane angles."""
    def rot(i, j, a):
        m = np.eye(4)
        c, s = math.cos(a), math.sin(a)
        m[i, i] = c
        m[j, j] = c
        m[i, j] = -s
        m[j, i] = s
        return m

    return rot(2, 3, zw) @ rot(1, 3, yw) @ rot(0, 3, xw)
