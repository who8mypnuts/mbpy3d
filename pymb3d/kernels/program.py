"""A *program*: the formula combination of a scene, compiled once per backend.

The program owns the generated formula code (C and Python), the hybrid
loops specialised for its slots (:mod:`loopgen`), the DE mode decision of
each hybrid part and the packing of a scene into the kernel parameter
block.  Two scenes with the same formula names (and DE combination layout)
share a program; their option values are runtime parameters, so changing a
slider never recompiles a kernel.
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
from . import loopgen

KERNEL_DIR = Path(__file__).parent / "clike"

DE_NUMERIC, DE_LINEAR, DE_LOG, DE_DIFS = 0, 1, 2, 3
DE_MODE_NAMES = {DE_NUMERIC: "numeric", DE_LINEAR: "linear", DE_LOG: "log", DE_DIFS: "dIFS"}


@dataclass
class ProgramSlot:
    spec: FormulaSpec
    values: dict[str, float]
    iterations: int

    @property
    def counted(self) -> bool:
        return self.spec.counted and self.spec.de != "difs_transform"


@dataclass
class Part:
    """One hybrid part: a range of program slots."""

    slots: list[int]
    de_mode: int
    de_scale: float = 1.0
    col_plus: float = 0.0
    col_var: float = 0.6

    @property
    def is_difs(self) -> bool:
        return self.de_mode == DE_DIFS


@dataclass
class Program:
    slots: list[ProgramSlot]
    analytic_de: bool = True
    #: first slot of the second part of a DE combination, None = one part
    start2: int | None = None
    #: interpolation hybrid of the first two slots (weights are parameters)
    ipol: bool = False
    _c_cache: str | None = field(default=None, repr=False)
    _py_cache: str | None = field(default=None, repr=False)
    _ipol_weights: tuple[float, float] = (1.0, 0.0)

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
        ipol = scene.interpolation is not None
        if ipol:
            if len(slots) < 2:
                raise ValueError("an interpolation hybrid needs two formulas")
            slots = slots[:2]
            for sl in slots:
                sl.iterations = 1
        start2 = None
        if scene.decomb.kind > 0 and not ipol:
            # scene slot index -> program slot index (inactive slots dropped)
            active = [i for i, s in enumerate(scene.slots[: PL.MAX_SLOTS]) if s.active]
            start2 = sum(1 for i in active if i < scene.decomb.start2)
            if not (1 <= start2 < len(slots)):
                start2 = None  # a part without active slots: plain hybrid
        p = cls(slots, analytic_de=scene.analytic_de, start2=start2, ipol=ipol)
        p.mix_kind = start2 is not None and scene.decomb.kind == 6
        p.validate()
        return p

    def validate(self) -> None:
        for part in self.parts:
            specs = [self.slots[i].spec for i in part.slots]
            difs = [s.is_difs for s in specs]
            if any(difs) and not all(difs):
                raise ValueError("dIFS formulas cannot be mixed with other formulas in one hybrid part")
            if all(difs) and not any(s.de == "difs_shape" for s in specs):
                raise ValueError("a dIFS hybrid needs a shape formula")
            if self.ipol and any(difs):
                raise ValueError("dIFS formulas do not work in an interpolation hybrid")
        if self.mix_kind and not self.parts[1].is_difs:
            raise ValueError("DE combination 'dIFS mix': the second hybrid part must be dIFS")

    # ---------------------------------------------------------------- keys
    @property
    def formula_names(self) -> tuple[str, ...]:
        return tuple(s.spec.name for s in self.slots)

    @property
    def key(self) -> str:
        """Identifies the generated code (formula names in order, the split)."""
        txt = "|".join(self.formula_names) + f"#{self.start2}#{int(self.analytic_de)}#{int(self.ipol)}"
        return hashlib.sha1(txt.encode()).hexdigest()[:12]

    @property
    def four_d(self) -> bool:
        return any(s.spec.four_d for s in self.slots)

    @property
    def parts(self) -> list[Part]:
        n = len(self.slots)
        if self.start2 is None:
            return [Part(list(range(n)), self._mode_of(range(n)))]
        return [Part(list(range(self.start2)), self._mode_of(range(self.start2))),
                Part(list(range(self.start2, n)), self._mode_of(range(self.start2, n)))]

    def _mode_of(self, indices) -> int:
        specs = [self.slots[i].spec for i in indices]
        if any(s.is_difs for s in specs):
            return DE_DIFS
        if not self.analytic_de:
            return DE_NUMERIC
        modes = {s.de for s in specs if (s.counted or self.ipol)}
        if modes == {"linear"}:
            return DE_LINEAR
        if modes == {"log"}:
            return DE_LOG
        return DE_NUMERIC

    #: DE combination kind 6 (dIFS mix) needs validation at program build
    mix_kind: bool = False

    @property
    def de_mode(self) -> int:
        return self.parts[0].de_mode

    @property
    def is_difs(self) -> bool:
        return self.parts[0].is_difs

    # ------------------------------------------------------------- sources
    def _helper_order(self) -> list[str]:
        seen: set[str] = set()
        order: list[str] = []
        for s in self.slots:
            for h in helpers_used(s.spec.func, seen):
                if h not in order:
                    order.append(h)
        return order

    def _loop_slots(self, part: Part) -> list[loopgen.LoopSlot]:
        return [loopgen.LoopSlot(i, self.slots[i].counted, f"P_SLOTS + {PL.SLOT_STRIDE * i}") for i in part.slots]

    def c_source(self) -> str:
        """Helpers, formulas, ``run_slot`` and the specialised loops."""
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
            ps = self.parts
            p1 = ps[0]
            p2 = ps[1] if len(ps) > 1 else None
            # only the loops a part uses are generated; the others are stubs
            if self.ipol:
                parts.append(loopgen.c_ipol("iterate", self._loop_slots(p1), self.four_d))
            else:
                parts.append(loopgen.c_iterate("iterate", self._loop_slots(p1), "P_REPEAT_FROM", self.four_d)
                             if not p1.is_difs else loopgen.c_iterate_stub("iterate"))
            parts.append(loopgen.c_difs("difs_orbit", self._loop_slots(p1), "P_REPEAT_FROM")
                         if p1.is_difs else loopgen.c_difs_stub("difs_orbit"))
            if p2 is None:
                parts.append(loopgen.c_iterate_stub("iterate_p2"))
                parts.append(loopgen.c_difs_stub("difs_orbit_p2"))
            else:
                parts.append(loopgen.c_iterate("iterate_p2", self._loop_slots(p2), "P_REPEAT_FROM2", self.four_d)
                             if not p2.is_difs else loopgen.c_iterate_stub("iterate_p2"))
                parts.append(loopgen.c_difs("difs_orbit_p2", self._loop_slots(p2), "P_REPEAT_FROM2")
                             if p2.is_difs else loopgen.c_difs_stub("difs_orbit_p2"))
            self._c_cache = "\n\n".join(parts) + "\n"
        return self._c_cache

    def py_source(self) -> str:
        """The complete numba module: prelude, helpers, formulas, run_slot,
        the loops and the marching core."""
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
            st = loopgen.PY_STATE
            lines = [loopgen.DECO, f"def run_slot(n, {st}, P):"]
            for i in range(len(self.slots)):
                kw = "if" if i == 0 else "elif"
                lines.append(f"    {kw} n == {i}:")
                lines.append(f"        return formula_{i}({st}, P)")
            lines.append(f"    return {loopgen.PY_CALL_RET}")
            parts.append("\n".join(lines))
            ps = self.parts
            p1 = ps[0]
            p2 = ps[1] if len(ps) > 1 else None
            if self.ipol:
                parts.append(loopgen.py_ipol("iterate", self._loop_slots(p1), self.four_d))
            else:
                parts.append(loopgen.py_iterate("iterate", self._loop_slots(p1), "P_REPEAT_FROM", self.four_d)
                             if not p1.is_difs else loopgen.py_iterate_stub("iterate"))
            parts.append(loopgen.py_difs("difs_orbit", self._loop_slots(p1), "P_REPEAT_FROM")
                         if p1.is_difs else loopgen.py_difs_stub("difs_orbit"))
            if p2 is None:
                parts.append("iterate_p2 = iterate\ndifs_orbit_p2 = difs_orbit")
            else:
                parts.append(loopgen.py_iterate("iterate_p2", self._loop_slots(p2), "P_REPEAT_FROM2", self.four_d)
                             if not p2.is_difs else loopgen.py_iterate_stub("iterate_p2"))
                parts.append(loopgen.py_difs("difs_orbit_p2", self._loop_slots(p2), "P_REPEAT_FROM2")
                             if p2.is_difs else loopgen.py_difs_stub("difs_orbit_p2"))
            parts.append(NUMBA_CORE)
            self._py_cache = "\n\n".join(parts) + "\n"
        return self._py_cache

    def glsl_source(self) -> str:
        return "\n".join([
            (KERNEL_DIR / "prelude_glsl.glsl").read_text(),
            PL.c_defines(),
            f"uniform float u_params[{PL.N_PARAMS}];",
            (KERNEL_DIR / "types.c").read_text(),
            (KERNEL_DIR / "orbit.c").read_text(),
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
            (KERNEL_DIR / "orbit.c").read_text(),
            self.c_source(),
            (KERNEL_DIR / "march_core.c").read_text(),
            (KERNEL_DIR / "wrapper_opencl.cl").read_text(),
        ])

    # -------------------------------------------------------------- params
    def effective_rstop(self, scene: Scene) -> float:
        if scene.rstop is not None:
            return max(1.0001, float(scene.rstop))
        return max([s.spec.rstop for s in self.slots] + [1.0001])

    def de_heuristics(self, part: Part, max_it: int) -> tuple[float, float, float]:
        """(DE scale, colour offset, colour/DE-stop coupling) of a part: the
        heuristics of `GetMCTparasFromHeader`."""
        if part.is_difs:
            return 1.0, max_it * 0.1, 0.0
        analytic = part.de_mode != DE_NUMERIC
        n = x1 = y1 = i = 0.0
        cy1 = cz1 = 0.0
        for k, idx in enumerate(part.slots):
            s = self.slots[idx]
            if not s.counted and not self.ipol:
                continue
            cnt = float(s.iterations) if not self.ipol else float(self._ipol_weights[k])
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
        de_scale = (x1 if self.ipol else x1 / n) if n > 0 else 1.0
        col_plus = (abs(y1 / i) ** 0.25 * 40.0 - 49.0) if (n > 0 and i > 0) else 0.0
        col_plus += max_it * 0.1
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
        P[PL.P_PIXEL_STEP] = 1.0
        P[PL.P_YSTART:PL.P_YSTART + 3] = cam.ystart
        P[PL.P_VGRADS:PL.P_VGRADS + 9] = cam.vgrads.reshape(-1)
        P[PL.P_FOV_Y] = cam.fov_y
        P[PL.P_FOVX_MUL] = cam.fovx_mul
        P[PL.P_FOVX_OFF] = cam.fovx_off
        P[PL.P_ZEND] = cam.zend
        P[PL.P_DE_STOP_FACTOR] = cam.de_stop_factor
        P[PL.P_STEP_WIDTH] = cam.step_width
        parts = self.parts
        p1 = parts[0]
        p2 = parts[1] if len(parts) > 1 else None
        max_it = max(1, scene.iterations)
        P[PL.P_MAX_IT] = max_it
        P[PL.P_INSIDE_IT] = max_it + (1 if p1.is_difs else 0)
        min_it = max(0, min(scene.min_iterations, max_it))
        rstop = self.effective_rstop(scene)
        P[PL.P_RSTOP2] = rstop * rstop
        P[PL.P_LN_RSTOP] = math.log(math.log(rstop))
        P[PL.P_DE_STOP] = max(0.001, scene.de_stop)
        zsd = min(1.0, max(0.0001, scene.z_step_div))
        ms_de_sub = 0.0
        if scene.step_sub_de_stop:
            zsd = zsd * zsd + (1.2 * zsd) * (1.0 - zsd)
            ms_de_sub = min(math.sqrt(zsd), 0.9)
        P[PL.P_Z_STEP_DIV] = zsd
        bin_search = max(0, scene.bin_search_steps)
        mh04zsd = cam.mh04zsd
        P[PL.P_FIRST_STEP_RANDOM] = 1.0 if scene.first_step_random else 0.0
        P[PL.P_DE_MODE] = p1.de_mode
        if self.ipol:
            w1, w2 = scene.interpolation or (1.0, 0.0)
            tot = w1 + w2
            w1, w2 = ((w1 / tot, w2 / tot) if tot > 1e-10 else (1.0, 0.0))
            self._ipol_weights = (w1, w2)
            P[PL.P_IPOL_W1], P[PL.P_IPOL_W2] = w1, w2
        P[PL.P_INSIDE_MODE] = min(2, max(0, scene.inside_mode))
        P[PL.P_SMOOTH_NORMALS] = min(8, max(0, scene.smooth_normals))
        P[PL.P_MC_SAMPLES] = scene.ao.samples if scene.ao.mode == "mc" else 0
        P[PL.P_MC_SEED] = 1.0
        P[PL.P_MC_RANGE] = max(1.0, scene.ao.radius) * 8.0
        vl = scene.lighting.vol_light
        P[PL.P_VOL_LIGHT] = vl if (0 < vl <= PL.MAX_LIGHTS and scene.lighting.lights[vl - 1].enabled) else 0
        P[PL.P_VOL_STEP] = 3
        P[PL.P_REFLECT] = min(1.0, max(0.0, scene.lighting.reflection))
        P[PL.P_SLICE_2D] = min(3, max(0, scene.slice_2d))
        de_scale, col_plus, col_var = self.de_heuristics(p1, max_it)
        P[PL.P_DE_SCALE] = de_scale
        P[PL.P_DE_OFFSET_PX] = min(P[PL.P_DE_STOP] * 0.1, 0.004)
        P[PL.P_JULIA] = 1.0 if scene.julia else 0.0
        P[PL.P_JULIA_C:PL.P_JULIA_C + 4] = scene.julia_c
        P[PL.P_NSLOTS] = len(p1.slots)
        P[PL.P_REPEAT_FROM] = self._repeat_index(scene.repeat_from, scene, p1)
        any_difs = p1.is_difs or (p2 is not None and p2.is_difs)
        if any_difs:
            # HeaderTrafos: more binary search steps, any iteration count valid
            bin_search = max(bin_search, int(round(min(zsd, 1.0) * 2.0)) + 2)
            if scene.decomb.kind < 5:
                min_it = 1
        if p1.is_difs and (p2 is None or p2.is_difs):
            mh04zsd = float(max(W, H))
            col_var = 0.0
        P[PL.P_MIN_IT] = min_it
        P[PL.P_BIN_SEARCH] = bin_search
        P[PL.P_MH04ZSD] = mh04zsd
        P[PL.P_COL_PLUS] = col_plus
        P[PL.P_COL_VAR] = col_var
        P[PL.P_COLOR_OPTION] = min(5, max(0, scene.color_option))
        P[PL.P_COLOR_MUL] = scene.color_mul * 512.0
        its = max_it
        if p2 is not None:
            its2 = max(1, scene.decomb.iterations2)
            its = max(max_it, its2)
            P[PL.P_DECOMB_KIND] = min(6, max(1, scene.decomb.kind))
            P[PL.P_DECOMB_SMOOTH] = max(1e-30, scene.decomb.smooth) / cam.step_width
            P[PL.P_DECOMB_MIX_POW] = scene.decomb.mix_pow
            P[PL.P_DECOMB_MIX_COL] = scene.decomb.mix_color
            P[PL.P_DE_MODE2] = p2.de_mode
            de_scale2, _, _ = self.de_heuristics(p2, its2)
            P[PL.P_DE_SCALE2] = de_scale2
            P[PL.P_MAX_IT2] = its2
            P[PL.P_INSIDE_IT2] = its2 + (1 if p2.is_difs else 0)
            P[PL.P_REPEAT_FROM2] = self._repeat_index(scene.decomb.repeat_from2, scene, p2)
            P[PL.P_NSLOTS2] = len(p2.slots)
        P[PL.P_MCTS_M] = 32767.0 / (its + 1)
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
        P[PL.P_MS_DE_SUB] = ms_de_sub
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

    def _repeat_index(self, scene_index: int, scene: Scene, part: Part) -> int:
        """The repeat slot of a part, relative to the part (scene slot
        indices count inactive slots)."""
        active = [i for i, s in enumerate(scene.slots[: PL.MAX_SLOTS]) if s.active]
        prog_index = sum(1 for i in active if i < scene_index)
        rel = prog_index - part.slots[0]
        return min(max(0, rel), len(part.slots) - 1)

    @staticmethod
    def _pack_lights(scene: Scene, P: np.ndarray) -> None:
        vg = scene.camera.vgrads
        n = 0
        for i, light in enumerate(scene.lighting.lights[: PL.MAX_LIGHTS]):
            o = PL.P_LIGHT_DIRS + 3 * i
            if light.positional:
                P[o:o + 3] = light.position
                P[PL.P_LIGHT_FLAGS + i] = 2.0 if (light.enabled and light.shadows) else 0.0
            else:
                d = np.asarray(light.direction_view(), dtype=float)
                world = d if light.relative_to_object else d @ vg
                P[o:o + 3] = world / (np.linalg.norm(world) or 1.0)
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
