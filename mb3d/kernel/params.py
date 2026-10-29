"""Scene -> kernel parameter arrays (port of ``GetMCTparasFromHeader``).

``KernelParams`` holds the three arrays every back end uploads: ``sp``
(reals), ``ip`` (ints) and ``fp`` (the formula options), plus what the
painter needs to interpret the G-buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mb3d.formulas import registry
from mb3d.formulas.spec import FormulaSpec
from mb3d.params.scene import CameraOptic, Scene

from .source import layout as L


class SceneError(ValueError):
    pass


@dataclass
class SlotConfig:
    spec: FormulaSpec
    iterations: int
    uncounted: bool
    options: list[float]


@dataclass
class RenderOptions:
    """Per render choices that are not part of the scene."""

    ao_samples: int = 0  # DE ambient occlusion samples in the kernel (0 = off)
    ao_length: float = 6.0  # in pixel steps
    shadow: bool = False  # soft shadow ray towards the first light
    shadow_dir: tuple[float, float, float] = (0.0, 0.0, 1.0)
    first_step_random: bool | None = None  # None: the scene's setting


@dataclass
class KernelParams:
    sp: np.ndarray
    ip: np.ndarray
    fp: np.ndarray
    width: int
    height: int
    step_width: float
    zend: float
    z_range: float
    de_stop: float
    slots: list[SlotConfig]
    de_mode: int
    max_it: int
    vgrads_unit: np.ndarray  # 3x3 unit rows
    ystart: np.ndarray
    fov_y: float

    @property
    def analytic(self) -> bool:
        return self.de_mode == 1


def resolve_slots(scene: Scene) -> list[SlotConfig]:
    slots = []
    for f in scene.formulas[:6]:
        if f.iterations == 0:
            continue
        spec = registry.find(f.name)
        if spec is None:
            raise SceneError(f"formula '{f.name}' is not available in this build")
        opts = list(f.options[:16]) + [0.0] * (16 - len(f.options[:16]))
        uncounted = f.iterations < 0 or spec.de_option < 0 or spec.de_option > 20
        slots.append(SlotConfig(spec, abs(f.iterations), uncounted, opts))
    if not slots:
        raise SceneError("no formula with iterations > 0")
    return slots


def _combine_de_option(slots: list[SlotConfig]) -> int:
    """``CheckFormulaOptions``: the DE option of the hybrid."""
    de_option = -1
    for s in slots:
        de = s.spec.de_option
        if de < 0 or de == de_option:
            continue
        if de_option in (-1, 21, 22):
            de_option = de
        elif de_option == 2:
            de_option = 2 if de in (2, 5, 6, 11) else 0
        elif de_option == 4:
            de_option = 4 if de in (5, 6) else 0
        elif de_option == 5:
            de_option = 4 if de == 4 else 2 if de in (2, 11) else 5 if de == 6 else 0
        elif de_option == 6:
            de_option = de if de in (2, 4, 5, 11) else 0
        elif de_option == 11:
            de_option = 2 if de in (2, 5) else 11 if de == 6 else 0
    return de_option


def _first_option(s: SlotConfig) -> float:
    """``first_option``: what MB3D reads at PVar - 16 (an integer power uses no memory)."""
    if not s.options:
        return 1.0
    if s.spec.options and s.spec.options[0].kind == "int":
        return s.options[1] if len(s.options) > 1 else 1.0
    return s.options[0]


def _de_scales(slots: list[SlotConfig], analytic: bool) -> tuple[float, float, float, list[float]]:
    """DE scale, colour offset, colour variation and 1/ln(power) per slot."""
    n = x1 = y1 = i = 0.0
    fhln = [1.0 / math.log(2.0)] * 6
    for k, s in enumerate(slots):
        if not s.uncounted:
            sip = s.spec.smooth_it_power(s.options)
            z1 = sip if sip > 1.0 else _first_option(s)
            fhln[k] = 1.0 / math.log(max(2.0, abs(z1)))
        cnt = float(s.iterations)
        z1 = cnt
        j = s.spec.de_option
        if analytic and not s.uncounted and j in (5, 11):
            n += cnt
            i += cnt
            x2 = _first_option(s)
            if x2 == 0.0:
                x2 = 1e-40
            if x2 < 0.0:
                x1 += 0.65 * z1
            else:
                x1 += (x2 * 1.2 + 1.0) / x2 * z1
            y1 += abs(x2) * z1
        elif j >= 0:
            n += cnt
            x1 += (s.spec.ade_scale if analytic else s.spec.de_scale) * z1
    d_de_scale = x1 / n if n > 0 else 1.0
    d_col_plus = 0.0
    if n > 0 and i > 0:
        d_col_plus = abs(y1 / i) ** 0.25 * 40.0 - 49.0
    # mctColVarDEstopMul
    col_var = 0.6
    y, z = 0.0, 0.0
    for s in slots:
        if s.uncounted:
            continue
        x2 = float(s.iterations)
        de = s.spec.de_option
        if de in (0, 4):
            y += x2 * 0.6
            z += x2
        elif 1 <= de <= 3 or 5 <= de <= 40:
            if s.spec.option_type_first() != "int":
                v = _first_option(s)
                if v > 1.0:
                    if v < 1.2:
                        v = 1.1 + (v - 1.0) * (v - 1.0) * 2.5
                    y += x2 * 0.5 / (math.log(v) + 0.03)
                    z += x2
                elif v < -1.0:
                    if v > -1.2:
                        v = -1.1 - (v + 1.0) * (v + 1.0) * 2.5
                    y += x2 * 0.5 / (math.log(-v) + 0.03)
                    z += x2
    if z > 1e-3:
        col_var = y / z
    return d_de_scale, d_col_plus, col_var, fhln


def build_smatrix4(xw: float, yw: float, zw: float) -> np.ndarray:
    def rot(i, j, a):
        m = np.eye(4)
        m[i, i] = m[j, j] = math.cos(a)
        m[i, j] = -math.sin(a)
        m[j, i] = math.sin(a)
        return m

    m = rot(0, 3, xw)
    m = rot(1, 3, yw) @ m
    m = rot(2, 3, zw) @ m
    return m.astype(np.float32).astype(np.float64)


def build_params(scene: Scene, options: RenderOptions | None = None) -> KernelParams:
    options = options or RenderOptions()
    slots = resolve_slots(scene)
    if scene.interpolation is not None or scene.decomb is not None:
        raise SceneError("interpolation hybrids and DE combinations are not supported yet")
    de_option = _combine_de_option(slots)
    if de_option < 0:
        raise SceneError("the formula combination has no valid DE option")
    if scene.disable_analytic_de:
        de_option = {2: 0, 11: 0, 5: 4, 6: 4}.get(de_option, de_option)
    if de_option >= 20:
        raise SceneError("dIFS formulas are not supported yet")
    analytic = de_option in (2, 5, 6, 11)
    four_d = de_option in (4, 5, 6)
    if analytic and four_d:
        raise SceneError("4D analytic DE is not supported yet")
    de_mode = 1 if analytic else 2 if four_d else 0

    width, height = scene.width, scene.height
    d_de_scale, d_col_plus, col_var, fhln = _de_scales(slots, analytic)
    rstop = scene.effective_rstop(lambda f: registry.find(f.name).rstop if registry.find(f.name) else 16.0)
    step_width = scene.step_width()
    fov_y = math.pi if scene.optic == CameraOptic.PANORAMA else math.radians(scene.fov_y)
    z_step_div0 = max(0.0001, scene.z_step_div)
    de_stop = max(0.001, scene.de_stop)
    x1c = max(0.01, min(1.5, fov_y * 0.5))
    pl_optic_z = math.cos(x1c) * x1c / math.sin(x1c)
    fovx_off = 0.5 * width
    fovx_mul = 2.0 * math.pi / width if scene.optic == CameraOptic.PANORAMA else fov_y / height
    zend = max(1e-10, (scene.z_end - scene.z_start) / step_width)
    mh04zsd = max(width, height) * 0.5 * math.sqrt(z_step_div0 + 0.001) * max(0.01, scene.raystep_limiter)
    if scene.vary_de_stop_on_fov:
        ze = max(1e-16, (scene.z_end - scene.z_start) / step_width)
        if scene.optic == CameraOptic.PANORAMA:
            x1 = 0.01 * height / (math.sin(0.01) * math.pi)
        else:
            x1 = 0.01 * height / (math.sin(0.01) * max(1.0 / 65535.0, math.radians(scene.fov_y)))
        x2 = step_width * (x1 + ze) / x1
        de_stop_factor = (x2 - step_width) / (step_width * max(1e-6, ze))
    else:
        de_stop_factor = 0.0
    # camera
    xh, yh = (0.0, 0.0) if scene.optic == CameraOptic.PANORAMA else (width * 0.5, height * 0.5)
    z1 = (scene.z_start - scene.mid[2]) / step_width
    vg = np.array(scene.vgrads, dtype=np.float64)
    norms = np.linalg.norm(vg, axis=1)
    if np.any(norms == 0):
        raise SceneError("degenerate camera matrix")
    vg_unit = vg / norms[:, None]
    vgrads = vg_unit * step_width
    mid = np.array(scene.mid, dtype=np.float64)
    ystart = mid + z1 * vgrads[2] - yh * vgrads[1] - xh * vgrads[0]
    de_offset = min(de_stop * 0.1, 0.004)
    if analytic:
        d_de_scale_k = d_de_scale / step_width
    else:
        d_de_scale_k = d_de_scale * de_offset
    de_offset006 = de_offset * 0.06
    de_offset_world = de_offset * step_width
    if scene.step_sub_de_stop:
        s = z_step_div0
        s = s * s + (1.2 * s) * (1.0 - s)
        z_step_div, ms_de_sub = s, min(0.9, math.sqrt(s))
    else:
        z_step_div, ms_de_sub = z_step_div0, 0.0
    max_it = max(1, scene.iterations)
    min_it = min(scene.min_iterations, max_it)
    ln_rstop = math.log(math.log(rstop))
    d_rstop = float(np.float32(rstop * rstop))
    rstop3d = float(np.float32(d_rstop * d_rstop * 64.0))
    mcts_m = 32767.0 / max(1, max_it + 1)
    d_col_plus += max_it * 0.1

    sp = np.zeros(L.SP_SIZE, dtype=np.float64)
    ip = np.zeros(L.IP_SIZE, dtype=np.int32)
    fp = np.zeros(L.FP_SIZE, dtype=np.float64)
    sp[L.SP_YSTART:L.SP_YSTART + 3] = ystart
    sp[L.SP_VG:L.SP_VG + 9] = vgrads.reshape(-1)
    sp[L.SP_FOVX_OFF] = fovx_off
    sp[L.SP_FOVX_MUL] = fovx_mul
    sp[L.SP_FOV_Y] = fov_y
    sp[L.SP_PL_OPTIC_Z] = pl_optic_z
    sp[L.SP_ZEND] = zend
    sp[L.SP_DE_STOP] = de_stop
    sp[L.SP_DE_STOP_FACTOR] = de_stop_factor
    sp[L.SP_Z_STEP_DIV] = z_step_div
    sp[L.SP_MS_DE_SUB] = ms_de_sub
    sp[L.SP_MH04ZSD] = mh04zsd
    sp[L.SP_RSTOP] = d_rstop
    sp[L.SP_RSTOP3D] = rstop3d
    sp[L.SP_LN_RSTOP] = ln_rstop
    sp[L.SP_D_DE_SCALE] = d_de_scale_k
    sp[L.SP_DE_OFFSET] = de_offset_world
    sp[L.SP_DE_OFFSET006] = de_offset006
    sp[L.SP_JU:L.SP_JU + 4] = scene.julia_c
    sp[L.SP_SMATRIX4:L.SP_SMATRIX4 + 16] = build_smatrix4(*scene.rot_4d).reshape(-1)
    sp[L.SP_FHLN:L.SP_FHLN + 6] = fhln
    sp[L.SP_STEP_WIDTH] = step_width
    sp[L.SP_COL_VAR] = col_var
    sp[L.SP_D_COL_PLUS] = d_col_plus
    sp[L.SP_MCTS_M] = mcts_m
    sp[L.SP_COLOR_MUL] = scene.color_mul * 512.0
    sp[L.SP_HEIGHT] = height
    sp[L.SP_WIDTH] = width
    sp[L.SP_CUT_POS:L.SP_CUT_POS + 3] = scene.cut_pos
    sp[L.SP_AO_LEN] = options.ao_length
    sd = np.array(options.shadow_dir, dtype=np.float64)
    sd /= max(1e-30, np.linalg.norm(sd))
    sp[L.SP_SHADOW_DIR:L.SP_SHADOW_DIR + 3] = sd

    for i, s in enumerate(slots):
        ip[L.IP_FID + i] = s.spec.fid
        ip[L.IP_ITS + i] = s.iterations
        ip[L.IP_UNCOUNTED + i] = int(s.uncounted)
        fp[i * L.FP_SLOT:(i + 1) * L.FP_SLOT] = s.options
    ip[L.IP_END_TO] = len(slots) - 1
    repeat_from = min(scene.repeat_from, len(slots) - 1)
    while repeat_from > 0 and slots[repeat_from].iterations <= 0:
        repeat_from -= 1
    ip[L.IP_REPEAT_FROM] = repeat_from
    ip[L.IP_MAX_IT] = max_it
    ip[L.IP_MIN_IT] = min_it
    ip[L.IP_DE_MODE] = de_mode
    ip[L.IP_JULIA] = int(scene.julia)
    ip[L.IP_BIN_STEPS] = scene.bin_search_steps
    ip[L.IP_SM_NORMALS] = scene.smooth_normals
    fsr = scene.first_step_random if options.first_step_random is None else options.first_step_random
    ip[L.IP_FIRST_STEP_RANDOM] = int(fsr)
    ip[L.IP_COLOR_OPTION] = scene.color_option
    ip[L.IP_OPTIC] = int(scene.optic)
    cut = scene.cut_options & 7
    if cut:
        cut |= (int(vgrads[2][0] > 0.0) << 4) | (int(vgrads[2][1] > 0.0) << 5) | (int(vgrads[2][2] > 0.0) << 6)
    ip[L.IP_CUT_OPTIONS] = cut
    ip[L.IP_AO_SAMPLES] = max(0, options.ao_samples)
    ip[L.IP_SHADOW] = int(options.shadow)
    return KernelParams(sp=sp, ip=ip, fp=fp, width=width, height=height, step_width=step_width, zend=zend,
                        z_range=scene.z_end - scene.z_start, de_stop=de_stop, slots=slots, de_mode=de_mode,
                        max_it=max_it, vgrads_unit=vg_unit, ystart=ystart, fov_y=fov_y)
