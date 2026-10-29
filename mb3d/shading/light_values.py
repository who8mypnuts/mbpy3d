"""Values derived from the lighting record (``MakeLightValsFromHeaderLight``)
and the camera values the painter needs (``PaintCamera``)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mb3d.params.lighting import Lighting
from mb3d.params.scene import CameraOptic, Scene


def _col255(c) -> np.ndarray:
    return np.array(c, dtype=np.float32)


def _col1(c) -> np.ndarray:
    return np.array(c, dtype=np.float32) / 255.0


@dataclass
class LightVal:
    idx: int
    ln: np.ndarray  # view space direction towards the light (unit)
    col: np.ndarray  # 0..255 scale times amplitude
    pow_func: int
    diff_func: int
    positional: bool
    position: np.ndarray
    lmax_l: float
    hs_enabled: bool


@dataclass
class LightVals:
    lights: list[LightVal]
    amb_col: np.ndarray
    amb_col2: np.ndarray
    depth_col: np.ndarray
    depth_col2: np.ndarray
    dyn_fog_col: np.ndarray
    dyn_fog_col2: np.ndarray
    s_depth: float
    s_shad: float
    s_shad_gr: float
    s_shad_zmul: float
    s_dyn_fog_mul: float
    s_amb_shad: float
    s_diff: float
    s_spec: float
    s_ind_light_reflect: float
    s_col_zmul: float
    s_c_start: float
    s_c_mul: float
    s_ci_start: float
    s_ci_mul: float
    s_roughness_factor: float
    col_cycling: bool
    col_on_otrap: bool
    far_fog: bool
    depth_func: int
    s_diffuse_shadowing: float
    gamma_h: int
    s_gamma: float
    col_dif: np.ndarray  # 10 x 3
    col_spe: np.ndarray
    col_pos: np.ndarray  # 10 ints
    s_c_div: np.ndarray
    col_int: np.ndarray  # 4 x 3
    icol_pos: np.ndarray
    s_ic_div: np.ndarray
    col_int_spec: np.ndarray
    sqr: bool
    no_col_ipol: bool
    amb_rel_obj: bool
    dfog_options: int
    ex_mode: bool

    @classmethod
    def from_lighting(cls, l: Lighting, z_step_div: float, dfog_on_it: int, z_range: float, step_width: float,
                      width: int, image_scale: float = 1.0) -> "LightVals":
        sqr = l.internal_gamma2
        light_scale = 1.5 if sqr else 1.0

        def lcol(c):
            return (_col255(c) ** 2 / 255.0) if sqr else _col255(c)

        def pcol(c):
            return (_col255(c) ** 2 * 0.0000153787) if sqr else _col1(c)

        lights = []
        for idx, li in enumerate(l.lights):
            if not li.on:
                continue
            dtmp = -li.x_angle
            dtmp2 = li.y_angle
            v = np.array([-math.sin(dtmp), math.sin(dtmp2), math.cos(dtmp2) * math.cos(dtmp)])
            ln = (-v / np.linalg.norm(v)).astype(np.float32)
            raw = lcol(li.color) * light_scale
            if li.positional:
                col = raw * li.amplitude * 1.3 * light_scale
                lmax_l = 800.0 * ((col[0] + col[1]) / 3.0 + 128.0 * li.amplitude)
            else:
                col = raw * li.amplitude
                lmax_l = 800.0 * (raw[0] + raw[1] + raw[2] + 128.0) * li.amplitude
            lights.append(LightVal(idx, ln, col.astype(np.float32), 2 << (li.spec_func & 7), li.diff_func & 3,
                                   li.positional, np.array(li.position, dtype=np.float64), lmax_l, li.hs_enabled))
        s_depth = l.depth_fog * 0.8e-6
        gr = (l.dyn_fog - 53.0) * image_scale * z_step_div * 0.00065
        dtmp = 2.2 / z_step_div
        mul = z_step_div * 0.015
        stmp = 128.0
        if dfog_on_it > 0:
            dtmp *= 0.25
            gr *= 4.0
            mul *= 4.0
        else:
            stmp = 137.0
        s_shad = (stmp - math.sqrt(max(0.0, l.fog_offset)) * 11.313708) * dtmp * 0.28
        s_shad_zmul = dtmp * 0.7 / z_range * (128.0 - math.sqrt(128.0) * 11.313708)
        s_amb_shad = l.amb_shadow / 53.0
        s_c_start = ((l.color_start + 30.0) / 90.0) ** 2 * 32767.0 - 10900.0
        s_c_mul = ((l.color_end + 30.0) / 90.0) ** 2 * 32767.0 - 10900.0 - s_c_start
        if l.fine_col_adj is not None:
            a1, a2 = l.fine_col_adj
            d = s_c_start + s_c_mul * (a2 - 30.0) * 0.0166666666666666
            s_c_start += s_c_mul * (a1 - 30.0) * 0.0166666666666666
            s_c_mul = d - s_c_start
        if abs(s_c_mul) > 0.001:
            s_c_mul = 2.0 / s_c_mul
        else:
            s_c_mul = -2000.0 if s_c_mul < 0 else 2000.0
        ti1, ti2 = l.interior_start, l.interior_end
        dt = ti1 ** 4
        s_ci_start = dt * 0.000158025 + 32768.0
        s_ci_mul = (ti2 ** 4 - dt) * 0.5 * 0.00031605
        if abs(s_ci_mul) > 1e-10:
            s_ci_mul = 1.0 / s_ci_mul
        gi = int(round(l.gamma))
        if gi == 32:
            gamma_h, s_gamma = 0, 0.0
        elif gi < 32:
            gamma_h, s_gamma = -1, 1.0 - gi / 32.0
        else:
            gamma_h, s_gamma = 1, (gi - 32) / 31.0
        amb_mul = l.ambient / 90.0
        if sqr:
            amb_mul = (amb_mul * amb_mul + amb_mul) * 0.5 * light_scale
        col_dif = np.array([pcol(p.diffuse) for p in l.palette], dtype=np.float32)
        col_spe = np.array([pcol(p.specular) for p in l.palette], dtype=np.float32)
        col_pos = np.array([p.position for p in l.palette], dtype=np.int64)
        s_c_div = np.ones(10, dtype=np.float32)
        for i in range(10):
            j = col_pos[i + 1] - col_pos[i] if i < 9 else 32767 - col_pos[9]
            s_c_div[i] = 1.0 / j if j > 1 else 1.0
        col_int = np.array([pcol(c) for _, c in l.interior], dtype=np.float32)
        icol_pos = np.array([p for p, _ in l.interior], dtype=np.int64)
        s_ic_div = np.ones(4, dtype=np.float32)
        for i in range(4):
            j = icol_pos[i + 1] - icol_pos[i] if i < 3 else 32767 - icol_pos[3]
            s_ic_div[i] = 1.0 / j if j > 1 else 1.0
        ind = max(-128.0, min(127.0, round(l.ind_light - 53.0)))
        s_ind = ((ind + 53.0) * 0.022) ** 2
        s_diff = l.diffuse * 0.02
        s_spec = max(l.specular * 0.02, 0.004)
        if sqr:
            s_ind *= 0.5
            s_diff = (s_diff * s_diff + s_diff) * 0.5
            s_spec = (s_spec * s_spec + s_spec) * 0.5
        col_int_spec = np.array([(a * a * 0.0000153787) if sqr else a / 255.0 for a in l.interior_spec], dtype=np.float32)
        return cls(
            lights=lights,
            amb_col=lcol(l.amb_top) * amb_mul,
            amb_col2=lcol(l.amb_bottom) * amb_mul,
            depth_col=lcol(l.depth_col),
            depth_col2=lcol(l.depth_col2),
            dyn_fog_col=(_col255(l.dyn_fog_col) ** 2 / 255.0) if sqr else _col255(l.dyn_fog_col),
            dyn_fog_col2=lcol(l.dyn_fog_col2),
            s_depth=s_depth, s_shad=s_shad, s_shad_gr=gr, s_shad_zmul=s_shad_zmul, s_dyn_fog_mul=mul,
            s_amb_shad=s_amb_shad, s_diff=s_diff, s_spec=s_spec, s_ind_light_reflect=s_ind,
            s_col_zmul=l.var_col_z * -0.005 / (step_width * width),
            s_c_start=s_c_start, s_c_mul=s_c_mul, s_ci_start=s_ci_start, s_ci_mul=s_ci_mul,
            s_roughness_factor=l.roughness / (255.0 * 255.0),
            col_cycling=l.color_cycling, col_on_otrap=l.color_on_otrap, far_fog=l.far_fog, depth_func=l.depth_func,
            s_diffuse_shadowing=l.diffuse_shadowing, gamma_h=gamma_h, s_gamma=s_gamma,
            col_dif=col_dif, col_spe=col_spe, col_pos=col_pos, s_c_div=s_c_div,
            col_int=col_int, icol_pos=icol_pos, s_ic_div=s_ic_div, col_int_spec=col_int_spec,
            sqr=sqr, no_col_ipol=l.no_col_ipol, amb_rel_obj=l.amb_rel_obj,
            dfog_options=1 if (l.dfog_options & 3) == 3 else (l.dfog_options & 3), ex_mode=l.ex_mode != 0,
        )


@dataclass
class PaintCamera:
    """Camera and z-buffer constants of the painter (``CalcPPZvals``)."""

    width: int
    height: int
    fov: float  # radians
    planar: int
    pl_optic_z: float
    zcorr: float
    zc_mul: float
    step_width: float
    zz_stmit_dif: float
    m: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))  # unit camera rows

    @classmethod
    def from_scene(cls, scene: Scene) -> "PaintCamera":
        step_width = scene.step_width()
        fov = math.pi if scene.optic == CameraOptic.PANORAMA else math.radians(scene.fov_y)
        zcorr = math.sin(max(math.radians(scene.fov_y), 1.0) / scene.height)
        zc_mul = 32767.0 * 256.0 / (math.sqrt((scene.z_end - scene.z_start) * zcorr / step_width + 1.0) - 0.999999999)
        x1c = max(0.01, min(1.5, fov * 0.5))
        vg = np.array(scene.vgrads, dtype=np.float64)
        vg = vg / np.linalg.norm(vg, axis=1)[:, None]
        return cls(scene.width, scene.height, fov, int(scene.optic), math.cos(x1c) * x1c / math.sin(x1c), zcorr, zc_mul,
                   step_width, scene.z_start - scene.mid[2], vg.astype(np.float32))

    def view_vectors(self) -> np.ndarray:
        """Unit view direction of every pixel in view space, shape (h, w, 3) (``CalcViewVec``)."""
        w, h = self.width, self.height
        x_pos = (np.arange(w, dtype=np.float32) + 1.0) / w
        y_pos = np.arange(h, dtype=np.float32) / h
        aspect = w / h
        cx = (0.5 - x_pos)[None, :] * self.fov * aspect
        cy = (y_pos - 0.5)[:, None] * self.fov
        cx = np.broadcast_to(cx, (h, w))
        cy = np.broadcast_to(cy, (h, w))
        if self.planar == 1:
            v = np.stack([-cx, cy, np.full((h, w), self.pl_optic_z, dtype=np.float32)], axis=-1)
        elif self.planar == 2:
            v = np.stack([-np.sin(cx) * np.cos(cy), np.sin(cy), np.cos(cy) * np.cos(cx)], axis=-1)
        else:
            v = np.stack([-np.sin(cx), np.sin(cy), np.cos(cy) * np.cos(cx)], axis=-1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def zpos_word(self, depth_steps: np.ndarray, hit: np.ndarray) -> np.ndarray:
        """MB3D's 16 bit z-buffer value (32767 at the start plane, 32768 = background)."""
        zz = np.maximum(depth_steps, 0.0)
        v = np.round(self.zc_mul * (np.sqrt(zz * self.zcorr + 1.0) - 1.0))
        word = np.clip(8388352 - v, 0, 8388352) / 256.0
        return np.where(hit > 0.5, word, 32768.0).astype(np.float32)
