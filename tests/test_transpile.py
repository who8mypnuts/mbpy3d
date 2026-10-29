"""Every formula: the plain Python reference, the numba code, the OpenCL
code and the GLSL code give the same results."""
import math

import numpy as np
import pytest

from pymb3d.core.scene import FormulaSlot, Scene
from pymb3d.formulas import registry
from pymb3d.formulas.dsl import State, vec3, helpers
from pymb3d.formulas.transpile import CEmitter, PyEmitter, helpers_used
from pymb3d.kernels import Program
from pymb3d.kernels.numba_loader import load_program_module

SPECS = registry.all_specs()


def random_states(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    io = np.zeros((n, 10))
    io[:, 0:3] = rng.uniform(-1.5, 1.5, (n, 3))       # z
    io[:, 3] = rng.uniform(-0.5, 0.5, n)              # w
    io[:, 4:7] = rng.uniform(-1.2, 1.2, (n, 3))       # c
    io[:, 7] = rng.uniform(-0.5, 0.5, n)              # cw
    io[:, 8] = rng.uniform(0.5, 3.0, n)               # dr
    io[:, 9] = np.sum(io[:, 0:3] ** 2, axis=1)        # r2
    return io


def reference(spec, io: np.ndarray) -> np.ndarray:
    out = io.copy()
    for i in range(io.shape[0]):
        s = State(vec3(*io[i, 0:3]), vec3(*io[i, 4:7]), w=io[i, 3], cw=io[i, 7], dr=io[i, 8], r2=io[i, 9])
        spec.run(s)
        out[i, 0:3] = (s.z.x, s.z.y, s.z.z)
        out[i, 3] = s.w
        out[i, 4:7] = (s.c.x, s.c.y, s.c.z)
        out[i, 8] = s.dr
    return out


def scene_for(spec) -> Scene:
    sc = Scene()
    sc.slots = [FormulaSlot(spec.name, 1, {})]
    return sc


def assert_close(ref: np.ndarray, got: np.ndarray, rtol: float, name: str) -> None:
    finite = np.isfinite(ref).all(axis=1) & np.isfinite(got).all(axis=1)
    assert finite.sum() >= ref.shape[0] * 0.8, f"{name}: too many non finite results"
    r, g = ref[finite], got[finite]
    # tolerance relative to the size of the whole result vector of a state
    scale = np.maximum(1.0, np.linalg.norm(r[:, :3], axis=1))[:, None]
    bad = np.abs(r - g) > rtol * scale
    assert not bad.any(), f"{name}: {bad.sum()} mismatches, worst {np.abs(r - g).max()}"


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_emitters_produce_code(spec):
    pidx = {p.name: i for i, p in enumerate(spec.params)}
    for h in helpers_used(spec.func):
        assert "FN_ARGS" in CEmitter(helpers()[h].func).emit()
        assert "def h_" in PyEmitter(helpers()[h].func).emit()
    assert f"State formula_0(FN_ARGS State s)" in CEmitter(spec.func, slot=0, param_index=pidx).emit()
    assert "def formula_0(" in PyEmitter(spec.func, slot=0, param_index=pidx, param_base=128).emit()


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_numba_matches_reference(spec):
    sc = scene_for(spec)
    prog = Program.from_scene(sc)
    mod = load_program_module(prog)
    P = prog.pack(sc)
    io = random_states(200)
    ref = reference(spec, io)
    got = io.copy()
    mod.formula_test(P, 0, got)
    assert_close(ref, got, 1e-9, spec.name)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_opencl_matches_reference(spec, opencl_backend):
    sc = scene_for(spec)
    prog = Program.from_scene(sc)
    P = prog.pack(sc)
    io = random_states(200)
    ref = reference(spec, io)
    got = opencl_backend.run_formula(prog, P, 0, io)
    rtol = 1e-9 if opencl_backend.precision == "float64" else 2e-4
    assert_close(ref[:, :9], got[:, :9], rtol, spec.name)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_glsl_matches_reference(spec, opengl_backend):
    sc = scene_for(spec)
    prog = Program.from_scene(sc)
    P = prog.pack(sc)
    io = random_states(100)
    ref = reference(spec, io)
    got = opengl_backend.run_formula(prog, P, 0, io)
    # single precision, and the bulb of power 8 amplifies rounding
    assert_close(ref[:, :9], got[:, :9], 5e-3, spec.name)


def test_hybrid_program_sources_include_all_slots():
    sc = Scene()
    sc.slots = [FormulaSlot("Amazing Box", 2, {"scale": -1.5}), FormulaSlot("Integer Power", 1, {"power": 3})]
    prog = Program.from_scene(sc)
    c = prog.c_source()
    assert "formula_0" in c and "formula_1" in c and "run_slot" in c
    py = prog.py_source()
    assert "def formula_1(" in py and "def render_rows(" in py
    assert prog.de_mode == 0  # mixed linear/log -> numeric gradient


def test_default_scene_smooth_pow_and_params():
    sc = Scene()
    sc.slots = [FormulaSlot("Integer Power", 1, {"power": 5})]
    prog = Program.from_scene(sc)
    P = prog.pack(sc)
    from pymb3d.core import params as PL

    assert P[PL.P_FORMULA] == 5.0
    assert math.isclose(P[PL.P_SLOTS + 2], 1.0 / math.log(5.0))
    assert prog.de_mode == 2
