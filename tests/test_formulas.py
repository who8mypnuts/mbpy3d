import math

import numpy as np
import pytest

from mb3d.formulas import registry
from mb3d.formulas.builtin import boxes, bulbs, ifs, quaternions
from mb3d.formulas.dispatch import cpu_dispatcher, dispatcher_source, gpu_dispatcher
from mb3d.kernel.dsl import GLSL, OPENCL_DOUBLE, transpile
from mb3d.kernel.source import layout as L


def state(v, j=(0.0, 0.0, 0.0, 0.0)):
    st = np.zeros(16)
    st[L.X:L.W + 1] = v
    st[L.JX:L.JW + 1] = j
    st[L.ROUT] = sum(c * c for c in v)
    return st


def fp_of(*opts):
    fp = np.zeros(96)
    fp[:len(opts)] = opts
    return fp


def triplex(n, v):
    """White/Nylander reference: r^n (cos n th cos n ph, sin n th cos n ph, -sin n ph)."""
    r = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    th = math.atan2(v[1], v[0]) * n
    ph = math.atan2(v[2], math.sqrt(v[0] ** 2 + v[1] ** 2)) * n
    rn = r ** n
    return [rn * math.cos(th) * math.cos(ph), rn * math.sin(th) * math.cos(ph), -rn * math.sin(ph)]


@pytest.mark.parametrize("n", range(2, 9))
def test_int_powers_match_spherical_triplex(n):
    p = (0.31, -0.42, 0.27, 0.0)
    st = state(p)
    bulbs.f_integer_power(st, fp_of(float(n), -1.0), 0, False)
    ref = triplex(n, p)
    assert np.allclose(st[L.X:L.Z + 1], ref, atol=1e-9)


def test_real_power_matches_int_power8():
    p = (0.31, -0.42, 0.27, 0.0)
    a, b = state(p), state(p)
    bulbs.f_integer_power(a, fp_of(8.0, -1.0), 0, False)
    bulbs.f_real_power(b, fp_of(8.0, -1.0), 0, False)
    assert np.allclose(a[L.X:L.Z + 1], b[L.X:L.Z + 1], atol=1e-9)


def test_box_scales_derivative():
    st = state((0.1, 0.1, 0.1, 1.0))
    boxes.f_amazing_box(st, fp_of(2.0, 0.5, 1.0), 0, True)
    assert st[L.W] == pytest.approx(8.0)  # r < minR^2 -> scale / minR^2
    assert np.allclose(st[L.X:L.Z + 1], [0.8, 0.8, 0.8])


def test_box_fold():
    assert bulbs.box_fold(0.3, 1.0) == pytest.approx(0.3)
    assert bulbs.box_fold(1.5, 1.0) == pytest.approx(0.5)
    assert bulbs.box_fold(-1.5, 1.0) == pytest.approx(-0.5)
    assert bulbs.box_fold(2.5, 1.0) == pytest.approx(-0.5)


def test_quaternion_square():
    st = state((0.5, 0.25, -0.125, 0.75), (0.1, 0.2, 0.3, 0.4))
    quaternions.f_quaternion(st, fp_of(-1.0, 0.0), 0, False)
    x, y, z, w = 0.5, 0.25, -0.125, 0.75
    assert st[L.X] == pytest.approx(x * x - y * y - z * z - w * w + 0.1)
    assert st[L.Y] == pytest.approx(2 * (y * x + z * w) + 0.2)
    assert st[L.Z] == pytest.approx(2 * (z * x - y * w) + 0.3)
    assert st[L.W] == pytest.approx(2 * (w * x + y * z) + 0.4)


def test_menger3_is_a_sponge():
    # a point of the unit cube's centre is mapped by the sponge step: |v| sorted, scaled by 3, shifted by 2
    st = state((0.2, -0.5, 0.1, 1.0))
    ifs.f_menger3(st, fp_of(3.0, 1.0, 1.0, 1.0), 0, True)
    assert st[L.X] == pytest.approx(0.5 * 3 - 2)  # largest |component| first
    assert st[L.Y] == pytest.approx(0.2 * 3 - 2)
    assert st[L.Z] == pytest.approx(0.1 * 3)  # below the half width: not shifted
    assert st[L.W] == pytest.approx(3.0)


def test_dispatcher_covers_registry():
    apply = cpu_dispatcher()
    for spec in registry.all_specs():
        st = state((0.3, 0.2, 0.1, 1.0), (0.3, 0.2, 0.1, 0.0))
        fp = np.zeros(96)
        fp[:len(spec.options)] = spec.defaults()[:len(spec.options)]
        apply(spec.fid, st, fp, 0, spec.analytic)
        assert np.all(np.isfinite(st)), spec.name


def test_dispatcher_source_lists_formulas():
    src = dispatcher_source(registry.all_specs()[:2])
    assert "fid == 1" in src and "f_integer_power(st, fp, b, ade)" in src


@pytest.mark.parametrize("target", [GLSL, OPENCL_DOUBLE])
def test_every_formula_transpiles(target):
    for spec in registry.all_specs():
        src = transpile(target, [registry.kernel_name(spec)], overrides={"apply_formula": gpu_dispatcher([spec])})
        assert registry.kernel_name(spec) in src


def test_registry_lookup():
    assert registry.find("amazing box").name == "Amazing Box"
    assert registry.find("Mandelbox").name == "Amazing Box"
    assert registry.find("no such formula") is None
    with pytest.raises(KeyError):
        registry.lookup("no such formula")
