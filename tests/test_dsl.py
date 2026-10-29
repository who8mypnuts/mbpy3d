import math

import numpy as np
import pytest

from mb3d.kernel.dsl import GLSL, OPENCL_DOUBLE, transpile

from . import dsl_sample as S


def test_numba_runs():
    p = np.array([2.0, 3.0])
    v = S.poly(0.7, p, True)
    assert isinstance(v, float)
    assert S.box_fold(0.3, 1.0) == pytest.approx(0.3)
    assert S.box_fold(1.5, 1.0) == pytest.approx(0.5)
    assert S.int_pow_loop(2.0, 3) == 8.0
    assert S.classify(-1.0) == -1


def test_transpile_glsl():
    src = transpile(GLSL, ["poly"])
    assert "#version 330 core" in src
    assert "uniform float p[2];" in src
    assert "#define SCALE 2.5" in src
    assert "float box_fold(float x, float fold)" in src
    # const arrays are dropped from parameter lists and call sites in GLSL
    assert "float sum_array(inout float a[4])" in src
    assert "sum_array(st)" in src
    assert "atan(x, 1.0)" in src
    assert "for (i = 0; i < n; i += 1)" in src
    assert "} else if (x == 0.0) {" in src


def test_transpile_opencl():
    src = transpile(OPENCL_DOUBLE, ["poly"])
    assert "#define REAL double" in src
    assert "double sum_array(double* a, __constant const double* p)" in src
    assert "sum_array(st, p)" in src
    assert "fabs(" in src and "atan2(" in src and "fmin(" in src
    assert "((double)(k))" in src


@pytest.fixture(scope="module")
def cl_ctx():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"no OpenCL device: {e}")
    return cl, ctx


def test_opencl_matches_numba(cl_ctx):
    cl, ctx = cl_ctx
    src = transpile(OPENCL_DOUBLE, ["poly"]) + """
__kernel void run(__global double* out, __constant const double* p, __global const double* xs) {
    int i = get_global_id(0);
    out[i] = poly(xs[i], p, i % 2 == 0);
}
"""
    prg = cl.Program(ctx, src).build()
    q = cl.CommandQueue(ctx)
    xs = np.linspace(-2, 2, 17)
    p = np.array([2.0, 3.0])
    mf = cl.mem_flags
    out = np.zeros_like(xs)
    bx = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xs)
    bp = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p)
    bo = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)
    prg.run(q, (len(xs),), None, bo, bp, bx)
    cl.enqueue_copy(q, out, bo)
    ref = np.array([S.poly(x, p, i % 2 == 0) for i, x in enumerate(xs)])
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_glsl_compiles():
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(backend="egl")
    except Exception:
        try:
            ctx = moderngl.create_standalone_context()
        except Exception as e:  # pragma: no cover
            pytest.skip(f"no OpenGL: {e}")
    frag = transpile(GLSL, ["poly"]) + """
out vec4 color;
void main() { color = vec4(poly(gl_FragCoord.x, true)); }
"""
    vert = "#version 330 core\nin vec2 in_pos; void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }"
    prog = ctx.program(vertex_shader=vert, fragment_shader=frag)
    assert prog is not None
    ctx.release()
