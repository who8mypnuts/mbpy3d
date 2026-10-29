"""Assembles the GPU kernel source for a set of formulas and a target."""

from __future__ import annotations

from functools import lru_cache

from mb3d.formulas import registry
from mb3d.formulas.dispatch import gpu_dispatcher

from .dsl import GLSL, OPENCL_DOUBLE, OPENCL_FLOAT, Target, transpile
from .source import layout as L
from .source import march as _march  # noqa: F401  (registers the kernel functions)

# the marcher and everything it calls
ENTRY = "march_pixel"

GLSL_MAIN = f"""
uniform int u_height;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;

void main() {{
    int ix = int(gl_FragCoord.x);
    int iy = u_height - 1 - int(gl_FragCoord.y);
    float st[{L.ST_SIZE}];
    float res[{L.OUT_SIZE}];
    for (int k = 0; k < {L.ST_SIZE}; k++) st[k] = 0.0;
    march_pixel(ix, iy, st, res);
    o0 = vec4(res[0], res[1], res[2], res[3]);
    o1 = vec4(res[4], res[5], res[6], res[7]);
    o2 = vec4(res[8], res[9], res[10], res[11]);
}}
"""

GLSL_VERTEX = """#version 330 core
const vec2 verts[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0); }
"""

OPENCL_MAIN = f"""
__kernel void render(__global REAL* gbuf, __constant const REAL* sp, __constant const REAL* fp,
                     __constant const int* ip, __global const int* rows, const int width) {{
    int ix = get_global_id(0);
    int ri = get_global_id(1);
    if (ix >= width) return;
    int iy = rows[ri];
    REAL st[{L.ST_SIZE}];
    REAL res[{L.OUT_SIZE}];
    for (int k = 0; k < {L.ST_SIZE}; k++) st[k] = (REAL)0;
    march_pixel(ix, iy, sp, fp, ip, st, res);
    __global REAL* o = gbuf + ((size_t)iy * width + ix) * {L.OUT_SIZE};
    for (int k = 0; k < {L.OUT_SIZE}; k++) o[k] = res[k];
}}
"""


def _specs_for(fids: tuple[int, ...]):
    return [registry.by_id(f) for f in fids]


@lru_cache(maxsize=64)
def kernel_source(target_name: str, fids: tuple[int, ...]) -> str:
    """The complete kernel source for ``target_name`` ("glsl", "opencl_double", "opencl_float")."""
    target: Target = {"glsl": GLSL, "opencl_double": OPENCL_DOUBLE, "opencl_float": OPENCL_FLOAT}[target_name]
    overrides = {"apply_formula": gpu_dispatcher(_specs_for(fids))}
    main = GLSL_MAIN if target.name == "glsl" else OPENCL_MAIN
    return transpile(target, [ENTRY], extra_source=main, overrides=overrides)


def formula_ids(params) -> tuple[int, ...]:
    return tuple(sorted({s.spec.fid for s in params.slots}))
