"""The flat parameter block of the kernels.

All three backends read the same ``float`` array (``float64`` for the CPU
and OpenCL-double, ``float32`` for OpenGL).  The indices below are the single
source of truth: :func:`c_defines` turns them into ``#define``s for the
C-like kernels and the numba kernel imports them as Python constants.
"""
from __future__ import annotations

MAX_SLOTS = 6
MAX_FORMULA_PARAMS = 32
MAX_LIGHTS = 6

# --- image / camera --------------------------------------------------------
P_WIDTH = 0
P_HEIGHT = 1
P_YSTART = 2          # 3 values
P_VGRADS = 5          # 9 values, rows scaled to step_width
P_FOV_Y = 14
P_FOVX_MUL = 15
P_FOVX_OFF = 16
P_ZEND = 17
P_DE_STOP_FACTOR = 18
P_MH04ZSD = 19
P_STEP_WIDTH = 20
# --- iteration --------------------------------------------------------------
P_MAX_IT = 21
P_MIN_IT = 22
P_RSTOP2 = 23         # squared bailout radius
P_LN_RSTOP = 24       # ln(ln(R))
P_DE_STOP = 25
P_Z_STEP_DIV = 26
P_BIN_SEARCH = 27
P_FIRST_STEP_RANDOM = 28
P_DE_MODE = 29        # 0 numeric gradient, 1 linear (r / dr), 2 logarithmic
P_DE_SCALE = 30       # combined DE scale
P_DE_OFFSET_PX = 31   # gradient offset (pixels)
P_INSIDE_IT = 32      # iteration count from which a point counts as inside (max_it, +1 for dIFS)
P_JULIA = 33
P_JULIA_C = 34        # 4 values
P_NSLOTS = 38
P_REPEAT_FROM = 39
P_COL_PLUS = 40
P_COL_VAR = 41
P_COLOR_OPTION = 42   # MB3D's 2nd colour choice 0..5
P_FOUR_D = 43
P_AO_MODE = 44        # 0 none, 1 ssao (outside the kernel), 2 DE AO
P_AO_SAMPLES = 45
P_AO_STRENGTH = 46
P_AO_RADIUS = 47
P_SHADOWS = 48        # 0/1
P_N_LIGHTS = 49
P_LIGHT_DIRS = 50     # 6 * 3 values (world space, towards the light)
P_LIGHT_FLAGS = 68    # 6 values: 1 = casts shadows
P_CUT_OPTIONS = 74
P_CUT_POS = 75        # 3 values
P_ROT4D = 78          # 16 values (4x4)
P_MS_DE_SUB = 94
P_SHADOW_SOFTNESS = 95
P_AO_FALLOFF = 96
P_RECT_X = 97         # origin of the calculated rectangle in the full image
P_RECT_Y = 98
P_FULL_WIDTH = 99     # size of the full image (the camera is set up for it)
P_FULL_HEIGHT = 100
P_COLOR_MUL = 101     # colour option 1 multiplier
P_MCTS_M = 102        # 32767 / (iterations + 1)
# --- DE combination (second hybrid part) -------------------------------------
P_DECOMB_KIND = 103   # 0 none, 1 min, 2 max, 3 inverted max, 4/5 smooth minima
P_DECOMB_SMOOTH = 104 # smoothing distance (pixels)
P_DE_MODE2 = 105
P_DE_SCALE2 = 106
P_MAX_IT2 = 107
P_INSIDE_IT2 = 108
P_REPEAT_FROM2 = 109
P_NSLOTS2 = 110
P_PIXEL_STEP = 111    # pixel stride of the calculated rectangle (1; the precision probe uses more)
# --- slot table: per slot (iterations, uncounted, fhln, reserved) ------------
P_SLOTS = 112
SLOT_STRIDE = 4
# --- formula options ---------------------------------------------------------
P_FORMULA = 144
FSTRIDE = MAX_FORMULA_PARAMS
N_PARAMS = P_FORMULA + MAX_SLOTS * FSTRIDE  # 336

# --- G-buffer channels --------------------------------------------------------
G_DEPTH = 0       # distance along the ray in pixel units, < 0 = background
G_NX = 1
G_NY = 2
G_NZ = 3
G_COLOR = 4       # MB3D's si_gradient 0..32767 (colour index)
G_OTRAP = 5       # MB3D's orbit trap colour value 0..32767 (2nd colour choice)
G_STEPS = 6       # ray steps (dynamic fog)
G_AO = 7          # 1 = unoccluded
G_SHADOW = 8      # 6 values: light factor 0..1 (1 = lit)
G_ITER = 14
G_DE = 15
N_CHANNELS = 16


def c_defines() -> str:
    """``#define`` lines of every ``P_*``, ``G_*`` and size constant."""
    lines = []
    for k, v in sorted(globals().items()):
        if k.startswith(("P_", "G_")) or k in ("MAX_SLOTS", "MAX_FORMULA_PARAMS", "MAX_LIGHTS", "SLOT_STRIDE",
                                                  "FSTRIDE", "N_PARAMS", "N_CHANNELS"):
            if isinstance(v, int):
                lines.append(f"#define {k} {v}")
    return "\n".join(lines) + "\n"


def py_constants() -> dict[str, int]:
    return {k: v for k, v in globals().items() if isinstance(v, int) and (k.startswith(("P_", "G_")) or k.isupper())}
