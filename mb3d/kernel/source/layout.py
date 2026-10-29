"""Index constants of the kernel arrays (shared by all back ends).

``st``: the per ray iteration state (``TIteration3Dext``),
``sp``: real scene parameters, ``ip``: integer scene parameters,
``fp``: the 16 options of each of the 6 formula slots.
The constants are plain module level ints so that the transpiler turns
them into ``#define``s and numba freezes them.
"""

# ----------------------------------------------------------------- st[16]
ST_SIZE = 16
X = 0
Y = 1
Z = 2
W = 3
ROUT = 4
ROLD = 5
OTRAP = 6
JX = 7
JY = 8
JZ = 9
JW = 10
CX = 11
CY = 12
CZ = 13
IT = 14  # iteration count (as a real)
SIT = 15  # smooth iteration count

# ------------------------------------------------------------- fp[6 * 16]
FP_SIZE = 96
FP_SLOT = 16

# ----------------------------------------------------------------- sp[80]
SP_SIZE = 80
SP_YSTART = 0  # 3: position of pixel (0, 0) on the start plane
SP_VG = 3  # 9: camera rows scaled to the pixel step (world units)
SP_FOVX_OFF = 12
SP_FOVX_MUL = 13
SP_FOV_Y = 14
SP_PL_OPTIC_Z = 15
SP_ZEND = 16
SP_DE_STOP = 17
SP_DE_STOP_FACTOR = 18
SP_Z_STEP_DIV = 19
SP_MS_DE_SUB = 20
SP_MH04ZSD = 21
SP_RSTOP = 22  # squared bailout
SP_RSTOP3D = 23
SP_LN_RSTOP = 24
SP_D_DE_SCALE = 25
SP_DE_OFFSET = 26  # world units
SP_DE_OFFSET006 = 27
SP_JU = 28  # 4: julia constant
SP_SMATRIX4 = 32  # 16: 4D rotation
SP_FHLN = 48  # 6: 1 / ln(power) per slot
SP_STEP_WIDTH = 54
SP_COL_VAR = 55
SP_D_COL_PLUS = 56
SP_MCTS_M = 57
SP_COLOR_MUL = 58
SP_HEIGHT = 59
SP_WIDTH = 60
SP_CUT_POS = 61  # 3
SP_AO_LEN = 64  # DE ambient occlusion: ray length in steps
SP_SHADOW_DIR = 65  # 3: world direction towards the shadow light (unit)
SP_NORMAL_OFFSET_MUL = 68

# ----------------------------------------------------------------- ip[32]
IP_SIZE = 32
IP_FID = 0  # 6: formula id per slot
IP_ITS = 6  # 6: iterations per slot (>= 0)
IP_UNCOUNTED = 12  # 6
IP_END_TO = 18
IP_REPEAT_FROM = 19
IP_MAX_IT = 20
IP_MIN_IT = 21
IP_DE_MODE = 22  # 0: numerical 3D, 1: analytic (w), 2: numerical 4D
IP_JULIA = 23
IP_BIN_STEPS = 24
IP_SM_NORMALS = 25
IP_FIRST_STEP_RANDOM = 26
IP_COLOR_OPTION = 27
IP_OPTIC = 28  # 0 common, 1 planar, 2 panorama
IP_CUT_OPTIONS = 29
IP_AO_SAMPLES = 30  # 0: no DE ambient occlusion
IP_SHADOW = 31  # 1: march a shadow ray towards SP_SHADOW_DIR

# --------------------------------------------------------------- out[12]
OUT_SIZE = 12
O_HIT = 0  # 0 background, 1 surface, 2 inside at the start plane
O_DEPTH = 1  # distance from the start plane in steps (mzz)
O_NX = 2
O_NY = 3
O_NZ = 4
O_SIG = 5  # colour gradient 0..32767 (>= 32768: inside colouring)
O_OTRAP = 6  # orbit trap colour value 0..32767
O_STEPS = 7  # ray steps (dynamic fog)
O_ITS = 8
O_ROUGH = 9
O_AO = 10  # 0..1 ambient occlusion (1 = open)
O_SHADOW = 11  # 0..1 (1 = lit)
