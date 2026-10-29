#version 330 core
/* GLSL prelude: single precision, parameters in a uniform array. */
#define real float
#define real3 vec3
#define real4 vec4
#define V3(a, b, c) vec3(a, b, c)
#define V3s(a) vec3(a)
#define TO_INT(x) int(x)
#define TO_REAL(x) float(x)
#define TO_UINT(x) uint(x)
#define fabs abs
#define fmin min
#define fmax max
#define fractx fract
#define modx mod
#define atan2(y, x) atan(y, x)
#define TINY 1e-30
#define PI_ 3.14159265358979
#define TWO_PI_ 6.28318530717959
#define FN_ARGS
#define FN_PASS
#define PARAM(i) u_params[i]
#define FPARAM(slot, k) u_params[P_FORMULA + (slot) * FSTRIDE + (k)]
#define STRUCT_BEGIN(n) struct n {
#define STRUCT_END(n) };
