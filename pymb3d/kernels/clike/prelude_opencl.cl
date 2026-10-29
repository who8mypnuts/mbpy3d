/* OpenCL prelude: double precision when USE_DOUBLE is defined. */
#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define real double
#define real3 double3
#define real4 double4
#define TINY 1e-40
#else
#define real float
#define real3 float3
#define real4 float4
#define TINY 1e-30
#endif
#define V3(a, b, c) ((real3)((real)(a), (real)(b), (real)(c)))
#define V3s(a) ((real3)((real)(a)))
#define TO_INT(x) ((int)(x))
#define TO_REAL(x) ((real)(x))
#define fractx(x) ((x) - floor(x))
#define modx(x, y) ((x) - (y) * floor((x) / (y)))
#define PI_ 3.14159265358979
#define TWO_PI_ 6.28318530717959
#define FN_ARGS __constant const real *P,
#define FN_PASS P,
#define PARAM(i) P[i]
#define FPARAM(slot, k) P[P_FORMULA + (slot) * FSTRIDE + (k)]
#define STRUCT_BEGIN(n) typedef struct {
#define STRUCT_END(n) } n;
