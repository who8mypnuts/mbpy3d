/* Struct types of the ray marching core (STRUCT_BEGIN/END come from the
 * preludes: GLSL declares `struct X {...};`, OpenCL a typedef). */

STRUCT_BEGIN(State)
    real3 z;
    real w;
    real3 c;
    real cw;
    real dr;
    real r2;
STRUCT_END(State)

STRUCT_BEGIN(Orbit)
    State s;
    real rold;
    real otrap;
    int it;
    int slot;
STRUCT_END(Orbit)

STRUCT_BEGIN(DeResult)
    Orbit o;
    real de;
STRUCT_END(DeResult)

STRUCT_BEGIN(Ray)
    real3 origin;
    real3 dir;
STRUCT_END(Ray)

STRUCT_BEGIN(Hit)
    bool hit;
    bool at_start;
    real t;
    real3 pos;
    real de;
    real steps;
    Orbit o;
STRUCT_END(Hit)

STRUCT_BEGIN(Pixel)
    real depth;
    real3 normal;
    real color;
    real otrap;
    real steps;
    real ao;
    real shadow[MAX_LIGHTS];
    real iter;
    real de;
STRUCT_END(Pixel)
