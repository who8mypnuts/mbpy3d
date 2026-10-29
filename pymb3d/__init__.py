"""pymb3d - a Python re-implementation of Mandelbulb3D.

Layout
------
core/       scene model: camera, lights, formula slots, kernel parameter layout
formulas/   the formula DSL (one Python source per formula) and its transpilers
            to GLSL / OpenCL C and to scalar numba code
kernels/    the ray marching kernels (a C-like core shared by GLSL and OpenCL,
            and a numba core for the CPU)
backends/   OpenGL (preview), OpenCL (precise final renders), CPU (numba, all cores)
render/     orchestration (tiles, progress, cancellation), deferred shading, AO
io/         scene JSON and Mandelbulb3D .m3p import
gui/        the PySide6 application (signals/slots, worker threads)
stats/      timing history (polars) used to pick the fastest backend
"""

__version__ = "0.1.0"
