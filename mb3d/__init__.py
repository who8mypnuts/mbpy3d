"""Mandelbulb3D in Python.

Packages
--------
params    .m3p parameter files and the Scene model
formulas  fractal formulas written once in the kernel DSL
kernel    the DSL (Python subset -> numba / GLSL / OpenCL) and the ray marcher
render    render back ends (CPU, OpenGL, OpenCL) and the progressive renderer
shading   G-buffer -> RGB (lights, palette, fog, ambient occlusion)
gui       PySide6 application
"""

__version__ = "0.1.0"
