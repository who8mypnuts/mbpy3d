# pymb3d — Mandelbulb3D in Python

A re-implementation of the Mandelbulb3D fractal renderer in Python: a PySide6
application with an interactive preview, GPU ray marching (OpenGL for the
preview, OpenCL in double precision for final renders), a multi-threaded CPU
fallback (numba on all cores) and an importer for Mandelbulb3D `.m3p`
parameter files.

```
pip install -e .[dev]          # or: pip install numpy numba polars PySide6 pillow moderngl pyopencl
python -m pymb3d                                   # the application
python -m pymb3d "M3Parameter/MengerTrees.m3p"     # open a MB3D parameter file
python -m pymb3d render scene.json -o out.png --width 1920 --backend opencl
pytest                                             # the test suite
```

## How it is built

```
pymb3d/
  core/        Scene, Camera (MB3D's conventions), lights, the kernel parameter layout
  formulas/    the formula DSL, registry, built-in formulas, transpilers
  kernels/     march_core.c (GLSL + OpenCL), numba_core.py (CPU), Program (per scene)
  backends/    OpenGL, OpenCL, CPU backends + device detection
  render/      Renderer (tiles, progress, cancellation), deferred shading, SSAO
  io/          JSON scenes, .m3p / MB3D text parameter import, PNG metadata
  gui/         PySide6: worker threads, controllers, panels, dialogs
  stats/       polars timing history used to rank devices
```

### One formula source, three targets

Each formula is written **once** as a small Python function on a state
object (`pymb3d/formulas/builtin/*.py`):

```python
@formula(name="Amazing Box", params=[Param("scale", 2.0), Param("min_r", 0.5), Param("fold", 1.0)],
         de="linear", de_scale=0.2, rstop=1024.0, scale_param="scale")
def amazing_box(s, p):
    f = box_fold(s.z, p.fold)
    m = sphere_fold_mul(dot(f, f), p.scale, p.min_r)
    s.dr = s.dr * fabs(m)
    s.z = f * m + s.c
```

The same source

* runs as plain Python (the reference the tests compare against),
* is transpiled to C (`formulas/transpile/emit_c.py`) — valid as **GLSL 3.30**
  for the OpenGL preview and as **OpenCL C** (double precision) for final
  renders, with two small preludes bridging the dialects,
* is transpiled to scalar Python (`emit_py.py`) that **numba** compiles for
  the CPU backend.

The ray marching core exists twice (`kernels/clike/march_core.c` shared by
GLSL and OpenCL, and `kernels/numba_core.py`), function for function in the
same order, so the three backends produce the same G-buffer: the OpenCL and
numba results are bit-identical, OpenGL differs by single precision only
(`tests/test_transpile.py`, `tests/test_render.py`).

### Rendering pipeline

1. `Program.from_scene` collects the formula slots, decides the distance
   estimation mode of each hybrid part (analytic *linear* for box type
   formulas, analytic *log* for bulbs, MB3D's 4-point numerical gradient for
   anything mixed, the *dIFS* loop for distance-estimated IFS shapes) and
   generates the code, including the hybrid loop **specialised for the
   slots** (`kernels/loopgen.py`: no per-iteration formula dispatch, no
   dynamic parameter indexing — 2.5x faster kernels than the generic loop).
   Option values, iteration counts, camera, lights … are **runtime
   parameters** (one flat float array, `core/params.py`), so sliders never
   recompile a kernel.  Two parts can be combined by their distance
   estimates (MB3D's *DE combination*: minimum, maximum, inverted maximum,
   two smooth minima).
2. The backend marches the image in bands (progress, cancellation, live
   display) and returns a G-buffer: depth, normal, colour index (smooth
   iteration count), orbit trap, ray steps, DE ambient occlusion, one soft
   shadow factor per light.
3. `render/shading.py` shades the G-buffer with numpy, following MB3D's
   `CalcPixelColor2`: six lights with MB3D's diffuse functions and Phong
   specular, ambient top/bottom, the 10-stop palette with colour start/end,
   depth fog, dynamic fog, gamma.  Screen-space ambient occlusion
   (`render/ssao.py`) runs on the depth buffer.  Because shading is separate
   from marching, **lighting changes re-shade the kept G-buffer instantly**.

Images beyond a size limit are marched in strips (bounded memory).

### Camera and parameters

The camera follows MB3D: rays start on the start plane (`z_start`) and fan
out with the field of view; distances are counted in pixel units
(`step_width = 2.1345 / (zoom * width)`), the DE stop is in pixels and grows
with the distance from the camera, the raystep multiplier, the binary search
refinement, the step limiter and the colour/DE-stop coupling heuristics of
`GetMCTparasFromHeader` are ported.  `.m3p` files therefore load with their
camera, iteration settings, lights and palette (`io/m3p.py`); all 80 example
parameter files of MB3D parse, those using only the formulas implemented
here render to MB3D's look.

### GUI

* Preview: left drag rotates, right drag pans, wheel moves forward/back,
  Shift+wheel zooms, WASD/QE/arrows/+/- move, double click centres a surface
  point.  A coarse pass renders while interacting, the full preview after.
* Every change is a signal on the `SceneModel`; the preview controller
  debounces requests and the render worker (a `QObject` in its own
  `QThread`, one per backend) cancels a running render when a newer request
  arrives ("latest wins").  The UI thread never blocks.
* The final render window uses its own worker thread and the chosen precise
  device, shows rows as they arrive (with an ETA), supports 1x/2x/3x
  supersampling, transparent and 16 bit PNG, a Z-buffer image, "copy to
  clipboard", saving / loading render data and **re-lighting** the finished
  G-buffer without marching again.  A cancelled render keeps its finished
  rows and can be **resumed**.
* The main window has MB3D's menus (File / Edit / View / Render / Tools /
  Prefs / Help), a toolbar, the image size bar (width, height, aspect lock,
  ½ / ×2, *Calculate 3D*), a messages dock, hotkeys `Ctrl+1..9` for the tool
  windows, a scene open dialog with thumbnail preview, a searchable formula
  picker and eight colour themes (glossy dark, graphite, midnight, ember,
  forest, daylight, high contrast, system; *Prefs > Theme*).  Layout, theme
  and recent files persist in `QSettings`.
* `Edit > Paste parameters` accepts MB3D text parameters
  (`Mandelbulb3Dv18{...}`) or pymb3d JSON from the clipboard.

### Devices and GPU speed

`backends/detect.py` lists the numba CPU backend, every OpenCL device (with
`cl_khr_fp64` where available) and an OpenGL 3.3 context (EGL/GLX on Linux,
WGL on Windows, CGL on macOS through moderngl).  Defaults: preview on OpenGL
of a real GPU, final render on an OpenCL GPU, the CPU otherwise; software
OpenGL (llvmpipe) and CPU OpenCL runtimes (pocl) are available but ranked
below numba.  `PYMB3D_BACKEND=cpu|opencl|opengl` forces a backend.

Consumer graphics cards run double precision at 1/16 to 1/64 of their
single precision rate.  The OpenCL backend therefore defaults to **auto
precision**: before a render it marches a coarse version of the image
(every k-th pixel) in both precisions and compares hits, depths and colour
indices (`backends/precision.py`); single precision is used unless they
differ (deep zooms).  The devices list also offers fixed single / double
precision and **mixed** precision: the GPU marches in single precision and
the numba CPU re-finishes every hit in double (`finish_rows`), so consumer
cards keep their speed while the surface positions stay exact.  Kernel launches are sized adaptively to about 0.1 s so the
display driver stays responsive and progress / cancellation are quick; the
coarse preview pass skips shadows and occlusion.  Render timings are recorded with polars
(`stats/benchmark.py`, `~/.cache/pymb3d/render_history.parquet`) and shown
on the Info tab.  Generated numba modules are cached in
`~/.cache/pymb3d/gen` (`PYMB3D_CACHE` overrides the location), so a formula
combination compiles once per machine.

### More features

* **dIFS** (MB3D's distance estimated IFS): `SphereIFS`, `boxIFS`,
  `cylinderIFS` shapes and the `transformIFS` transform, ported from the
  machine code of the `.m3f` files (option order kept, so `.m3p` files load).
* **DE combinations** of two hybrid parts (min, max, inverted max, two smooth
  minima, the dIFS mix), **interpolation hybrids**, **inside / in-and-outside
  rendering**, the **2D slice calculation**, MB3D's **2nd colour choice**
  (orbit trap modes), **positional lights**, 4D rotation, exact
  `3SingleAngles` rotation matrices, "step sub DE stop".
* **Smooth normals** (1..8, MB3D's tangent sweeps / 5x5x5 cube) with the
  measured **surface roughness** softening the lights through MB3D's cos
  tables.
* **Reflections**: a second marched layer (the mirrored ray's hit with its
  own normal, colour, AO and shadows) shaded with the same lighting and
  blended by the reflection amount and the specular colour.
* **Monte Carlo sky visibility** as ambient occlusion (random hemisphere rays
  from a hash RNG, passes averaged progressively in the final render),
  **volumetric light** (lit path length along the view ray), **visible light
  sources** (glow).
* **Background pictures** (sphere or image mapped), **light maps** (ambient
  from an image by the normal), **diffuse colour maps** (by normals, by
  iterations + orbit trap, wrapped), with MB3D's map numbers resolved in the
  `M3Maps` folder next to a parameter file.
* **Depth of field** (layered blur by the distance from the focus plane,
  Shift + double click sets the focus), **post processing** (brightness,
  contrast, saturation, sharpen, a DCT **denoiser** for Monte Carlo
  passes), **stereo pairs** (side by side, cross-eyed, anaglyph), **16 bit
  PNG**.
* **Navigator window** (MB3D's Navigator): a fast own preview with move /
  turn / roll buttons, step and angle sizes, continuous flight, mouse and
  keyboard navigation, live transfer of the camera to the main window.
* **MutaGen**: a grid of random variations (formula options, lights,
  palette) rendered as thumbnails; click one to adopt it.
* **Batch rendering** (window and `pymb3d batch`), **voxel / mesh export**
  (PNG slices, OBJ through marching cubes), palette tools (random, from an
  image, reverse, spread), recent files, preview quality (DE stop multiplier).
* **Animation** (`core/animation.py`, *Render > Animation*): keyframes taken
  from the scene; every numeric setting is a column of a polars DataFrame
  interpolated per frame (linear, eased or a Catmull-Rom spline through the
  keyframes), the camera rotation by quaternion slerp, the zoom
  logarithmically.  Frames render in a worker
  thread to numbered PNGs, and to an MP4 when ffmpeg is installed.
* **Export** to `.m3p` / MB3D text parameters (*File > Export*, *Edit > Copy
  parameters*), lighting presets (`.json`, MB3D `.m3l`), render data files
  (`.npz` with the G-buffer and the scene) that can be re-lit later.

## Formulas included

MB3D's internal formulas ported from `formulas.pas`: Integer Power (2..8),
Real Power, Quaternion, Tricorn, Amazing Box, Bulbox, Folding Int Pow,
Aexion C.  Custom formulas re-implemented from their `.m3f` machine code
(names match, so `.m3p` files find them): Menger3, MengerIFS, Menger4 (4D),
SierpHilbert, Amazing Surf, ABoxMod1, ABoxModKali, genIFS, SphereIFS,
boxIFS, cylinderIFS, transformIFS, _Rotate, _RotateC, _Translate,
_PolyFold-sym, _SphereFolding1; plus Mandelbulb IQ, ABox Mod, Sierpinski3, Octahedral KIFS
and a Sphere Inversion transform.  The rest of MB3D's 460 `.m3f` formulas are x86
machine code and are ported one by one.

## Not yet ported (next steps)

Full Monte Carlo path tracing with transparency (the MC renderer here does
sky visibility and mirror reflections), MB3D's exact volumetric light maps,
hotkey configuration, interface scaling and language files, the remaining
`.m3f` formulas, a wgpu/WebGPU backend, map sequences for animations.
