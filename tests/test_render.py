"""Rendering small scenes on every backend, shading and the renderer."""
import threading

import numpy as np
import pytest

from pymb3d.core import params as PL
from pymb3d.core.scene import FormulaSlot, Scene, default_scene
from pymb3d.kernels import Program
from pymb3d.render import Renderer, RenderOptions, relight
from pymb3d.render.shading import shade, to_uint8
from pymb3d.render.ssao import ssao


def small_scene(w=64, h=48) -> Scene:
    sc = default_scene()
    sc.width, sc.height = w, h
    sc.camera.z_start = -2.0
    sc.camera.zoom = 1.4
    return sc


def gbuffer(backend, sc: Scene) -> np.ndarray:
    prog = Program.from_scene(sc)
    P = prog.pack(sc, dtype=backend.dtype)
    return backend.render(prog, P, sc.width, sc.height)


def test_cpu_gbuffer_has_hits(cpu_backend):
    g = gbuffer(cpu_backend, small_scene())
    depth = g[..., PL.G_DEPTH]
    assert g.shape == (48, 64, PL.N_CHANNELS)
    assert 0.1 < (depth >= 0).mean() < 0.9
    n = g[..., PL.G_NX:PL.G_NZ + 1][depth >= 0]
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)


def test_opencl_matches_cpu(cpu_backend, opencl_backend):
    sc = small_scene()
    a = gbuffer(cpu_backend, sc)
    opencl_backend.precision_mode = "float64" if opencl_backend.double_ok else "float32"
    b = gbuffer(opencl_backend, sc)
    assert ((a[..., 0] >= 0) == (b[..., 0] >= 0)).all()
    tol = 1e-6 if opencl_backend.double_ok else 0.1
    assert np.abs(a[..., 0] - b[..., 0]).max() < tol
    opencl_backend.precision_mode = "auto"


def test_opencl_precision_probe(opencl_backend):
    if not opencl_backend.double_ok:
        pytest.skip("no double precision")
    sc = small_scene(64, 48)
    opencl_backend.precision_mode = "auto"
    g = gbuffer(opencl_backend, sc)
    probe = opencl_backend.last_probe
    assert probe is not None and probe.single_ok, probe.summary
    assert opencl_backend.last_precision == "float32"
    assert (g[..., 0] >= 0).any()
    # a deep zoom onto the surface needs double precision: refine a hit
    # point onto the surface, then look at it from 300 pixels away
    from pymb3d.kernels.numba_loader import load_program_module

    hit = g[..., 0] >= 0
    ys, xs = np.nonzero(hit)
    y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    setup = sc.camera.kernel_setup(64, 48, sc.raystep_limiter, sc.z_step_div, sc.vary_de_stop_on_fov)
    start, d = setup.ray(x, y)
    point = start + d * float(g[y, x, 0])
    prog = Program.from_scene(sc)
    mod = load_program_module(prog)
    P = prog.pack(sc)
    sw0 = float(P[PL.P_STEP_WIDTH])
    dn = d / np.linalg.norm(d)
    for _ in range(200):
        r = mod.calc_de(point[0], point[1], point[2], 1e-6, False, P)
        if r[0] * sw0 < 1e-9:
            break
        point = point + dn * r[0] * sw0 * 0.5
    deep = small_scene(64, 48)
    deep.camera.mid = point
    deep.camera.zoom = 1e7
    sw = deep.camera.step_width(64)
    deep.camera.z_start = float(point[2]) - 300 * sw
    deep.camera.z_end = float(point[2]) + 3000 * sw
    gbuffer(opencl_backend, deep)
    assert opencl_backend.last_precision == "float64", opencl_backend.last_probe.summary


def test_opengl_matches_cpu(cpu_backend, opengl_backend):
    sc = small_scene()
    a = gbuffer(cpu_backend, sc)
    b = gbuffer(opengl_backend, sc)
    hit_a, hit_b = a[..., 0] >= 0, b[..., 0] >= 0
    assert (hit_a != hit_b).mean() < 0.02
    both = hit_a & hit_b
    assert np.abs(a[..., 0] - b[..., 0])[both].max() < 0.5


def test_hybrid_numeric_de_and_shadows(cpu_backend):
    sc = small_scene(48, 32)
    sc.slots = [FormulaSlot("Amazing Box", 3, {"scale": 2.0}), FormulaSlot("Integer Power", 1, {"power": 3})]
    sc.iterations = 10
    sc.shadows = True
    sc.ao.mode = "de"
    sc.camera.zoom = 0.5
    g = gbuffer(cpu_backend, sc)
    assert Program.from_scene(sc).de_mode == 0
    hit = g[..., 0] >= 0
    assert hit.any()
    assert (g[..., PL.G_AO][hit] <= 1.0).all()
    assert (g[..., PL.G_SHADOW][hit] <= 1.0).all()


def test_julia_and_cut_plane(cpu_backend):
    sc = small_scene(32, 24)
    sc.julia = True
    sc.julia_c = (0.2, 0.4, -0.3, 0.0)
    sc.cut_options = 2
    sc.cut_pos = (0.0, 0.0, 0.0)
    g = gbuffer(cpu_backend, sc)
    assert g.shape[0] == 24


def test_renderer_produces_image_and_bands(cpu_backend):
    sc = small_scene()
    bands = []
    progress = []
    r = Renderer(cpu_backend)
    res = r.render(sc, progress=lambda f, m: progress.append(f), band=lambda y, rgb: bands.append((y, rgb)))
    assert res.image.shape == (48, 64, 3) and res.image.dtype == np.uint8
    assert res.gbuffer is not None
    assert progress and progress[-1] == 1.0
    assert sum(b[1].shape[0] for b in bands) == 48
    # relighting without marching gives the same image
    assert np.array_equal(relight(res.gbuffer, sc), res.image)
    sc2 = sc.copy()
    sc2.lighting.lights[0].color = (1.0, 0.0, 0.0)
    assert not np.array_equal(relight(res.gbuffer, sc2), res.image)


def test_renderer_strips_for_big_images(cpu_backend):
    sc = small_scene(40, 30)
    r = Renderer(cpu_backend, RenderOptions(gbuffer_limit=100, strip_rows=8))
    res = r.render(sc)
    assert res.image.shape == (30, 40, 3) and res.gbuffer is None
    full = Renderer(cpu_backend).render(sc).image
    # strips differ from the whole image only through the SSAO margin
    assert np.abs(full.astype(int) - res.image.astype(int)).mean() < 3.0


def test_render_cancel(cpu_backend):
    from pymb3d.backends.base import RenderCancelled

    sc = small_scene(200, 200)
    sc.iterations = 30
    abort = threading.Event()
    r = Renderer(cpu_backend)

    def progress(f, m):
        if f > 0.1:
            abort.set()

    with pytest.raises(RenderCancelled):
        r.render(sc, progress=progress, abort=abort)


def test_shading_background_and_gamma():
    sc = small_scene(8, 8)
    g = np.zeros((8, 8, PL.N_CHANNELS), dtype=np.float32)
    g[..., PL.G_DEPTH] = -1.0
    g[..., PL.G_AO] = 1.0
    g[..., PL.G_SHADOW:PL.G_SHADOW + 6] = 1.0
    img = shade(g, sc)
    assert img.shape == (8, 8, 3)
    top = np.asarray(sc.lighting.depth_color_top)
    assert np.allclose(img[0, 0], top, atol=1 / 255)
    sc.lighting.gamma = 50
    bright = shade(g, sc)
    assert bright.mean() >= img.mean()
    assert to_uint8(img).dtype == np.uint8


def test_ssao_darkens_corners():
    depth = np.full((40, 40), 100.0)
    depth[:, 20:] = 60.0  # a step: the far side next to the edge is occluded
    ao = ssao(depth, radius=8, strength=1.0)
    assert ao[20, 18] < ao[20, 5]
    assert (ao <= 1.0).all() and (ao >= 0.0).all()


def test_difs_scene_renders(cpu_backend):
    from pymb3d.core.scene import Scene
    from pymb3d.core.camera import Camera

    sc = Scene()
    sc.width, sc.height = 48, 36
    sc.camera = Camera(zoom=0.6, z_start=-3.0, z_end=6.0)
    sc.camera.set_angles_deg(-20, 30, 0)
    sc.slots = [FormulaSlot("SphereIFS", 1, {"radius": 1.0, "x_add": 0.0, "scale": 1.0}),
                FormulaSlot("transformIFS", 1, {"add_x": 1.5, "scale": 1.0, "abs_y": 1}),
                FormulaSlot("boxIFS", 1, {"x_add": 0.0, "scale": 1.0})]
    sc.iterations = 3
    prog = Program.from_scene(sc)
    assert prog.is_difs
    g = gbuffer(cpu_backend, sc)
    hit = g[..., 0] >= 0
    assert 0.05 < hit.mean() < 0.95
    assert (g[..., PL.G_ITER][hit] >= 1).all()


def test_difs_mixing_rejected():
    from pymb3d.core.scene import Scene

    sc = Scene()
    sc.slots = [FormulaSlot("SphereIFS", 1, {}), FormulaSlot("Integer Power", 1, {})]
    with pytest.raises(ValueError):
        Program.from_scene(sc)


def test_de_combination(cpu_backend):
    from pymb3d.core.scene import DeCombination

    sc = small_scene(48, 36)
    sc.slots = [FormulaSlot("Integer Power", 1, {"power": 8}), FormulaSlot("Amazing Box", 1, {"scale": 2.0})]
    sc.iterations = 8
    base = gbuffer(cpu_backend, sc).copy()
    results = {}
    for kind in (1, 2, 4, 5):
        sc.decomb = DeCombination(kind=kind, start2=1, repeat_from2=1, iterations2=6, smooth=0.3)
        prog = Program.from_scene(sc)
        assert len(prog.parts) == 2
        results[kind] = gbuffer(cpu_backend, sc)
    # the maximum is a subset of the minimum, the minimum covers at least part 1
    assert (results[2][..., 0] >= 0).sum() <= (results[1][..., 0] >= 0).sum()
    assert (results[1][..., 0] >= 0).sum() >= (base[..., 0] >= 0).sum() * 0.9


def test_color_options(cpu_backend):
    sc = small_scene(32, 24)
    vals = []
    for opt in range(6):
        sc.color_option = opt
        g = gbuffer(cpu_backend, sc)
        hit = g[..., 0] >= 0
        assert (g[..., PL.G_OTRAP][hit] >= 0).all() and (g[..., PL.G_OTRAP][hit] <= 32767).all()
        vals.append(g[..., PL.G_OTRAP][hit].mean())
    assert len(set(round(v) for v in vals)) > 1


def test_positional_light_shadow(cpu_backend):
    sc = small_scene(32, 24)
    sc.shadows = True
    sc.lighting.lights[0].positional = True
    sc.lighting.lights[0].position = (2.0, 2.0, -2.0)
    g = gbuffer(cpu_backend, sc)
    hit = g[..., 0] >= 0
    assert hit.any() and (g[..., PL.G_SHADOW][hit] <= 1.0).all()


def test_inside_rendering(cpu_backend):
    sc = small_scene(32, 24)
    sc.camera.mid = np.array([0.0, 0.0, 0.0])
    sc.camera.z_start, sc.camera.z_end = -0.2, 3.0   # the start plane inside the bulb
    sc.inside_mode = 1
    g = gbuffer(cpu_backend, sc)
    hit = g[..., 0] >= 0
    assert hit.mean() > 0.5
    sc.inside_mode = 2
    g2 = gbuffer(cpu_backend, sc)
    assert (g2[..., 0] >= 0).any()


def test_interpolation_hybrid(cpu_backend):
    sc = small_scene(32, 24)
    sc.slots = [FormulaSlot("Integer Power", 1, {"power": 8}), FormulaSlot("Amazing Box", 1, {"scale": 2.0})]
    sc.interpolation = (0.7, 0.3)
    prog = Program.from_scene(sc)
    assert prog.ipol and prog.de_mode == 0
    g = gbuffer(cpu_backend, sc)
    assert (g[..., 0] >= 0).any()
    sc.interpolation = (1.0, 0.0)
    only = gbuffer(cpu_backend, sc)
    sc.interpolation = None
    sc.slots = [FormulaSlot("Integer Power", 1, {"power": 8})]
    sc.analytic_de = False
    plain = gbuffer(cpu_backend, sc)
    assert ((only[..., 0] >= 0) == (plain[..., 0] >= 0)).mean() > 0.95


def test_smooth_normals_and_roughness(cpu_backend):
    sc = small_scene(32, 24)
    g0 = gbuffer(cpu_backend, sc)
    sc.smooth_normals = 3
    g = gbuffer(cpu_backend, sc)
    hit = g[..., 0] >= 0
    assert np.allclose(np.linalg.norm(g[..., 1:4][hit], axis=1), 1.0, atol=1e-4)
    assert (g[..., PL.G_ROUGH][hit] >= 0).all() and (g[..., PL.G_ROUGH][hit] <= 1).all()
    assert (g[..., PL.G_ROUGH][hit] > 0).any()
    sc.smooth_normals = 8
    g8 = gbuffer(cpu_backend, sc)
    assert np.abs(g8[..., 1:4][hit] - g0[..., 1:4][hit]).mean() < 0.5


def test_mc_sky_visibility(cpu_backend):
    sc = small_scene(32, 24)
    sc.ao.mode = "mc"
    sc.ao.samples = 4
    g = gbuffer(cpu_backend, sc)
    hit = g[..., 0] >= 0
    ao = g[..., PL.G_AO][hit]
    assert (ao >= 0).all() and (ao <= 1).all() and ao.mean() < 1.0
    from pymb3d.render import Renderer

    sc.ao.passes = 2
    res = Renderer(cpu_backend).render(sc)
    assert res.gbuffer is not None and (res.gbuffer[..., PL.G_AO][hit] <= 1).all()


def test_difs_mix_combination(cpu_backend):
    from pymb3d.core.scene import DeCombination

    sc = small_scene(32, 24)
    sc.camera.zoom = 0.6
    sc.slots = [FormulaSlot("Amazing Box", 2, {"scale": 2.0}), FormulaSlot("SphereIFS", 1, {"x_add": 0.0, "scale": 1.0})]
    sc.decomb = DeCombination(kind=6, start2=1, repeat_from2=1, iterations2=2)
    prog = Program.from_scene(sc)
    assert prog.mix_kind and prog.parts[1].is_difs
    g = gbuffer(cpu_backend, sc)
    assert g.shape[0] == 24


def test_post_processing_and_maps(cpu_backend, tmp_path):
    from PIL import Image
    from pymb3d.render import Renderer
    from pymb3d.render.post import post_process, depth_of_field

    sc = small_scene(48, 36)
    res = Renderer(cpu_backend).render(sc)
    base = res.image.copy()
    # post settings change the picture but keep its shape and range
    sc.post.contrast = 0.5
    sc.post.saturation = -0.5
    sc.post.sharpen = 0.5
    img2 = relight(res.gbuffer, sc)
    assert img2.shape == base.shape and not np.array_equal(img2, base)
    sc.post.dof = True
    sc.post.dof_focus = 0.0
    sc.post.dof_aperture = 30.0
    img3 = relight(res.gbuffer, sc)
    assert img3.shape == base.shape
    # a background picture, a light map and a diffuse map
    grad = np.zeros((16, 32, 3), dtype=np.uint8)
    grad[..., 0] = np.linspace(0, 255, 32)[None, :]
    grad[..., 2] = 200
    Image.fromarray(grad).save(tmp_path / "bg.png")
    sc.post = type(sc.post)()
    sc.lighting.bg_image = str(tmp_path / "bg.png")
    sc.lighting.bg_brightness = 1.0
    img4 = relight(res.gbuffer, sc)
    bgpix = img4[res.gbuffer[..., 0] < 0]
    assert bgpix.size and bgpix[:, 2].mean() > 100
    sc.lighting.bg_direct = True
    sc.lighting.bg_ambient = True
    relight(res.gbuffer, sc)
    sc.lighting.light_map = str(tmp_path / "bg.png")
    sc.lighting.diffuse_map = str(tmp_path / "bg.png")
    for mode in ("normals", "wrap", "iter_otrap"):
        sc.lighting.diffuse_map_mode = mode
        img5 = relight(res.gbuffer, sc)
        assert img5.shape == base.shape
    sc.lighting.diffuse_map_combine = True
    relight(res.gbuffer, sc)


def test_volumetric_light_and_glow(cpu_backend):
    from pymb3d.render import Renderer

    sc = small_scene(32, 24)
    sc.lighting.vol_light = 1
    sc.lighting.lights[0].visible = 2
    g = gbuffer(cpu_backend, sc)
    assert (g[..., PL.G_VOL] >= 0).all() and g[..., PL.G_VOL].max() > 0
    res = Renderer(cpu_backend).render(sc)
    assert res.image.shape == (24, 32, 3)


def test_reflections(cpu_backend):
    from pymb3d.render import Renderer

    sc = small_scene(40, 30)
    sc.camera.zoom = 0.8
    sc.lighting.reflection = 0.8
    sc.shadows = True
    sc.ao.mode = "de"
    g = gbuffer(cpu_backend, sc)
    hit = g[..., 0] >= 0
    r_hit = g[..., PL.G_R_DEPTH] >= 0
    assert r_hit.any() and (r_hit & ~hit).sum() == 0
    assert np.allclose(np.linalg.norm(g[..., PL.G_R_NX:PL.G_R_NZ + 1][r_hit], axis=1), 1.0, atol=1e-4)
    res = Renderer(cpu_backend).render(sc)
    sc2 = sc.copy()
    sc2.lighting.reflection = 0.0
    assert not np.array_equal(relight(res.gbuffer, sc2), res.image)


def test_slice_2d(cpu_backend):
    sc = small_scene(32, 24)
    sc.camera.z_start = 0.0
    sc.slice_2d = 1
    g = gbuffer(cpu_backend, sc)
    assert (g[..., 0] >= 0).all()
    its = g[..., PL.G_ITER]
    assert its.max() > its.min()


def test_mixed_precision_finish(cpu_backend, opencl_backend):
    """A deep zoom marched in single precision and finished in double on
    the CPU matches the CPU's own double precision render."""
    from pymb3d.render import Renderer

    deep = small_scene(48, 36)
    deep.camera.mid = np.array([-0.56645371, 0.27057946, -0.53137482])
    deep.camera.zoom = 3e5
    sw = deep.camera.step_width(48)
    deep.camera.z_start = float(deep.camera.mid[2]) - 300 * sw
    deep.camera.z_end = float(deep.camera.mid[2]) + 3000 * sw
    ref = gbuffer(cpu_backend, deep)
    if not opencl_backend.double_ok:
        pytest.skip("no double precision")
    opencl_backend.precision_mode = "mixed"
    r = Renderer(opencl_backend)
    res = r.render(deep)
    g = res.gbuffer
    assert opencl_backend.needs_finish or opencl_backend.last_probe.single_ok
    hit_ref, hit = ref[..., 0] >= 0, g[..., 0] >= 0
    assert (hit_ref != hit).mean() < 0.05
    both = hit_ref & hit
    if both.any():
        ds = deep.de_stop * (1.0 + ref[..., 0][both] * 0.0)
        assert np.median(np.abs(ref[..., 0][both] - g[..., 0][both])) < 2.0
    opencl_backend.precision_mode = "auto"


def test_dct_denoise_reduces_noise():
    from pymb3d.render.post import dct_denoise

    rng = np.random.default_rng(0)
    clean = np.zeros((40, 48, 3), dtype=np.float32)
    clean[:, :24] = 0.3
    clean[:, 24:] = 0.7
    noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape).astype(np.float32), 0, 1)
    den = dct_denoise(noisy, 8.0)
    assert den.shape == clean.shape
    assert np.abs(den - clean).mean() < np.abs(noisy - clean).mean() * 0.7


def test_render_resume(cpu_backend):
    from pymb3d.backends.base import RenderCancelled
    from pymb3d.render import Renderer

    sc = small_scene(64, 64)
    sc.iterations = 20
    abort = threading.Event()
    r = Renderer(cpu_backend)

    def progress(f, m):
        if f > 0.3:
            abort.set()

    try:
        r.render(sc, progress=progress, abort=abort)
        partial = None
    except RenderCancelled as e:
        partial = e.partial
    assert partial is not None and 0.0 < partial.fraction < 1.0
    done_before = partial.done_rows.copy()
    res = Renderer(cpu_backend).render(sc, resume=partial)
    assert res.image.shape == (64, 64, 3)
    full = Renderer(cpu_backend).render(sc)
    assert np.array_equal(res.gbuffer[done_before], full.gbuffer[done_before])
    assert np.array_equal(res.image, full.image)
