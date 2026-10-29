import math

import numpy as np

from pymb3d.core.animation import Animation, quat_from_matrix, matrix_from_quat, slerp
from pymb3d.core.scene import default_scene
from pymb3d.core.vec import vgrads_from_angles


def test_quaternion_round_trip():
    for a in [(0.1, 0.2, 0.3), (-1.0, 0.5, 2.0), (0, 0, 0), (3.0, -1.5, 0.7)]:
        m = vgrads_from_angles(*a)
        assert np.allclose(matrix_from_quat(quat_from_matrix(m)), m, atol=1e-9)
    q0, q1 = quat_from_matrix(vgrads_from_angles(0, 0, 0)), quat_from_matrix(vgrads_from_angles(0.5, 0, 0))
    assert np.allclose(slerp(q0, q1, 0.0), q0) and np.allclose(np.abs(slerp(q0, q1, 1.0)), np.abs(q1))


def test_animation_interpolates(tmp_path):
    a = default_scene()
    a.camera.zoom = 1.0
    a.camera.set_angles_deg(0, 0, 0)
    a.slots[0].params["power"] = 2.0
    a.iterations = 10
    a.lighting.lights[0].amplitude = 1.0
    b = a.copy()
    b.camera.zoom = 100.0
    b.camera.set_angles_deg(0, 90, 0)
    b.slots[0].params["power"] = 8.0
    b.iterations = 20
    b.lighting.lights[0].amplitude = 3.0
    anim = Animation(fps=10)
    anim.set_keyframe(0, a)
    anim.set_keyframe(20, b)
    assert anim.frame_count == 21
    mid = anim.scene_at(10)
    assert math.isclose(mid.camera.zoom, 10.0, rel_tol=1e-6)          # logarithmic
    assert math.isclose(mid.slots[0].params["power"], 5.0, abs_tol=1e-9)
    assert mid.iterations == 15
    assert math.isclose(mid.lighting.lights[0].amplitude, 2.0, abs_tol=1e-9)
    ay = mid.camera.angles_deg[1]
    assert 40 < ay < 50                                                  # slerp half way
    assert np.allclose(mid.camera.vgrads @ mid.camera.vgrads.T, np.eye(3), atol=1e-9)
    assert anim.scene_at(0).to_dict() == a.to_dict() or math.isclose(anim.scene_at(0).camera.zoom, 1.0)
    # eased: still the same end points, slower start
    anim.easing = "ease"
    assert anim.scene_at(2).iterations <= 11
    p = tmp_path / "anim.json"
    anim.save(p)
    back = Animation.load(p)
    assert back.frame_count == 21 and back.easing == "ease" and back.fps == 10


def test_render_animation(cpu_backend, tmp_path):
    from pymb3d.render.animation import render_animation

    a = default_scene()
    a.width, a.height = 32, 24
    a.camera.z_start, a.camera.zoom = -2.0, 1.4
    b = a.copy()
    b.camera.zoom = 2.0
    anim = Animation()
    anim.set_keyframe(0, a)
    anim.set_keyframe(2, b)
    calls = []
    files = render_animation(anim, cpu_backend, tmp_path, progress=lambda k, n, f, m: calls.append((k, n)))
    assert len(files) == 3 and all(f.exists() for f in files)
    assert calls and calls[-1][1] == 3


def test_spline_interpolation():
    from pymb3d.core.animation import catmull_rom

    xk = np.array([0.0, 10.0, 20.0, 30.0])
    yk = np.array([0.0, 1.0, 0.0, 1.0])
    y = catmull_rom(xk, yk, np.array([0.0, 5.0, 10.0, 25.0, 30.0]))
    assert np.allclose(y[[0, 2, 4]], [0.0, 1.0, 1.0])
    assert 0.0 < y[1] < 1.0
    a = default_scene()
    anim = Animation(easing="spline")
    for f, z in ((0, 1.0), (10, 2.0), (20, 8.0)):
        b = a.copy()
        b.camera.zoom = z
        b.iterations = 10 + f
        anim.set_keyframe(f, b)
    mid = anim.scene_at(10)
    assert math.isclose(mid.camera.zoom, 2.0, rel_tol=1e-6) and mid.iterations == 20
    assert 1.0 < anim.scene_at(5).camera.zoom < 2.0
