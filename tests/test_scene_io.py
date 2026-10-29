import glob
import json
import math
import os

import numpy as np
import pytest

from pymb3d.core.camera import Camera
from pymb3d.core.scene import FormulaSlot, Scene, default_scene
from pymb3d.core.vec import angles_from_vgrads, vgrads_from_angles
from pymb3d.io import load_scene, save_scene, scene_from_clipboard_text
from pymb3d.io.m3p import load, raw_from_text, raw_to_text, parse, HEADER_SIZE, ADDON_SIZE

PARAM_DIR = "/workspace/mb3d-code/mb3d/M3Parameter"


def test_scene_json_round_trip(tmp_path):
    sc = default_scene()
    sc.slots.append(FormulaSlot("Amazing Box", 3, {"scale": -1.7}))
    sc.lighting.lights[1].enabled = True
    sc.camera.set_angles_deg(10, 20, 30)
    p = tmp_path / "scene.json"
    save_scene(sc, p)
    back, warnings = load_scene(p)
    assert warnings == []
    assert back.to_dict() == sc.to_dict()
    assert back.slots[1].params["scale"] == -1.7


def test_camera_angles_round_trip():
    for ax, ay, az in [(0.1, 0.2, 0.3), (-1.0, 0.5, 2.0), (0.0, 0.0, 0.0), (0.3, -1.2, -0.4)]:
        m = vgrads_from_angles(ax, ay, az)
        bx, by, bz = angles_from_vgrads(m)
        assert np.allclose(vgrads_from_angles(bx, by, bz), m, atol=1e-9)


def test_camera_navigation_keeps_orthonormal():
    c = Camera()
    c.set_angles_deg(-20, 30, 0)
    c.rotate(np.array([0, 1, 0]), 0.3)
    c.rotate(np.array([1, 0, 0]), -0.2)
    c.move_pixels(10, -5, 3, 640)
    assert np.allclose(c.vgrads @ c.vgrads.T, np.eye(3), atol=1e-9)
    setup = c.kernel_setup(640, 480)
    start, d = setup.ray(320, 240)
    assert math.isclose(np.linalg.norm(d), setup.step_width, rel_tol=1e-9)


@pytest.mark.skipif(not os.path.isdir(PARAM_DIR), reason="MB3D example parameters not available")
def test_all_example_m3p_files_parse():
    files = sorted(glob.glob(os.path.join(PARAM_DIR, "*.m3p")))
    assert files
    for f in files:
        res = load(f)
        assert 1 <= res.scene.width < 32768
        assert res.scene.slots


@pytest.mark.skipif(not os.path.isdir(PARAM_DIR), reason="MB3D example parameters not available")
def test_known_m3p_values():
    res = load(os.path.join(PARAM_DIR, "MengerTrees.m3p"))
    sc = res.scene
    assert [s.name for s in sc.slots] == ["Menger3"]
    assert sc.iterations == 60
    assert sc.shadows
    assert math.isclose(sc.camera.zoom, 20.7, rel_tol=0.01)
    assert len(sc.lighting.palette) == 10


def test_text_parameters_round_trip():
    raw = bytes(HEADER_SIZE + ADDON_SIZE)
    import struct

    raw = bytearray(raw)
    raw[0:4] = struct.pack("<i", 44)
    raw[4:8] = struct.pack("<i", 320)
    raw[8:12] = struct.pack("<i", 240)
    raw[12:16] = struct.pack("<i", 12)
    raw[84:92] = struct.pack("<d", 1.5)
    raw[92:100] = struct.pack("<d", 16.0)
    raw[108:116] = struct.pack("<d", 60.0)
    raw[840] = 16
    raw[848:852] = struct.pack("<i", 1)
    raw[852:856] = struct.pack("<i", 0)
    raw[856:860] = struct.pack("<i", 2)
    raw[860:873] = b"Integer Power"
    raw[908:916] = struct.pack("<d", 8.0)
    raw[916:924] = struct.pack("<d", -1.0)
    for i in range(3):
        raw[246 + (i * 3 + i) * 8:246 + (i * 3 + i) * 8 + 8] = struct.pack("<d", 1.0)
    text = raw_to_text(bytes(raw), "round trip")
    back, title = raw_from_text(text)
    assert title == "round trip"
    assert back[:HEADER_SIZE + 8 + 188] == bytes(raw[:HEADER_SIZE + 8 + 188])
    sc, warnings = scene_from_clipboard_text(text)
    assert sc.width == 320 and sc.slots[0].name == "Integer Power" and sc.slots[0].params["power"] == 8.0
    assert math.isclose(sc.camera.zoom, 1.5)
