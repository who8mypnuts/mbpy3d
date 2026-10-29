"""The command line renderer."""
import os

from PIL import Image

from pymb3d.__main__ import main
from pymb3d.core.scene import default_scene
from pymb3d.io import save_scene


def test_render_cli(tmp_path):
    sc = default_scene()
    sc.width, sc.height = 48, 32
    sc.camera.z_start, sc.camera.zoom = -2.0, 1.4
    scene_path = tmp_path / "s.json"
    save_scene(sc, scene_path)
    out = tmp_path / "o.png"
    assert main(["render", str(scene_path), "-o", str(out), "--backend", "cpu"]) == 0
    assert os.path.exists(out)
    assert Image.open(out).size == (48, 32)


def test_render_cli_resize(tmp_path):
    sc = default_scene()
    sc.width, sc.height = 64, 32
    scene_path = tmp_path / "s.json"
    save_scene(sc, scene_path)
    out = tmp_path / "o.png"
    assert main(["render", str(scene_path), "-o", str(out), "--width", "32", "--backend", "cpu"]) == 0
    assert Image.open(out).size == (32, 16)
