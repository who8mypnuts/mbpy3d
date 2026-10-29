"""GUI smoke tests (offscreen Qt): the window renders previews, panels
drive the model, lighting changes re-shade without marching, the final
render dialog finishes and saves."""
import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pymb3d.backends.detect import DeviceChoice  # noqa: E402

CPU = DeviceChoice("cpu", "numba (all cores)", "float64", "cpu")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(cond, timeout=60.0):
    t0 = time.time()
    while not cond():
        QCoreApplication.processEvents()
        time.sleep(0.01)
        if time.time() - t0 > timeout:
            raise TimeoutError("condition not met")


@pytest.fixture
def window(app):
    from pymb3d.gui.main_window import MainWindow

    w = MainWindow(devices=[CPU])
    w.controller.preview_width = 96
    images = []
    results = []
    w.controller.image_ready.connect(lambda img: images.append(img))
    w.controller.result_ready.connect(lambda r: results.append(r))
    w.show()
    w._test_images = images
    w._test_results = results
    yield w
    w.close()
    QCoreApplication.processEvents()


def test_preview_renders(window):
    wait_until(lambda: len(window._test_results) >= 1)
    res = window._test_results[-1]
    assert res.image.shape[1] == 96 and res.gbuffer is not None
    assert window._test_images[-1].shape == res.image.shape


def test_lighting_change_relights_without_march(window):
    wait_until(lambda: len(window._test_results) >= 1)
    n_results = len(window._test_results)
    n_images = len(window._test_images)
    window.light_panel.global_editor.colors["ambient_top"].set_rgb((1.0, 0.0, 0.0))
    window.light_panel.global_editor.colors["ambient_top"].colorChanged.emit((1.0, 0.0, 0.0))
    wait_until(lambda: len(window._test_images) > n_images, timeout=10)
    QCoreApplication.processEvents()
    assert len(window._test_results) == n_results  # no new march
    assert window.model.scene.lighting.ambient_top == (1.0, 0.0, 0.0)


def test_formula_change_triggers_march(window):
    wait_until(lambda: len(window._test_results) >= 1)
    n = len(window._test_results)
    editor = window.formula_panel.editors[0]
    editor.combo.setCurrentIndex(editor.combo.findData("Amazing Box"))
    wait_until(lambda: len(window._test_results) > n)
    assert window.model.scene.slots[0].name == "Amazing Box"
    assert window._test_results[-1].scene.slots[0].name == "Amazing Box"


def test_navigation_changes_camera(window):
    wait_until(lambda: len(window._test_results) >= 1)
    mid0 = window.model.scene.camera.mid.copy()
    vg0 = window.model.scene.camera.vgrads.copy()
    window.preview.rotate.emit(10.0, 0.0)
    window.preview.move_forward.emit(1.0)
    window.preview.gesture_end.emit()
    assert not np.allclose(window.model.scene.camera.vgrads, vg0)
    assert not np.allclose(window.model.scene.camera.mid, mid0)
    assert np.allclose(window.model.scene.camera.vgrads @ window.model.scene.camera.vgrads.T, np.eye(3), atol=1e-9)


def test_open_m3p(window):
    path = "/workspace/mb3d-code/mb3d/M3Parameter/ABoxScale2Start.m3p"
    if not os.path.exists(path):
        pytest.skip("no example files")
    window.open_path(path)
    assert window.model.scene.slots[0].name == "Amazing Box"
    assert window.camera_panel.zoom.value() == pytest.approx(window.model.scene.camera.zoom)


def test_final_render_dialog(app, tmp_path):
    from pymb3d.core.scene import default_scene
    from pymb3d.gui.render_dialog import RenderDialog
    from pymb3d.io import load_scene

    sc = default_scene()
    sc.width, sc.height = 64, 40
    sc.camera.z_start, sc.camera.zoom = -2.0, 1.4
    dlg = RenderDialog(sc, CPU)
    dlg.show()
    wait_until(lambda: dlg.result is not None, timeout=120)
    assert dlg.result.image.shape == (40, 64, 3)
    out = tmp_path / "out.png"
    from pymb3d.gui.render_dialog import save_image

    save_image(out, dlg.result.image, sc)
    back, _ = load_scene(out)
    assert back.width == 64 and back.slots[0].name == sc.slots[0].name
    dlg.close()


def test_undo_redo(window):
    wait_until(lambda: len(window._test_results) >= 1)
    it0 = window.model.scene.iterations
    window.formula_panel.iterations.setValue(it0 + 3)
    assert window.model.scene.iterations == it0 + 3
    assert window.model.undo()
    assert window.model.scene.iterations == it0
    assert window.model.redo()
    assert window.model.scene.iterations == it0 + 3


def test_animation_dialog(window, tmp_path):
    from pymb3d.gui.animation_dialog import AnimationDialog

    wait_until(lambda: len(window._test_results) >= 1)
    dlg = AnimationDialog(window.model, CPU, window)
    dlg.frame_spin.setValue(0)
    dlg._add()
    window.model.scene.camera.zoom *= 2
    dlg.frame_spin.setValue(2)
    dlg._add()
    assert dlg.anim.frame_count == 3
    dlg.width.setValue(24)
    dlg.height.setValue(16)
    dlg.out_dir.setText(str(tmp_path))
    dlg._render()
    wait_until(lambda: dlg._thread is None, timeout=120)
    assert sorted(p.name for p in tmp_path.glob("frame_*.png")) == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    dlg.scrub.setValue(1)
    assert abs(window.model.scene.camera.zoom - dlg.anim.scene_at(1).camera.zoom) < 1e-9
    dlg.close()


def test_render_panel_devices_and_export(window, tmp_path):
    wait_until(lambda: len(window._test_results) >= 1)
    assert window.render_panel.final_choice.backend == "cpu"
    from pymb3d.io import save_scene, load_scene

    p = tmp_path / "x.m3p"
    save_scene(window.model.scene, p)
    sc, _ = load_scene(p)
    assert sc.slots[0].name == window.model.scene.slots[0].name


def test_render_dialog_data_round_trip(app, tmp_path):
    from pymb3d.core.scene import default_scene
    from pymb3d.gui.render_dialog import RenderDialog
    from pymb3d.io.gbuffer_file import load_gbuffer, save_gbuffer

    sc = default_scene()
    sc.width, sc.height = 40, 30
    sc.camera.z_start, sc.camera.zoom = -2.0, 1.4
    dlg = RenderDialog(sc, CPU)
    wait_until(lambda: dlg.result is not None, timeout=120)
    p = tmp_path / "r.npz"
    save_gbuffer(p, dlg.result.gbuffer, dlg.result.scene)
    g, back = load_gbuffer(p)
    assert g.shape == dlg.result.gbuffer.shape and back.width == 40
    dlg.close()
