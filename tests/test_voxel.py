import numpy as np
import pytest

from pymb3d.core.scene import default_scene
from pymb3d.render.voxel import VoxelBox, export_mesh, inside_mask, mesh_from_field, sample_de_grid, save_slices


def test_voxel_sampling_and_export(tmp_path):
    sc = default_scene()
    box = VoxelBox((0.0, 0.0, 0.0), 3.0, 16)
    calls = []
    field = sample_de_grid(sc, box, progress=lambda f, m: calls.append(f))
    assert field.shape == (17, 17, 17) and calls[-1] == 1.0
    mask = inside_mask(field, box.step)
    assert 0.01 < mask.mean() < 0.9
    files = save_slices(field, box.step, tmp_path / "slices")
    assert len(files) == 17
    pytest.importorskip("skimage")
    verts, faces = mesh_from_field(field, box)
    assert len(faces) > 0 and verts.shape[1] == 3
    assert np.all(np.abs(verts) <= 1.6)
    n = export_mesh(sc, box, tmp_path / "m.obj")
    assert n == len(faces) and (tmp_path / "m.obj").read_text().startswith("# pymb3d")
