"""Voxel and mesh export (MB3D's voxel export / BulbTracer): the distance
estimate is sampled on a grid inside a box; the slices can be written as
PNG images (for 3D printing slicers) and the iso surface as an OBJ mesh
(marching cubes, needs scikit-image)."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..backends.base import RenderCancelled
from ..core.scene import Scene
from ..kernels.numba_loader import load_program_module
from ..kernels.program import Program


@dataclass
class VoxelBox:
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: float = 3.0           # edge length of the cube (world units)
    resolution: int = 64        # cells per edge

    @property
    def step(self) -> float:
        return self.size / max(1, self.resolution)

    @property
    def origin(self) -> tuple[float, float, float]:
        h = self.size * 0.5
        return (self.center[0] - h, self.center[1] - h, self.center[2] - h)


def sample_de_grid(scene: Scene, box: VoxelBox, progress: Callable[[float, str], None] | None = None,
                   abort: threading.Event | None = None, threads: int = 0) -> np.ndarray:
    """The DE field (world units, -1 inside) on the box's grid (z, y, x)."""
    from concurrent.futures import ThreadPoolExecutor

    from ..backends.cpu_backend import cpu_count

    prog = Program.from_scene(scene)
    mod = load_program_module(prog)
    P = prog.pack(scene)
    n = box.resolution + 1
    out = np.zeros((n, n, n), dtype=np.float64)
    abort = abort or threading.Event()
    flag = np.zeros(1, dtype=np.int32)
    ox, oy, oz = box.origin
    threads = threads or cpu_count()
    chunk = max(1, n // (threads * 4))
    jobs = [(k, min(n, k + chunk)) for k in range(0, n, chunk)]
    done = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = [pool.submit(mod.de_grid, P, ox, oy, oz, box.step, out, k0, k1, flag) for k0, k1 in jobs]
        for f in futs:
            f.result()
            done += 1
            if abort.is_set():
                flag[0] = 1
                raise RenderCancelled()
            if progress:
                progress(done / len(jobs), f"sampled {done}/{len(jobs)} slabs")
    return out


def inside_mask(field: np.ndarray, step: float) -> np.ndarray:
    return (field < 0) | (field < step * 0.5)


def save_slices(field: np.ndarray, step: float, out_dir: str | Path, prefix: str = "slice") -> list[Path]:
    """Writes one PNG per z slice (white = inside)."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = inside_mask(field, step)
    files = []
    for k in range(mask.shape[0]):
        p = out_dir / f"{prefix}_{k:04d}.png"
        Image.fromarray((mask[k] * 255).astype(np.uint8)).save(p)
        files.append(p)
    return files


def mesh_from_field(field: np.ndarray, box: VoxelBox) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (world units) and triangle faces of the iso surface."""
    try:
        from skimage import measure
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("mesh export needs scikit-image (pip install scikit-image)") from e
    f = np.where(field < 0, -box.step, field)
    level = box.step * 0.5
    if f.min() >= level or f.max() <= level:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    verts, faces, _normals, _vals = measure.marching_cubes(f, level=level, spacing=(box.step, box.step, box.step))
    # the field is indexed (z, y, x): bring the vertices to (x, y, z) world order
    ox, oy, oz = box.origin
    world = np.stack([verts[:, 2] + ox, verts[:, 1] + oy, verts[:, 0] + oz], axis=1)
    return world, faces


def save_obj(path: str | Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("# pymb3d mesh export\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def export_mesh(scene: Scene, box: VoxelBox, path: str | Path, progress=None, abort=None) -> int:
    """Samples the box and writes an OBJ; returns the number of triangles."""
    field = sample_de_grid(scene, box, progress, abort)
    if progress:
        progress(0.95, "marching cubes ...")
    verts, faces = mesh_from_field(field, box)
    save_obj(path, verts, faces)
    return int(len(faces))
