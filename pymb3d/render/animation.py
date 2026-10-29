"""Renders the frames of an animation to numbered PNG files (and a video
when ffmpeg is available)."""
from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

from ..backends.base import Backend, RenderCancelled
from ..core.animation import Animation
from .renderer import Renderer, RenderOptions

ProgressFn = Callable[[int, int, float, str], None]  # frame, frames, fraction within the frame, message


def frame_path(out_dir: Path, frame: int) -> Path:
    return out_dir / f"frame_{frame:05d}.png"


def render_animation(anim: Animation, backend: Backend, out_dir: str | Path, width: int | None = None,
                     height: int | None = None, progress: ProgressFn | None = None,
                     abort: threading.Event | None = None, frames: list[int] | None = None,
                     skip_existing: bool = True) -> list[Path]:
    """Renders ``frames`` (default all) and returns the files written."""
    from ..gui.render_dialog import save_image  # PNG with the scene embedded

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    abort = abort or threading.Event()
    renderer = Renderer(backend, RenderOptions(live_bands=False))
    n = anim.frame_count
    todo = frames if frames is not None else list(range(n))
    done: list[Path] = []
    for k, f in enumerate(todo):
        if abort.is_set():
            raise RenderCancelled()
        path = frame_path(out_dir, f)
        if skip_existing and path.exists():
            done.append(path)
            continue
        scene = anim.scene_at(f)
        if width:
            scene.width = width
            scene.height = height or int(round(width * scene.height / max(1, scene.width)))
        res = renderer.render(scene, progress=(lambda fr, m, _f=f, _k=k: progress(_k, len(todo), fr, m)) if progress else None,
                              abort=abort)
        save_image(path, res.image, scene)
        done.append(path)
    return done


def encode_video(out_dir: str | Path, video_path: str | Path, fps: float = 25.0) -> bool:
    """Encodes the PNG frames with ffmpeg (H.264); False when ffmpeg is missing."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", str(Path(out_dir) / "frame_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(video_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return True
