"""``python -m pymb3d [scene.json | file.m3p]`` starts the application;
``python -m pymb3d render SCENE -o out.png [--width W --height H --backend cpu|opencl|opengl]``
renders from the command line, ``python -m pymb3d batch FILES... -o DIR`` a list of files."""
from __future__ import annotations

import argparse
import sys


def render_cli(args: argparse.Namespace) -> int:
    from .backends.detect import available_devices, choose, create_backend
    from .io import load_scene
    from .render import Renderer
    from .gui.render_dialog import save_image

    scene, warnings = load_scene(args.scene)
    for w in warnings:
        print("note:", w, file=sys.stderr)
    if args.width:
        aspect = scene.height / max(1, scene.width)
        scene.width = args.width
        scene.height = args.height or max(1, int(round(args.width * aspect)))
    elif args.height:
        scene.height = args.height
    devices = available_devices(probe_opengl=args.backend in (None, "opengl"))
    if args.backend:
        devices = [d for d in devices if d.backend == args.backend] or devices
    choice = choose(devices, "final")
    backend = create_backend(choice)
    print(f"rendering {scene.width}x{scene.height} on {backend.info.label}", file=sys.stderr)
    last = [0]

    def progress(f, m):
        pct = int(f * 100)
        if pct // 10 != last[0] // 10:
            print(f"{pct:3d}% {m}", file=sys.stderr)
        last[0] = pct

    res = Renderer(backend).render(scene, progress=progress)
    save_image(args.output, res.image, scene)
    print(f"saved {args.output} ({res.seconds:.1f} s)", file=sys.stderr)
    return 0


def batch_cli(args: argparse.Namespace) -> int:
    from .backends.detect import available_devices, choose
    from .gui.batch_dialog import render_files

    devices = available_devices(probe_opengl=args.backend in (None, "opengl"))
    if args.backend:
        devices = [d for d in devices if d.backend == args.backend] or devices
    choice = choose(devices, "final")
    last = [-1]

    def progress(k, n, f, m):
        if k != last[0]:
            print(f"[{k + 1}/{n}] {args.files[k]}", file=sys.stderr)
            last[0] = k

    done = render_files(args.files, args.output, choice, args.width, progress=progress)
    print(f"{len(done)} pictures written to {args.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help"):
        print("usage: pymb3d [scene]            open the GUI (optionally with a scene)\n"
              "       pymb3d render SCENE -o OUT.png [--width W] [--height H] [--backend B]\n"
              "       pymb3d batch FILES... -o DIR [--width W] [--backend B]")
        return 0
    if argv and argv[0] == "batch":
        p = argparse.ArgumentParser(prog="pymb3d batch")
        p.add_argument("files", nargs="+")
        p.add_argument("-o", "--output", default="batch_out")
        p.add_argument("--width", type=int, default=0)
        p.add_argument("--backend", choices=["cpu", "opencl", "opengl"], default=None)
        return batch_cli(p.parse_args(argv[1:]))
    if argv and argv[0] == "render":
        p = argparse.ArgumentParser(prog="pymb3d render")
        p.add_argument("scene")
        p.add_argument("-o", "--output", default="render.png")
        p.add_argument("--width", type=int, default=0)
        p.add_argument("--height", type=int, default=0)
        p.add_argument("--backend", choices=["cpu", "opencl", "opengl"], default=None)
        return render_cli(p.parse_args(argv[1:]))
    from .gui.app import run

    return run([sys.argv[0]] + argv)


if __name__ == "__main__":
    sys.exit(main())
