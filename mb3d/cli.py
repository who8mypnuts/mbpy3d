"""The command line: render parameter files, list devices, benchmark.

    mb3d render scene.m3p -o out.png [--backend cpu|opengl|opencl|auto] [--scale 0.5]
    mb3d info scene.m3p
    mb3d devices
    mb3d bench scene.m3p [--backends cpu,opengl,opencl]
"""

from __future__ import annotations

import argparse
import sys
import time

import polars as pl

from mb3d.params import load


def _add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("file")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--backend", default="auto", choices=["auto", "cpu", "opengl", "opencl"])
    p.add_argument("--scale", type=float, default=1.0, help="image size factor (the look is kept)")
    p.add_argument("--ao", type=int, default=6, help="DE ambient occlusion samples (0 = off)")
    p.add_argument("--no-shadow", action="store_true")
    p.add_argument("--preview", action="store_true", help="single precision / speed over accuracy")
    p.add_argument("--quiet", action="store_true")


def cmd_render(args) -> int:
    from mb3d.render import Renderer, RenderSettings
    from mb3d.io.image_io import save_image

    scene = load(args.file)
    if args.scale != 1.0:
        scene.scale_image(args.scale)
    settings = RenderSettings(backend=args.backend, precise=not args.preview, ao_samples=args.ao, shadow=not args.no_shadow,
                              progressive_scales=(1.0,))
    last = [0]

    def progress(done, total, scale):
        if not args.quiet and done * 20 // max(1, total) != last[0]:
            last[0] = done * 20 // max(1, total)
            print(f"\r{done}/{total} rows", end="", file=sys.stderr, flush=True)

    t0 = time.time()
    r = Renderer().render(scene, settings, on_progress=progress)
    out = args.output or (args.file.rsplit(".", 1)[0] + ".png")
    save_image(r.image, out)
    if not args.quiet:
        print(f"\r{out}: {r.width}x{r.height} on {r.backend} ({r.precision}), march {r.march_seconds:.2f}s, "
              f"paint {r.paint_seconds:.2f}s, total {time.time() - t0:.2f}s", file=sys.stderr)
    return 0


def cmd_info(args) -> int:
    from mb3d.formulas import registry

    s = load(args.file)
    print(f"{args.file}: {s.width}x{s.height}, MandId {s.mand_id}, iterations {s.iterations}, zoom {s.zoom:g}")
    print(f"mid {s.mid}, z_start {s.z_start:g}, z_end {s.z_end:g}, FOV {s.fov_y:g}, DE stop {s.de_stop:g}, raystep {s.z_step_div:g}")
    rows = []
    for i, f in enumerate(s.formulas):
        spec = registry.find(f.name)
        rows.append({"slot": i + 1, "formula": f.name, "iterations": f.iterations, "fnr": f.fnr,
                     "native": spec is not None, "options": ", ".join(f"{v:g}" for v in f.options[:f.option_count])})
    with pl.Config(fmt_str_lengths=80, tbl_width_chars=140):
        print(pl.DataFrame(rows))
    lights = [i + 1 for i, li in enumerate(s.lighting.lights) if li.on]
    print(f"lights on: {lights}, palette start/end {s.lighting.color_start:g}/{s.lighting.color_end:g}, "
          f"ao {'on' if s.ao else 'off'}, shadows {'on' if s.shadows else 'off'}")
    return 0


def cmd_devices(args) -> int:
    from mb3d.render.device import probe

    rep = probe(allow_software_gl=True)
    for d in rep.available:
        print(f"{d.kind:7s} {d.precision:7s} {d.name}")
    for k, v in rep.errors.items():
        print(f"{k:7s} unavailable: {v}")
    try:
        from mb3d.render.cl_backend import list_devices

        for platform, dev, fp64 in list_devices():
            print(f"opencl device: {dev} [{platform}] fp64={fp64}")
    except Exception:
        pass
    return 0


def cmd_bench(args) -> int:
    from mb3d.render import BackendPool, Renderer, RenderSettings
    from mb3d.render.stats import RenderStats

    scene = load(args.file)
    if args.scale != 1.0:
        scene.scale_image(args.scale)
    stats = RenderStats()
    pool = BackendPool(allow_software_gl=True)
    renderer = Renderer(pool)
    for kind in args.backends.split(","):
        try:
            backend = pool.get(kind)
        except Exception as e:
            print(f"{kind}: unavailable ({e})", file=sys.stderr)
            continue
        settings = RenderSettings(backend=kind, ao_samples=args.ao, shadow=not args.no_shadow, progressive_scales=(1.0,))
        for _ in range(args.repeat):
            r = renderer.render_pass(scene, settings, backend)
            stats.add(r, args.file)
    with pl.Config(tbl_width_chars=140):
        print(stats.frame().select(["backend", "precision", "width", "height", "march_s", "paint_s", "avg_steps", "avg_iterations"]))
        print(stats.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mb3d", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render", help="render a parameter file to an image")
    _add_render_args(p)
    p.set_defaults(func=cmd_render)
    p = sub.add_parser("info", help="print the parameters of a file")
    p.add_argument("file")
    p.set_defaults(func=cmd_info)
    p = sub.add_parser("devices", help="list render devices")
    p.set_defaults(func=cmd_devices)
    p = sub.add_parser("bench", help="time the back ends on a file")
    _add_render_args(p)
    p.add_argument("--backends", default="cpu,opengl,opencl")
    p.add_argument("--repeat", type=int, default=1)
    p.set_defaults(func=cmd_bench)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
