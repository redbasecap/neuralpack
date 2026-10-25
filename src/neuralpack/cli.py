"""``npk`` command line interface."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import __version__
from .codec import DEFAULT_OVERLAP, DEFAULT_TILE, decode_file, encode_file
from .container import NpkFormatError, inspect
from .models import DEFAULT_MODEL, DEFAULT_QUALITY, MODELS, available_devices
from .video import VIDEO_SUFFIXES, decode_video, encode_video

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".ppm", ".pgm",
                  ".heic", ".heif", ".jp2", ".dng"}


def _human(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000
    return f"{n:.2f} TB"


class _Progress:
    def __init__(self, label: str, enabled: bool):
        self.label, self.enabled, self.t0 = label, enabled and sys.stderr.isatty(), time.perf_counter()

    def __call__(self, done: int, total: int | None) -> None:
        if not self.enabled:
            return
        tot = f"/{total}" if total else ""
        sys.stderr.write(f"\r{self.label} {done}{tot}  {time.perf_counter() - self.t0:.1f}s")
        sys.stderr.flush()
        if total and done >= total:
            sys.stderr.write("\n")


def _add_common_encode(p: argparse.ArgumentParser) -> None:
    p.add_argument("-q", "--quality", type=int, default=DEFAULT_QUALITY,
                   help=f"1 (smallest) .. 8 (near transparent); default {DEFAULT_QUALITY}")
    p.add_argument("-m", "--model", choices=sorted(MODELS), default=DEFAULT_MODEL,
                   help=f"shared prior to use (default {DEFAULT_MODEL})")
    p.add_argument("--tile", type=int, default=DEFAULT_TILE,
                   help=f"coded tile edge in px, multiple of 64; 0 = whole image (default {DEFAULT_TILE})")
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                   help=f"halo around tiles in px (default {DEFAULT_OVERLAP})")
    p.add_argument("--no-metadata", action="store_true", help="drop EXIF / ICC profile")
    p.add_argument("--no-web", action="store_true",
                   help="use the float64 hyper-synthesis instead of the integer one; ~0.1 %% smaller "
                        "but the file cannot be opened by the browser viewer")


def _add_device(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="where to run the transforms (default auto)")


def _out_path(src: Path, out: str | None, many: bool, suffix: str) -> Path:
    if out is None:
        return src.with_suffix(suffix)
    o = Path(out)
    if many or o.is_dir():
        o.mkdir(parents=True, exist_ok=True)
        return o / src.with_suffix(suffix).name
    return o


def cmd_encode(a: argparse.Namespace) -> int:
    many = len(a.inputs) > 1
    kw = {"quality": a.quality, "model": a.model, "device": a.device, "tile": a.tile, "overlap": a.overlap,
              "keep_metadata": not a.no_metadata}
    total_in = total_out = 0
    for src in map(Path, a.inputs):
        t0 = time.perf_counter()
        if src.suffix.lower() in VIDEO_SUFFIXES:
            dst = _out_path(src, a.output, many, ".npkv")
            encode_video(src, dst, progress=_Progress(f"{src.name}: frames", not a.quiet), **kw)
        else:
            dst = _out_path(src, a.output, many, ".npk")
            encode_file(src, dst, progress=_Progress(f"{src.name}: tiles", not a.quiet), **kw)
        n_in, n_out = src.stat().st_size, dst.stat().st_size
        total_in += n_in
        total_out += n_out
        if not a.quiet:
            print(f"{src.name} -> {dst.name}: {_human(n_in)} -> {_human(n_out)} "
                  f"({n_in / max(n_out, 1):.1f}x smaller, {time.perf_counter() - t0:.1f}s)")
    if many and not a.quiet:
        print(f"total: {_human(total_in)} -> {_human(total_out)} ({total_in / max(total_out, 1):.1f}x)")
    return 0


def cmd_decode(a: argparse.Namespace) -> int:
    many = len(a.inputs) > 1
    suffix = "." + a.format.lower().lstrip(".")
    for src in map(Path, a.inputs):
        t0 = time.perf_counter()
        if src.suffix.lower() == ".npkv":
            dst = _out_path(src, a.output, many, ".mp4" if suffix == ".png" else suffix)
            decode_video(src, dst, device=a.device,
                         progress=_Progress(f"{src.name}: frames", not a.quiet))
        else:
            dst = _out_path(src, a.output, many, suffix)
            decode_file(src, dst, device=a.device,
                        progress=_Progress(f"{src.name}: tiles", not a.quiet))
        if not a.quiet:
            print(f"{src.name} -> {dst} ({time.perf_counter() - t0:.1f}s)")
    return 0


def cmd_info(a: argparse.Namespace) -> int:
    for src in map(Path, a.inputs):
        data = src.read_bytes()
        try:
            info = inspect(data)
        except NpkFormatError as e:
            print(f"{src}: {e}", file=sys.stderr)
            return 1
        if a.json:
            print(json.dumps({"file": str(src), **info}, indent=2))
            continue
        print(f"{src}")
        for k, v in info.items():
            print(f"  {k:26s} {v}")
    return 0


def cmd_open(a: argparse.Namespace) -> int:
    src = Path(a.input)
    tmp = Path(tempfile.mkdtemp(prefix="npk-")) / (src.stem + ".png")
    decode_file(src, tmp, device=a.device)
    if sys.platform == "darwin":
        subprocess.run(["open", str(tmp)], check=False)
    elif os.name == "nt":  # pragma: no cover
        os.startfile(str(tmp))  # type: ignore[attr-defined]
    else:  # pragma: no cover
        subprocess.run(["xdg-open", str(tmp)], check=False)
    print(tmp)
    return 0


def cmd_models(a: argparse.Namespace) -> int:
    print(f"devices: {', '.join(available_devices())}")
    for name, spec in MODELS.items():
        mark = " (default)" if name == DEFAULT_MODEL else ""
        print(f"{name}{mark}: id={spec.id} qualities={spec.qualities[0]}-{spec.qualities[-1]}  {spec.description}")
    if a.download:
        from .models import load_model
        for name in (a.download if a.download != ["all"] else sorted(MODELS)):
            for q in MODELS[name].qualities:
                load_model(name, q, "cpu")
                print(f"  cached {name} q{q}")
    return 0


def cmd_export_web(a: argparse.Namespace) -> int:
    from .webexport import default_models_dir, export_model, exportable

    out = Path(a.output) if a.output else default_models_dir()
    targets = exportable() if not a.qualities else [(a.model, q) for q in a.qualities]
    for model, q in targets:
        d = export_model(out, model, q, device="cpu")
        size = sum(f.stat().st_size for f in d.iterdir())
        print(f"{model} q{q} -> {d} ({_human(size)})")
    print(f"index: {out / 'index.json'}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .serve import serve

    return serve(port=a.port, models_dir=Path(a.models) if a.models else None, open_browser=not a.no_open,
                 export_missing=not a.no_export)


def cmd_bench(a: argparse.Namespace) -> int:
    from .bench import run_bench

    run_bench(a.inputs, qualities=a.qualities, model=a.model, device=a.device, tile=a.tile,
              overlap=a.overlap, json_out=a.json, baselines=not a.no_baselines)
    return 0


def _qualities(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="npk", description="Neural image & video compression with a shared learned prior.")
    p.add_argument("--version", action="version", version=f"npk {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="image/video -> .npk/.npkv")
    e.add_argument("inputs", nargs="+")
    e.add_argument("-o", "--output", help="output file, or directory for several inputs")
    _add_common_encode(e)
    _add_device(e)
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_encode)

    d = sub.add_parser("decode", help=".npk/.npkv -> image/video")
    d.add_argument("inputs", nargs="+")
    d.add_argument("-o", "--output", help="output file, or directory for several inputs")
    d.add_argument("-f", "--format", default="png", help="png (default), jpg, webp, tiff; mp4/mkv for video")
    _add_device(d)
    d.add_argument("--quiet", action="store_true")
    d.set_defaults(func=cmd_decode)

    i = sub.add_parser("info", help="show what is inside an .npk file")
    i.add_argument("inputs", nargs="+")
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_info)

    o = sub.add_parser("open", help="decode to a temp file and open it in the system viewer")
    o.add_argument("input")
    _add_device(o)
    o.set_defaults(func=cmd_open)

    m = sub.add_parser("models", help="list priors and devices; --download caches weights")
    m.add_argument("--download", nargs="*", metavar="MODEL", help="pre-download weights (no arg = all)")
    m.set_defaults(func=cmd_models)

    x = sub.add_parser("export-web", help="export priors for the browser viewer (ONNX + integer tables)")
    x.add_argument("-o", "--output", help="target directory (default ~/.cache/neuralpack/web-models)")
    x.add_argument("-m", "--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    x.add_argument("-q", "--qualities", type=_qualities, default=None, help="default: all calibrated")
    x.set_defaults(func=cmd_export_web)

    sv = sub.add_parser("serve", help="serve the browser viewer locally (exports missing priors first)")
    sv.add_argument("-p", "--port", type=int, default=8765)
    sv.add_argument("--models", help="exported models directory (default ~/.cache/neuralpack/web-models)")
    sv.add_argument("--no-open", action="store_true", help="do not open the browser")
    sv.add_argument("--no-export", action="store_true", help="do not export missing priors")
    sv.set_defaults(func=cmd_serve)

    b = sub.add_parser("bench", help="rate/quality table vs JPEG & WebP at equal file size")
    b.add_argument("inputs", nargs="+")
    b.add_argument("-q", "--qualities", type=_qualities, default=[1, 3, 5, 8])
    b.add_argument("-m", "--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    b.add_argument("--tile", type=int, default=0)
    b.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    b.add_argument("--json", help="also write results to this JSON file")
    b.add_argument("--no-baselines", action="store_true")
    _add_device(b)
    b.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (NpkFormatError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"npk: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
