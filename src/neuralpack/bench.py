"""``npk bench``: rate/quality of the neural codec vs JPEG and WebP at *equal file size*."""

from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from .codec import decode_bytes, encode_image
from .metrics import classic_at_size, decode_classic, ms_ssim, ms_ssim_db, psnr


def bench_image(path: str | Path, qualities: list[int], model: str, device: str, tile: int,
                overlap: int, baselines: bool = True) -> list[dict]:
    img = Image.open(path)
    img.load()
    img = img.convert("RGB")
    px = img.width * img.height
    rows = []
    for q in qualities:
        res = encode_image(img, quality=q, model=model, device=device, tile=tile, overlap=overlap)
        t0 = time.perf_counter()
        rec = decode_bytes(res.data, device=device)
        t_dec = time.perf_counter() - t0
        row = {"image": Path(path).name, "codec": f"npk/{model}", "quality": q, "bytes": res.bytes,
               "bpp": res.bpp, "psnr": psnr(img, rec), "msssim_db": ms_ssim_db(ms_ssim(img, rec)),
               "enc_s": res.seconds, "dec_s": t_dec}
        rows.append(row)
        if baselines:
            for fmt in ("JPEG", "WEBP"):
                cq, data = classic_at_size(img, fmt, res.bytes)
                crec = decode_classic(data)
                rows.append({"image": Path(path).name, "codec": f"{fmt.lower()}@same-size",
                             "quality": cq, "bytes": len(data), "bpp": len(data) * 8 / px,
                             "psnr": psnr(img, crec), "msssim_db": ms_ssim_db(ms_ssim(img, crec)),
                             "enc_s": None, "dec_s": None})
    return rows


def format_table(rows: list[dict]) -> str:
    hdr = f"{'image':14s} {'codec':20s} {'q':>3s} {'bytes':>9s} {'bpp':>6s} {'PSNR':>6s} {'MS-SSIM dB':>10s} {'enc s':>6s} {'dec s':>6s}"
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        ms = "-" if r["msssim_db"] is None else f"{r['msssim_db']:.2f}"
        enc = "-" if r["enc_s"] is None else f"{r['enc_s']:.2f}"
        dec = "-" if r["dec_s"] is None else f"{r['dec_s']:.2f}"
        out.append(f"{r['image'][:14]:14s} {r['codec']:20s} {r['quality']:3d} {r['bytes']:9d} "
                   f"{r['bpp']:6.3f} {r['psnr']:6.2f} {ms:>10s} {enc:>6s} {dec:>6s}")
    return "\n".join(out)


def run_bench(inputs: list[str], qualities: list[int], model: str, device: str, tile: int,
              overlap: int, json_out: str | None = None, baselines: bool = True) -> list[dict]:
    rows: list[dict] = []
    for p in inputs:
        rows += bench_image(p, qualities, model, device, tile, overlap, baselines)
    print(format_table(rows))
    if json_out:
        Path(json_out).write_text(json.dumps(rows, indent=2))
    return rows
