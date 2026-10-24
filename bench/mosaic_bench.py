"""Headline benchmark on one large photo: ratio vs. the original file, quality vs. JPEG/WebP
at equal size, and the JPEG size needed to *match* npk quality. Usage:
    python bench/mosaic_bench.py path/to/large_photo.jpg [--json out.json]
"""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

from PIL import Image

import neuralpack as npk
from neuralpack.metrics import (
    classic_at_size,
    decode_classic,
    encode_classic,
    ms_ssim,
    ms_ssim_db,
    psnr,
)

src = Path(sys.argv[1]); json_out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
img = Image.open(src); img.load(); img = img.convert("RGB")
px = img.width * img.height
print(f"{src.name}: {img.width}x{img.height} ({px/1e6:.1f} MP), source file {src.stat().st_size/1e6:.2f} MB, raw {px*3/1e6:.1f} MB")
results = []
for q in (3, 5, 8):
    res = npk.encode_image(img, quality=q, tile=1024, overlap=16)
    t0 = time.perf_counter(); rec = npk.decode_bytes(res.data); dec = time.perf_counter() - t0
    p, ms = psnr(img, rec), ms_ssim_db(ms_ssim(img, rec))
    jq, jdata = classic_at_size(img, "JPEG", res.bytes); jp = psnr(img, decode_classic(jdata))
    wq, wdata = classic_at_size(img, "WEBP", res.bytes); wp = psnr(img, decode_classic(wdata))
    # JPEG size needed to match npk PSNR
    lo, hi, match = 1, 100, None
    while lo <= hi:
        mid = (lo + hi) // 2; d = encode_classic(img, "JPEG", mid)
        if psnr(img, decode_classic(d)) >= p: match, hi = len(d), mid - 1
        else: lo = mid + 1
    row = {"quality": q, "bytes": res.bytes, "bpp": res.bpp, "psnr": p, "msssim_db": ms, "enc_s": res.seconds, "dec_s": dec,
               "ratio_vs_source": src.stat().st_size / res.bytes, "ratio_vs_raw": px * 3 / res.bytes,
               "jpeg_same_size_psnr": jp, "jpeg_same_size_q": jq, "webp_same_size_psnr": wp, "webp_same_size_q": wq,
               "jpeg_bytes_for_same_psnr": match, "tiles": res.tiles}
    results.append(row)
    print(f"q{q}: {res.bytes/1e6:.3f} MB  ({row['ratio_vs_source']:.1f}x vs source, {row['ratio_vs_raw']:.0f}x vs raw)  "
          f"{res.bpp:.3f} bpp  PSNR {p:.2f} dB  MS-SSIM {ms:.2f} dB  enc {res.seconds:.1f}s dec {dec:.1f}s | "
          f"same-size JPEG q{jq}: {jp:.2f} dB, WebP q{wq}: {wp:.2f} dB | JPEG needs {match/1e6 if match else float('nan'):.2f} MB for {p:.1f} dB")
# tiling overhead: whole picture in one piece vs 1024 tiles
for tile in (0, 2048, 1024, 512):
    try:
        r = npk.encode_image(img, quality=5, tile=tile, overlap=16)
        print(f"tile {tile:4d}: {r.bytes} bytes, {r.tiles} tiles, enc {r.seconds:.1f}s")
        results.append({"tile_test": tile, "bytes": r.bytes, "tiles": r.tiles, "enc_s": r.seconds})
    except (RuntimeError, MemoryError) as e:
        print(f"tile {tile}: failed: {type(e).__name__}: {str(e)[:100]}")
if json_out:
    Path(json_out).write_text(json.dumps(results, indent=2))
