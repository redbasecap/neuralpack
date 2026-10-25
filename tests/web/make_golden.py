"""Produce golden vectors for the JavaScript decoder: an .npk file plus, per tile, CRC32s of
the integer hyper-synthesis output, the Gaussian indexes and the decoded y symbols, and the
Python-decoded PNG. Usage: python tests/web/make_golden.py OUTDIR [IMAGE]"""
import json
import sys
import zlib
from pathlib import Path

import torch
from PIL import Image

import neuralpack as npk
from neuralpack import container
from neuralpack.codec import STRIDE
from neuralpack.metrics import psnr
from neuralpack.models import load_model
from neuralpack.tiling import tile_grid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import synthetic_photo

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
img = Image.open(sys.argv[2]).convert("RGB") if len(sys.argv) > 2 else synthetic_photo(333, 271, seed=7)
cases = {"single": {"tile": 0, "overlap": 0}, "tiled": {"tile": 128, "overlap": 16}}
golden = {}
for name, geo in cases.items():
    res = npk.encode_image(img, quality=3, model="mean-scale", device="cpu", web=True, **geo)
    (out / f"{name}.npk").write_bytes(res.data)
    header, tiles = container.unpack(res.data)
    lm = load_model("mean-scale", 3, "cpu")
    grid = tile_grid(header.width, header.height, header.tile, header.overlap)
    recs = []
    for t, (yb, zb) in zip(grid, tiles):
        h, w = t.ey1 - t.ey0, t.ex1 - t.ex0
        ph, pw = h + (-h) % STRIDE, w + (-w) % STRIDE
        z_hat = lm.entropy_bottleneck.decompress([zb], (ph // STRIDE, pw // STRIDE))
        q = lm.int_hyper.forward_int(z_hat)
        scales, means = lm.entropy_params(z_hat, integer=True)
        indexes = lm.gaussian_conditional.build_indexes(scales)
        y_hat = lm.gaussian_conditional.decompress([yb], indexes, means=means)
        symbols = torch.round(y_hat - means).int()
        recs.append({
            "q_crc": zlib.crc32(q.to(torch.int32).contiguous().numpy().tobytes()) & 0xFFFFFFFF,
            "indexes_crc": zlib.crc32(indexes.to(torch.uint8).contiguous().numpy().tobytes()) & 0xFFFFFFFF,
            "ysym_crc": zlib.crc32(symbols.contiguous().numpy().tobytes()) & 0xFFFFFFFF,
            "hy": int(ph // 16), "wy": int(pw // 16),
        })
    dec = npk.decode_bytes(res.data, device="cpu")
    dec.save(out / f"{name}.decoded.png")
    golden[name] = {"file": f"{name}.npk", "width": header.width, "height": header.height,
                    "fingerprint": header.table_fingerprint, "tiles": recs,
                    "psnr_python": float(psnr(img, dec))}
img.save(out / "original.png")
(out / "golden.json").write_text(json.dumps(golden, indent=1))
print(json.dumps({k: (v["width"], v["height"], len(v["tiles"]), round(v["psnr_python"], 2)) for k, v in golden.items()}))
