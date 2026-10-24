"""Quality metrics and classic-codec baselines for benchmarking."""

from __future__ import annotations

import io
import math

import numpy as np
import torch
from PIL import Image


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(img.convert("RGB"), dtype=np.uint8)).permute(2, 0, 1).float().div(255).unsqueeze(0)


def psnr(a: Image.Image, b: Image.Image) -> float:
    x, y = to_tensor(a), to_tensor(b)
    mse = torch.mean((x - y) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def ms_ssim(a: Image.Image, b: Image.Image) -> float | None:
    """Multi-scale SSIM in [0,1]; None if pytorch-msssim is not installed or image is tiny."""
    try:
        from pytorch_msssim import ms_ssim as _ms
    except ImportError:
        return None
    x, y = to_tensor(a), to_tensor(b)
    if min(x.shape[-2:]) < 161:
        return None
    return float(_ms(x, y, data_range=1.0).item())


def ms_ssim_db(v: float | None) -> float | None:
    return None if v is None else (-10 * math.log10(1 - v) if v < 1 else float("inf"))


def encode_classic(img: Image.Image, fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    kw = {"quality": quality}
    if fmt == "JPEG":
        kw.update(optimize=True, subsampling="4:2:0" if quality < 90 else "4:4:4")
    if fmt == "WEBP":
        kw.update(method=6)
    img.convert("RGB").save(buf, fmt, **kw)
    return buf.getvalue()


def classic_at_size(img: Image.Image, fmt: str, target_bytes: int) -> tuple[int, bytes]:
    """Largest classic-codec quality whose output is <= target_bytes (binary search)."""
    lo, hi = 1, 100
    best_q, best = 1, encode_classic(img, fmt, 1)
    while lo <= hi:
        mid = (lo + hi) // 2
        data = encode_classic(img, fmt, mid)
        if len(data) <= target_bytes:
            best_q, best = mid, data
            lo = mid + 1
        else:
            hi = mid - 1
    return best_q, best


def decode_classic(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as im:
        return im.convert("RGB")
