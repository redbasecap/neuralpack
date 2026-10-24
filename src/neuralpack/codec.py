"""Encoder / decoder: image <-> .npk bytes.

Pipeline for one tile::

    pixels --g_a--> y --h_a--> z --EB.compress--> z bytes
                                 z_hat = EB.decompress(z bytes)      (same on both sides)
                                 scales, means = h_s64(z_hat)        (CPU, float64)
                    y --GC.compress(indexes(scales), means)--> y bytes

    decode: z bytes -> z_hat -> scales, means -> y_hat -> g_s -> pixels

``EB`` is the factorised entropy bottleneck for the hyper-latent, ``GC`` the
conditional Gaussian entropy model for the main latent; both drive the range
coder (rANS) shipped with CompressAI.
"""

from __future__ import annotations

import base64
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import container
from .models import (
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    MODELS,
    MODELS_BY_ID,
    STRIDE,
    LoadedModel,
    load_model,
)
from .tiling import tile_grid

DEFAULT_TILE = 1024
DEFAULT_OVERLAP = 16
MAX_DIM = 2**32 - 1

ProgressFn = Callable[[int, int], None]


@dataclass
class EncodeResult:
    data: bytes
    width: int
    height: int
    model: str
    quality: int
    tiles: int
    seconds: float

    @property
    def bytes(self) -> int:
        return len(self.data)

    @property
    def bpp(self) -> float:
        return len(self.data) * 8 / (self.width * self.height)


# --------------------------------------------------------------------------- helpers

def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def _to_image(x: torch.Tensor) -> Image.Image:
    arr = x.squeeze(0).clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(arr), "RGB")


def _pad(x: torch.Tensor, stride: int = STRIDE) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % stride, (-w) % stride
    if ph == 0 and pw == 0:
        return x
    # Replicate padding gives the analysis transform smooth borders (fewer bits,
    # fewer ringing artefacts than zero padding). The decoder simply crops.
    return F.pad(x, (0, pw, 0, ph), mode="replicate")


def _extract_metadata(img: Image.Image) -> dict:
    meta: dict = {}
    exif = img.info.get("exif")
    if exif:
        meta["exif"] = base64.b64encode(exif).decode("ascii")
    icc = img.info.get("icc_profile")
    if icc:
        meta["icc"] = base64.b64encode(icc).decode("ascii")
    if img.format:
        meta["source_format"] = img.format
    dpi = img.info.get("dpi")
    if dpi:
        meta["dpi"] = [float(dpi[0]), float(dpi[1])]
    return meta


# --------------------------------------------------------------------------- tile coding

@torch.no_grad()
def encode_tile(model: LoadedModel, x: torch.Tensor) -> tuple[bytes, bytes]:
    """Encode one (1,3,H,W) float tensor in [0,1] to (y_bytes, z_bytes)."""
    net = model.net
    xp = _pad(x).to(model.device)
    y = net.g_a(xp)
    z = net.h_a(torch.abs(y) if model.spec.hyper_abs else y)
    z_cpu = z.cpu()
    z_strings = model.entropy_bottleneck.compress(z_cpu)
    z_hat = model.entropy_bottleneck.decompress(z_strings, z_cpu.shape[-2:])
    scales, means = model.entropy_params(z_hat)
    indexes = model.gaussian_conditional.build_indexes(scales)
    y_strings = model.gaussian_conditional.compress(y.cpu(), indexes, means=means)
    return y_strings[0], z_strings[0]


@torch.no_grad()
def decode_tile(model: LoadedModel, y_bytes: bytes, z_bytes: bytes, h: int, w: int) -> torch.Tensor:
    """Decode one tile of size (h, w) back to a (1,3,h,w) tensor on ``model.device``."""
    ph, pw = h + (-h) % STRIDE, w + (-w) % STRIDE
    z_shape = (ph // STRIDE, pw // STRIDE)
    z_hat = model.entropy_bottleneck.decompress([z_bytes], z_shape)
    scales, means = model.entropy_params(z_hat)
    indexes = model.gaussian_conditional.build_indexes(scales)
    y_hat = model.gaussian_conditional.decompress([y_bytes], indexes, means=means)
    x_hat = model.net.g_s(y_hat.to(model.device))
    return x_hat[..., :h, :w].clamp_(0, 1)


# --------------------------------------------------------------------------- public API

def encode_image(img: Image.Image, quality: int = DEFAULT_QUALITY, model: str = DEFAULT_MODEL,
                 device: str = "auto", tile: int = DEFAULT_TILE, overlap: int = DEFAULT_OVERLAP,
                 keep_metadata: bool = True, progress: ProgressFn | None = None) -> EncodeResult:
    """Compress a PIL image into .npk bytes.

    quality   1 (smallest, ~0.08 bpp) .. 8 (near-transparent, ~0.8 bpp)
    model     "mean-scale" (default) or "hyperprior"
    tile      max tile edge in pixels; 0 codes the picture in one piece
    overlap   halo around each tile (pixels) to hide seams
    """
    t0 = time.perf_counter()
    if img.width > MAX_DIM or img.height > MAX_DIM:
        raise ValueError("image too large for the .npk header")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        warnings.warn("alpha channel is not supported by the prior and will be dropped", stacklevel=2)
    grayscale = img.mode in ("L", "LA", "I", "I;16", "1")
    meta = _extract_metadata(img) if keep_metadata else {}
    rgb = img.convert("RGB")

    lm = load_model(model, quality, device)
    grid = tile_grid(rgb.width, rgb.height, tile, overlap)
    payloads: list[tuple[bytes, bytes]] = []
    for i, t in enumerate(grid):
        region = rgb.crop((t.ex0, t.ey0, t.ex1, t.ey1))
        payloads.append(encode_tile(lm, _to_tensor(region)))
        if progress:
            progress(i + 1, len(grid))

    header = container.NpkHeader(
        model_id=lm.spec.id, quality=quality, width=rgb.width, height=rgb.height,
        tile=tile if len(grid) > 1 else 0, overlap=overlap if len(grid) > 1 else 0,
        ntiles=len(grid), table_fingerprint=lm.table_fingerprint,
        grayscale=grayscale, metadata=meta,
    )
    data = container.pack(header, payloads)
    return EncodeResult(data, rgb.width, rgb.height, lm.spec.name, quality, len(grid),
                        time.perf_counter() - t0)


def decode_bytes(data: bytes, device: str = "auto", progress: ProgressFn | None = None,
                 strict: bool = True) -> Image.Image:
    """Decode .npk bytes back into a PIL image (RGB, or L for grayscale sources)."""
    header, payloads = container.unpack(data)
    spec = MODELS_BY_ID.get(header.model_id)
    if spec is None:
        raise container.NpkFormatError(
            f"file needs model id {header.model_id}, which this version does not know")
    lm = load_model(spec.name, header.quality, device)
    if lm.table_fingerprint != header.table_fingerprint:
        msg = (f"entropy tables of the local {spec.name} q{header.quality} model "
               f"({lm.table_fingerprint:08x}) differ from the encoder's "
               f"({header.table_fingerprint:08x}); the weights are a different version")
        if strict:
            raise container.NpkFormatError(msg)
        warnings.warn(msg + " - decoding anyway", stacklevel=2)

    grid = tile_grid(header.width, header.height, header.tile, header.overlap)
    if len(grid) != header.ntiles:
        raise container.NpkFormatError("tile count does not match geometry")
    canvas = torch.empty(1, 3, header.height, header.width)
    for i, (t, (yb, zb)) in enumerate(zip(grid, payloads)):
        xt = decode_tile(lm, yb, zb, t.ey1 - t.ey0, t.ex1 - t.ex0)
        left, top, right, bottom = t.crop
        canvas[..., t.y0:t.y1, t.x0:t.x1] = xt[..., top:bottom, left:right].cpu()
        if progress:
            progress(i + 1, len(grid))
    img = _to_image(canvas)
    if header.grayscale:
        img = img.convert("L")
    _attach_metadata(img, header.metadata)
    return img


def _attach_metadata(img: Image.Image, meta: dict) -> None:
    if "exif" in meta:
        img.info["exif"] = base64.b64decode(meta["exif"])
    if "icc" in meta:
        img.info["icc_profile"] = base64.b64decode(meta["icc"])
    if "dpi" in meta:
        img.info["dpi"] = tuple(meta["dpi"])


def decode_image(data: bytes, **kw) -> Image.Image:
    """Alias of :func:`decode_bytes`."""
    return decode_bytes(data, **kw)


def encode_file(src: str | os.PathLike, dst: str | os.PathLike | None = None, **kw) -> EncodeResult:
    """Compress an image file to ``dst`` (default: same name with .npk)."""
    src = Path(src)
    dst = Path(dst) if dst else src.with_suffix(".npk")
    with Image.open(src) as img:
        img.load()
        res = encode_image(img, **kw)
    dst.write_bytes(res.data)
    return res


def decode_file(src: str | os.PathLike, dst: str | os.PathLike | None = None,
                device: str = "auto", progress: ProgressFn | None = None, **save_kw) -> Path:
    """Decode an .npk file to an image file (default: PNG next to it)."""
    src = Path(src)
    dst = Path(dst) if dst else src.with_suffix(".png")
    img = decode_bytes(src.read_bytes(), device=device, progress=progress)
    save_kw = dict(save_kw)
    if dst.suffix.lower() in (".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".png"):
        if "exif" in img.info and "exif" not in save_kw:
            save_kw["exif"] = img.info["exif"]
        if "icc_profile" in img.info and dst.suffix.lower() != ".tif":
            save_kw.setdefault("icc_profile", img.info["icc_profile"])
        if "dpi" in img.info:
            save_kw.setdefault("dpi", img.info["dpi"])
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        save_kw.setdefault("quality", 97)
        save_kw.setdefault("subsampling", 0)
    img.save(dst, **save_kw)
    return dst


def model_names() -> list[str]:
    return sorted(MODELS)
