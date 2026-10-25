""".npk container format (version 1).

All integers little-endian. Layout::

    +--------+---------------------------------------------------------------+
    | 4      | magic  b"NPK\\x01"                                             |
    | 1      | model id (see models.MODELS_BY_ID)                            |
    | 1      | quality (1..8)                                                 |
    | 1      | flags  bit0 grayscale, bit1 metadata, bit2 integer hyper-synth  |
    | 1      | reserved (0)                                                   |
    | 4 + 4  | width, height in pixels                                        |
    | 2 + 2  | tile size, tile overlap in pixels (tile 0 = single tile)       |
    | 4      | number of tiles                                                |
    | 4      | entropy-table fingerprint of the model (CRC32)                 |
    | 4      | metadata chunk length (zlib-compressed JSON) – may be 0        |
    | n      | metadata chunk                                                 |
    | tiles  | per tile: u32 len(y), u32 len(z), y bytes, z bytes             |
    | 4      | CRC32 of everything above                                      |
    +--------+---------------------------------------------------------------+

The header is 32 bytes, the per-tile overhead 8 bytes. A whole 24 MP photo at
quality 5 is ~800 kB of payload, so the container costs < 0.02 %.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field

MAGIC = b"NPK\x01"
_HDR = struct.Struct("<4sBBBBIIHHIII")
HEADER_SIZE = _HDR.size  # 32
_TILE = struct.Struct("<II")

FLAG_GRAYSCALE = 1 << 0
FLAG_METADATA = 1 << 1
FLAG_INT_HYPER = 1 << 2  # entropy parameters from the integer hyper-synthesis (web-decodable)


class NpkFormatError(ValueError):
    """Raised for a file that is not a valid .npk stream."""


@dataclass
class NpkHeader:
    model_id: int
    quality: int
    width: int
    height: int
    tile: int
    overlap: int
    ntiles: int
    table_fingerprint: int
    grayscale: bool = False
    metadata: dict = field(default_factory=dict)
    int_hyper: bool = False

    @property
    def flags(self) -> int:
        return ((FLAG_GRAYSCALE if self.grayscale else 0) | (FLAG_METADATA if self.metadata else 0)
                | (FLAG_INT_HYPER if self.int_hyper else 0))


def pack(header: NpkHeader, tiles: list[tuple[bytes, bytes]]) -> bytes:
    """Serialise header + tile payloads into one .npk byte string."""
    if len(tiles) != header.ntiles:
        raise ValueError(f"header says {header.ntiles} tiles, got {len(tiles)}")
    meta = b""
    if header.metadata:
        meta = zlib.compress(json.dumps(header.metadata, separators=(",", ":")).encode("utf-8"), 9)
    parts = [
        _HDR.pack(MAGIC, header.model_id, header.quality, header.flags, 0, header.width,
                  header.height, header.tile, header.overlap, header.ntiles,
                  header.table_fingerprint, len(meta)),
        meta,
    ]
    for y, z in tiles:
        parts.append(_TILE.pack(len(y), len(z)))
        parts.append(y)
        parts.append(z)
    body = b"".join(parts)
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def unpack(data: bytes) -> tuple[NpkHeader, list[tuple[bytes, bytes]]]:
    """Parse and verify a .npk byte string."""
    if len(data) < HEADER_SIZE + 4:
        raise NpkFormatError("file too short to be an .npk stream")
    if data[:4] != MAGIC:
        if data[:3] == b"NPK":
            raise NpkFormatError(f"unsupported .npk version {data[3]}")
        raise NpkFormatError("not an .npk file (bad magic)")
    body, crc = data[:-4], struct.unpack("<I", data[-4:])[0]
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise NpkFormatError("CRC mismatch: the file is corrupted or truncated")
    (_, model_id, quality, flags, _res, width, height, tile, overlap, ntiles, fp,
     meta_len) = _HDR.unpack_from(body, 0)
    pos = HEADER_SIZE
    metadata: dict = {}
    if flags & FLAG_METADATA:
        try:
            metadata = json.loads(zlib.decompress(body[pos:pos + meta_len]).decode("utf-8"))
        except (zlib.error, ValueError) as e:  # pragma: no cover - defensive
            raise NpkFormatError(f"corrupt metadata chunk: {e}") from e
    pos += meta_len
    tiles: list[tuple[bytes, bytes]] = []
    for _ in range(ntiles):
        if pos + _TILE.size > len(body):
            raise NpkFormatError("truncated tile table")
        ly, lz = _TILE.unpack_from(body, pos)
        pos += _TILE.size
        if pos + ly + lz > len(body):
            raise NpkFormatError("truncated tile payload")
        tiles.append((body[pos:pos + ly], body[pos + ly:pos + ly + lz]))
        pos += ly + lz
    if pos != len(body):
        raise NpkFormatError(f"{len(body) - pos} unexpected trailing bytes")
    header = NpkHeader(model_id, quality, width, height, tile, overlap, ntiles, fp,
                       grayscale=bool(flags & FLAG_GRAYSCALE), metadata=metadata,
                       int_hyper=bool(flags & FLAG_INT_HYPER))
    return header, tiles


def inspect(data: bytes) -> dict:
    """Human-readable summary of an .npk stream (used by ``npk info``)."""
    header, tiles = unpack(data)
    from .models import MODELS_BY_ID  # local import to keep container.py torch-free

    spec = MODELS_BY_ID.get(header.model_id)
    y_bytes = sum(len(y) for y, _ in tiles)
    z_bytes = sum(len(z) for _, z in tiles)
    px = header.width * header.height
    return {
        "model": spec.name if spec else f"unknown(id={header.model_id})",
        "quality": header.quality,
        "width": header.width,
        "height": header.height,
        "megapixels": round(px / 1e6, 2),
        "grayscale": header.grayscale,
        "web_decodable": header.int_hyper,
        "tile": header.tile,
        "overlap": header.overlap,
        "tiles": header.ntiles,
        "file_bytes": len(data),
        "latent_bytes": y_bytes,
        "hyper_bytes": z_bytes,
        "container_overhead_bytes": len(data) - y_bytes - z_bytes,
        "bits_per_pixel": round(len(data) * 8 / px, 4) if px else None,
        "table_fingerprint": f"{header.table_fingerprint:08x}",
        "metadata_keys": sorted(header.metadata),
    }
