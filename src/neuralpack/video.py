"""Experimental video support: every frame is an independent .npk picture.

Container ``.npkv``::

    5   magic b"NPKV\\x01"
    4+4 width, height
    4+4 frame-rate numerator / denominator
    4   frame count
    per frame: u32 length + complete .npk stream
    4   CRC32 of everything above

This is *intra-only* coding (like Motion-JPEG, but with a neural codec). It is
excellent for frame sequences, time-lapses and archival stills, and it makes
the whole toolchain work end-to-end on video files. It does **not** exploit
motion between frames, so a modern inter-frame codec (HEVC/AV1) will beat it
on ordinary footage at equal bitrate. Audio is not carried. Needs ``ffmpeg``.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from .codec import decode_bytes, encode_image
from .container import NpkFormatError

MAGIC = b"NPKV\x01"
_HDR = struct.Struct("<5sIIIII")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".gif"}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _require_ffmpeg() -> None:
    if not ffmpeg_available():
        raise RuntimeError("video support needs ffmpeg and ffprobe on PATH")


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    nframes: int | None


def probe(path: str | Path) -> VideoInfo:
    _require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    nf = s.get("nb_frames")
    return VideoInfo(int(s["width"]), int(s["height"]), Fraction(s["r_frame_rate"]),
                     int(nf) if nf and nf.isdigit() else None)


def iter_frames(path: str | Path, info: VideoInfo) -> Iterator[Image.Image]:
    _require_ffmpeg()
    nbytes = info.width * info.height * 3
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            arr = np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
            yield Image.fromarray(arr.copy(), "RGB")
    finally:
        proc.stdout.close()
        proc.wait()


def encode_video(src: str | Path, dst: str | Path | None = None,
                 progress: Callable[[int, int | None], None] | None = None, **encode_kw) -> Path:
    """Encode a video file to .npkv. ``encode_kw`` go to :func:`encode_image`."""
    src = Path(src)
    dst = Path(dst) if dst else src.with_suffix(".npkv")
    info = probe(src)
    frames: list[bytes] = []
    for i, frame in enumerate(iter_frames(src, info)):
        frames.append(encode_image(frame, **encode_kw).data)
        if progress:
            progress(i + 1, info.nframes)
    body = _HDR.pack(MAGIC, info.width, info.height, info.fps.numerator, info.fps.denominator,
                     len(frames))
    body += b"".join(struct.pack("<I", len(f)) + f for f in frames)
    dst.write_bytes(body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF))
    return dst


def unpack_video(data: bytes) -> tuple[VideoInfo, list[bytes]]:
    if len(data) < _HDR.size + 4 or data[:5] != MAGIC:
        raise NpkFormatError("not an .npkv stream")
    body, crc = data[:-4], struct.unpack("<I", data[-4:])[0]
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise NpkFormatError("CRC mismatch: the file is corrupted or truncated")
    _, w, h, num, den, n = _HDR.unpack_from(body, 0)
    pos, frames = _HDR.size, []
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", body, pos)
        pos += 4
        frames.append(body[pos:pos + ln])
        pos += ln
    if pos != len(body):
        raise NpkFormatError("truncated .npkv stream")
    return VideoInfo(w, h, Fraction(num, den), n), frames


def decode_video(src: str | Path, dst: str | Path | None = None, device: str = "auto",
                 crf: int = 12, progress: Callable[[int, int | None], None] | None = None) -> Path:
    """Decode .npkv to a video (H.264 .mp4 / lossless FFV1 .mkv) or a directory of PNGs."""
    src = Path(src)
    dst = Path(dst) if dst else src.with_suffix(".mp4")
    info, frames = unpack_video(src.read_bytes())
    if dst.suffix == "" or dst.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            decode_bytes(f, device=device).save(dst / f"frame_{i:06d}.png")
            if progress:
                progress(i + 1, len(frames))
        return dst
    _require_ffmpeg()
    if dst.suffix.lower() == ".mkv":
        vcodec = ["-c:v", "ffv1", "-pix_fmt", "rgb24"]
    else:
        vcodec = ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p"]
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{info.width}x{info.height}", "-r", str(info.fps), "-i", "pipe:0", *vcodec, str(dst)],
        stdin=subprocess.PIPE)
    try:
        for i, f in enumerate(frames):
            img = decode_bytes(f, device=device).convert("RGB")
            proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
            if progress:
                progress(i + 1, len(frames))
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing the decoded video")
    return dst
