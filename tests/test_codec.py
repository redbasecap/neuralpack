"""End-to-end tests. They need the pretrained weights (downloaded on first use, ~30-70 MB)."""

import io

import numpy as np
import pytest
from PIL import Image

import neuralpack as npk
from neuralpack.container import NpkFormatError
from neuralpack.metrics import psnr
from neuralpack.models import MODELS, available_devices

pytestmark = pytest.mark.weights

Q = 3  # small weights, fast


def _arr(img):
    return np.asarray(img.convert("RGB")).astype(np.int16)


def test_roundtrip_quality_and_size(photo):
    res = npk.encode_image(photo, quality=Q)
    rec = npk.decode_bytes(res.data)
    assert rec.size == photo.size and rec.mode == "RGB"
    assert psnr(photo, rec) > 24
    raw = photo.width * photo.height * 3
    assert res.bytes < raw / 10, f"only {raw / res.bytes:.1f}x smaller than raw"


def test_encoding_is_deterministic(photo):
    a = npk.encode_image(photo, quality=Q).data
    b = npk.encode_image(photo, quality=Q).data
    assert a == b


@pytest.mark.parametrize("size", [(1, 1), (7, 5), (63, 65), (64, 64), (129, 1)])
def test_odd_sizes(size):
    img = Image.new("RGB", size, (120, 30, 200))
    rec = npk.decode_bytes(npk.encode_image(img, quality=Q).data)
    assert rec.size == size
    assert np.abs(_arr(rec) - _arr(img)).max() < 40


def test_tiled_matches_untiled(big_photo):
    whole = npk.decode_bytes(npk.encode_image(big_photo, quality=Q, tile=0).data)
    res = npk.encode_image(big_photo, quality=Q, tile=128, overlap=16)
    assert res.tiles > 4
    tiled = npk.decode_bytes(res.data)
    assert tiled.size == big_photo.size
    # tiles with halo must not be visibly worse than coding the picture in one piece
    assert psnr(big_photo, tiled) > psnr(big_photo, whole) - 0.5
    assert np.abs(_arr(tiled) - _arr(whole)).mean() < 6


def test_grayscale_roundtrip(photo):
    g = photo.convert("L")
    rec = npk.decode_bytes(npk.encode_image(g, quality=Q).data)
    assert rec.mode == "L" and rec.size == g.size
    assert psnr(g.convert("RGB"), rec.convert("RGB")) > 24


def test_metadata_survives(photo, tmp_path):
    exif = Image.Exif()
    exif[0x010F] = "neuralpack test camera"
    buf = io.BytesIO()
    photo.save(buf, "JPEG", quality=95, exif=exif.tobytes(), dpi=(300, 300))
    src = tmp_path / "in.jpg"
    src.write_bytes(buf.getvalue())
    res = npk.encode_file(src, tmp_path / "x.npk", quality=Q)
    assert npk.inspect(res.data)["metadata_keys"] == ["dpi", "exif", "source_format"]
    out = npk.decode_file(tmp_path / "x.npk", tmp_path / "out.jpg")
    with Image.open(out) as back:
        assert back.getexif()[0x010F] == "neuralpack test camera"
        assert tuple(round(d) for d in back.info["dpi"]) == (300, 300)
    no_meta = npk.encode_image(photo, quality=Q, keep_metadata=False)
    assert npk.inspect(no_meta.data)["metadata_keys"] == []


def test_alpha_is_dropped_with_warning(photo):
    rgba = photo.convert("RGBA")
    with pytest.warns(UserWarning, match="alpha"):
        res = npk.encode_image(rgba, quality=Q)
    assert npk.decode_bytes(res.data).mode == "RGB"


def test_corrupt_payload_is_rejected(photo):
    data = bytearray(npk.encode_image(photo, quality=Q).data)
    data[len(data) // 2] ^= 0x55
    with pytest.raises(NpkFormatError, match="CRC"):
        npk.decode_bytes(bytes(data))


def test_fingerprint_mismatch_is_rejected(photo):
    from neuralpack import container

    header, tiles = container.unpack(npk.encode_image(photo, quality=Q).data)
    header.table_fingerprint ^= 1
    with pytest.raises(NpkFormatError, match="entropy tables"):
        npk.decode_bytes(container.pack(header, tiles))
    with pytest.warns(UserWarning):
        npk.decode_bytes(container.pack(header, tiles), strict=False)


@pytest.mark.parametrize("model", sorted(MODELS))
def test_every_model_family(photo, model):
    res = npk.encode_image(photo, quality=Q, model=model)
    rec = npk.decode_bytes(res.data)
    assert npk.inspect(res.data)["model"] == model
    assert psnr(photo, rec) > 24


@pytest.mark.skipif(len(available_devices()) < 2, reason="needs an accelerator besides the CPU")
def test_cross_device_bitstream_compatibility(big_photo):
    """The parse of the bitstream must not depend on the device; pixels may differ by rounding."""
    acc = available_devices()[0]
    enc_acc = npk.encode_image(big_photo, quality=Q, device=acc, tile=128, overlap=16).data
    enc_cpu = npk.encode_image(big_photo, quality=Q, device="cpu", tile=128, overlap=16).data
    on_cpu = npk.decode_bytes(enc_acc, device="cpu")
    on_acc = npk.decode_bytes(enc_acc, device=acc)
    assert np.abs(_arr(on_cpu) - _arr(on_acc)).max() <= 3
    assert np.abs(_arr(on_cpu) - _arr(npk.decode_bytes(enc_cpu, device="cpu"))).max() <= 3
    assert abs(len(enc_acc) - len(enc_cpu)) < len(enc_cpu) * 0.01
