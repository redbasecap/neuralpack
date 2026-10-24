import pytest

from neuralpack import container
from neuralpack.container import NpkFormatError, NpkHeader, pack, unpack


def _hdr(**kw):
    base = {"model_id": 2, "quality": 5, "width": 100, "height": 50, "tile": 0, "overlap": 0, "ntiles": 1,
                "table_fingerprint": 0xDEADBEEF}
    base.update(kw)
    return NpkHeader(**base)


def test_roundtrip_single_tile():
    data = pack(_hdr(), [(b"yyy", b"zz")])
    h, tiles = unpack(data)
    assert (h.width, h.height, h.quality, h.model_id) == (100, 50, 5, 2)
    assert tiles == [(b"yyy", b"zz")]
    assert h.table_fingerprint == 0xDEADBEEF
    assert len(data) == container.HEADER_SIZE + 8 + 5 + 4


def test_roundtrip_metadata_and_flags():
    meta = {"exif": "AAEC", "source_format": "JPEG", "dpi": [72.0, 72.0]}
    data = pack(_hdr(grayscale=True, metadata=meta, ntiles=2), [(b"a", b""), (b"", b"b")])
    h, tiles = unpack(data)
    assert h.grayscale is True
    assert h.metadata == meta
    assert tiles == [(b"a", b""), (b"", b"b")]


def test_tile_count_mismatch():
    with pytest.raises(ValueError):
        pack(_hdr(ntiles=2), [(b"a", b"b")])


@pytest.mark.parametrize("mutation", ["magic", "crc", "truncate", "version"])
def test_corruption_is_detected(mutation):
    data = bytearray(pack(_hdr(), [(b"payload" * 10, b"side")]))
    if mutation == "magic":
        data[0:3] = b"XXX"
    elif mutation == "version":
        data[3] = 9
    elif mutation == "crc":
        data[40] ^= 0xFF
    elif mutation == "truncate":
        data = data[:-3]
    with pytest.raises(NpkFormatError):
        unpack(bytes(data))


def test_inspect_reports_sizes():
    data = pack(_hdr(), [(b"y" * 100, b"z" * 10)])
    info = container.inspect(data)
    assert info["latent_bytes"] == 100 and info["hyper_bytes"] == 10
    assert info["model"] == "mean-scale"
    assert info["container_overhead_bytes"] == len(data) - 110
