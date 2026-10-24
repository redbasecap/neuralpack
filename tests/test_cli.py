import json
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from neuralpack.cli import main
from neuralpack.metrics import psnr

pytestmark = pytest.mark.weights


def test_encode_decode_info_cli(photo, tmp_path, capsys):
    src = tmp_path / "pic.png"
    photo.save(src)
    assert main(["encode", str(src), "-q", "3", "--quiet"]) == 0
    npk_file = tmp_path / "pic.npk"
    assert npk_file.exists() and npk_file.stat().st_size < src.stat().st_size / 5
    capsys.readouterr()  # drop anything printed so far (e.g. first-run weight download notices)
    assert main(["info", str(npk_file), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["width"] == photo.width and info["quality"] == 3
    assert main(["decode", str(npk_file), "-o", str(tmp_path / "back.png"), "--quiet"]) == 0
    with Image.open(tmp_path / "back.png") as back:
        assert psnr(photo, back) > 24


def test_many_inputs_into_directory(photo, tmp_path):
    for i in range(2):
        photo.save(tmp_path / f"p{i}.png")
    out = tmp_path / "out"
    assert main(["encode", str(tmp_path / "p0.png"), str(tmp_path / "p1.png"), "-o", str(out), "-q", "3", "--quiet"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["p0.npk", "p1.npk"]
    dec = tmp_path / "dec"
    assert main(["decode", str(out / "p0.npk"), str(out / "p1.npk"), "-o", str(dec), "-f", "jpg", "--quiet"]) == 0
    assert sorted(p.name for p in dec.iterdir()) == ["p0.jpg", "p1.jpg"]


def test_bad_file_gives_error_code(tmp_path, capsys):
    bad = tmp_path / "bad.npk"
    bad.write_bytes(b"not an npk file at all" * 3)
    assert main(["decode", str(bad)]) == 1
    assert "npk:" in capsys.readouterr().err


def test_models_and_bench(photo, tmp_path, capsys):
    assert main(["models"]) == 0
    assert "mean-scale (default)" in capsys.readouterr().out
    src = tmp_path / "pic.png"
    photo.save(src)
    js = tmp_path / "b.json"
    assert main(["bench", str(src), "-q", "3", "--json", str(js)]) == 0
    rows = json.loads(js.read_text())
    codecs = {r["codec"] for r in rows}
    assert {"npk/mean-scale", "jpeg@same-size", "webp@same-size"} <= codecs
    ours = next(r for r in rows if r["codec"].startswith("npk"))
    jpeg = next(r for r in rows if r["codec"].startswith("jpeg"))
    assert jpeg["bytes"] <= ours["bytes"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_roundtrip(tmp_path):
    src = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=96x64:rate=4",
                    "-t", "1", "-pix_fmt", "yuv420p", str(src)], check=True)
    assert main(["encode", str(src), "-q", "3", "--quiet"]) == 0
    npkv = tmp_path / "clip.npkv"
    assert npkv.exists()
    out_dir = tmp_path / "frames"
    assert main(["decode", str(npkv), "-o", str(out_dir), "--quiet"]) == 0
    frames = sorted(out_dir.glob("*.png"))
    assert len(frames) == 4
    with Image.open(frames[0]) as f:
        assert f.size == (96, 64)
    assert main(["decode", str(npkv), "-o", str(tmp_path / "back.mp4"), "--quiet"]) == 0
    assert (tmp_path / "back.mp4").stat().st_size > 0
    assert main(["decode", str(npkv), "-o", str(tmp_path / "back.mkv"), "--quiet"]) == 0
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                            str(tmp_path / "back.mkv")], check=True, capture_output=True, text=True)
    assert probe.stdout.strip() == "4"
    arr = np.asarray(Image.open(frames[0]))
    assert arr.std() > 20  # not a blank frame
