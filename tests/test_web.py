"""Browser-decoder pipeline: integer hyper-synthesis, web export and the JavaScript decoder."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import neuralpack as npk
from neuralpack import container
from neuralpack.container import NpkFormatError
from neuralpack.inthyper import IntHyperSynthesis
from neuralpack.metrics import psnr
from neuralpack.models import load_model

pytestmark = pytest.mark.weights
Q = 3
ROOT = Path(__file__).resolve().parents[1]


def test_integer_hyper_is_deterministic_and_close_to_float(big_photo):
    lm = load_model("mean-scale", Q, "cpu")
    assert lm.int_hyper is not None
    import torch

    from neuralpack.codec import _pad, _to_tensor
    with torch.no_grad():
        y = lm.net.g_a(_pad(_to_tensor(big_photo)))
        z = lm.net.h_a(y)
        z_hat = lm.entropy_bottleneck.decompress(lm.entropy_bottleneck.compress(z), z.shape[-2:])
    s_int, m_int = lm.entropy_params(z_hat, integer=True)
    s_int2, _ = lm.entropy_params(z_hat, integer=True)
    assert torch.equal(s_int, s_int2)
    s_f, m_f = lm.entropy_params(z_hat, integer=False)
    rel = ((s_int - s_f).abs() / s_f.clamp(min=0.11)).median().item()
    assert rel < 0.01, f"integer scales deviate {rel:.3%} (median) from float64"
    assert (m_int - m_f).abs().median().item() < 0.01
    # re-quantising from the float weights with the stored shifts gives the same fingerprint
    again = IntHyperSynthesis.quantize(lm.net.h_s, True, shifts=lm.int_hyper.shifts, in_shift=lm.int_hyper.in_shift)
    assert again.fingerprint == lm.int_hyper.fingerprint


def test_web_and_float_files_are_distinct_and_both_decode(photo):
    web = npk.encode_image(photo, quality=Q, web=True)
    flt = npk.encode_image(photo, quality=Q, web=False)
    assert npk.inspect(web.data)["web_decodable"] is True
    assert npk.inspect(flt.data)["web_decodable"] is False
    assert abs(web.bytes - flt.bytes) < 0.05 * flt.bytes
    for res in (web, flt):
        assert psnr(photo, npk.decode_bytes(res.data)) > 24
    # a float file relabelled as integer must be rejected by the fingerprint, not decoded to noise
    header, tiles = container.unpack(flt.data)
    header.int_hyper = True
    with pytest.raises(NpkFormatError, match="entropy tables"):
        npk.decode_bytes(container.pack(header, tiles))


def test_export_web_layout(tmp_path):
    from neuralpack.webexport import export_model

    d = export_model(tmp_path, "mean-scale", Q)
    for f in ("manifest.json", "g_s.onnx", "hyper.json", "hyper_weights.bin", "tables.json"):
        assert (d / f).exists()
    manifest = json.loads((d / "manifest.json").read_text())
    lm = load_model("mean-scale", Q, "cpu")
    assert manifest["fingerprint"] == lm.int_fingerprint
    hyper = json.loads((d / "hyper.json").read_text())
    n_weights = sum(layer["weight_count"] for layer in hyper["layers"])
    assert (d / "hyper_weights.bin").stat().st_size == 2 * n_weights
    index = json.loads((tmp_path / "index.json").read_text())
    assert index[0]["dir"] == d.name


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_decoder_is_bit_exact(tmp_path):
    from neuralpack.webexport import export_model

    export_model(tmp_path / "models", "mean-scale", Q)
    subprocess.run([sys.executable, str(ROOT / "tests/web/make_golden.py"), str(tmp_path / "golden")],
                   check=True, capture_output=True)
    r = subprocess.run(["node", str(ROOT / "tests/web/test_npk.mjs"), str(tmp_path / "golden"),
                        str(tmp_path / "models")], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "bit for bit" in r.stdout


def test_onnx_synthesis_matches_torch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    import torch

    from neuralpack.webexport import export_g_s

    lm = load_model("mean-scale", Q, "cpu")
    export_g_s(lm, tmp_path / "g_s.onnx")
    sess = ort.InferenceSession(str(tmp_path / "g_s.onnx"), providers=["CPUExecutionProvider"])
    y = torch.randn(1, lm.net.g_s[0].in_channels, 6, 9)
    with torch.no_grad():
        ref = lm.net.g_s(y).numpy()
    out = sess.run(None, {"y_hat": y.numpy()})[0]
    assert out.shape == ref.shape == (1, 3, 96, 144)
    assert np.abs(out - ref).max() < 1e-3
