"""Export everything the browser decoder needs for one (model, quality).

Output directory layout (``npk export-web``)::

    <out>/<model>-q<q>/manifest.json      ids, shapes, fingerprint, file names
    <out>/<model>-q<q>/g_s.onnx           synthesis transform (float32, dynamic H/W)
    <out>/<model>-q<q>/hyper.json         integer hyper-synthesis layer descriptions
    <out>/<model>-q<q>/hyper_weights.bin  int16 little-endian weights, layer after layer
    <out>/<model>-q<q>/tables.json        range-coder tables (CDFs, offsets, medians, scale table)
    <out>/index.json                      list of exported model directories
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch

from .models import MODELS, LoadedModel, load_model

ONNX_OPSET = 17


def export_g_s(lm: LoadedModel, path: Path) -> None:
    net = lm.net
    m = net.g_s[0].in_channels
    dummy = torch.zeros(1, m, 4, 4, device=lm.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(net.g_s, dummy, str(path), input_names=["y_hat"], output_names=["x_hat"],
                          dynamic_axes={"y_hat": {2: "h", 3: "w"}, "x_hat": {2: "h16", 3: "w16"}},
                          opset_version=ONNX_OPSET, dynamo=False)


def export_tables(lm: LoadedModel) -> dict:
    eb, gc = lm.entropy_bottleneck, lm.gaussian_conditional
    medians = eb._get_medians().detach().reshape(-1).float().tolist()
    return {
        "precision": 16,
        "eb": {"cdf": eb._quantized_cdf.int().tolist(), "cdf_length": eb._cdf_length.reshape(-1).int().tolist(),
               "offset": eb._offset.reshape(-1).int().tolist(), "medians": medians},
        "gc": {"cdf": gc._quantized_cdf.int().tolist(), "cdf_length": gc._cdf_length.reshape(-1).int().tolist(),
               "offset": gc._offset.reshape(-1).int().tolist(),
               "scale_table": gc.scale_table.float().tolist()},
    }


def export_model(out: Path, model: str, quality: int, device: str = "cpu") -> Path:
    lm = load_model(model, quality, device)
    if lm.int_hyper is None:
        raise ValueError(f"{model} q{quality} has no integer hyper-synthesis calibration; cannot export")
    d = out / f"{model}-q{quality}"
    d.mkdir(parents=True, exist_ok=True)
    export_g_s(lm, d / "g_s.onnx")
    hyper = lm.int_hyper.export()
    blobs, offset = [], 0
    for layer in hyper["layers"]:
        import base64

        raw = base64.b64decode(layer.pop("weight_b64"))
        layer["weight_offset"] = offset
        layer["weight_count"] = len(raw) // 2
        offset += len(raw)
        blobs.append(raw)
        # biases are exact integers below 2^52 (JSON numbers); the shaders add them as 64-bit
        if max(abs(b) for b in layer["bias"]) >= 2**52 or not 0 <= layer["shift"] <= 31:
            raise RuntimeError("integer layer parameters out of the range the shaders support")
    (d / "hyper_weights.bin").write_bytes(b"".join(blobs))
    (d / "hyper.json").write_text(json.dumps(hyper, separators=(",", ":")))
    (d / "tables.json").write_text(json.dumps(export_tables(lm), separators=(",", ":")))
    g_s_in = lm.net.g_s[0].in_channels
    manifest = {
        "format": 1, "model": model, "quality": quality, "model_id": lm.spec.id,
        "fingerprint": lm.int_fingerprint, "n_latent": g_s_in,
        "n_hyper": lm.net.h_a[-1].out_channels if hasattr(lm.net.h_a[-1], "out_channels") else None,
        "stride": 64, "latent_stride": 16,
        "files": {"g_s": "g_s.onnx", "hyper": "hyper.json", "weights": "hyper_weights.bin",
                  "tables": "tables.json"},
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    _update_index(out)
    return d


def _update_index(out: Path) -> None:
    entries = []
    for mf in sorted(out.glob("*/manifest.json")):
        m = json.loads(mf.read_text())
        entries.append({"dir": mf.parent.name, "model": m["model"], "quality": m["quality"],
                        "model_id": m["model_id"], "fingerprint": m["fingerprint"]})
    (out / "index.json").write_text(json.dumps(entries, indent=1))


def exportable() -> list[tuple[str, int]]:
    from .inthyper import stored_shifts

    return [(name, q) for name, spec in MODELS.items() for q in spec.qualities
            if stored_shifts(name, q) is not None]


def default_models_dir() -> Path:
    return Path.home() / ".cache" / "neuralpack" / "web-models"
