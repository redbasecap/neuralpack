"""Integer-only hyper-synthesis: the key to a bit-exact decoder on *any* device.

The entropy coder needs, for every latent, the scale/mean predicted by the
hyper-synthesis ``h_s``. If the decoder's floating-point result differs from the
encoder's by a single ulp at a threshold of the scale table, the arithmetic
decoder desynchronises and the rest of the tile becomes noise. Measured: plain
float32 on CPU/GPU/ONNX Runtime flips 3-9 indexes per 2 M latents - that is
several corrupted tiles per picture.

So ``h_s`` is evaluated in **integer arithmetic only** (Ballé et al., *Integer
networks for data compression with latent-variable models*, ICLR 2019), here as
a post-training quantisation of the published float weights:

* weights int16, quantised per output channel; the per-channel scales are folded
  into the *next* layer's weights (LeakyReLU/ReLU are positively homogeneous),
* activations int16 (|a| <= 32767), one power-of-two requantisation shift per layer,
* every product fits int32 (2^15 * 2^15 = 2^30); sums are accumulated *exactly*:
  float64 in PyTorch and JavaScript (|sum| < 2^43 << 2^53), a carry-tracked
  (hi, lo) i32 pair in WebGPU shaders,
* the final integer outputs are turned into float32 scales/means by one IEEE
  multiply, which is correctly rounded everywhere.

Every operation is an integer add, multiply, compare or shift, so PyTorch, plain
JavaScript, WebGPU and WebAssembly all produce identical results. Measured cost on
Kodak versus the float64 reference: well under one percent of bitrate. The
quantised weights are derived deterministically from the float checkpoint; only
the per-layer shifts are stored (``inthyper_shifts.json``) because they come from
a calibration run.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from importlib import resources

import torch
import torch.nn.functional as F
from torch import nn

ACT_MAX = 32767  # int16 activations
W_MAX = 32767  # int16 weights
LEAKY_NUM, LEAKY_SHIFT = 41, 12  # 41 / 4096 = 0.010009765625 ~ LeakyReLU(0.01)


@dataclass
class IntLayer:
    kind: str  # "deconv" | "conv"
    weight: torch.Tensor  # float64 tensor holding int16 values, PyTorch layout
    bias: torch.Tensor  # float64 tensor holding int32 values
    shift: int  # requantisation shift after bias
    act: str  # "leaky" | "relu" | "none"
    stride: int
    padding: int
    output_padding: int
    out_scale: torch.Tensor  # float64 per-out-channel real step (only used for the last layer)


INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def round_shift(x: torch.Tensor, s: int) -> torch.Tensor:
    """(x + 2^(s-1)) >> s with floor semantics and int32 saturation, on integer-valued
    float64 tensors (exact: every value stays far below 2^53)."""
    if s > 0:
        x = torch.floor((x + float(1 << (s - 1))) / float(1 << s))
    return x.clamp_(INT32_MIN, INT32_MAX)


def leaky_int(a: torch.Tensor) -> torch.Tensor:
    neg = torch.floor(a * LEAKY_NUM / float(1 << LEAKY_SHIFT))
    return torch.where(a >= 0, a, neg)


@dataclass
class IntHyperSynthesis:
    layers: list[IntLayer]
    in_shift: int  # z_hat is scaled by 2^in_shift before rounding
    has_means: bool
    n_scales: int  # number of scale channels (M)
    fingerprint: int = 0
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """z_hat (1,N,h,w) float -> (scales, means) float32, bit-exact across devices."""
        a = self.forward_int(z_hat)
        # one IEEE multiply per value: float32(q) * float32(step) is correctly rounded everywhere
        q = a.float()
        step = self.layers[-1].out_scale.float().view(1, -1, 1, 1)
        out = q * step
        if self.has_means:
            return out[:, :self.n_scales].contiguous(), out[:, self.n_scales:].contiguous()
        return out, None

    @torch.no_grad()
    def forward_int(self, z_hat: torch.Tensor) -> torch.Tensor:
        """The integer outputs of the last layer (1, 2M or M, 4h, 4w) as an integer-valued float64 tensor."""
        a = torch.round(z_hat.detach().cpu().double() * float(1 << self.in_shift))
        a = a.clamp_(-ACT_MAX, ACT_MAX)
        last = len(self.layers) - 1
        for i, lyr in enumerate(self.layers):
            if lyr.kind == "deconv":
                acc = F.conv_transpose2d(a, lyr.weight, None, stride=lyr.stride,
                                         padding=lyr.padding, output_padding=lyr.output_padding)
            else:
                acc = F.conv2d(a, lyr.weight, None, stride=lyr.stride, padding=lyr.padding)
            acc = acc + lyr.bias.view(1, -1, 1, 1)
            a = round_shift(acc, lyr.shift)
            if i != last:
                a = a.clamp_(-ACT_MAX, ACT_MAX)
                if lyr.act == "leaky":
                    a = leaky_int(a)
                elif lyr.act == "relu":
                    a = a.clamp_(min=0)
        return a

    __call__ = forward

    # ------------------------------------------------------------- quantisation
    @classmethod
    def quantize(cls, h_s: nn.Sequential, has_means: bool, shifts: list[int] | None = None,
                 in_shift: int | None = None, calib: list[torch.Tensor] | None = None) -> IntHyperSynthesis:
        """Post-training quantisation of a CompressAI hyper-synthesis.

        Either pass the stored ``shifts``/``in_shift`` (deployment) or ``calib`` z_hat
        tensors to derive them (one-off calibration, result must then be stored).
        """
        mods = [m for m in h_s if not isinstance(m, (nn.LeakyReLU, nn.ReLU))]
        acts = []
        for i, m in enumerate(h_s):
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nxt = h_s[i + 1] if i + 1 < len(h_s) else None
                acts.append("leaky" if isinstance(nxt, nn.LeakyReLU) else
                            "relu" if isinstance(nxt, nn.ReLU) else "none")
        calibrating = shifts is None
        if calibrating:
            if not calib:
                raise ValueError("need calibration tensors or stored shifts")
            zmax = max(float(z.abs().max()) for z in calib)
            in_shift = int(max(0, min(12, math.floor(math.log2(ACT_MAX / max(zmax, 1e-6))))))
            shifts = []
        assert in_shift is not None
        layers: list[IntLayer] = []
        step_in = torch.full((mods[0].in_channels,), 2.0 ** -in_shift, dtype=torch.float64)
        cal_act = ([torch.round(z.double() * (1 << in_shift)).clamp(-ACT_MAX, ACT_MAX) for z in calib]
                   if calibrating else [])
        n_layers = len(mods)
        for li, m in enumerate(mods):
            w = m.weight.detach().double()  # deconv: (in, out, k, k); conv: (out, in, k, k)
            is_deconv = isinstance(m, nn.ConvTranspose2d)
            # fold the real step of each input channel into the weights
            w = w * (step_in.view(-1, 1, 1, 1) if is_deconv else step_in.view(1, -1, 1, 1))
            out_dim = 1 if is_deconv else 0
            wmax = w.abs().amax(dim=[d for d in range(4) if d != out_dim])
            wmax = torch.where(wmax > 0, wmax, torch.ones_like(wmax))
            s_out = wmax / float(W_MAX)
            wq = torch.round(w / (s_out.view(1, -1, 1, 1) if is_deconv else s_out.view(-1, 1, 1, 1)))
            b = m.bias.detach().double() if m.bias is not None else torch.zeros(len(s_out), dtype=torch.float64)
            bq = torch.round(b / s_out)
            act = acts[li]
            if calibrating:
                mx = 0.0
                accs = []
                for a in cal_act:
                    if is_deconv:
                        acc = F.conv_transpose2d(a, wq, None, stride=m.stride[0], padding=m.padding[0],
                                                 output_padding=m.output_padding[0])
                    else:
                        acc = F.conv2d(a, wq, None, stride=m.stride[0], padding=m.padding[0])
                    acc = acc + bq.view(1, -1, 1, 1)
                    accs.append(acc)
                    mx = max(mx, float(acc.abs().max()))
                if li == n_layers - 1:
                    # keep ~2^22 of integer range for the outputs: plenty of precision
                    sh = int(max(0, math.ceil(math.log2(max(mx, 1.0) / 2.0 ** 22))))
                else:
                    sh = int(max(0, math.ceil(math.log2(max(mx, 1.0) * 1.25 / ACT_MAX))))
                shifts.append(sh)
            sh = shifts[li]
            step_out = s_out * float(1 << sh)
            layers.append(IntLayer("deconv" if is_deconv else "conv", wq, bq, sh, act, m.stride[0],
                                   m.padding[0], m.output_padding[0] if is_deconv else 0, step_out))
            if calibrating and li != n_layers - 1:
                nxt = []
                for acc in accs:
                    a = round_shift(acc, sh).clamp(-ACT_MAX, ACT_MAX)
                    a = leaky_int(a) if act == "leaky" else a.clamp(min=0) if act == "relu" else a
                    nxt.append(a)
                cal_act = nxt
            step_in = step_out
        n_out = layers[-1].weight.shape[1 if layers[-1].kind == "deconv" else 0]
        obj = cls(layers, in_shift, has_means, n_out // 2 if has_means else n_out)
        obj.fingerprint = obj._fingerprint()
        return obj

    def _fingerprint(self) -> int:
        h = zlib.crc32(bytes([self.in_shift]))
        for lyr in self.layers:
            h = zlib.crc32(lyr.weight.to(torch.int16).contiguous().numpy().tobytes(), h)
            h = zlib.crc32(lyr.bias.to(torch.int32).contiguous().numpy().tobytes(), h)
            h = zlib.crc32(lyr.out_scale.float().contiguous().numpy().tobytes(), h)
            h = zlib.crc32(bytes([lyr.shift]), h)
        return h & 0xFFFFFFFF

    @property
    def shifts(self) -> list[int]:
        return [lyr.shift for lyr in self.layers]

    # ------------------------------------------------------------------ export
    def export(self) -> dict:
        """JSON-able description for the browser decoder (weights as little-endian bytes)."""
        import base64

        out = {"in_shift": self.in_shift, "has_means": self.has_means, "n_scales": self.n_scales,
               "act_max": ACT_MAX, "w_max": W_MAX, "leaky": [LEAKY_NUM, LEAKY_SHIFT],
               "fingerprint": self.fingerprint,
               "layers": []}
        for lyr in self.layers:
            w = lyr.weight
            if lyr.kind == "deconv":  # store as (out, in, k, k) for a uniform gather kernel
                w = w.permute(1, 0, 2, 3)
            w = w.contiguous()
            out["layers"].append({
                "kind": lyr.kind, "out": int(w.shape[0]), "in": int(w.shape[1]), "k": int(w.shape[2]),
                "stride": lyr.stride, "padding": lyr.padding, "output_padding": lyr.output_padding,
                "shift": lyr.shift, "act": lyr.act,
                "weight_b64": base64.b64encode(w.to(torch.int16).numpy().tobytes()).decode(),
                "bias": [int(v) for v in lyr.bias.tolist()],
                "out_scale": [float(v) for v in lyr.out_scale.float().tolist()],
            })
        return out


# ----------------------------------------------------------------------- shifts

def _shift_store() -> dict:
    try:
        with resources.files("neuralpack").joinpath("inthyper_shifts.json").open() as fh:
            return json.load(fh)
    except FileNotFoundError:  # pragma: no cover
        return {}


def stored_shifts(model: str, quality: int) -> tuple[int, list[int]] | None:
    entry = _shift_store().get(f"{model}:{quality}")
    if not entry:
        return None
    return int(entry["in_shift"]), [int(s) for s in entry["shifts"]]
