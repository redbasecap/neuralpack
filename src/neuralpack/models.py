"""Model registry: the shared learned priors that sender and receiver both hold.

Every model here is a *hyperprior* codec (Ballé et al. 2018, Minnen et al. 2018)
from CompressAI with published pretrained weights:

* ``g_a``  analysis transform  – a learned, non-linear generalisation of the
  DCT/KLT "rotation": pixels -> latent coefficients ``y``.
* ``h_a``/``h_s``  hyper transforms – a second, much smaller latent ``z`` that
  tells the entropy coder how ``y`` is distributed (the "side information").
* ``g_s``  synthesis transform – latents back to pixels.
* entropy models – learned probability tables driving a range coder.

Only models with *parallel* entropy decoding are registered. Autoregressive
context models (mbt2018, cheng2020) are 2-4 % smaller but decode a 24 MP image
in minutes on a CPU, which defeats the "open and it is just there" goal.
"""

from __future__ import annotations

import copy
import os
import sys
import types
import warnings
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="compressai")



def _stub_torch_geometric() -> dict[str, types.ModuleType]:
    """The handful of torch_geometric / torchvision names CompressAI touches at import."""

    class BaseTransform:
        def __call__(self, data):  # pragma: no cover - never used for images
            return data

    class Data(dict):
        pass

    def functional_transform(name):
        return lambda cls: cls

    pkg = types.ModuleType("torch_geometric")
    pkg.__path__ = []  # type: ignore[attr-defined]  # behave like a package
    tr = types.ModuleType("torch_geometric.transforms")
    tr.BaseTransform, tr.Center = BaseTransform, BaseTransform  # type: ignore[attr-defined]
    data = types.ModuleType("torch_geometric.data")
    data.__path__ = []  # type: ignore[attr-defined]
    data.Data = Data  # type: ignore[attr-defined]
    dp = types.ModuleType("torch_geometric.data.datapipes")
    dp.functional_transform = functional_transform  # type: ignore[attr-defined]
    pkg.transforms, pkg.data, data.datapipes = tr, data, dp  # type: ignore[attr-defined]
    stubs = {"torch_geometric": pkg, "torch_geometric.transforms": tr,
             "torch_geometric.data": data, "torch_geometric.data.datapipes": dp}
    # torchvision is imported for the same registry dict only; its import drags in
    # torch._dynamo and costs several seconds more.
    if "torchvision" not in sys.modules:
        tv = types.ModuleType("torchvision")
        tv.__path__ = []  # type: ignore[attr-defined]
        tvt = types.ModuleType("torchvision.transforms")
        tv.transforms = tvt  # type: ignore[attr-defined]
        stubs.update({"torchvision": tv, "torchvision.transforms": tvt})
    return stubs


def _import_compressai_zoo():
    """Import CompressAI without paying for torch_geometric.

    ``compressai/__init__`` eagerly imports its point-cloud code, which imports
    ``torch_geometric`` unguarded; that pulls in a graph/LLM stack and costs 10-20 s
    of start-up for a tool that only needs two image models. CompressAI only needs
    a few names from it at import time, so a throw-away stub satisfies it. The stub
    is removed again afterwards, so a later real ``import torch_geometric`` in the
    same process still works. Set ``NEURALPACK_NO_IMPORT_SHIM=1`` to disable.
    """
    shim = ("torch_geometric" not in sys.modules
            and "compressai" not in sys.modules
            and not os.environ.get("NEURALPACK_NO_IMPORT_SHIM"))
    stubs = _stub_torch_geometric() if shim else {}
    sys.modules.update(stubs)
    try:
        from compressai.zoo import bmshj2018_hyperprior, mbt2018_mean
    except ImportError:
        if not shim:
            raise
        for k in stubs:  # stub insufficient for this CompressAI version: do it the slow way
            sys.modules.pop(k, None)
        for k in [m for m in sys.modules if m.startswith("compressai")]:
            del sys.modules[k]
        from compressai.zoo import bmshj2018_hyperprior, mbt2018_mean
    finally:
        for k in stubs:
            if sys.modules.get(k) is stubs[k]:
                del sys.modules[k]
    return bmshj2018_hyperprior, mbt2018_mean


bmshj2018_hyperprior, mbt2018_mean = _import_compressai_zoo()


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one model family."""

    id: int  # numeric id written into the .npk header (never reuse ids)
    name: str  # CLI / API name
    factory: Callable[..., torch.nn.Module]
    qualities: tuple[int, ...]
    has_means: bool  # hyper-synthesis predicts (scale, mean) instead of scale only
    hyper_abs: bool  # hyper-analysis sees |y| (bmshj2018) or y (mbt2018)
    description: str


MODELS: dict[str, ModelSpec] = {
    "hyperprior": ModelSpec(
        id=1,
        name="hyperprior",
        factory=bmshj2018_hyperprior,
        qualities=tuple(range(1, 9)),
        has_means=False,
        hyper_abs=True,
        description="Ballé 2018 scale hyperprior. Smallest/fastest prior (20-48 MB).",
    ),
    "mean-scale": ModelSpec(
        id=2,
        name="mean-scale",
        factory=mbt2018_mean,
        qualities=tuple(range(1, 9)),
        has_means=True,
        hyper_abs=False,
        description="Minnen 2018 mean-scale hyperprior. Default: ~5-10 % smaller files.",
    ),
}
MODELS_BY_ID: dict[int, ModelSpec] = {m.id: m for m in MODELS.values()}
DEFAULT_MODEL = "mean-scale"
DEFAULT_QUALITY = 5

# The transforms are convolutional with total stride 64 (16 for y, another 4 for z).
STRIDE = 64


def available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.insert(0, "cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devs.insert(0 if "cuda" not in devs else 1, "mps")
    return devs


def pick_device(preference: str | None = "auto") -> torch.device:
    """Resolve ``auto``/``cuda``/``mps``/``cpu`` to a torch device, falling back to CPU."""
    if preference in (None, "", "auto"):
        return torch.device(available_devices()[0])
    if preference == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available, using CPU", stacklevel=2)
        return torch.device("cpu")
    if preference == "mps" and not torch.backends.mps.is_available():
        warnings.warn("MPS requested but not available, using CPU", stacklevel=2)
        return torch.device("cpu")
    return torch.device(preference)


class LoadedModel:
    """A pretrained prior ready for coding on a specific device.

    The heavy transforms (``g_a``, ``g_s``) run on ``device``. Everything that
    decides *how the bitstream is parsed* – the hyper-synthesis that produces
    the entropy parameters, and the range coder itself – runs on the CPU in
    float64. That keeps encoder and decoder in lockstep across GPUs, CPUs and
    operating systems; a 1-ulp difference in a scale value could otherwise
    desynchronise the arithmetic decoder and corrupt the whole picture.
    """

    def __init__(self, spec: ModelSpec, quality: int, device: torch.device):
        if quality not in spec.qualities:
            raise ValueError(f"{spec.name}: quality must be in {spec.qualities}, got {quality}")
        self.spec = spec
        self.quality = quality
        self.device = device
        net = spec.factory(quality=quality, pretrained=True, progress=True).eval()
        net.update(force=True)  # build the quantised CDF tables for the range coder
        # CPU-resident copies of everything the range coder touches, taken before
        # the transforms move to the accelerator.
        self.entropy_bottleneck = copy.deepcopy(net.entropy_bottleneck).cpu().eval()
        self.gaussian_conditional = copy.deepcopy(net.gaussian_conditional).cpu().eval()
        # Float64 CPU twin of the hyper-synthesis for bit-exact entropy parameters.
        self.h_s64 = copy.deepcopy(net.h_s).cpu().double().eval()
        # Integer-only hyper-synthesis (see inthyper.py): identical on every device and
        # in the browser. Available for qualities with stored calibration shifts.
        from .inthyper import IntHyperSynthesis, stored_shifts

        self.int_hyper: IntHyperSynthesis | None = None
        stored = stored_shifts(spec.name, quality)
        if stored is not None:
            in_shift, shifts = stored
            self.int_hyper = IntHyperSynthesis.quantize(net.h_s, spec.has_means, shifts=shifts,
                                                        in_shift=in_shift)
        self.net = net.to(device)
        self.table_fingerprint = self._fingerprint()
        self.int_fingerprint = (zlib.crc32(self.int_hyper.fingerprint.to_bytes(4, "little"),
                                           self.table_fingerprint) & 0xFFFFFFFF
                                if self.int_hyper is not None else None)

    def _fingerprint(self) -> int:
        """CRC32 over the entropy coder tables. Stored in every file so a decoder with
        subtly different tables fails loudly instead of producing garbage."""
        h = 0
        for mod in (self.entropy_bottleneck, self.gaussian_conditional):
            for t in (mod._quantized_cdf, mod._offset, mod._cdf_length):
                h = zlib.crc32(t.detach().cpu().contiguous().numpy().tobytes(), h)
        h = zlib.crc32(self.gaussian_conditional.scale_table.cpu().numpy().tobytes(), h)
        return h & 0xFFFFFFFF

    @torch.no_grad()
    def entropy_params(self, z_hat: torch.Tensor, integer: bool = False
                       ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(scales, means) for the latent ``y`` given dequantised ``z_hat`` (CPU, float32).

        ``integer=True`` uses the integer-only hyper-synthesis (bit-exact everywhere, needed
        for files the browser decoder must read); otherwise the float64 CPU path.
        """
        if integer:
            if self.int_hyper is None:
                raise ValueError(f"{self.spec.name} q{self.quality} has no integer hyper-synthesis")
            return self.int_hyper(z_hat)
        params = self.h_s64(z_hat.cpu().double())
        if self.spec.has_means:
            scales, means = params.chunk(2, 1)
            return scales.float(), means.float()
        return params.float(), None

    def fingerprint(self, integer: bool) -> int:
        if integer:
            if self.int_fingerprint is None:
                raise ValueError(f"{self.spec.name} q{self.quality} has no integer hyper-synthesis")
            return self.int_fingerprint
        return self.table_fingerprint


_CACHE: dict[tuple[str, int, str], LoadedModel] = {}


def load_model(name: str = DEFAULT_MODEL, quality: int = DEFAULT_QUALITY,
               device: str | torch.device = "auto") -> LoadedModel:
    """Load (and cache) a pretrained prior. Weights are downloaded once (~20-70 MB)."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    dev = pick_device(device) if isinstance(device, str) else device
    key = (name, int(quality), str(dev))
    if key not in _CACHE:
        _CACHE[key] = LoadedModel(MODELS[name], int(quality), dev)
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()
