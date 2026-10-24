"""neuralpack: neural image & video compression with a shared learned prior.

The sender and the receiver both hold the same pretrained model (the "prior",
a learned dictionary of image structure). A ``.npk`` file contains nothing but
entropy-coded latent indexes into that prior, so it is tiny; the receiver's
model turns the indexes back into a full-resolution picture.
"""

from .codec import (
    EncodeResult,
    decode_bytes,
    decode_file,
    decode_image,
    encode_file,
    encode_image,
)
from .container import NpkHeader, inspect
from .models import DEFAULT_MODEL, MODELS, available_devices, pick_device

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "EncodeResult",
    "NpkHeader",
    "__version__",
    "available_devices",
    "decode_bytes",
    "decode_file",
    "decode_image",
    "encode_file",
    "encode_image",
    "inspect",
    "pick_device",
]
