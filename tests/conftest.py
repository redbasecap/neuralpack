import numpy as np
import pytest
from PIL import Image


def synthetic_photo(w: int, h: int, seed: int = 0) -> Image.Image:
    """Smooth gradients + edges + texture + noise, so it behaves like a photo for the codec."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.5 + 0.5 * np.sin(xx / max(w, 1) * 6.0 + yy / max(h, 1) * 2.0)
    g = 0.5 + 0.5 * np.cos(yy / max(h, 1) * 5.0)
    b = ((xx // 23 + yy // 17) % 2) * 0.6 + 0.2
    img = np.stack([r, g, b], -1)
    img[h // 4:h // 2, w // 4:w // 2] = [0.9, 0.2, 0.1]  # a flat coloured block with hard edges
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")


@pytest.fixture(scope="session")
def photo() -> Image.Image:
    return synthetic_photo(200, 136)


@pytest.fixture(scope="session")
def big_photo() -> Image.Image:
    return synthetic_photo(333, 271, seed=1)
