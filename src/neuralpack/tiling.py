"""Tile geometry for arbitrarily large images.

A 100 MP picture does not fit through a convolutional codec in one piece on
most machines, so the image is cut into a grid of *core* tiles. Each core tile
is encoded together with a small *halo* of neighbouring pixels (``overlap``),
which the decoder crops away again. The halo gives the synthesis transform real
context at tile borders, so seams are invisible at a cost of a few percent of
extra bits. Geometry is a pure function of (width, height, tile, overlap), so
nothing per tile needs to be stored in the file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    # core region (what this tile is responsible for), in image coordinates
    x0: int
    y0: int
    x1: int
    y1: int
    # expanded region that is actually coded (core + halo, clipped to the image)
    ex0: int
    ey0: int
    ex1: int
    ey1: int

    @property
    def crop(self) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) of the core inside the decoded expanded tile."""
        return (self.x0 - self.ex0, self.y0 - self.ey0, self.x1 - self.ex0, self.y1 - self.ey0)


def tile_grid(width: int, height: int, tile: int, overlap: int) -> list[Tile]:
    """Row-major list of tiles.

    ``tile`` is the edge of the *coded* block (core + halo on both sides), so with
    a multiple of 64 no interior tile needs padding. The core stride is therefore
    ``tile - 2 * overlap``. ``tile == 0`` codes the whole picture in one piece.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image must have positive size")
    if tile <= 0 or (width <= tile and height <= tile):
        return [Tile(0, 0, width, height, 0, 0, width, height)]
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    stride = tile - 2 * overlap
    if stride <= 0:
        raise ValueError("tile must be larger than 2 * overlap")
    tiles = []
    for y0 in range(0, height, stride):
        for x0 in range(0, width, stride):
            x1, y1 = min(x0 + stride, width), min(y0 + stride, height)
            tiles.append(Tile(x0, y0, x1, y1,
                              max(0, x0 - overlap), max(0, y0 - overlap),
                              min(width, x1 + overlap), min(height, y1 + overlap)))
    return tiles
