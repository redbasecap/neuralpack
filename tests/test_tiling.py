import pytest

from neuralpack.tiling import tile_grid


def test_single_tile_when_small_or_disabled():
    assert len(tile_grid(500, 300, 1024, 16)) == 1
    assert len(tile_grid(5000, 3000, 0, 16)) == 1
    t = tile_grid(500, 300, 1024, 16)[0]
    assert (t.x0, t.y0, t.x1, t.y1) == (0, 0, 500, 300)
    assert t.crop == (0, 0, 500, 300)


@pytest.mark.parametrize("w,h,tile,ov", [(1000, 700, 256, 16), (1025, 1, 1024, 8), (64, 4000, 64, 0),
                                         (3333, 2222, 1024, 16), (300, 300, 128, 60)])
def test_cores_partition_image_exactly(w, h, tile, ov):
    grid = tile_grid(w, h, tile, ov)
    covered = 0
    for t in grid:
        assert 0 <= t.ex0 <= t.x0 < t.x1 <= t.ex1 <= w
        assert 0 <= t.ey0 <= t.y0 < t.y1 <= t.ey1 <= h
        assert t.ex1 - t.ex0 <= tile and t.ey1 - t.ey0 <= tile
        covered += (t.x1 - t.x0) * (t.y1 - t.y0)
        left, top, right, bottom = t.crop
        assert (right - left, bottom - top) == (t.x1 - t.x0, t.y1 - t.y0)
    assert covered == w * h
    cores = [(t.x0, t.y0, t.x1, t.y1) for t in grid]
    assert len(set(cores)) == len(cores)


def test_interior_tiles_are_exactly_tile_sized():
    grid = tile_grid(4000, 4000, 1024, 16)
    interior = [t for t in grid if t.ex0 > 0 and t.ey0 > 0 and t.ex1 < 4000 and t.ey1 < 4000]
    assert interior and all((t.ex1 - t.ex0, t.ey1 - t.ey0) == (1024, 1024) for t in interior)


def test_bad_geometry():
    with pytest.raises(ValueError):
        tile_grid(100, 100, 64, 32)
    with pytest.raises(ValueError):
        tile_grid(0, 10, 64, 0)
