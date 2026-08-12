"""The WRG record array must keep every column on the same point ordering.

Regression for a transposed elevation field. Sector histograms are indexed from
`speed.reshape(n_time, -1)` of a (ny, nx) grid -- row-major -- while the WRG
format lists points column-major. `export_wrg` converts the whole record array
once at the end, so anything handed to `build_records` must arrive row-major.

`_elevation_on_grid` returned column-major, so elevation was converted twice: a
transpose relative to its own coordinates. It measured 37 m RMS on a 60x60 tile
and produced 7.8x elevation steps at tile seams in the concatenated domain WRG,
while every wind column stayed correct -- the kind of fault that looks like a
physical feature rather than a bug.
"""

from __future__ import annotations

import numpy as np

from fuxicfd_wrf.wrg_export import _reorder_f


def test_reorder_f_is_row_major_to_column_major():
    """_reorder_f must map C-order rows onto the WRG's column-major listing."""
    ny, nx = 3, 4
    # Column 0 encodes the (row, col) of each point in C order.
    rec = np.arange(ny * nx, dtype=float).reshape(-1, 1)
    out = _reorder_f(rec, ny, nx)
    # Column-major listing: for each x (col), all y (rows) ascending.
    expected = np.array([r * nx + c for c in range(nx) for r in range(ny)], dtype=float)
    assert np.allclose(out.ravel(), expected)


def test_applying_reorder_twice_transposes():
    """Double conversion is exactly the bug this guards against.

    A square grid makes it a clean transpose, which is why it survived review on
    60x60 tiles and only became obvious once the domain was assembled.
    """
    n = 5
    field = np.arange(n * n, dtype=float).reshape(n, n)
    once = _reorder_f(field.reshape(-1, 1), n, n).ravel()
    twice = _reorder_f(once.reshape(-1, 1), n, n).ravel()
    assert np.allclose(once.reshape(n, n), field.T)
    assert np.allclose(twice.reshape(n, n), field)     # back to start
    assert not np.allclose(once, field.ravel())        # so once != identity


def test_elevation_is_returned_row_major(tmp_path, monkeypatch):
    """_elevation_on_grid must hand back C-order, matching histogram indexing."""
    from fuxicfd_wrf import wrg_export
    from fuxicfd_wrf.grids import Tile
    from fuxicfd_wrf.terrain.build import TerrainTile

    ny, nx = 8, 8
    res = 30.0
    tile = Tile("t", x=500000.0, y=3850000.0, crs="EPSG:32613")
    gx, gy = tile.terrain_axes()
    grid = wrg_export.WrgGrid(x=gx[:nx] * 0 + gx[0] + np.arange(nx) * res,
                              y=gy[0] - np.arange(ny) * res, res_m=res,
                              crs="EPSG:32613")

    # A DEM that varies only with x: the result must vary only along axis 1.
    dem = np.tile(np.arange(301, dtype=np.float32), (301, 1))
    monkeypatch.setattr(
        wrg_export, "build_terrain",
        lambda t, c, **k: TerrainTile(t.tile_id, dem, np.full_like(dem, 0.03),
                                      np.full_like(dem, 30)),
        raising=False,
    )
    monkeypatch.setattr(
        "fuxicfd_wrf.terrain.build.build_terrain",
        lambda t, c, **k: TerrainTile(t.tile_id, dem, np.full_like(dem, 0.03),
                                      np.full_like(dem, 30)),
    )
    (tmp_path / "tiles").mkdir()
    (tmp_path / "tiles" / "t_raw.npz").write_bytes(b"x")   # existence check only

    out = wrg_export._elevation_on_grid(grid, [tile], tmp_path).reshape(ny, nx)
    # x-varying field: rows identical, columns increasing.
    assert np.allclose(out[0], out[-1]), "row-major expected; got a transposed field"
    assert (np.diff(out[0]) > 0).all()
