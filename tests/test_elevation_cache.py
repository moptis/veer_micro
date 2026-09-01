"""The WRG elevation column must actually find the cached tile DEMs.

This lookup has failed silently twice. `build_terrain` names the raw cache
`{tile_id}-{geo_key}_raw.npz` (or `..._raw_{source}.npz`), and `_elevation_on_grid`
has to recognise both halves of that stem. First the `_{source}` suffix was
missed, so every `landcover_source: nlcd` run fell through; then `geo_key` was
introduced -- to stop one domain serving another's terrain -- and the `-{geo_key}`
infix broke the patterns again, so *every* run fell through.

Both times the WRG still wrote, with an elevation column of zeros and only a
warning on stdout. A consumer cannot tell that from a genuinely flat site, which
is what makes it worth a test rather than a careful reading.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuxicfd_wrf.grids import Tile
from fuxicfd_wrf.wrg_export import _elevation_on_grid, build_wrg_grid

CRS = "EPSG:32613"


def _tile(tmp_path, source="worldcover", elevation=1500.0):
    """Write a raw terrain cache entry under the exact name build_terrain uses."""
    from fuxicfd_wrf.terrain.build import TERRAIN_SHAPE

    tile = Tile(tile_id="t_003_004", x=430_000.0, y=3_790_000.0, crs=CRS)
    d = tmp_path / "tiles"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{tile.tile_id}-{tile.geo_key}"
    name = f"{stem}_raw.npz" if source == "worldcover" else f"{stem}_raw_{source}.npz"
    np.savez(
        d / name,
        dem=np.full(TERRAIN_SHAPE, elevation, dtype="float32"),
        roughness=np.full(TERRAIN_SHAPE, 0.03, dtype="float32"),
        landcover=np.full(TERRAIN_SHAPE, 30, dtype="int32"),
    )
    return tile


def _grid(tile):
    minx, miny, maxx, maxy = tile.bounds()
    return build_wrg_grid(minx, miny, maxx, maxy, res_m=200.0, crs=CRS)


@pytest.mark.parametrize("source", ["worldcover", "nlcd"])
def test_elevation_found_for_both_cache_names(tmp_path, source):
    tile = _tile(tmp_path, source=source)
    elev = _elevation_on_grid(_grid(tile), [tile], tmp_path)
    assert np.all(elev == pytest.approx(1500.0)), (
        f"elevation column is zero for a {source}-named cache entry -- the glob in "
        "_elevation_on_grid has drifted from build_terrain's naming again"
    )


def test_elevation_lookup_matches_build_terrain_naming(tmp_path):
    """Pin the two against each other rather than trusting a restated pattern."""
    import re

    from fuxicfd_wrf.terrain import build as build_mod

    tile = _tile(tmp_path)
    written = {p.name for p in (tmp_path / "tiles").glob("*_raw*.npz")}
    matched = {
        p.name
        for p in (tmp_path / "tiles").glob(f"{tile.tile_id}-{tile.geo_key}_raw*.npz")
    }
    assert written and written == matched

    # The stem must carry the geo_key: tile ids repeat across domains, and a
    # collision hands one site another site's terrain with no error at all.
    src = build_mod.__file__
    assert re.search(r'f"\{tile\.tile_id\}-\{tile\.geo_key\}"', open(src).read()), (
        "build_terrain no longer keys the raw cache on geo_key; the elevation "
        "lookup and the cross-domain collision guard both depend on it"
    )


def test_missing_cache_still_returns_zeros_not_an_error(tmp_path):
    """A missing cache degrades to a zero column -- loud, but not fatal."""
    tile = Tile(tile_id="t_000_000", x=430_000.0, y=3_790_000.0, crs=CRS)
    elev = _elevation_on_grid(_grid(tile), [tile], tmp_path)
    assert elev.shape == (_grid(tile).n_points,)
    assert np.all(elev == 0.0)
