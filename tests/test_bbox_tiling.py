"""Restricting a domain sweep to a lon/lat bounding box.

The coarse WRF domain is usually far larger than the area anyone wants at 30 m,
and inference cost is linear in tile count, so this is the main lever on run
cost. Two properties have to hold for it to be trustworthy:

* **Coverage.** Every point inside the requested box that the domain can reach
  must land in some selected tile's core. A subset that quietly leaves a hole is
  worse than no subset, because nothing downstream reports it.
* **Alignment.** Tiles must be *filtered* from the full-domain lattice, never
  re-originated on the box. Otherwise a subset run and a full run put their tile
  centres in different places and the two stores stop being comparable.
"""

from __future__ import annotations

import numpy as np
import pyproj
import pytest
from pydantic import ValidationError

from fuxicfd_wrf.config import BBox, TilingConfig
from fuxicfd_wrf.grids import GEOGRAPHIC_CRS, DomainGeometry
from fuxicfd_wrf.pipeline import (
    _check_bbox_coverage,
    _check_lattice_is_rectangular,
)

from test_geometry_and_winds import _synthetic_domain

STRIDE = 6000.0


def _geom(rotation_deg: float = -0.34) -> DomainGeometry:
    return DomainGeometry(*_synthetic_domain(rotation_deg=rotation_deg))


def _lonlat_of(geom, x, y):
    tf = pyproj.Transformer.from_crs(geom.crs, GEOGRAPHIC_CRS, always_xy=True)
    return tf.transform(x, y)


def _domain_lonlat_box(geom, shrink=0.35):
    """A lon/lat box covering the middle `shrink` of the domain."""
    minx, miny, maxx, maxy = geom.extent
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    hx, hy = (maxx - minx) * shrink / 2, (maxy - miny) * shrink / 2
    lon0, lat0 = _lonlat_of(geom, cx - hx, cy - hy)
    lon1, lat1 = _lonlat_of(geom, cx + hx, cy + hy)
    return (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))


def test_bbox_selects_a_strict_subset_of_the_same_lattice():
    geom = _geom()
    full = geom.make_tiles(stride_m=STRIDE)
    sub = geom.make_tiles(stride_m=STRIDE, bbox=_domain_lonlat_box(geom))

    assert 0 < len(sub) < len(full)
    by_id = {t.tile_id: t for t in full}
    for t in sub:
        # Same id AND same centre: filtered, not re-originated.
        assert t.tile_id in by_id
        assert (t.x, t.y) == (by_id[t.tile_id].x, by_id[t.tile_id].y)


def test_bbox_spanning_everything_returns_every_tile():
    geom = _geom()
    full = geom.make_tiles(stride_m=STRIDE)
    everything = geom.make_tiles(stride_m=STRIDE, bbox=(-180.0, -90.0, 180.0, 90.0))
    assert [t.tile_id for t in everything] == [t.tile_id for t in full]


def test_selected_tiles_cover_every_requested_point():
    """The real guarantee: no holes inside the requested box.

    Sampled on a grid in projected space, since that is where the cores are
    rectangles. Points outside the domain's own reach are excluded -- the bbox
    cannot conjure coverage the WRF domain does not have.
    """
    geom = _geom()
    bbox = _domain_lonlat_box(geom)
    sub = geom.make_tiles(stride_m=STRIDE, bbox=bbox)
    full = geom.make_tiles(stride_m=STRIDE)

    cores = np.array([geom.core_bounds(t, STRIDE) for t in sub])
    full_cores = np.array([geom.core_bounds(t, STRIDE) for t in full])

    minx, miny, maxx, maxy = (full_cores[:, 0].min(), full_cores[:, 1].min(),
                              full_cores[:, 2].max(), full_cores[:, 3].max())
    gx, gy = np.meshgrid(np.linspace(minx, maxx, 60), np.linspace(miny, maxy, 60))
    lon, lat = _lonlat_of(geom, gx.ravel(), gy.ravel())

    inside_request = ((lon >= bbox[0]) & (lon <= bbox[2])
                      & (lat >= bbox[1]) & (lat <= bbox[3]))
    px, py = gx.ravel(), gy.ravel()

    def covered(boxes, i):
        return np.any((boxes[:, 0] <= px[i]) & (px[i] <= boxes[:, 2])
                      & (boxes[:, 1] <= py[i]) & (py[i] <= boxes[:, 3]))

    reachable = [i for i in np.nonzero(inside_request)[0] if covered(full_cores, i)]
    assert reachable, "test box selected nothing the full domain covers"
    missed = [i for i in reachable if not covered(cores, i)]
    assert not missed, f"{len(missed)} of {len(reachable)} requested points uncovered"


def test_bbox_outside_the_domain_selects_nothing():
    geom = _geom()
    assert geom.make_tiles(stride_m=STRIDE, bbox=(120.0, 10.0, 121.0, 11.0)) == []


def test_selection_uses_the_core_not_the_full_footprint():
    """A tile whose 9 km footprint overlaps the box but whose 6 km core does not
    contributes nothing after the 1.5 km trim, so it must not be selected."""
    geom = _geom(rotation_deg=0.0)
    full = geom.make_tiles(stride_m=STRIDE)
    t = full[len(full) // 2]

    # A box just outside this tile's core, but inside its 9 km footprint.
    cminx, cminy, cmaxx, cmaxy = geom.core_bounds(t, STRIDE)
    gap_x = cmaxx + (t.x + 4500.0 - cmaxx) / 2
    lon0, lat0 = _lonlat_of(geom, gap_x - 100, t.y - 100)
    lon1, lat1 = _lonlat_of(geom, gap_x + 100, t.y + 100)
    box = (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))

    picked = {x.tile_id for x in geom.make_tiles(stride_m=STRIDE, bbox=box)}
    assert t.tile_id not in picked
    assert picked, "the neighbouring tile whose core does contain the box should be picked"


# --- config validation -------------------------------------------------------

def test_bbox_rejects_reversed_bounds():
    with pytest.raises(ValidationError, match="min_lon"):
        BBox(min_lon=-105.0, min_lat=34.0, max_lon=-106.0, max_lat=35.0)
    with pytest.raises(ValidationError, match="min_lat"):
        BBox(min_lon=-106.0, min_lat=35.0, max_lon=-105.0, max_lat=34.0)


def test_bbox_rejects_swapped_lon_lat():
    """lat in a lon slot is in range; lon in a lat slot is not -- catch it."""
    with pytest.raises(ValidationError, match="out of range"):
        BBox(min_lon=34.0, min_lat=-106.0, max_lon=35.0, max_lat=-105.0)


def test_bbox_is_rejected_in_sites_mode():
    with pytest.raises(ValidationError, match="mode='domain'"):
        TilingConfig(
            mode="sites",
            sites=[{"id": "a", "lon": -105.0, "lat": 34.5}],
            bbox=BBox(min_lon=-106.0, min_lat=34.0, max_lon=-105.0, max_lat=35.0),
        )


# --- coverage and lattice guards ---------------------------------------------

def _tiles_from(geom, bbox):
    return geom.make_tiles(stride_m=STRIDE, bbox=bbox)


def test_coverage_is_full_for_a_box_inside_the_domain():
    geom = _geom()
    bbox = _domain_lonlat_box(geom)
    frac = _check_bbox_coverage(geom, _tiles_from(geom, bbox), STRIDE, bbox, 0.9)
    assert frac > 0.999


def test_coverage_guard_rejects_a_box_mostly_outside_the_domain():
    """The real failure: a box hanging far off the domain edge still selects
    tiles, so nothing errors and the run proceeds on a fraction of the request."""
    geom = _geom()
    lon0, lat0, lon1, lat1 = _domain_lonlat_box(geom, shrink=0.3)
    # Extend the southern edge well past the domain: the overhang dominates.
    bbox = (lon0, lat0 - 3 * (lat1 - lat0), lon1, lat1)
    tiles = _tiles_from(geom, bbox)
    assert tiles, "the box still intersects the domain, so tiles are selected"

    with pytest.raises(ValueError, match="of the requested box is covered"):
        _check_bbox_coverage(geom, tiles, STRIDE, bbox, 0.9)


def test_coverage_guard_can_be_disabled():
    geom = _geom()
    lon0, lat0, lon1, lat1 = _domain_lonlat_box(geom, shrink=0.3)
    bbox = (lon0, lat0 - 3 * (lat1 - lat0), lon1, lat1)
    frac = _check_bbox_coverage(geom, _tiles_from(geom, bbox), STRIDE, bbox, 0.0)
    assert 0.0 < frac < 0.9


def test_lattice_check_counts_tiles_absent_from_the_bounding_block():
    geom = _geom(rotation_deg=-8.0)     # exaggerated, to guarantee a ragged edge
    tiles = geom.make_tiles(stride_m=STRIDE)
    missing = _check_lattice_is_rectangular(tiles)

    rc = [tuple(map(int, t.tile_id.split("_")[1:])) for t in tiles]
    rows = {r for r, _ in rc}
    cols = {c for _, c in rc}
    assert missing == len(rows) * len(cols) - len(tiles)
    assert missing > 0, "a strongly rotated domain must give a ragged lattice"


def test_lattice_check_is_silent_for_site_tiles():
    from fuxicfd_wrf.grids import Tile
    assert _check_lattice_is_rectangular([Tile("mast_a", x=0.0, y=0.0)]) == 0
