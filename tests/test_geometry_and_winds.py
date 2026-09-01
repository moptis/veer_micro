"""Geometry, wind conversion and interpolation tests.

These need no model assets and no GPU, so they are the fast feedback loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuxicfd_wrf.constants import (
    COARSE_RES_M, COARSE_SHAPE, TERRAIN_RES_M, TERRAIN_SHAPE, TILE_SPAN_M,
    WRF_GRID_SHAPE,
)
from fuxicfd_wrf.grids import COARSE_HALF_EXTENT_M as COARSE_HALF
from fuxicfd_wrf.grids import DomainGeometry, Tile
from fuxicfd_wrf.winds import (
    build_stencil_interpolator, grid_convergence_deg, rotate_to_grid,
    speeddir_to_uv, stencil_uv, stencil_winds, uv_to_speeddir,
)


# --- grid consistency --------------------------------------------------------

def test_terrain_and_coarse_grids_span_the_same_box():
    """301 nodes x 30 m and 9 cells x 1 km must both describe 9 km.

    An off-by-one here (300 nodes, or 9 km measured centre-to-centre) misaligns
    terrain against the wind stencil by half a cell everywhere.
    """
    assert (TERRAIN_SHAPE[0] - 1) * TERRAIN_RES_M == TILE_SPAN_M
    assert COARSE_SHAPE[0] * COARSE_RES_M == TILE_SPAN_M


def test_tile_axes_are_north_up_and_centred():
    t = Tile("t", x=500000.0, y=3850000.0)
    xs, ys = t.terrain_axes()
    assert len(xs) == len(ys) == TERRAIN_SHAPE[0]
    assert xs[0] == pytest.approx(t.x - TILE_SPAN_M / 2)
    assert xs[-1] == pytest.approx(t.x + TILE_SPAN_M / 2)
    # y descends: row 0 is the northern edge.
    assert ys[0] > ys[-1]
    assert ys[0] == pytest.approx(t.y + TILE_SPAN_M / 2)

    cx, cy = t.coarse_axes()
    assert len(cx) == COARSE_SHAPE[0]
    assert cx[0] == pytest.approx(t.x - 4000.0)
    assert cy[0] > cy[-1]
    # Coarse cell centres sit 4 km out; their edges reach the terrain extent.
    assert cx[-1] - cx[0] == pytest.approx(8000.0)


# --- wind conversion ---------------------------------------------------------

@pytest.mark.parametrize(
    "direction,expect",
    [
        (270.0, "westerly"),   # from the west -> u > 0
        (90.0, "easterly"),    # from the east -> u < 0
        (180.0, "southerly"),  # from the south -> v > 0
        (0.0, "northerly"),    # from the north -> v < 0
    ],
)
def test_meteorological_direction_convention(direction, expect):
    u, v = speeddir_to_uv(np.array([10.0]), np.array([direction]))
    signs = {
        "westerly": (u[0] > 0, abs(v[0]) < 1e-9),
        "easterly": (u[0] < 0, abs(v[0]) < 1e-9),
        "southerly": (abs(u[0]) < 1e-9, v[0] > 0),
        "northerly": (abs(u[0]) < 1e-9, v[0] < 0),
    }[expect]
    assert all(signs), f"{direction} deg should be {expect}, got u={u[0]:.3f} v={v[0]:.3f}"


def test_speeddir_uv_roundtrip():
    rng = np.random.default_rng(0)
    speed = rng.uniform(0.1, 30.0, 500)
    direction = rng.uniform(0.0, 360.0, 500)
    s2, d2 = uv_to_speeddir(*speeddir_to_uv(speed, direction))
    assert np.allclose(speed, s2)
    assert np.allclose((direction - d2 + 180) % 360 - 180, 0, atol=1e-9)


def test_rotation_preserves_speed():
    """Frame rotation must not change wind speed, only its decomposition."""
    u, v = speeddir_to_uv(np.array([7.5]), np.array([215.0]))
    ur, vr = rotate_to_grid(u, v, 12.0)
    assert np.hypot(ur, vr) == pytest.approx(np.hypot(u, v))


def test_grid_convergence_is_small_near_central_meridian():
    """Broadview sits near the UTM 13N central meridian (-105 deg)."""
    conv = grid_convergence_deg(np.array(-103.035), np.array(34.817), "EPSG:32613")
    assert abs(float(conv)) < 1.5


# --- domain geometry ---------------------------------------------------------

def _synthetic_domain(rotation_deg: float = -0.34):
    """A Broadview-like WRF domain: regular in a projected CRS, given as lat/lon.

    Built by laying out a regular 802 x 798 m grid in UTM 13N and converting to
    geographic, which is what a Lambert-conformal WRF grid looks like from the
    outside: very nearly affine in UTM (the real domain measures 10.25 m residual).

    Do *not* build this as a regular lat/lon grid. That curves in a projected CRS --
    203 m of non-affinity across this domain -- and would be testing a case the
    real data never presents.
    """
    import pyproj

    nrow, ncol = WRF_GRID_SHAPE
    x0, y0 = 675220.0 - 66 * 802.0, 3850009.0 - 39 * 798.0
    rr, cc = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
    th = np.deg2rad(rotation_deg)
    dx, dy = 802.0, 798.0
    x = x0 + dx * cc * np.cos(th) - dy * rr * np.sin(th)
    y = y0 + dx * cc * np.sin(th) + dy * rr * np.cos(th)
    tf = pyproj.Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(x, y)
    return lat.ravel(), lon.ravel()


def test_regular_latlon_grid_is_rejected_as_non_affine():
    """The affine shortcut must refuse a grid it cannot represent.

    A regular lat/lon grid is exactly such a case. This guards the guard: if the
    residual check were ever loosened, interpolation would silently gain ~200 m of
    positional error.
    """
    nrow, ncol = WRF_GRID_SHAPE
    rr, cc = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
    lat = 34.54 + 0.00720 * rr
    lon = -103.61 + 0.008728 * cc
    with pytest.raises(ValueError, match="not affine"):
        DomainGeometry(lat.ravel(), lon.ravel())


def _curved_domain(nrow: int = 150, ncol: int = 177):
    """A domain large enough that one global UTM fit is not good enough.

    Laid out regular in a Lambert conformal projection and handed back as lat/lon,
    which is what WRF actually produces. Viewed in UTM the misfit is a smooth bowl
    that grows with the square of the domain size: Broadview (106 x 64 km) misses
    by 10 m, Pattern ph1 (180 x 153 km, reproduced here) by 42.7 m.
    """
    import pyproj

    lcc = ("+proj=lcc +lat_1=35.5 +lat_2=35.5 +lat_0=35.45 +lon_0=-101.4 "
           "+R=6370000 +units=m +no_defs")
    rr, cc = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
    x = -0.5 * ncol * 1000.0 + 1000.0 * cc
    y = 0.5 * nrow * 1000.0 - 1000.0 * rr
    tf = pyproj.Transformer.from_crs(lcc, "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(x, y)
    return lat.ravel(), lon.ravel(), (nrow, ncol)


def test_large_domain_is_rejected_globally_but_fits_locally():
    """The guard should send you to a local fit, not to a dead end.

    A 180 x 153 km domain is past what one affine map can represent in UTM, but it
    is perfectly affine over any 17-cell window. Failing outright would mean no
    such domain could ever run.
    """
    lat, lon, shape = _curved_domain()
    with pytest.raises(ValueError, match="affine_fit: local"):
        DomainGeometry(lat, lon, crs="auto", grid_shape=shape)

    geom = DomainGeometry(lat, lon, crs="auto", grid_shape=shape, affine_fit="local")
    assert geom.affine_residual_m > DomainGeometry.MAX_AFFINE_RESIDUAL_M
    assert geom.local_residual_m < 1.0
    assert geom.local_residual_m < geom.affine_residual_m / 10


def test_local_fit_puts_the_stencil_where_the_grid_actually_is(monkeypatch):
    """What the fit is *for*: recovering a node's own index from its coordinates.

    Error here is a positional bias in every interpolated wind value, so this
    measures the quantity that matters rather than the residual that stands in for
    it. The global fit is allowed through only to show the size of what it costs.
    """
    lat, lon, shape = _curved_domain()
    monkeypatch.setattr(DomainGeometry, "MAX_AFFINE_RESIDUAL_M", 1e9)
    glob = DomainGeometry(lat, lon, crs="auto", grid_shape=shape)
    loc = DomainGeometry(lat, lon, crs="auto", grid_shape=shape, affine_fit="local")

    rows, cols = np.meshgrid(np.arange(0, shape[0], 17), np.arange(0, shape[1], 19),
                             indexing="ij")
    x, y = glob.x[rows, cols], glob.y[rows, cols]

    gr, gc = glob.xy_to_rowcol(x, y)
    lr, lc = loc.xy_to_rowcol(x, y)
    global_err = np.hypot(gr - rows, gc - cols).max()
    local_err = np.hypot(lr - rows, lc - cols).max()
    assert global_err > 0.02          # ~43 m at 1 km spacing
    assert local_err < 0.002          # ~2 m, the float32 lon/lat floor
    assert local_err < global_err / 10


def test_rowcol_xy_roundtrip_is_consistent_under_a_local_fit():
    """The two maps must stay mutual inverses, or tiles land off their own grid."""
    lat, lon, shape = _curved_domain()
    geom = DomainGeometry(lat, lon, crs="auto", grid_shape=shape, affine_fit="local")
    rows = np.array([0.0, 40.5, 149.0])
    cols = np.array([0.0, 88.25, 176.0])
    r2, c2 = geom.xy_to_rowcol(*geom.rowcol_to_xy(rows, cols))
    assert np.allclose(rows, r2, atol=1e-6)
    assert np.allclose(cols, c2, atol=1e-6)


def test_local_fit_matches_global_where_global_is_already_valid():
    """On a domain the global fit handles, switching must not move anything much.

    Broadview and WSW are in that regime, so this pins that `local` is a
    refinement of `global` and not a different convention.
    """
    lat, lon = _synthetic_domain()
    glob = DomainGeometry(lat, lon)
    loc = DomainGeometry(lat, lon, affine_fit="local")
    tile = Tile("t", x=(glob.extent[0] + glob.extent[2]) / 2,
                y=(glob.extent[1] + glob.extent[3]) / 2)
    gx, gy = tile.coarse_points()
    gr, gc = glob.xy_to_rowcol(gx, gy)
    lr, lc = loc.xy_to_rowcol(gx, gy)
    # This domain is affine in UTM by construction, so both maps are exact.
    assert np.hypot(gr - lr, gc - lc).max() < 0.01


def test_domain_affine_fit_is_accurate():
    geom = DomainGeometry(*_synthetic_domain())
    assert geom.affine_residual_m < DomainGeometry.MAX_AFFINE_RESIDUAL_M


def test_rowcol_xy_roundtrip():
    geom = DomainGeometry(*_synthetic_domain())
    rows = np.array([0.0, 10.5, 77.0])
    cols = np.array([0.0, 60.25, 131.0])
    r2, c2 = geom.xy_to_rowcol(*geom.rowcol_to_xy(rows, cols))
    assert np.allclose(rows, r2, atol=1e-6)
    assert np.allclose(cols, c2, atol=1e-6)


def test_tiles_outside_domain_are_rejected():
    geom = DomainGeometry(*_synthetic_domain())
    minx, miny, maxx, maxy = geom.extent
    inside = Tile("in", x=(minx + maxx) / 2, y=(miny + maxy) / 2)
    outside = Tile("out", x=minx - 1000.0, y=miny - 1000.0)
    assert geom.covers_stencil(inside)
    assert not geom.covers_stencil(outside)


def test_make_tiles_all_covered_and_overlap_increases_count():
    geom = DomainGeometry(*_synthetic_domain())
    plain = geom.make_tiles()
    overlapped = geom.make_tiles(stride_m=6000.0)
    assert plain and all(geom.covers_stencil(t) for t in plain)
    assert len(overlapped) > len(plain)


def test_bounding_box_corner_is_rejected_on_a_rotated_domain():
    """Regression: containment must be tested in index space, not by bbox.

    The WRF domain is a rotated rectangle in UTM, so its axis-aligned bounding box
    is strictly larger than the domain. A bbox test accepts corner tiles that are
    actually off-grid, and they then blow up inside the interpolator with an
    extrapolation error. Rotation is exaggerated here to make the gap unambiguous.
    """
    geom = DomainGeometry(*_synthetic_domain(rotation_deg=-8.0))
    minx, miny, maxx, maxy = geom.extent
    corner = Tile("corner", x=minx + COARSE_HALF, y=miny + COARSE_HALF)
    assert not geom.covers_stencil(corner)


def test_every_generated_tile_builds_an_interpolator():
    """covers_stencil and the interpolator must agree on what is in-domain.

    If they disagree, `plan` and `run` crash partway through a sweep rather than
    at setup -- which is how this bug first showed up.
    """
    geom = DomainGeometry(*_synthetic_domain(rotation_deg=-3.0))
    for tile in geom.make_tiles(stride_m=6000.0):
        build_stencil_interpolator(geom, tile)  # must not raise


# --- interpolation -----------------------------------------------------------

def test_stencil_interpolator_reproduces_a_linear_field():
    """Bilinear interpolation is exact for a field linear in x and y.

    This validates the affine index-space shortcut against an analytic answer
    rather than against another interpolator with the same possible bug.
    """
    lat, lon = _synthetic_domain()
    geom = DomainGeometry(lat, lon)
    minx, miny, maxx, maxy = geom.extent
    tile = Tile("t", x=(minx + maxx) / 2, y=(miny + maxy) / 2)
    interp = build_stencil_interpolator(geom, tile)

    field = 3.0 * geom.x.ravel() - 2.0 * geom.y.ravel() + 17.0
    got = interp.apply(field)
    gx, gy = tile.coarse_points()
    expected = 3.0 * gx - 2.0 * gy + 17.0
    assert got.shape == COARSE_SHAPE
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-4)


def test_stencil_weights_sum_to_one():
    """Partition of unity: a constant field must interpolate to that constant."""
    lat, lon = _synthetic_domain()
    geom = DomainGeometry(lat, lon)
    minx, miny, maxx, maxy = geom.extent
    tile = Tile("t", x=(minx + maxx) / 2, y=(miny + maxy) / 2)
    interp = build_stencil_interpolator(geom, tile)
    assert np.allclose(interp.matrix.sum(axis=1), 1.0)
    assert np.allclose(interp.apply(np.full(geom.x.size, 4.2)), 4.2)


def test_batched_apply_matches_single():
    lat, lon = _synthetic_domain()
    geom = DomainGeometry(lat, lon)
    minx, miny, maxx, maxy = geom.extent
    interp = build_stencil_interpolator(
        geom, Tile("t", x=(minx + maxx) / 2, y=(miny + maxy) / 2)
    )
    rng = np.random.default_rng(1)
    batch = rng.normal(size=(5, geom.x.size))
    out = interp.apply(batch)
    assert out.shape == (5, *COARSE_SHAPE)
    for i in range(5):
        assert np.allclose(out[i], interp.apply(batch[i]))


def test_stencil_uv_matches_the_speeddir_path_exactly():
    """The hoisted-conversion path must be numerically identical, not merely close.

    The run loop converts speed/direction to components once per read block and
    then calls stencil_uv per tile, instead of re-converting inside the tile loop.
    That is a pure refactor of where the work happens, so any difference at all
    means the two paths have diverged.
    """
    geom = DomainGeometry(*_synthetic_domain())
    tile = Tile("t", x=(geom.extent[0] + geom.extent[2]) / 2,
                y=(geom.extent[1] + geom.extent[3]) / 2)
    interp = build_stencil_interpolator(geom, tile)

    rng = np.random.default_rng(0)
    n_pts = geom.grid_shape[0] * geom.grid_shape[1]
    speed = rng.uniform(0.5, 25.0, (6, n_pts)).astype(np.float32)
    direction = rng.uniform(0.0, 360.0, (6, n_pts)).astype(np.float32)

    u_old, v_old = stencil_winds(interp, speed, direction)
    u_new, v_new = stencil_uv(interp, *speeddir_to_uv(speed, direction))
    assert np.array_equal(u_old, u_new)
    assert np.array_equal(v_old, v_new)


def test_same_tile_id_in_different_domains_gets_a_different_cache_key():
    """Regression: tile ids repeat across domains, so they cannot key the terrain cache.

    `t_008_010` is a lattice index, not a place. The WSW domain and the Panhandle
    domain both generate it, and keying the cache on the id alone served 23 tiles of
    New Mexico DEM (~1900 m) to a Texas site (~1000 m) -- corrupting the model's dem
    channel, not just the WRG elevation column, with no error anywhere.
    """
    wsw = Tile("t_008_010", x=447510.0, y=3786960.0, crs="EPSG:32613")
    panhandle = Tile("t_008_010", x=255450.0, y=3953100.0, crs="EPSG:32614")
    assert wsw.tile_id == panhandle.tile_id
    assert wsw.geo_key != panhandle.geo_key


def test_geo_key_is_stable_against_float_noise_but_tracks_real_moves():
    """A key that changed on rounding noise would silently rebuild terrain forever."""
    a = Tile("t", x=255450.0, y=3953100.0)
    b = Tile("t", x=255450.0000001, y=3953099.9999999)
    assert a.geo_key == b.geo_key
    assert Tile("t", x=255480.0, y=3953100.0).geo_key != a.geo_key
    # The CRS is part of the identity: the same numbers mean different places.
    # (a uses the WORKING_CRS default, so name a different zone explicitly.)
    assert Tile("t", x=255450.0, y=3953100.0, crs="EPSG:32614").geo_key != a.geo_key
