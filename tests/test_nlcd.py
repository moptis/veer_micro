"""NLCD land-cover source and its roughness mapping."""

from __future__ import annotations

import numpy as np
import pytest

from fuxicfd_wrf.terrain import nlcd
from fuxicfd_wrf.terrain.roughness import (
    NLCD_CLASS_NAMES, NLCD_Z0, WORLDCOVER_Z0, class_histogram, classes_to_z0,
)


def test_developed_classes_follow_wrfs_collapsed_gradient():
    """Development is only a two-step gradient once WRF has had its way with it.

    NLCD resolves four development intensities, and that was a stated reason for
    preferring it over WorldCover's single built-up class: farm tracks and verges
    stop inheriting a city-centre z0, which is what produced single-pixel spikes.

    Matching WRF gives that up, deliberately. Noah remaps NLCD40 categories 23 and
    25 onto 24 at init (module_physics_init.F:3268), so open space, low and medium
    intensity all run at 0.70 m and only high intensity stands apart at 2.00. The
    four-step gradient collapses to two, and open space at 0.70 is no longer below
    WorldCover's blanket 0.80 -- the earlier form of this test asserted exactly
    that and now fails by design.

    The trade is pipeline consistency: these are the values the surface actually
    had in the WRF run whose winds drive the model. Revisit if a domain with real
    development is run, where reintroducing the speckle would matter -- WSW is
    ~0% developed, which is why it is acceptable here.
    """
    z = [NLCD_Z0[c] for c in (21, 22, 23, 24)]
    assert z == sorted(z), "z0 must not decrease with development intensity"
    assert z[0] == z[1] == z[2], "Noah collapses 23 and 25 onto 24: expect one value"
    assert z[3] > z[2], "high intensity must remain distinct"
    assert z[-1] > z[0] * 2, "the developed gradient must actually span a range"


def test_every_class_has_a_name_and_a_z0():
    assert set(NLCD_Z0) == set(NLCD_CLASS_NAMES)
    assert all(v > 0 for v in NLCD_Z0.values())


def test_classes_to_z0_uses_the_selected_table():
    """Code 21 means different things in the two schemes; the source must decide.

    In WorldCover 21 is unmapped (fallback); in NLCD it is developed open space.
    """
    codes = np.array([[21, 71], [82, 42]])
    nl = classes_to_z0(codes, "nlcd")
    assert nl[0, 0] == pytest.approx(NLCD_Z0[21])
    assert nl[1, 1] == pytest.approx(NLCD_Z0[42])

    wc = classes_to_z0(np.array([[10, 30]]), "worldcover")
    assert wc[0, 0] == pytest.approx(WORLDCOVER_Z0[10])


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown land-cover source"):
        classes_to_z0(np.zeros((2, 2)), "corine")


def test_class_histogram_respects_source():
    h = class_histogram(np.full((10, 10), 42), "nlcd")
    assert "evergreen forest" in h


def test_url_shape():
    u = nlcd.url_for(2023)
    assert u.startswith("/vsis3/usgs-landcover/")
    assert "Annual_NLCD_LndCov_2023_CU_C1V0.tif" in u


# --- z0 table provenance and cache keying ------------------------------------

def test_shrub_is_smoother_than_grass_as_wrf_has_it():
    """The pairing that dominates US domains -- and the two tables disagree on sign.

    Shrub/scrub and grassland cover ~95% of WSW, so their relative roughness sets
    the spatial pattern over nearly the whole domain. UL's table made shrub the
    rougher of the two (0.10 vs 0.06); WRF's VEGPARM makes it markedly smoother
    (0.03 vs 0.11). Those are not a matter of degree -- they put the roughness
    gradient in opposite directions across the dominant land-cover boundary.

    Pinning the WRF ordering guards against silently reverting to a table that
    would anti-correlate the static channel with the winds driving it.
    """
    import numpy as np
    from fuxicfd_wrf.terrain.roughness import classes_to_z0

    z = classes_to_z0(np.array([[52, 71]], dtype=np.int32), "nlcd")
    assert (z[0, 0], z[0, 1]) == (np.float32(0.03), np.float32(0.11))
    assert z[0, 0] < z[0, 1], "WRF has shrub/scrub smoother than grassland"


def test_fingerprint_changes_when_a_z0_value_changes():
    """Derived terrain is cached on disk, so the cache key must see value edits.

    Keying on the source name alone is what would have served tensors built with
    the old table after UL's values replaced them.
    """
    from fuxicfd_wrf.terrain import roughness as R

    before = R.table_fingerprint("nlcd")
    original = R.NLCD_Z0[71]
    try:
        R.NLCD_Z0[71] = original + 0.01
        assert R.table_fingerprint("nlcd") != before
    finally:
        R.NLCD_Z0[71] = original
    assert R.table_fingerprint("nlcd") == before
    assert R.table_fingerprint("nlcd") != R.table_fingerprint("worldcover")


def test_static_cache_key_carries_the_table_fingerprint():
    import inspect
    from fuxicfd_wrf.terrain import build
    src = inspect.getsource(build.static_channels)
    assert "table_fingerprint" in src


def test_constant_mode_zeroes_the_normalized_channel():
    """`constant` must land the channel at exactly 0 sigma, not merely flatten it.

    The point of the mode is attribution: with roughness carrying no signal, any
    surviving spatial structure is attributable to terrain and the wind stencil.
    That only holds if the constant is the *training mean* -- flattening to the
    tile mean would leave a non-zero offset, and flattening to 0.0 would sit at
    -4.05 sigma, further off-distribution than any real surface.
    """
    import numpy as np
    from fuxicfd_wrf.terrain.build import TerrainTile, apply_modes

    in_stats = {"high_mean": np.array([351.92178, 0.8173138]),
                "high_std": np.array([227.55724, 0.20186852])}
    t = TerrainTile(tile_id="t", dem=np.full((301, 301), 1955.0),
                    roughness=np.random.default_rng(0).uniform(0.03, 1.2, (301, 301)),
                    landcover=np.full((301, 301), 52, dtype=np.int32))
    _, z0 = apply_modes(t, in_stats, dem_mode="recentered", roughness_mode="constant")
    assert z0.std() == 0.0
    normalized = (z0 - in_stats["high_mean"][1]) / in_stats["high_std"][1]
    assert np.allclose(normalized, 0.0)
