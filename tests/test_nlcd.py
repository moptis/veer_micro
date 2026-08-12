"""NLCD land-cover source and its roughness mapping."""

from __future__ import annotations

import numpy as np
import pytest

from fuxicfd_wrf.terrain import nlcd
from fuxicfd_wrf.terrain.roughness import (
    NLCD_CLASS_NAMES, NLCD_Z0, WORLDCOVER_Z0, class_histogram, classes_to_z0,
)


def test_developed_classes_are_ordered_by_intensity():
    """The whole point of NLCD here: development is a gradient, not a switch.

    WorldCover's single built-up class sends everything from a farm track to a
    city centre to one z0, producing single-pixel spikes in the model's input.
    Whatever table is in use, NLCD must at least resolve that into a monotone
    gradient with open space well below the blanket value.

    The margin is deliberately loose. This assertion originally demanded open
    space be >10x below WorldCover's 0.8, which was really a restatement of the
    unsourced legacy value of 0.05. UL puts it at 0.20 -- a 4x reduction, not
    16x -- so the strong form was asserting a guess, not a property.
    """
    z = [NLCD_Z0[c] for c in (21, 22, 23, 24)]
    assert z == sorted(z), "z0 must increase with development intensity"
    assert z[0] < WORLDCOVER_Z0[50] / 2, (
        "developed-open-space must sit meaningfully below WorldCover's blanket "
        "built-up value, or NLCD buys nothing at the speckle"
    )
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

def test_grassland_and_shrub_carry_ULs_values():
    """The pairing that dominates US domains, and the reason the table changed.

    The removed in-house table put grassland at 0.03 while agreeing with UL on
    shrub/scrub at 0.10 -- roughly double the roughness contrast across the two
    classes covering 92% of WSW. Pinning both guards against drifting back.
    """
    import numpy as np
    from fuxicfd_wrf.terrain.roughness import classes_to_z0

    z = classes_to_z0(np.array([[52, 71]], dtype=np.int32), "nlcd")
    assert (z[0, 0], z[0, 1]) == (np.float32(0.10), np.float32(0.06))
    assert np.log(z[0, 0] / z[0, 1]) < 0.6


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
