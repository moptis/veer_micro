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

    WorldCover's single built-up class forced rural verges to the urban z0 of
    0.8 m, producing 3.8-sigma single-pixel spikes in the model's input.
    """
    z = [NLCD_Z0[c] for c in (21, 22, 23, 24)]
    assert z == sorted(z), "z0 must increase with development intensity"
    assert z[0] < WORLDCOVER_Z0[50] / 10, (
        "developed-open-space must be far below WorldCover's blanket built-up value"
    )
    assert z[-1] >= WORLDCOVER_Z0[50], "high-intensity urban should be at least as rough"


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
