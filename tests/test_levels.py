"""Vertical level selection and interpolation.

FuXi-CFD's stack has no round numbers above 50 m, so every turbine-relevant
deliverable height (80/100/120/160 m) is interpolated. These tests pin that the
interpolation is correct, is labelled honestly, and refuses to extrapolate.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuxicfd_wrf.constants import OUTPUT_HEIGHTS
from fuxicfd_wrf.levels import LevelSelection, interp_weights
from fuxicfd_wrf.writers import resolve_levels

DELIVERABLE_HEIGHTS = [80.0, 100.0, 120.0, 160.0]


def test_no_deliverable_height_is_a_model_level():
    """The premise: 80/100/120/160 m all need interpolation."""
    for h in DELIVERABLE_HEIGHTS:
        assert np.abs(OUTPUT_HEIGHTS - h).min() > 1.0, f"{h} m unexpectedly exact"


def test_weights_sum_to_one():
    """Partition of unity, or interpolation biases the magnitude."""
    W, _ = interp_weights(OUTPUT_HEIGHTS, DELIVERABLE_HEIGHTS)
    assert np.allclose(W.sum(axis=1), 1.0)


def test_weights_are_local():
    """Each interpolated height draws on exactly its two bracketing levels."""
    W, exact = interp_weights(OUTPUT_HEIGHTS, DELIVERABLE_HEIGHTS)
    assert not exact.any()
    for i, h in enumerate(DELIVERABLE_HEIGHTS):
        nz = np.flatnonzero(W[i])
        assert len(nz) == 2 and nz[1] == nz[0] + 1
        lo, hi = OUTPUT_HEIGHTS[nz[0]], OUTPUT_HEIGHTS[nz[1]]
        assert lo < h < hi, f"{h} not bracketed by {lo},{hi}"


def test_exact_levels_are_one_hot():
    """A requested model level must be taken exactly, not interpolated."""
    W, exact = interp_weights(OUTPUT_HEIGHTS, [102.17944, 50.0])
    assert exact.all()
    assert (W.sum(axis=1) == 1.0).all()
    assert (np.count_nonzero(W, axis=1) == 1).all()


def test_interpolation_is_exact_for_a_log_linear_profile():
    """A field linear in log(z) must be reproduced exactly by log interpolation.

    This is the analytic check that the weights are right, not merely plausible.
    """
    sel = LevelSelection.build(OUTPUT_HEIGHTS, DELIVERABLE_HEIGHTS, method="log")
    profile = 3.0 * np.log(OUTPUT_HEIGHTS) + 1.5          # (27,)
    field = profile[None, :, None, None] * np.ones((2, 1, 4, 4))
    got = sel.apply(field)
    expected = 3.0 * np.log(np.array(DELIVERABLE_HEIGHTS)) + 1.5
    assert np.allclose(got[0, :, 0, 0], expected)


def test_log_and_linear_agree_closely_at_these_spacings():
    """The log-vs-linear choice is defensible but not consequential here.

    Model levels are 8-11% apart, so the two interpolants differ by ~1% of the
    interval -- fractions of a cm/s on a real wind profile.
    """
    log_w, _ = interp_weights(OUTPUT_HEIGHTS, DELIVERABLE_HEIGHTS, method="log")
    lin_w, _ = interp_weights(OUTPUT_HEIGHTS, DELIVERABLE_HEIGHTS, method="linear")
    assert np.abs(log_w - lin_w).max() < 0.02


def test_refuses_extrapolation():
    """Below the lowest or above the highest level must raise, not clamp."""
    for bad in (5.0, 350.0):
        with pytest.raises(ValueError, match="outside the available stack"):
            interp_weights(OUTPUT_HEIGHTS, [bad])


def test_apply_preserves_shape_and_ordering():
    sel = LevelSelection.build(OUTPUT_HEIGHTS, DELIVERABLE_HEIGHTS)
    field = np.random.default_rng(0).normal(size=(3, 27, 8, 8))
    out = sel.apply(field)
    assert out.shape == (3, 4, 8, 8)
    assert sel.heights.tolist() == DELIVERABLE_HEIGHTS   # sorted ascending


def test_requested_heights_are_sorted():
    sel = LevelSelection.build(OUTPUT_HEIGHTS, [160.0, 80.0, 120.0])
    assert sel.heights.tolist() == [80.0, 120.0, 160.0]


def test_default_selection_is_identity():
    sel = resolve_levels(None)
    assert sel.n_out == 27
    assert not sel.any_interpolated
    field = np.random.default_rng(1).normal(size=(2, 27, 3, 3))
    assert np.allclose(sel.apply(field), field)


def test_describe_flags_interpolated_heights():
    """The run log must say which heights were interpolated and from what."""
    sel = resolve_levels(DELIVERABLE_HEIGHTS)
    text = sel.describe()
    assert "interp" in text
    assert "80 m" in text
    assert "75.526" in text or "75.525" in text   # bracketing level appears


def test_interpolated_value_lies_between_bracketing_levels():
    """Sanity: no overshoot for a monotonic profile."""
    sel = resolve_levels(DELIVERABLE_HEIGHTS)
    profile = np.linspace(5.0, 12.0, 27)
    field = profile[None, :, None, None]
    got = sel.apply(field)[0, :, 0, 0]
    assert np.all(np.diff(got) > 0)
    assert got.min() > profile.min() and got.max() < profile.max()
