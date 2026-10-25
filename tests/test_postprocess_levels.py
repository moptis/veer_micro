"""``denormalize_levels`` must equal ``denormalize`` restricted to the same slice.

The WRG path uses it to avoid materialising 27 levels x 4 variables in float64
when it wants two planes, so any drift between the two would show up as a wrong
hub-height field rather than as an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuxicfd_wrf.constants import N_LEVELS, OUTPUT_VARS
from fuxicfd_wrf.levels import LevelSelection
from fuxicfd_wrf.model import postprocess

H = W = 8


def _stats(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "mean": rng.normal(size=(N_LEVELS, len(OUTPUT_VARS))),
        "std": rng.uniform(0.5, 2.0, size=(N_LEVELS, len(OUTPUT_VARS))),
    }


@pytest.mark.parametrize("lo,hi", [(0, 1), (19, 20), (15, 17), (0, N_LEVELS), (25, 27)])
@pytest.mark.parametrize("ndim", [4, 5])
def test_matches_full_denormalize(lo, hi, ndim):
    rng = np.random.default_rng(1)
    shape = (N_LEVELS, len(OUTPUT_VARS), H, W)
    if ndim == 5:
        shape = (3, *shape)
    pred = rng.normal(size=shape).astype(np.float32)
    st = _stats()

    full = postprocess.denormalize(pred, st)
    full = full[lo:hi] if ndim == 4 else full[:, lo:hi]
    assert np.allclose(postprocess.denormalize_levels(pred, st, lo, hi), full, atol=0)


def test_rejects_bad_slice():
    pred = np.zeros((N_LEVELS, len(OUTPUT_VARS), H, W), dtype=np.float32)
    for lo, hi in [(-1, 3), (5, 5), (0, N_LEVELS + 1), (4, 2)]:
        with pytest.raises(ValueError, match="level slice"):
            postprocess.denormalize_levels(pred, _stats(), lo, hi)


@pytest.mark.parametrize("height", [100.0, 80.0, 120.0, 160.0])
def test_sliced_selection_matches_full_pipeline(height):
    """End to end: slice + apply_sliced == denormalize + apply, for real heights."""
    from fuxicfd_wrf.constants import OUTPUT_HEIGHTS

    rng = np.random.default_rng(2)
    pred = rng.normal(size=(3, N_LEVELS, len(OUTPUT_VARS), H, W)).astype(np.float32)
    st = _stats(3)
    sel = LevelSelection.build(OUTPUT_HEIGHTS, [height])
    lo, hi = sel.support

    want = sel.apply(postprocess.split(postprocess.denormalize(pred, st))["u"])[:, 0]
    got = sel.apply_sliced(
        postprocess.split(postprocess.denormalize_levels(pred, st, lo, hi))["u"],
        lo, hi, level_axis=1,
    )[:, 0]
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12)
    # An exact level needs one plane; an interpolated one needs two. If this ever
    # widens, the memory argument for the sliced path quietly evaporates.
    assert hi - lo == (1 if height in (80.0, 100.0) else 2)
