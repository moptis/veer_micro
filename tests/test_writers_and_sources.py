"""Output store and time-handling tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fuxicfd_wrf.constants import N_LEVELS, OUTPUT_HEIGHTS
from fuxicfd_wrf.grids import Tile
from fuxicfd_wrf.sources.base import WrfSource
from fuxicfd_wrf.writers import (
    TileStore, cosine_taper, from_epoch_ns, resolve_levels, shard_bounds, to_epoch_ns,
)


# --- time encoding -----------------------------------------------------------

def test_epoch_ns_roundtrip_utc():
    times = pd.DatetimeIndex(pd.to_datetime(
        ["2000-04-01 00:00:00", "2013-03-02 19:50:00"], utc=True
    ))
    assert from_epoch_ns(to_epoch_ns(times)).equals(times)


def test_epoch_ns_survives_microsecond_resolution_index():
    """Regression: a microsecond-unit index must not be reinterpreted as nanoseconds.

    `DatetimeIndex.view("int64")` on a us-resolution index yields microsecond
    counts. Decoded as nanoseconds those become 1970 dates -- wrong, but plausible
    enough to ship unnoticed. This is exactly what happened in the first smoke run.
    """
    us = pd.DatetimeIndex(
        pd.to_datetime(["2000-04-01 00:00:00"], utc=True)
    ).as_unit("us")
    decoded = from_epoch_ns(to_epoch_ns(us))
    assert decoded[0].year == 2000, f"decoded to {decoded[0]}, expected year 2000"


# --- level selection ---------------------------------------------------------

def test_resolve_levels_default_is_all():
    sel = resolve_levels(None)
    assert sel.n_out == N_LEVELS
    assert np.allclose(sel.heights, OUTPUT_HEIGHTS)
    assert not sel.any_interpolated


def test_resolve_levels_exact_match():
    sel = resolve_levels([102.17944, 50.0])
    assert sel.heights.tolist() == [50.0, 102.17944]  # sorted by height
    assert sel.exact.all()


def test_resolve_levels_interpolates_non_model_height():
    """100 m is not a model level, so it is interpolated -- and labelled 100 m.

    Earlier this raised. Interpolating is the right behaviour (deliverables are
    specified at round heights); what must never happen is *snapping* to 102.18 m
    while still calling the field 100 m.
    """
    sel = resolve_levels([100.0])
    assert sel.heights.tolist() == [100.0]
    assert not sel.exact[0]
    assert np.count_nonzero(sel.weights[0]) == 2


# --- sharding ----------------------------------------------------------------

@pytest.mark.parametrize("n_times,chunk,n_shards", [(100, 6, 4), (52560, 6, 64), (7, 6, 3), (5, 6, 2)])
def test_shard_bounds_are_chunk_aligned_and_cover_everything(n_times, chunk, n_shards):
    bounds = shard_bounds(n_times, chunk, n_shards)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n_times
    for i, (a, b) in enumerate(bounds):
        assert a % chunk == 0, "shard start must be on a chunk boundary"
        if b != n_times:
            assert b % chunk == 0, "shard end must be on a chunk boundary or the array end"
        if i:
            assert a == bounds[i - 1][1], "shards must be contiguous with no gap"


def test_shard_bounds_handles_more_shards_than_chunks():
    """Asking for 100 shards of 2 chunks must not produce empty or duplicate work."""
    bounds = shard_bounds(10, 6, 100)
    assert len(bounds) == 2
    assert all(b > a for a, b in bounds)


# --- blending ----------------------------------------------------------------

def test_cosine_taper_is_zero_at_edges_and_one_inside():
    w = cosine_taper((100, 100), overlap_px=20)
    assert w[0, 50] < 0.01 and w[-1, 50] < 0.01
    assert w[50, 0] < 0.01 and w[50, -1] < 0.01
    assert w[50, 50] == pytest.approx(1.0)


def test_cosine_taper_complements_across_a_seam():
    """Two tiles overlapping by their taper width must sum to ~1 across the seam.

    Without this, a mosaic shows bright or dark bands wherever tiles meet.
    """
    n, ov = 200, 40
    w = cosine_taper((n, n), overlap_px=ov)[n // 2]
    left = w[n - ov:]           # right edge of the western tile
    right = w[:ov]              # left edge of the eastern tile
    assert np.allclose(left + right, 1.0, atol=1e-5)


def test_cosine_taper_disabled_when_no_overlap():
    assert np.all(cosine_taper((50, 50), overlap_px=0) == 1.0)


# --- store -------------------------------------------------------------------

def _store(tmp_path, n_times=12, chunk=6):
    times = pd.DatetimeIndex(pd.date_range("2000-04-01", periods=n_times, freq="10min", tz="UTC"))
    tiles = [Tile("t0", x=500000.0, y=3850000.0)]
    sel = resolve_levels(None)
    return TileStore(
        tmp_path / "s.zarr", times=times, tiles=tiles, levels=sel.heights,
        dtype="float16", time_chunk=chunk,
    ), times


def test_store_roundtrip(tmp_path):
    import zarr

    store, times = _store(tmp_path)
    store.create()
    rng = np.random.default_rng(0)
    block = {v: rng.normal(size=(6, N_LEVELS, 300, 300)).astype("float32")
             for v in ["u", "v", "w", "k"]}
    store.write_block("t0", 0, block)

    root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
    assert from_epoch_ns(np.asarray(root["time"]))[0] == times[0]
    got = np.asarray(root["t0"]["u"][0:6])
    assert got.shape == (6, N_LEVELS, 300, 300)
    # float16 storage: agreement to ~3 decimal digits is the expected fidelity.
    assert np.allclose(got.astype("float32"), block["u"], atol=1e-2)


def test_store_rejects_unaligned_write(tmp_path):
    """A misaligned write is how two Slurm tasks would corrupt a shared chunk."""
    store, _ = _store(tmp_path)
    store.create()
    block = {v: np.zeros((6, N_LEVELS, 300, 300), dtype="float32") for v in ["u", "v", "w", "k"]}
    with pytest.raises(ValueError, match="not a multiple of time_chunk"):
        store.write_block("t0", 3, block)


def test_store_allows_short_final_block(tmp_path):
    """The last block may be shorter than a chunk without tripping the guard."""
    store, _ = _store(tmp_path, n_times=9, chunk=6)
    store.create()
    block = {v: np.zeros((3, N_LEVELS, 300, 300), dtype="float32") for v in ["u", "v", "w", "k"]}
    store.write_block("t0", 6, block)


# --- source time selection ---------------------------------------------------

class _FakeSource(WrfSource):
    """Mimics control.h5's out-of-order, discontinuous time index."""

    def __init__(self):
        self._times = pd.DatetimeIndex(pd.to_datetime([
            "2002-01-01 00:00", "2002-01-01 00:10",
            "2013-03-02 19:50",                      # backward jump, as in the real file
            "2002-01-01 00:20", "2018-09-30 23:50",
        ], utc=True))

    @property
    def times(self):
        return self._times

    def coordinates(self):
        raise NotImplementedError

    def read(self, variables, rows):
        raise NotImplementedError


def test_select_returns_chronological_rows_not_file_order():
    src = _FakeSource()
    rows = src.select()
    assert list(rows) == [0, 1, 3, 2, 4], "rows must be ordered by timestamp"
    assert src.times[rows].is_monotonic_increasing


def test_select_respects_time_window():
    src = _FakeSource()
    rows = src.select(start="2002-01-01 00:00", end="2002-01-01 00:15")
    assert list(rows) == [0, 1]


def test_select_stride_is_in_time_not_file_order():
    src = _FakeSource()
    rows = src.select(stride=2)
    assert list(rows) == [0, 3, 4]
    assert src.times[rows].is_monotonic_increasing
