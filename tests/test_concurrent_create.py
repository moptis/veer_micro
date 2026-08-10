"""Concurrent store creation must be safe.

Regression for the first production launch, where 11 of 16 Slurm array tasks died
within 11 seconds with::

    zarr.errors.ContainsArrayError: An array exists in store
      broadview_sites.zarr at path 'bv_center/k'

`_create_array` used a bare ``if name in group`` check, which is check-then-act:
several tasks see the node as absent, all try to create it, and every loser
raises. Losing that race is the normal outcome, not an error.

These tests use real processes, not threads. The GIL serialises enough of the
zarr metadata path that a threaded test can pass while the process case still
fails, so threads would give false confidence.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fuxicfd_wrf.grids import Tile
from fuxicfd_wrf.writers import TileStore, resolve_levels

N_WORKERS = 12


def _make_store(path: Path, n_times: int = 24, chunk: int = 6) -> TileStore:
    times = pd.DatetimeIndex(
        pd.date_range("2000-04-01", periods=n_times, freq="10min", tz="UTC")
    )
    tiles = [
        Tile("bv_center", x=675220.0, y=3854509.0),
        Tile("bv_north", x=675220.0, y=3860509.0),
    ]
    sel = resolve_levels([102.17944])
    return TileStore(
        path, times=times, tiles=tiles, levels=sel.heights,
        dtype="float16", time_chunk=chunk,
    )


def _worker_create(path_str: str) -> str:
    """Create the store; return 'ok' or the exception text."""
    try:
        _make_store(Path(path_str)).create()
        return "ok"
    except Exception as exc:  # noqa: BLE001 - we want the message
        return f"{type(exc).__name__}: {exc}"


def _worker_create_and_write(args) -> str:
    path_str, shard = args
    try:
        store = _make_store(Path(path_str))
        store.create()
        block = {
            v: np.full((6, 1, 300, 300), float(shard), dtype="float32")
            for v in ("u", "v", "w", "k")
        }
        store.write_block("bv_center", shard * 6, block)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


@pytest.mark.slow
def test_concurrent_create_all_succeed(tmp_path):
    """Every task must survive simultaneous creation of the same store."""
    path = str(tmp_path / "race.zarr")
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS) as pool:
        results = pool.map(_worker_create, [path] * N_WORKERS)

    failures = [r for r in results if r != "ok"]
    assert not failures, (
        f"{len(failures)}/{N_WORKERS} tasks failed creating the store "
        f"concurrently:\n  " + "\n  ".join(sorted(set(failures)))
    )


@pytest.mark.slow
def test_concurrent_create_yields_one_complete_store(tmp_path):
    """The racing tasks must converge on a single, structurally complete store."""
    import zarr

    path = str(tmp_path / "race2.zarr")
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS) as pool:
        pool.map(_worker_create, [path] * N_WORKERS)

    root = zarr.open_group(path, mode="r")
    assert sorted(root.group_keys()) == ["bv_center", "bv_north"]
    for tile in ("bv_center", "bv_north"):
        assert sorted(root[tile].array_keys()) == ["k", "u", "v", "w", "x", "y"]
    assert root["time"].shape == (24,)
    assert root["level"].shape == (1,)


@pytest.mark.slow
def test_concurrent_create_then_disjoint_writes(tmp_path):
    """The real pattern: create together, then write disjoint chunk-aligned shards."""
    import zarr

    path = str(tmp_path / "race3.zarr")
    n_shards = 4
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_shards) as pool:
        results = pool.map(
            _worker_create_and_write, [(path, s) for s in range(n_shards)]
        )
    failures = [r for r in results if r != "ok"]
    assert not failures, "concurrent create+write failed:\n  " + "\n  ".join(failures)

    u = np.asarray(zarr.open_group(path, mode="r")["bv_center"]["u"])
    # Each shard stamped its own index; no shard may have clobbered another.
    for shard in range(n_shards):
        block = u[shard * 6 : (shard + 1) * 6]
        assert np.allclose(block, shard), (
            f"shard {shard} chunk holds {np.unique(block)}, expected {shard} -- "
            "a concurrent write corrupted it"
        )
