"""Drive a WRG export from a FuXi-CFD Zarr store.

Reads either output layout:

* **tile store** -- `<store>/<tile_id>/{u,v}`. Tiles are sampled onto the WRG grid
  and blended where they overlap, so no full-domain mosaic has to be materialised
  first.
* **mosaic store** -- `<store>/{u,v}` already on a uniform grid; sampled directly.

Per time block the u/v fields are blended in *component* space (never
speed/direction -- see winds.py), converted to speed and direction, and folded
into per-point sector histograms.

Parallelism
-----------
Work is split over **time**, not space. Sector histograms are additive, so each
worker streams a disjoint time range into its own histogram and the parent sums
them -- an exact decomposition, not an approximation. (Contrast the reference
`create_wrg.py`, which parallelises over pixels because its per-pixel scipy fit is
the bottleneck. Here the fit is vectorised and takes seconds; the cost is
streaming.)

The bottleneck is I/O, and knowing why matters: stores are chunked
``(6, 27, 300, 300)`` -- the level axis is *not* chunked -- so reading a single
level still pulls all 27 off disk. A 52560-step single-level export therefore
moves ~511 GB. Slicing the level range in Python does not avoid that; only
re-chunking the store would. Until then, concurrent readers are what recovers the
throughput.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from .constants import TILE_SPAN_M
from .grids import Tile
from .levels import EXACT_TOL_M, LevelSelection
from .winds import uv_to_speeddir
from .wrg import (
    DEFAULT_AIR_DENSITY, DEFAULT_BIN_SIZE_DEG, SectorHistograms, build_records,
    nearest_model_level, sector_edges, write_wrg,
)
from .writers import _interp_matrix, cosine_taper, from_epoch_ns


@dataclass
class WrgGrid:
    """Uniform target grid for the WRG file."""

    x: np.ndarray   # ascending
    y: np.ndarray   # descending (north-up)
    res_m: float
    crs: str

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.y), len(self.x))

    @property
    def n_points(self) -> int:
        return len(self.x) * len(self.y)

    def meshed(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y)


def _buffer(lo: float, hi: float, frac: float) -> tuple[float, float]:
    span = hi - lo
    return lo + span * frac, hi - span * frac


def build_wrg_grid(
    minx: float, miny: float, maxx: float, maxy: float,
    *, res_m: float, buffer: float = 0.0, crs: str,
) -> WrgGrid:
    if buffer:
        minx, maxx = _buffer(minx, maxx, buffer)
        miny, maxy = _buffer(miny, maxy, buffer)
    x = np.arange(np.ceil(minx / res_m) * res_m, maxx + 1e-6, res_m)
    y = np.arange(np.ceil(miny / res_m) * res_m, maxy + 1e-6, res_m)[::-1]
    return WrgGrid(x=x, y=y, res_m=res_m, crs=crs)


@dataclass
class _Sampler:
    """Precomputed sampling of one source patch onto the WRG grid."""

    row: slice
    col: slice
    wy: np.ndarray
    wx: np.ndarray
    weight: np.ndarray

    def sample(self, src: np.ndarray) -> np.ndarray:
        """(n_time, h, w) -> (n_time, n_dst_row, n_dst_col)."""
        tmp = np.einsum("ij,njk->nik", self.wy, src)
        return np.einsum("nik,lk->nil", tmp, self.wx)


def _make_sampler(
    grid: WrgGrid, sx: np.ndarray, sy: np.ndarray, *, overlap_px: int
) -> _Sampler | None:
    # Include a source patch's own end points even when float error puts them a
    # hair inside. `Tile.model_axes` accumulates 300 steps of 30.1003 m and lands
    # ~1.5e-9 m short of the tile bound the output grid was built from, so a
    # strict `<=` dropped the grid's final column: on sww that was 1,500 of
    # 1,951,500 points written as uncovered, at the edge of a *fully covered*
    # domain. The tolerance is a millionth of a cell -- far below any real
    # geometry, and `_interp_matrix` clips the resulting index, so a point at the
    # boundary is evaluated exactly rather than extrapolated.
    tol_x = 1e-6 * (sx[-1] - sx[0]) / (len(sx) - 1)
    tol_y = 1e-6 * (sy[0] - sy[-1]) / (len(sy) - 1)
    inside_x = np.flatnonzero((grid.x >= sx[0] - tol_x) & (grid.x <= sx[-1] + tol_x))
    inside_y = np.flatnonzero((grid.y <= sy[0] + tol_y) & (grid.y >= sy[-1] - tol_y))
    if inside_x.size == 0 or inside_y.size == 0:
        return None
    res_x = (sx[-1] - sx[0]) / (len(sx) - 1)
    res_y = (sy[0] - sy[-1]) / (len(sy) - 1)
    wx = _interp_matrix((grid.x[inside_x] - sx[0]) / res_x, len(sx))
    wy = _interp_matrix((sy[0] - grid.y[inside_y]) / res_y, len(sy))
    weight = cosine_taper((inside_y.size, inside_x.size), overlap_px)
    return _Sampler(
        row=slice(inside_y[0], inside_y[-1] + 1),
        col=slice(inside_x[0], inside_x[-1] + 1),
        wy=wy, wx=wx, weight=weight,
    )


def _tile_from_group(g) -> Tile:
    return Tile(
        tile_id=str(g.attrs["tile_id"]),
        x=float(g.attrs["center_x"]),
        y=float(g.attrs["center_y"]),
        crs=str(g.attrs["crs"]),
    )


# --- shared setup -------------------------------------------------------------

def _build_context(spec: dict, *, quiet: bool = False) -> dict:
    """Resolve store, grid, samplers and level selection from a picklable spec.

    Both the parent and each spawned worker call this; the spec holds only plain
    values so it survives pickling, and rebuilding takes well under a second.
    """
    root = zarr.open_group(spec["store_path"], mode="r")
    is_mosaic = bool(root.attrs.get("mosaic", False))
    hub = spec["hub_height"]

    levels = np.asarray(root["level"])
    if levels.size == 1:
        sel, lev_lo, lev_hi = None, 0, 1
        actual_h = float(levels[0])
        if abs(actual_h - hub) > EXACT_TOL_M and not quiet:
            print(
                f"[wrg] NOTE: store holds only {actual_h:.5f} m, so {hub} m cannot "
                "be interpolated. Using the single available level."
            )
    else:
        sel = LevelSelection.build(levels, [hub])
        lev_lo, lev_hi = sel.support
        actual_h = float(hub)
        if not quiet:
            print(
                f"[wrg] vertical: {sel.describe()}"
                f"  [reading source levels {lev_lo}:{lev_hi} of {levels.size}]"
            )

    if is_mosaic:
        sx, sy = np.asarray(root["x"]), np.asarray(root["y"])
        crs = str(root.attrs.get("crs", "EPSG:32613"))
        grid = build_wrg_grid(sx.min(), sy.min(), sx.max(), sy.max(),
                              res_m=spec["res_m"], buffer=spec["buffer"], crs=crs)
        sources = [("__mosaic__", _make_sampler(grid, sx, sy, overlap_px=0))]
        tiles, tile_ids = [], []
    else:
        available = sorted(root.group_keys())
        if not available:
            raise ValueError(f"no tile groups found in {spec['store_path']}")
        requested = spec.get("tile_ids") or []
        if requested:
            missing = [t for t in requested if t not in available]
            if missing:
                raise ValueError(f"tiles {missing} not in store. Available: {available}")
            tile_ids = list(requested)
        else:
            tile_ids = available
        tiles = [_tile_from_group(root[t]) for t in tile_ids]
        crs = tiles[0].crs
        half = TILE_SPAN_M / 2
        # A single tile has no neighbours to blend with, so its unbuffered outer
        # margin has to be discarded rather than smoothed away.
        trim = spec["trim_edge_m"] if len(tiles) == 1 else 0.0
        if trim and not quiet:
            usable = (TILE_SPAN_M - 2 * trim) / 1000
            print(f"[wrg] trimming {trim:.0f} m from each edge: single-tile export, "
                  f"usable core {usable:.1f} x {usable:.1f} km")
        grid = build_wrg_grid(
            min(t.x for t in tiles) - half + trim, min(t.y for t in tiles) - half + trim,
            max(t.x for t in tiles) + half - trim, max(t.y for t in tiles) + half - trim,
            res_m=spec["res_m"], buffer=spec["buffer"], crs=crs,
        )
        xs = sorted({t.x for t in tiles})
        stride = min((b - a for a, b in zip(xs, xs[1:])), default=TILE_SPAN_M)
        overlap_px = max(0, int(round((TILE_SPAN_M - stride) / spec["res_m"])))
        sources = [
            (tid, _make_sampler(grid, *tile.model_axes(), overlap_px=overlap_px))
            for tid, tile in zip(tile_ids, tiles)
        ]

    return {
        "root": root, "is_mosaic": is_mosaic, "grid": grid, "sources": sources,
        "sel": sel, "lev_lo": lev_lo, "lev_hi": lev_hi, "actual_h": actual_h,
        "tiles": tiles, "tile_ids": tile_ids,
        "n_sectors": len(sector_edges(spec["bin_size_deg"])) - 1,
    }


# --- parallel accumulation ----------------------------------------------------

def _resolve_workers(n: int | None) -> int:
    """Default to the Slurm CPU allocation, capped so memory stays sane.

    Each worker holds its own histogram plus a ~512 MB bincount temporary, so
    more workers is not free.
    """
    if n and n > 0:
        return int(n)
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    avail = int(slurm) if slurm and slurm.isdigit() else (os.cpu_count() or 1)
    return max(1, min(avail, 12))


def _time_shards(n_times: int, n_workers: int) -> list[tuple[int, int]]:
    """Contiguous, non-overlapping time ranges -- one per worker."""
    if n_workers <= 1:
        return [(0, n_times)]
    per = int(np.ceil(n_times / n_workers))
    out = []
    for i in range(n_workers):
        a = i * per
        if a >= n_times:
            break
        out.append((a, min(a + per, n_times)))
    return out


def _accumulate_range(spec: dict, t0: int, t1: int, time_block: int, hist=None):
    """Stream [t0, t1) into a SectorHistograms."""
    ctx = _build_context(spec, quiet=True)
    grid, sources, sel = ctx["grid"], ctx["sources"], ctx["sel"]
    lev_lo, lev_hi, root = ctx["lev_lo"], ctx["lev_hi"], ctx["root"]
    if hist is None:
        hist = SectorHistograms(grid.n_points, ctx["n_sectors"])
    ny, nx = grid.shape

    for b0 in range(t0, t1, time_block):
        b1 = min(b0 + time_block, t1)
        nt = b1 - b0
        acc_u = np.zeros((nt, ny, nx), dtype=np.float32)
        acc_v = np.zeros((nt, ny, nx), dtype=np.float32)
        wsum = np.zeros((ny, nx), dtype=np.float32)

        for tid, smp in sources:
            if smp is None:
                continue
            src = root if tid == "__mosaic__" else root[tid]
            if sel is None:
                u = np.asarray(src["u"][b0:b1, 0], dtype=np.float32)
                v = np.asarray(src["v"][b0:b1, 0], dtype=np.float32)
            else:
                u = sel.apply_sliced(
                    np.asarray(src["u"][b0:b1, lev_lo:lev_hi], dtype=np.float32),
                    lev_lo, lev_hi)[:, 0]
                v = sel.apply_sliced(
                    np.asarray(src["v"][b0:b1, lev_lo:lev_hi], dtype=np.float32),
                    lev_lo, lev_hi)[:, 0]
            acc_u[:, smp.row, smp.col] += smp.sample(u) * smp.weight
            acc_v[:, smp.row, smp.col] += smp.sample(v) * smp.weight
            wsum[smp.row, smp.col] += smp.weight

        safe = np.where(wsum > 0, wsum, np.nan)
        acc_u /= safe
        acc_v /= safe
        speed, direction = uv_to_speeddir(acc_u, acc_v)
        hist.add(speed.reshape(nt, -1), direction.reshape(nt, -1))
        print(f"[wrg]   shard [{t0},{t1}): {b1 - t0}/{t1 - t0}", flush=True)
    return hist


def _worker_accumulate(payload):
    spec, t0, t1, time_block = payload
    h = _accumulate_range(spec, t0, t1, time_block)
    return h.counts, h.sum_u, h.sum_u3, h.n_samples


# --- helpers ------------------------------------------------------------------

def _reorder_f(records: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Reorder point-major records from C (row-major) to F (column-major) order."""
    return records.reshape(ny, nx, -1).transpose(1, 0, 2).reshape(ny * nx, -1)


def _elevation_on_grid(grid: WrgGrid, tiles: list[Tile], terrain_cache) -> np.ndarray:
    """Sample cached SRTM tile DEMs onto the WRG grid, in **row-major** order.

    Row-major matters: the sector histograms are indexed by `speed.reshape(nt, -1)`
    of a (ny, nx) field, i.e. C order, and `build_records` puts elevation in the
    same array as those statistics. `export_wrg` then reorders the whole record
    array C -> F once, to match the column-major point order the WRG format uses.

    Returning column-major here made elevation get reordered *twice* while every
    other column was reordered once, transposing the elevation field relative to
    its own coordinates -- 37 m RMS error on a 60x60 tile, and 7.8x steps at tile
    seams once concatenated. Keep this C-ordered and let the single conversion in
    `export_wrg` handle it.

    Returns zeros (with a warning) if unavailable -- a WRG with zero elevation is
    still usable for wind statistics, and failing the export over a missing cache
    would be worse.
    """
    ny, nx = grid.shape
    if not tiles or terrain_cache is None:
        print("[wrg] WARNING: no terrain cache given; elevation column will be 0")
        return np.zeros(grid.n_points)

    from .terrain.build import build_terrain

    cache = Path(terrain_cache)
    acc = np.zeros((ny, nx), dtype=np.float64)
    wsum = np.zeros((ny, nx), dtype=np.float64)
    for tile in tiles:
        # build_terrain names the raw cache `{tile_id}-{geo_key}_raw.npz` for
        # WorldCover and `{tile_id}-{geo_key}_raw_{source}.npz` otherwise. Both
        # halves of that stem matter, and both have been got wrong here before:
        # first the `_{source}` suffix, so every `landcover_source: nlcd` run
        # missed; then, when geo_key was introduced to stop one domain serving
        # another's terrain, the `-{geo_key}` infix broke the patterns again and
        # *every* run missed. Each time the WRG still wrote, with a silently
        # zeroed elevation column.
        #
        # Glob the whole stem so this tracks whatever `_cache` produces rather
        # than restating it. The DEM is identical whichever land-cover raster was
        # fetched, so any match serves.
        tile_cache = cache / "tiles"
        if not any(tile_cache.glob(f"{tile.tile_id}-{tile.geo_key}_raw*.npz")):
            continue
        dem = build_terrain(tile, cache).dem.astype(np.float32)
        smp = _make_sampler(grid, *tile.terrain_axes(), overlap_px=0)
        if smp is None:
            continue
        acc[smp.row, smp.col] += smp.sample(dem[None])[0]
        wsum[smp.row, smp.col] += 1.0
    if wsum.max() == 0:
        print("[wrg] WARNING: no cached tile DEMs found; elevation column will be 0")
        return np.zeros(grid.n_points)
    elev = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), 0.0)
    return elev.ravel()          # C order, same as the histogram indexing


# --- entry point --------------------------------------------------------------

def export_wrg(
    store_path: Path | str,
    out_path: Path | str,
    *,
    hub_height: float = 100.0,
    res_m: float = 100.0,
    bin_size_deg: float = DEFAULT_BIN_SIZE_DEG,
    buffer: float = 0.0,
    air_density: float = DEFAULT_AIR_DENSITY,
    time_block: int = 240,
    terrain_cache: Path | str | None = None,
    max_hist_gb: float = 8.0,
    tile_ids: list[str] | None = None,
    min_coverage: float = 0.98,
    allow_partial: bool = False,
    trim_edge_m: float = 1500.0,
    n_workers: int | None = None,
) -> dict:
    """Build a WRG file from a FuXi-CFD store.

    `tile_ids` restricts the export to a subset of tiles -- what site-mode runs
    want, since the grid then spans only those tiles rather than the bounding box
    of all of them.

    `trim_edge_m` discards the outer margin of a single-tile export. FuXi-CFD's
    training CFD had a 1.5 km buffer around its 9 km window and the deployed model
    never gets one, so tile edges are its least constrained region: measured on
    bv_center the elevation-removed residual is 0.044 m/s in the core and
    0.129 m/s in the outermost 300 m, 1.9x inflated over the outer 1.5 km. This is
    the same correction the reference `create_wrg.py` applies via `buffer=0.2`
    (0.2 x 9 km = 1.8 km). Overlapping tiles plus mosaic blending fix it without
    trimming, because each point is then carried by a tile interior.

    `min_coverage` refuses a grid the tiles do not fill. A WRG is a complete
    rectangular grid, so scattered site tiles produce a mostly-empty file -- five
    9 km tiles over the Broadview domain cover 6% of their own bounding box.
    """
    store_path = Path(store_path)
    spec = dict(
        store_path=str(store_path), tile_ids=list(tile_ids or []),
        hub_height=hub_height, res_m=res_m, bin_size_deg=bin_size_deg,
        buffer=buffer, trim_edge_m=trim_edge_m,
    )
    ctx = _build_context(spec)
    grid, sources = ctx["grid"], ctx["sources"]
    times = from_epoch_ns(np.asarray(ctx["root"]["time"]))
    n_times = len(times)

    ny, nx = grid.shape
    covered = np.zeros((ny, nx), dtype=bool)
    for _tid, _smp in sources:
        if _smp is not None:
            covered[_smp.row, _smp.col] = True
    coverage = float(covered.mean())
    if coverage < min_coverage and not allow_partial:
        raise ValueError(
            f"tiles cover only {coverage:.1%} of the {grid.shape} WRG grid "
            f"(threshold {min_coverage:.0%}).\n"
            "A WRG is a complete rectangular grid, so the remaining "
            f"{1 - coverage:.0%} would have no Weibull fit.\n"
            "Options:\n"
            "  * export one tile at a time:  --tile <tile_id>\n"
            "  * use a full-domain (mosaic or dense-tiling) run\n"
            "  * override deliberately:      --allow-partial"
        )
    if coverage < 1.0:
        print(f"[wrg] WARNING: only {coverage:.1%} of the grid is covered by tiles")

    hist = SectorHistograms(grid.n_points, ctx["n_sectors"])
    gb = hist.nbytes() / 1e9
    if gb > max_hist_gb:
        raise MemoryError(
            f"sector histograms need {gb:.1f} GB for a {grid.shape} grid at "
            f"{res_m} m. Increase --resolution or raise max_hist_gb."
        )
    workers = _resolve_workers(n_workers)
    shards = _time_shards(n_times, workers)

    # Budget up front. The first full-domain attempt died after 80 minutes
    # because only the histogram was counted: the bincount temporary and the
    # per-block u/v accumulators together were larger than it.
    per_worker = hist.estimate_peak_gb(time_block, grid.n_points)
    total = per_worker * len(shards) + 2 * gb          # + parent hist and one incoming
    print(f"[wrg] grid {grid.shape} @ {res_m} m = {grid.n_points:,} points, "
          f"{ctx['n_sectors']} sectors, histograms {gb:.2f} GB")
    print(f"[wrg] memory estimate: ~{per_worker:.1f} GB/worker x {len(shards)} "
          f"+ {2*gb:.1f} GB parent = ~{total:.0f} GB peak")
    slurm_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem and slurm_mem.isdigit():
        avail = int(slurm_mem) / 1024
        if total > 0.85 * avail:
            raise MemoryError(
                f"estimated peak ~{total:.0f} GB exceeds 85% of the {avail:.0f} GB "
                f"allocation. Reduce --workers, --time-block, or --resolution."
            )
        print(f"[wrg] allocation {avail:.0f} GB -- OK")
    print(f"[wrg] accumulating {n_times:,} timesteps over {len(shards)} worker(s)")

    if len(shards) == 1:
        hist = _accumulate_range(spec, shards[0][0], shards[0][1], time_block, hist)
    else:
        import multiprocessing as mp

        payloads = [(spec, a, b, time_block) for a, b in shards]
        # imap_unordered, not map: map materialises every worker's histogram in
        # the parent at once. Over a full domain each is GBs (1.08M points x 16
        # sectors x 100 bins), so 8 workers would land ~55 GB simultaneously.
        # Summing as results arrive keeps the parent at one extra histogram.
        with mp.get_context("spawn").Pool(len(shards)) as pool:
            done = 0
            for counts, su, su3, ns in pool.imap_unordered(
                _worker_accumulate, payloads
            ):
                # Histograms are additive: summing shards is exact.
                hist.counts += counts
                hist.sum_u += su
                hist.sum_u3 += su3
                hist.n_samples += ns
                del counts, su, su3, ns
                done += 1
                print(f"[wrg]   merged shard {done}/{len(shards)}", flush=True)

    elevation = _elevation_on_grid(grid, ctx["tiles"], terrain_cache)
    records = build_records(
        hist, elevation=elevation, hub_height=ctx["actual_h"], air_density=air_density
    )
    gx, gy = grid.meshed()
    out = write_wrg(
        out_path, gx.ravel(order="F"), gy.ravel(order="F"),
        _reorder_f(records, ny, nx), resolution_m=res_m,
    )
    n_fit = int(np.isfinite(records[:, 2]).sum())
    print(f"[wrg] wrote {out} -- {n_fit:,}/{grid.n_points:,} points with a fit")

    _, model_h = nearest_model_level(hub_height)
    return {
        "path": str(out),
        "grid_shape": list(grid.shape),
        "n_points": grid.n_points,
        "resolution_m": res_m,
        "level_m": ctx["actual_h"],
        "model_level_m": model_h,
        "n_sectors": ctx["n_sectors"],
        "n_times": n_times,
        "time_range": [str(times.min()), str(times.max())],
        "points_with_fit": n_fit,
        "grid_coverage": coverage,
        "tiles": ctx["tile_ids"] or None,
        "workers": len(shards),
        "power_density": "standard_W_m2 (0.5*rho*mean(U^3))",
        "air_density": air_density,
    }
