"""Drive a WRG export from a FuXi-CFD Zarr store.

Reads either output layout:

* **tile store** -- `<store>/<tile_id>/{u,v}`. Tiles are sampled onto the WRG grid
  and blended where they overlap, so no full-domain mosaic has to be materialised
  first. For a WRG you need one level and two variables, so going straight from
  tiles avoids writing a multi-TB intermediate.
* **mosaic store** -- `<store>/{u,v}` already on a uniform grid; sampled directly.

The time axis is streamed in blocks. Per block the u/v fields are blended in
*component* space (never speed/direction -- see winds.py), converted to speed and
direction, and folded into per-point sector histograms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from .constants import MODEL_SHAPE, TILE_SPAN_M
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
        gx, gy = np.meshgrid(self.x, self.y)
        return gx, gy


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
    """Weights mapping a source grid (ascending sx, descending sy) onto `grid`."""
    inside_x = np.flatnonzero((grid.x >= sx[0]) & (grid.x <= sx[-1]))
    inside_y = np.flatnonzero((grid.y <= sy[0]) & (grid.y >= sy[-1]))
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


def _open(store_path: Path):
    root = zarr.open_group(str(store_path), mode="r")
    is_mosaic = bool(root.attrs.get("mosaic", False))
    return root, is_mosaic


def _tile_from_group(g) -> Tile:
    return Tile(
        tile_id=str(g.attrs["tile_id"]),
        x=float(g.attrs["center_x"]),
        y=float(g.attrs["center_y"]),
        crs=str(g.attrs["crs"]),
    )


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
) -> dict:
    """Build a WRG file from a FuXi-CFD store."""
    store_path = Path(store_path)
    root, is_mosaic = _open(store_path)

    levels = np.asarray(root["level"])
    # Interpolate to the requested hub height rather than snapping to the nearest
    # stored level. Snapping and then labelling the file "80 m" would misreport
    # what it contains; interpolating and labelling it 80 m is accurate.
    if levels.size == 1:
        sel = None
        actual_h = float(levels[0])
        if abs(actual_h - hub_height) > EXACT_TOL_M:
            print(
                f"[wrg] NOTE: store holds only {actual_h:.5f} m, so {hub_height} m "
                "cannot be interpolated. Using the single available level."
            )
    else:
        sel = LevelSelection.build(levels, [hub_height])
        actual_h = float(hub_height)
        print(f"[wrg] vertical: {sel.describe()}")
    _, model_h = nearest_model_level(hub_height)
    times = from_epoch_ns(np.asarray(root["time"]))
    n_times = len(times)

    n_sectors = len(sector_edges(bin_size_deg)) - 1

    # --- target grid + samplers ---------------------------------------------
    if is_mosaic:
        sx = np.asarray(root["x"])
        sy = np.asarray(root["y"])
        crs = str(root.attrs.get("crs", "EPSG:32613"))
        grid = build_wrg_grid(
            sx.min(), sy.min(), sx.max(), sy.max(),
            res_m=res_m, buffer=buffer, crs=crs,
        )
        sources = [("__mosaic__", _make_sampler(grid, sx, sy, overlap_px=0))]
        tiles = []
    else:
        tile_ids = [k for k in root.group_keys()]
        tiles = [_tile_from_group(root[t]) for t in tile_ids]
        if not tiles:
            raise ValueError(f"no tile groups found in {store_path}")
        crs = tiles[0].crs
        half = TILE_SPAN_M / 2
        grid = build_wrg_grid(
            min(t.x for t in tiles) - half, min(t.y for t in tiles) - half,
            max(t.x for t in tiles) + half, max(t.y for t in tiles) + half,
            res_m=res_m, buffer=buffer, crs=crs,
        )
        # Overlap width in WRG pixels, inferred from the tightest tile spacing.
        xs = sorted({t.x for t in tiles})
        stride = min((b - a for a, b in zip(xs, xs[1:])), default=TILE_SPAN_M)
        overlap_px = max(0, int(round((TILE_SPAN_M - stride) / res_m)))
        sources = []
        for tid, tile in zip(tile_ids, tiles):
            tx, ty = tile.model_axes()
            sources.append((tid, _make_sampler(grid, tx, ty, overlap_px=overlap_px)))

    hist = SectorHistograms(grid.n_points, n_sectors)
    gb = hist.nbytes() / 1e9
    if gb > max_hist_gb:
        raise MemoryError(
            f"sector histograms need {gb:.1f} GB for a {grid.shape} grid at "
            f"{res_m} m. Increase --resolution (WRG files are rarely finer than "
            f"100 m) or raise max_hist_gb."
        )
    print(
        f"[wrg] grid {grid.shape} @ {res_m} m = {grid.n_points:,} points, "
        f"{n_sectors} sectors, histograms {gb:.2f} GB"
    )

    # --- stream time ---------------------------------------------------------
    ny, nx = grid.shape
    for t0 in range(0, n_times, time_block):
        t1 = min(t0 + time_block, n_times)
        nt = t1 - t0
        acc_u = np.zeros((nt, ny, nx), dtype=np.float32)
        acc_v = np.zeros((nt, ny, nx), dtype=np.float32)
        wsum = np.zeros((ny, nx), dtype=np.float32)

        for tid, smp in sources:
            if smp is None:
                continue
            src = root if tid == "__mosaic__" else root[tid]
            if sel is None:
                u = np.asarray(src["u"][t0:t1, 0], dtype=np.float32)
                v = np.asarray(src["v"][t0:t1, 0], dtype=np.float32)
            else:
                u = sel.apply(np.asarray(src["u"][t0:t1], dtype=np.float32))[:, 0]
                v = sel.apply(np.asarray(src["v"][t0:t1], dtype=np.float32))[:, 0]
            acc_u[:, smp.row, smp.col] += smp.sample(u) * smp.weight
            acc_v[:, smp.row, smp.col] += smp.sample(v) * smp.weight
            wsum[smp.row, smp.col] += smp.weight

        safe = np.where(wsum > 0, wsum, np.nan)
        acc_u /= safe
        acc_v /= safe
        speed, direction = uv_to_speeddir(acc_u, acc_v)
        hist.add(speed.reshape(nt, -1), direction.reshape(nt, -1))
        print(f"[wrg]   accumulated {t1}/{n_times} timesteps", flush=True)

    # --- elevation -----------------------------------------------------------
    elevation = _elevation_on_grid(grid, tiles, terrain_cache)

    records = build_records(
        hist, elevation=elevation, hub_height=actual_h,
        air_density=air_density,
    )
    gx, gy = grid.meshed()
    # Column-major so X varies slowest, matching the reference writer's ordering.
    out = write_wrg(
        out_path,
        gx.ravel(order="F"),
        gy.ravel(order="F"),
        _reorder_f(records, ny, nx),
        resolution_m=res_m,
    )
    covered = int(np.isfinite(records[:, 2]).sum())
    print(f"[wrg] wrote {out} -- {covered:,}/{grid.n_points:,} points with a fit")
    return {
        "path": str(out),
        "grid_shape": list(grid.shape),
        "n_points": grid.n_points,
        "resolution_m": res_m,
        "level_m": actual_h,
        "model_level_m": model_h,
        "n_sectors": n_sectors,
        "n_times": n_times,
        "time_range": [str(times.min()), str(times.max())],
        "points_with_fit": covered,
        "power_density": "standard_W_m2 (0.5*rho*mean(U^3))",
        "air_density": air_density,
    }


def _reorder_f(records: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Reorder point-major records from C (row-major) to F (column-major) order."""
    return records.reshape(ny, nx, -1).transpose(1, 0, 2).reshape(ny * nx, -1)


def _elevation_on_grid(
    grid: WrgGrid, tiles: list[Tile], terrain_cache: Path | str | None
) -> np.ndarray:
    """Sample cached SRTM tile DEMs onto the WRG grid.

    Returns zeros (with a warning) if the terrain cache is unavailable -- a WRG
    with zero elevation is still usable for wind statistics, and failing the whole
    export over a missing cache would be worse.
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
        raw = cache / "tiles" / f"{tile.tile_id}_raw.npz"
        if not raw.exists():
            continue
        dem = build_terrain(tile, cache).dem.astype(np.float32)
        tx, ty = tile.terrain_axes()
        smp = _make_sampler(grid, tx, ty, overlap_px=0)
        if smp is None:
            continue
        acc[smp.row, smp.col] += smp.sample(dem[None])[0]
        wsum[smp.row, smp.col] += 1.0
    if wsum.max() == 0:
        print("[wrg] WARNING: no cached tile DEMs found; elevation column will be 0")
        return np.zeros(grid.n_points)
    elev = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), 0.0)
    return _reorder_f(elev.reshape(-1, 1), ny, nx).ravel()
