"""Orchestration: WRF time series x tiles -> FuXi-CFD -> Zarr.

Work is organised as time-blocks, not as individual timesteps, because three
different costs all favour blocking:

* control.h5 chunks are (411, 81), so a one-timestep read costs the same I/O as a
  411-timestep read (see sources/h5_source.py).
* Per-tile static channels and interpolation weights are constant in time, so
  they are built once and reused across the whole block.
* GPU inference wants a batch, and (tile x time) pairs within a block supply one.

The only per-timestep work is a sparse matmul for the 9x9 stencil and a small zoom.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig
from .constants import MODEL_SHAPE, OUTPUT_VARS
from .grids import GEOGRAPHIC_CRS, DomainGeometry, Tile
from .model import assets as assets_mod
from .model import postprocess, preprocess
from .model.session import FuxiCFDSession
from .sources.base import WrfSource
from .sources.h5_source import H5WrfSource
from .sources.npz_source import NpzWrfSource
from .terrain.build import static_channels
from .winds import StencilInterpolator, build_stencil_interpolator, stencil_winds
from .levels import LevelSelection
from .writers import MosaicStore, TileStore, resolve_levels, shard_bounds


def open_source(cfg: RunConfig) -> WrfSource:
    if cfg.source.kind == "h5":
        return H5WrfSource(cfg.source.path)
    return NpzWrfSource(cfg.source.path, coordinates_from=cfg.source.coordinates_from)


def build_tiles(cfg: RunConfig, geom: DomainGeometry) -> list[Tile]:
    if cfg.tiling.mode == "sites":
        tiles = [
            geom.tile_at_lonlat(s.lon, s.lat, tile_id=s.id) for s in cfg.tiling.sites
        ]
        rejected = [t.tile_id for t in tiles if not geom.covers_stencil(t)]
        if rejected:
            raise ValueError(
                f"site tiles fall outside the WRF domain (their 9x9 stencil needs "
                f"4 km of margin on every side): {rejected}"
            )
        return tiles
    stride = cfg.tiling.stride_m
    tiles = geom.make_tiles(stride_m=stride)
    if cfg.tiling.bbox is not None:
        bbox = cfg.tiling.bbox.as_tuple()
        n_full = len(tiles)
        tiles = geom.make_tiles(stride_m=stride, bbox=bbox)
        if not tiles:
            # An empty list alone does not distinguish "your box is outside the
            # domain" from "it is inside but too close to the edge for a 9x9
            # stencil", and those need different fixes. Report the reachable
            # range so the answer is in the error.
            full = geom.make_tiles(stride_m=stride)
            lo0, la0, lo1, la1 = _tiles_lonlat_extent(geom, full, stride)
            raise ValueError(
                f"tiling.bbox selected 0 tiles.\n"
                f"  requested : lon {bbox[0]:.4f}..{bbox[2]:.4f}  "
                f"lat {bbox[1]:.4f}..{bbox[3]:.4f}\n"
                f"  available : lon {lo0:.4f}..{lo1:.4f}  lat {la0:.4f}..{la1:.4f}\n"
                f"The available range is what the {len(full)} tiles fitting a full "
                f"9x9 km stencil inside the WRF domain cover, so it is inset ~4.5 km "
                f"from the domain edge."
            )
        lo0, la0, lo1, la1 = _tiles_lonlat_extent(geom, tiles, stride)
        print(f"[tiles] bbox selected {len(tiles)} of {n_full} domain tiles; "
              f"covering lon {lo0:.4f}..{lo1:.4f}  lat {la0:.4f}..{la1:.4f}")
        _check_bbox_coverage(geom, tiles, stride, bbox,
                             cfg.tiling.min_bbox_coverage)
    if cfg.tiling.mode == "domain":
        _check_lattice_is_rectangular(tiles)
    if cfg.tiling.max_tiles:
        tiles = tiles[: cfg.tiling.max_tiles]
    return tiles


def _tiles_lonlat_extent(geom: DomainGeometry, tiles: list[Tile], stride_m: float):
    """Geographic extent actually covered by a tile list's cores."""
    if not tiles:
        return (float("nan"),) * 4
    b = np.array([geom.core_lonlat_bounds(t, stride_m) for t in tiles])
    return float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())


def _check_bbox_coverage(geom, tiles, stride_m, bbox, minimum: float) -> float:
    """How much of the requested box the selected tiles actually cover.

    A bbox that overhangs the WRF domain is not an error -- tiles are selected by
    intersection, so the run simply gets fewer of them. But "fewer" and "half of
    what you asked for" deserve different reactions, and reporting only the
    covered extent (as this did originally) buries the difference: it takes
    comparing two lon/lat ranges by eye to notice that most of the request was
    unreachable. That is worth catching before the GPU hours, not after.

    Measured by sampling the requested box and testing membership in the union of
    tile cores. Sampling rather than polygon intersection because cores are
    rectangles in the projected CRS while the box is in lon/lat, so the exact
    intersection is a clipped quad union and not worth the dependency.
    """
    import pyproj

    minlon, minlat, maxlon, maxlat = bbox
    n = 200
    glon, glat = np.meshgrid(np.linspace(minlon, maxlon, n),
                             np.linspace(minlat, maxlat, n))
    tf = pyproj.Transformer.from_crs(GEOGRAPHIC_CRS, geom.crs, always_xy=True)
    px, py = tf.transform(glon.ravel(), glat.ravel())

    cores = np.array([geom.core_bounds(t, stride_m) for t in tiles])
    inside = np.zeros(px.size, dtype=bool)
    for cminx, cminy, cmaxx, cmaxy in cores:
        inside |= ((px >= cminx) & (px <= cmaxx) & (py >= cminy) & (py <= cmaxy))
    frac = float(inside.mean())

    print(f"[tiles] requested box coverage: {100 * frac:.1f}%")
    if frac >= 0.999:
        return frac

    # Name the edges that fell short, so the fix is obvious rather than a hunt.
    edges = []
    grid_ok = inside.reshape(n, n)
    if not grid_ok[:, 0].any():
        edges.append("west")
    if not grid_ok[:, -1].any():
        edges.append("east")
    if not grid_ok[0, :].any():
        edges.append("south")
    if not grid_ok[-1, :].any():
        edges.append("north")
    detail = (f" The {' and '.join(edges)} edge{'s' if len(edges) > 1 else ''} "
              f"of the box lies entirely outside the WRF domain." if edges else "")

    msg = (f"tiling.bbox: only {100 * frac:.1f}% of the requested box is covered "
           f"by the selected tiles.{detail} The rest is outside the WRF domain "
           f"(or too close to its edge for a 9x9 km stencil), so no run can "
           f"produce it.")
    if frac < minimum:
        raise ValueError(
            msg + f"\nThis is below tiling.min_bbox_coverage={minimum}. Fix the "
            f"box, or lower/disable the threshold (0.0) if partial is intended."
        )
    print(f"[tiles] WARNING: {msg}")
    return frac


def _check_lattice_is_rectangular(tiles: list[Tile]) -> int:
    """Warn when the selected tiles do not form a full rectangle.

    A WRG is a rectangular grid. The WRF domain is rotated in UTM, so its
    boundary cuts diagonally across the tile lattice and a selection near an edge
    comes back L-shaped -- which concatenates into a grid with holes. Discovering
    that at WRG time means the inference has already been paid for, so say it
    here instead.
    """
    rc = []
    for t in tiles:
        m = re.match(r"^t_(\d+)_(\d+)$", t.tile_id)
        if not m:
            return 0                    # site tiles have no lattice
        rc.append((int(m.group(1)), int(m.group(2))))
    if not rc:
        return 0
    have = set(rc)
    rows = {r for r, _ in rc}
    cols = {c for _, c in rc}
    missing = len(rows) * len(cols) - len(have)
    if missing:
        print(
            f"[tiles] NOTE: the {len(have)} selected tiles do not fill their "
            f"{len(rows)}x{len(cols)} bounding block ({missing} absent, cut off by "
            f"the rotated domain boundary). A WRG is rectangular, so `wrg-concat` "
            f"will need --largest-rect to trim to a complete block."
        )
    return missing


@dataclass
class Prepared:
    """Everything derived from config that does not depend on the time shard."""

    source: WrfSource
    geom: DomainGeometry
    tiles: list[Tile]
    rows: np.ndarray            # file row indices, chronological
    times: pd.DatetimeIndex
    levels: LevelSelection
    store: TileStore
    in_stats: dict
    out_stats: dict
    interpolators: dict[str, StencilInterpolator]
    statics: dict[str, np.ndarray]


def prepare(cfg: RunConfig, *, build_static: bool = True) -> Prepared:
    """Resolve geometry, tiles, times and per-tile static inputs."""
    src = open_source(cfg)
    lat, lon = src.coordinates()
    # Grid shape and UTM zone come from the store, not from constants -- domains
    # differ (Broadview 78x132 at ~800 m, WSW 96x114 at ~1000 m).
    shape = getattr(src, "domain_shape", None)
    geom = DomainGeometry(
        lat, lon,
        crs=cfg.tiling.working_crs,
        **({"grid_shape": shape} if shape else {}),
    )
    tiles = build_tiles(cfg, geom)
    if not tiles:
        raise ValueError("no tiles selected -- check tiling config against the domain extent")

    rows = src.select(
        start=cfg.time.start, end=cfg.time.end,
        stride=cfg.time.stride, limit=cfg.time.limit,
    )
    if rows.size == 0:
        raise ValueError(
            f"no timesteps in [{cfg.time.start}, {cfg.time.end}].\n" + src.time_report()
        )
    times = src.times[rows]

    A = assets_mod.resolve(cfg.model.assets_root)
    in_stats = A.load_input_scaler()
    out_stats = A.load_output_scaler()

    levels = resolve_levels(cfg.output.levels)
    if levels.any_interpolated:
        print(f"[run] output levels: {levels.describe()}")
    if cfg.output.mosaic:
        store = MosaicStore(
            cfg.output.path, times=times, tiles=tiles, levels=levels.heights,
            variables=cfg.output.variables, dtype=cfg.output.dtype,
            time_chunk=cfg.output.time_chunk, stride_m=cfg.tiling.stride_m,
        )
    else:
        store = TileStore(
            cfg.output.path, times=times, tiles=tiles, levels=levels.heights,
            variables=cfg.output.variables,
            dtype=cfg.output.dtype, time_chunk=cfg.output.time_chunk,
        )

    interpolators, statics = {}, {}
    for t in tiles:
        interpolators[t.tile_id] = build_stencil_interpolator(geom, t)
        if build_static:
            statics[t.tile_id] = static_channels(
                t, in_stats, cfg.terrain.cache_root,
                dem_mode=cfg.terrain.dem_mode,
                roughness_mode=cfg.terrain.roughness_mode,
                roughness_despeckle=cfg.terrain.roughness_despeckle,
                landcover_source=cfg.terrain.landcover_source,
            )

    return Prepared(
        source=src, geom=geom, tiles=tiles, rows=rows, times=times,
        levels=levels, store=store,
        in_stats=in_stats, out_stats=out_stats,
        interpolators=interpolators, statics=statics,
    )


def run(
    cfg: RunConfig,
    *,
    shard: int = 0,
    n_shards: int = 1,
    progress_every: int = 1,
) -> dict:
    """Execute one shard of the run.

    Shards split the time axis on Zarr chunk boundaries, so array tasks can write
    to the same store concurrently without locking.
    """
    prep = prepare(cfg)
    prep.store.create(config_json=cfg.model_dump_json())

    bounds = shard_bounds(len(prep.times), cfg.output.time_chunk, n_shards)
    if shard >= len(bounds):
        print(f"[run] shard {shard}/{n_shards}: nothing to do ({len(bounds)} non-empty shards)")
        return {"shard": shard, "n_written": 0}
    t_start, t_stop = bounds[shard]
    print(
        f"[run] shard {shard}/{n_shards}: times [{t_start}, {t_stop}) of "
        f"{len(prep.times)}, {len(prep.tiles)} tiles",
        flush=True,
    )

    session = FuxiCFDSession(
        assets_mod.resolve(cfg.model.assets_root).onnx, device=cfg.model.device
    )
    print(f"[run] {session.describe()}")
    if not session.on_gpu:
        print(
            "[run] WARNING: running on CPU (~19 s per tile-timestep). "
            "A production run needs a GPU node."
        )
    batch = session.max_batch(cfg.model.batch_size)

    n_written = 0
    t0 = time.time()
    # Read the WRF winds in large blocks, then walk output chunks inside them.
    #
    # control.h5 is chunked (411, 81), so any read touching one timestep pulls
    # ~68 MB off disk for both wind variables. Reading per output chunk
    # (time_chunk=6) therefore re-read each block ~137 times and pinned the first
    # production run at 4.8 tile-steps/s against ~42 on the GPU -- the pipeline was
    # I/O bound by an order of magnitude. Reading `read_block` rows at once
    # amortises that back down.
    read_block = max(cfg.model.read_block, cfg.output.time_chunk)
    read_block -= read_block % cfg.output.time_chunk      # keep chunk alignment
    cache_start, cache_speed, cache_dir = -1, None, None

    for b_start in range(t_start, t_stop, cfg.output.time_chunk):
        b_stop = min(b_start + cfg.output.time_chunk, t_stop)

        if cache_speed is None or b_start >= cache_start + read_block or b_start < cache_start:
            cache_start = b_start
            cache_stop = min(b_start + read_block, t_stop)
            cache_speed, cache_dir = prep.source.read_wind_100m(
                prep.rows[cache_start:cache_stop]
            )
        lo = b_start - cache_start
        speed = cache_speed[lo : lo + (b_stop - b_start)]
        direction = cache_dir[lo : lo + (b_stop - b_start)]

        per_tile = {}
        for tile in prep.tiles:
            interp = prep.interpolators[tile.tile_id]
            u9, v9 = stencil_winds(interp, speed, direction)
            winds = preprocess.normalize_winds(u9, v9, prep.in_stats)
            x = preprocess.assemble(winds, prep.statics[tile.tile_id])

            outs = []
            for i in range(0, x.shape[0], batch):
                raw = session.run(x[i : i + batch])
                outs.append(postprocess.denormalize(raw, prep.out_stats))
            pred = np.concatenate(outs, axis=0)  # (n, 27, 4, 300, 300)

            fields = postprocess.split(pred)
            fields = {
                v: prep.levels.apply(fields[v]) for v in cfg.output.variables
            }
            if cfg.output.mosaic:
                # Hold the block until every tile is in: a domain pixel may draw
                # from up to four overlapping tiles and must see all of them
                # before the weighted mean is normalised.
                per_tile[tile.tile_id] = fields
            else:
                prep.store.write_block(tile.tile_id, b_start, fields)

        if cfg.output.mosaic:
            prep.store.write_block(b_start, prep.store.blend(per_tile))

        n_written += b_stop - b_start
        if progress_every and (b_start // cfg.output.time_chunk) % progress_every == 0:
            done = b_stop - t_start
            rate = done / max(1e-9, time.time() - t0)
            eta = (t_stop - b_stop) / max(1e-9, rate)
            print(
                f"[run]   t={b_stop}/{t_stop}  {rate:.2f} timesteps/s  "
                f"({rate * len(prep.tiles):.2f} tile-steps/s)  ETA {eta / 60:.1f} min",
                flush=True,   # else this buffers and the log looks hung
            )

    elapsed = time.time() - t0
    print(
        f"[run] shard {shard} done: {n_written} timesteps x {len(prep.tiles)} tiles "
        f"in {elapsed / 60:.1f} min"
    )
    return {
        "shard": shard,
        "n_written": n_written,
        "n_tiles": len(prep.tiles),
        "elapsed_s": elapsed,
        "store": str(cfg.output.path),
    }


def plan(cfg: RunConfig) -> dict:
    """Dry run: report tiles, timesteps and output size without touching the GPU."""
    prep = prepare(cfg, build_static=False)
    itemsize = 2 if cfg.output.dtype == "float16" else 4
    per_tile_step = (
        prep.levels.n_out * len(cfg.output.variables) * MODEL_SHAPE[0] * MODEL_SHAPE[1] * itemsize
    )
    if cfg.output.mosaic:
        ny, nx = prep.store.ny, prep.store.nx
        itemsize2 = itemsize
        total = (
            prep.levels.n_out * len(cfg.output.variables) * ny * nx * itemsize2
            * len(prep.times)
        )
    else:
        total = per_tile_step * len(prep.tiles) * len(prep.times)
    return {
        "n_tiles": len(prep.tiles),
        "n_times": len(prep.times),
        "time_range": [str(prep.times.min()), str(prep.times.max())],
        "levels_m": prep.levels.heights.tolist(),
        "variables": cfg.output.variables,
        "dtype": cfg.output.dtype,
        "mb_per_tile_timestep": per_tile_step / 1e6,
        "total_gb": total / 1e9,
        "affine_residual_m": prep.geom.affine_residual_m,
        "tile_ids": [t.tile_id for t in prep.tiles[:10]],
        "mosaic": cfg.output.mosaic,
        "mosaic_shape": ([prep.store.ny, prep.store.nx] if cfg.output.mosaic else None),
    }
