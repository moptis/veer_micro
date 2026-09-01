"""Downscale a coarse ``.wrg`` to a fine ``.wrg`` -- statistics in, statistics out.

One inference per sector per tile, instead of one per timestep. For a 57-tile
domain at 12 sectors that is 684 inferences, ~16 s at the measured 42
tile-steps/s on an H100, against ~20 H100-hours and 58 TB for the equivalent
timestep sweep. Nothing is streamed and nothing is stored in between, so none of
the sharding, resume, chunk-alignment or float16 machinery in ``pipeline.py`` and
``writers.py`` applies here.

Pipeline
--------
1. Read the coarse WRG; each sector becomes an inflow state (`WrgSectorSource`).
2. Tile the domain exactly as the timestep path does -- same lattice, same ids.
3. Per tile, run every state in one batch and reduce to speed + deflection at the
   target height (`transfer.sector_transfer`).
4. Blend tiles onto the output grid with the same cosine-squared taper the mosaic
   writer uses.
5. Interpolate the coarse sector Weibulls onto that grid, apply the transfer, and
   write the file.

Steps 3 and 4 are deliberately ordered that way: blending is linear, and the
nonlinear parts -- sector redistribution and moment matching -- happen once on the
final grid rather than once per tile.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DownscaleConfig
from .constants import OUTPUT_HEIGHTS
from .grids import DomainGeometry, Tile
from .levels import LevelSelection
from .model import assets as assets_mod
from .model import postprocess, preprocess
from .model.session import FuxiCFDSession
from .pipeline import build_tiles
from .sources.wrg_source import WrgSectorSource
from .terrain.build import static_channels
from .transfer import (
    apply_transfer, combine_sectors, power_density, sector_transfer, wrap180,
)
from .constants import TILE_SPAN_M
from .winds import build_stencil_interpolator, stencil_uv
from .wrg import write_wrg
from .wrg_export import (
    WrgGrid, _elevation_on_grid, _make_sampler, _reorder_f, build_wrg_grid,
)


@dataclass
class DownscaleResult:
    path: Path
    grid: WrgGrid
    n_tiles: int
    n_states: int
    elapsed_s: float
    diagnostics: dict


def _fine_grid(tiles: list[Tile], res_m: float, crs: str, buffer: float) -> WrgGrid:
    b = np.array([t.bounds() for t in tiles])
    return build_wrg_grid(
        float(b[:, 0].min()), float(b[:, 1].min()),
        float(b[:, 2].max()), float(b[:, 3].max()),
        res_m=res_m, buffer=buffer, crs=crs,
    )


def _coarse_on_grid(
    src: WrgSectorSource, grid: WrgGrid, field: np.ndarray
) -> np.ndarray:
    """Bilinearly interpolate an (n_sec, ny_c, nx_c) coarse field onto `grid`."""
    smp = _make_sampler(grid, src.x, src.y, overlap_px=0)
    if smp is None:
        raise ValueError(
            "the output grid does not intersect the coarse WRG at all -- check "
            "that source.crs matches the projection the coarse file was written in"
        )
    ny, nx = grid.shape
    out = np.full((field.shape[0], ny, nx), np.nan)
    out[:, smp.row, smp.col] = smp.sample(np.asarray(field, dtype=np.float64))
    n_gap = int(np.isnan(out[0]).sum())
    if n_gap:
        print(
            f"[downscale] WARNING: {n_gap:,}/{ny * nx:,} output points fall "
            "outside the coarse WRG and will be written as zeros"
        )
    return out


def run(cfg: DownscaleConfig) -> DownscaleResult:
    t0 = time.time()

    src = WrgSectorSource(
        cfg.source.path, crs=cfg.source.crs, speed_bins=cfg.source.speed_bins
    )
    print("[downscale] coarse source:\n" + src.describe())

    # The coarse WRG is already a regular lattice in its own projection, so
    # laying tiles out in that same CRS makes DomainGeometry's affine fit exact.
    # Anything else re-projects a grid that did not need re-projecting.
    crs = src.crs if cfg.tiling.working_crs == "auto" else cfg.tiling.working_crs
    lat, lon = src.coordinates()
    geom = DomainGeometry(
        lat, lon, crs=crs, grid_shape=src.domain_shape,
        affine_fit=cfg.tiling.affine_fit,
    )
    print(
        f"[downscale] geometry: {src.domain_shape[0]}x{src.domain_shape[1]} in "
        f"{geom.crs}, affine residual {geom.affine_residual_m:.2f} m"
    )

    tiles = build_tiles(cfg, geom)
    grid = _fine_grid(tiles, cfg.output.resolution_m, geom.crs, cfg.output.buffer)
    ny, nx = grid.shape
    print(
        f"[downscale] {len(tiles)} tiles -> {ny} x {nx} grid @ "
        f"{cfg.output.resolution_m:g} m, {src.n_states} states/tile "
        f"({len(tiles) * src.n_states} inferences)"
    )

    levels = LevelSelection.build(OUTPUT_HEIGHTS, [cfg.output.hub_height])
    print(f"[downscale] target height: {levels.describe()}")

    A = assets_mod.resolve(cfg.model.assets_root)
    in_stats, out_stats = A.load_input_scaler(), A.load_output_scaler()
    session = FuxiCFDSession(A.onnx, device=cfg.model.device)
    print(f"[downscale] {session.describe()}")
    batch = session.max_batch(cfg.model.batch_size)

    states = np.arange(src.n_states)
    u_flat, v_flat = src.read_uv(states)

    # Accumulators on the output grid. dtheta is blended through its sine and
    # cosine rather than directly: a weighted mean of angles is wrong across the
    # +/-180 branch cut, and while deflections that large are pathological, the
    # correct version costs one extra array.
    n_st = src.n_states
    acc = {
        "speed_fine": np.zeros((n_st, ny, nx)),
        "speed_coarse": np.zeros((n_st, ny, nx)),
        "sin_dth": np.zeros((n_st, ny, nx)),
        "cos_dth": np.zeros((n_st, ny, nx)),
    }
    wsum = np.zeros((ny, nx))

    overlap_px = max(
        0, int(round((TILE_SPAN_M - cfg.tiling.stride_m) / cfg.output.resolution_m))
    )
    n_done = 0
    for tile in tiles:
        interp = build_stencil_interpolator(geom, tile)
        static = static_channels(
            tile, in_stats, cfg.terrain.cache_root,
            dem_mode=cfg.terrain.dem_mode,
            roughness_mode=cfg.terrain.roughness_mode,
            roughness_despeckle=cfg.terrain.roughness_despeckle,
            landcover_source=cfg.terrain.landcover_source,
        )

        u9, v9 = stencil_uv(interp, u_flat, v_flat)
        x = preprocess.assemble(preprocess.normalize_winds(u9, v9, in_stats), static)

        outs = []
        for i in range(0, x.shape[0], batch):
            outs.append(
                postprocess.denormalize(session.run(x[i : i + batch]), out_stats)
            )
        fields = postprocess.split(np.concatenate(outs, axis=0))

        # (n_state, 1, 300, 300) -> (n_state, 300, 300) at the target height.
        u_fine = levels.apply(fields["u"])[:, 0]
        v_fine = levels.apply(fields["v"])[:, 0]
        tr = sector_transfer(u_fine, v_fine, u9, v9)

        smp = _make_sampler(grid, *tile.model_axes(), overlap_px=overlap_px)
        if smp is None:
            continue
        w = smp.weight
        dth = np.deg2rad(tr.dtheta)
        for key, val in (
            ("speed_fine", tr.speed_fine),
            ("speed_coarse", tr.speed_coarse),
            ("sin_dth", np.sin(dth)),
            ("cos_dth", np.cos(dth)),
        ):
            acc[key][:, smp.row, smp.col] += w * smp.sample(val)
        wsum[smp.row, smp.col] += w

        n_done += 1
        if n_done % 10 == 0 or n_done == len(tiles):
            print(f"[downscale]   {n_done}/{len(tiles)} tiles", flush=True)

    covered = wsum > 0
    if not covered.any():
        raise ValueError("no tile covered any output point")
    norm = np.where(covered, wsum, np.nan)
    for key in acc:
        acc[key] /= norm

    speed_fine = acc["speed_fine"]
    speed_coarse = acc["speed_coarse"]

    # Speedup at the state level, before any collapsing -- this is what the
    # speed-invariance test needs, and it is the quantity a timestep-derived
    # transfer is compared against.
    den = np.where(speed_coarse > 1e-6, speed_coarse, np.nan)
    S_states = np.where(covered[None], speed_fine / den, np.nan)

    # Collapse speed bins back to sectors. Bins are equal-probability quantiles,
    # so an unweighted mean over them estimates the sector mean. Deflection stays
    # in sine/cosine space until the very end -- averaging wrapped angles is the
    # same branch-cut mistake as averaging directions.
    nb = src.speed_bins
    sin_dth, cos_dth = acc["sin_dth"], acc["cos_dth"]
    if nb > 1:
        shape = (src.n_sectors, nb, ny, nx)
        S_bins = S_states.reshape(shape)
        S_sector = np.nanmean(S_bins, axis=1)
        speed_fine = speed_fine.reshape(shape).mean(axis=1)
        speed_coarse = speed_coarse.reshape(shape).mean(axis=1)
        sin_dth = sin_dth.reshape(shape).mean(axis=1)
        cos_dth = cos_dth.reshape(shape).mean(axis=1)
    else:
        S_bins, S_sector = None, S_states
    dtheta = wrap180(np.rad2deg(np.arctan2(sin_dth, cos_dth)))

    diagnostics = _diagnose(src, S_sector, S_bins, dtheta, covered)

    A_c = _coarse_on_grid(src, grid, src.A)
    k_c = _coarse_on_grid(src, grid, src.k)
    f_c = _coarse_on_grid(src, grid, src.freq)

    A_f, k_f, freq_f = apply_transfer(
        A_c, k_c, f_c, speed_fine, dtheta, deflect=cfg.output.deflect
    )
    A_all, k_all = combine_sectors(freq_f, A_f, k_f)
    pd_all = power_density(A_all, k_all, cfg.output.air_density)

    elevation = _elevation_on_grid(grid, tiles, cfg.terrain.cache_root)

    n_sec = src.n_sectors
    n_pts = ny * nx
    rec = np.zeros((n_pts, 6 + 3 * n_sec))
    rec[:, 0] = elevation
    rec[:, 1] = cfg.output.hub_height
    rec[:, 2] = A_all.ravel()
    rec[:, 3] = k_all.ravel()
    rec[:, 4] = pd_all.ravel()
    rec[:, 5] = n_sec
    fsum = np.nansum(freq_f, axis=0)
    fsum = np.where(fsum > 0, fsum, np.nan)
    for s in range(n_sec):
        rec[:, 6 + 3 * s + 0] = (freq_f[s] / fsum).ravel() * 1000.0
        rec[:, 6 + 3 * s + 1] = A_f[s].ravel() * 10.0
        rec[:, 6 + 3 * s + 2] = k_f[s].ravel() * 100.0

    gx, gy = grid.meshed()
    out = write_wrg(
        cfg.output.path,
        gx.T.ravel(), gy.T.ravel(),
        _reorder_f(rec, ny, nx),
        resolution_m=grid.res_m,
    )

    if cfg.output.save_transfer:
        tp = Path(cfg.output.path).with_suffix(".transfer.npz")
        np.savez_compressed(
            tp,
            x=grid.x, y=grid.y,
            sector_centers=src.centers,
            speed_fine=speed_fine.astype(np.float32),
            speed_coarse=speed_coarse.astype(np.float32),
            dtheta=dtheta.astype(np.float32),
            hub_height=cfg.output.hub_height,
        )
        print(f"[downscale] transfer fields -> {tp}")

    elapsed = time.time() - t0
    print(
        f"[downscale] wrote {out} ({n_pts:,} points, {n_sec} sectors) "
        f"in {elapsed:.1f} s"
    )
    print("[downscale] speedup by sector:")
    for row in diagnostics["per_sector"]:
        print(
            f"[downscale]   {row['center_deg']:5.1f} deg  "
            f"S p05/p50/p95 = {row['S_p05']:.3f}/{row['S_p50']:.3f}/"
            f"{row['S_p95']:.3f}   |dtheta| p95 = {row['abs_dtheta_p95']:.1f} deg"
        )
    if diagnostics["speed_invariance"] is not None:
        si = diagnostics["speed_invariance"]
        print(
            f"[downscale] speed-invariance of S across {nb} bins: "
            f"max spread {si['max_range']:.4f}, median spread "
            f"{si['median_range']:.4f}"
        )

    report = Path(cfg.output.path).with_suffix(".diagnostics.json")
    report.write_text(json.dumps(diagnostics, indent=2))

    return DownscaleResult(
        path=out, grid=grid, n_tiles=len(tiles), n_states=src.n_states,
        elapsed_s=elapsed, diagnostics=diagnostics,
    )


def _diagnose(src, S, S_bins, dtheta, covered):
    """Summary statistics of the transfer, for the run log and a JSON report.

    `S` is the per-sector speedup on the output grid; `S_bins` is the same
    quantity split by inflow speed bin, or None for a single-bin run.
    """
    nb = src.speed_bins
    per_sector = []
    for s in range(src.n_sectors):
        v = S[s][np.isfinite(S[s])]
        d = np.abs(dtheta[s][covered])
        d = d[np.isfinite(d)]
        per_sector.append({
            "sector": s,
            "center_deg": float(src.centers[s]),
            "S_p05": float(np.percentile(v, 5)) if v.size else float("nan"),
            "S_p50": float(np.percentile(v, 50)) if v.size else float("nan"),
            "S_p95": float(np.percentile(v, 95)) if v.size else float("nan"),
            "abs_dtheta_p95": float(np.percentile(d, 95)) if d.size else float("nan"),
        })

    invariance = None
    if S_bins is not None:
        # Does the speedup depend on inflow magnitude? In neutral RANS it should
        # not -- Reynolds-number independence at high Re is exactly what lets a
        # sector-mean inflow stand in for the whole speed distribution. But
        # FuXi-CFD never sees inflow speed as an explicit input, so the
        # invariance is a property of the learned operator and has to be
        # measured. Spread here is (max - min) / mean of S across the bins of one
        # sector, per grid point.
        rng = np.nanmax(S_bins, axis=1) - np.nanmin(S_bins, axis=1)
        mean = np.nanmean(S_bins, axis=1)
        rel = rng / np.where(mean > 1e-6, mean, np.nan)
        invariance = {
            "n_bins": int(nb),
            "max_range": float(np.nanmax(rel)),
            "median_range": float(np.nanmedian(rel)),
            "p95_range": float(np.nanpercentile(rel, 95)),
            "note": (
                "relative spread of the speedup S across inflow speed bins. "
                "Small (<~0.02) means one inference per sector suffices; large "
                "means the transfer needs speed-binned functions, and that "
                "dependence is also the k adjustment transfer.apply_transfer "
                "currently omits."
            ),
        }

    return {
        "n_sectors": int(src.n_sectors),
        "speed_bins": int(nb),
        "hub_height_m": float(src.hub_height),
        "per_sector": per_sector,
        "speed_invariance": invariance,
        "S_all": {
            "p05": float(np.nanpercentile(S, 5)),
            "p50": float(np.nanpercentile(S, 50)),
            "p95": float(np.nanpercentile(S, 95)),
        },
    }
