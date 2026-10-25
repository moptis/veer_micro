"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .constants import TERRAIN_RES_M


def _cmd_fetch_model(args) -> int:
    from .model import assets as assets_mod

    a = assets_mod.fetch(args.assets_root, force=args.force)
    print(f"[fetch] assets at {a.root}")
    missing = a.missing()
    if missing:
        print(f"[fetch] STILL MISSING: {missing}", file=sys.stderr)
        return 1
    from . import constants

    constants.validate_heights_against_asset(a.height)
    print("[fetch] all assets present; output heights verified against height.npy")
    return 0


def _cmd_inspect(args) -> int:
    """Summarise a WRF source: grid geometry and the (surprising) time structure."""
    from .grids import DomainGeometry
    from .sources.h5_source import H5WrfSource
    from .sources.npz_source import NpzWrfSource

    src = (
        H5WrfSource(args.path)
        if args.kind == "h5"
        else NpzWrfSource(args.path, coordinates_from=args.coordinates_from)
    )
    print("=== time ===")
    print(src.time_report())

    lat, lon = src.coordinates()
    # Grid shape comes from the store: it differs per domain (Broadview 78x132,
    # WSW 96x114, Pattern ph1 150x177) and the constant is only a fallback.
    shape = getattr(src, "domain_shape", None)
    kw = {"crs": "auto", **({"grid_shape": shape} if shape else {})}
    try:
        geom = DomainGeometry(lat, lon, **kw)
        fit_note = "(index-space bilinear is valid)"
    except ValueError as exc:
        if "not affine" not in str(exc):
            raise
        # Report rather than fail: knowing the domain needs a local fit is exactly
        # what you ran `inspect` to find out.
        geom = DomainGeometry(lat, lon, affine_fit="local", **kw)
        fit_note = "-- too curved for one global fit; set tiling.affine_fit: local"
    minx, miny, maxx, maxy = geom.extent
    print("\n=== grid ===")
    print(f"shape            : {geom.grid_shape}")
    print(f"lat range        : {lat.min():.4f} .. {lat.max():.4f}")
    print(f"lon range        : {lon.min():.4f} .. {lon.max():.4f}")
    print(f"extent ({geom.crs}) : {maxx - minx:.0f} m x {maxy - miny:.0f} m")
    print(f"affine residual  : {geom.affine_residual_m:.2f} m global {fit_note}")
    if geom.local_residual_m is not None:
        print(f"                 : {geom.local_residual_m:.2f} m over "
              f"{2 * geom.LOCAL_FIT_HALF + 1}-cell windows")
    tiles = geom.make_tiles()
    over = geom.make_tiles(stride_m=6000.0)
    print(f"tiles @ 9 km     : {len(tiles)}")
    print(f"tiles @ 6 km     : {len(over)} (3 km overlap, for mosaicking)")
    return 0


def _cmd_plan(args) -> int:
    from .config import RunConfig
    from .pipeline import plan

    cfg = RunConfig.from_yaml(args.config)
    print(json.dumps(plan(cfg), indent=2))
    return 0


def _cmd_init(args) -> int:
    """Create the output store and warm the terrain cache, once, before an array job.

    Array tasks starting together would otherwise all race to create the same Zarr
    arrays and download the same terrain tiles. Creation is idempotent now, but
    doing it once up front is cheaper and removes the contention entirely.
    """
    from .config import RunConfig
    from .pipeline import prepare

    cfg = RunConfig.from_yaml(args.config)
    prep = prepare(cfg)               # builds terrain + static channels for every tile
    prep.store.create(config_json=cfg.model_dump_json())
    print(
        f"[init] store ready at {cfg.output.path}\n"
        f"[init] {len(prep.tiles)} tiles, {len(prep.times)} timesteps, "
        f"{prep.levels.n_out} levels, terrain cached"
    )
    return 0


def _cmd_run(args) -> int:
    from .config import RunConfig
    from .pipeline import run

    cfg = RunConfig.from_yaml(args.config)
    if args.limit:
        cfg.time.limit = args.limit
    if args.device:
        cfg.model.device = args.device
    result = run(cfg, shard=args.shard, n_shards=args.n_shards, resume=args.resume)
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_validate(args) -> int:
    from .config import RunConfig
    from .model import assets as assets_mod
    from .model.session import FuxiCFDSession
    from .pipeline import prepare
    from .terrain.build import static_channels
    from .validate import (
        Report, check_closure, check_distribution, check_profile,
        check_upstream_parity, run_inference_sample,
    )

    report = Report()
    print("[validate] 1/2: upstream parity + error floor ...")
    parity, floor = check_upstream_parity(args.assets_root, device="cpu")
    report.add("1_upstream_parity", parity.pop("passed"), parity)
    report.add("2_error_floor", floor.pop("passed"), floor)

    if not args.config:
        print(report.render())
        if args.out:
            report.to_json(args.out)
        return 0 if report.checks["1_upstream_parity"]["passed"] else 1

    cfg = RunConfig.from_yaml(args.config)
    cfg.time.limit = args.n_times
    prep = prepare(cfg, build_static=False)
    tiles = prep.tiles[: args.n_tiles]
    print(f"[validate] using {len(tiles)} tiles x {len(prep.rows)} timesteps")

    print("[validate] 3: terrain distribution shift ...")
    dist, _ = check_distribution(tiles, cfg.terrain.cache_root, prep.in_stats,
                                 source=cfg.terrain.landcover_source)
    report.add("3_distribution_shift", dist.pop("passed"), dist)

    session = FuxiCFDSession(
        assets_mod.resolve(cfg.model.assets_root).onnx, device=cfg.model.device
    )
    print(f"[validate] {session.describe()}")

    modes = ["absolute", "recentered"] if args.compare_modes else [cfg.terrain.dem_mode]
    mode_results = {}
    for mode in modes:
        print(f"[validate] 4/5: inference with dem_mode={mode} ...")
        statics = {
            t.tile_id: static_channels(
                t, prep.in_stats, cfg.terrain.cache_root,
                dem_mode=mode, roughness_mode=cfg.terrain.roughness_mode,
                roughness_despeckle=cfg.terrain.roughness_despeckle,
                landcover_source=cfg.terrain.landcover_source,
            )
            for t in tiles
        }
        sample = run_inference_sample(prep, tiles, prep.rows, session, statics)
        closure = check_closure(sample)
        profile = check_profile(sample, prep, prep.rows, tiles)
        mode_results[mode] = {
            "closure": closure,
            "profile": profile,
            "mean_profile_rmse": float(
                np.mean([p["rmse_overall"] for p in profile["per_tile"].values()])
            ) if profile.get("per_tile") else None,
        }
        report.add(f"4_closure[{mode}]", closure.pop("passed"), closure)
        report.add(f"5_profile[{mode}]", profile.pop("passed"), profile)

    if args.compare_modes:
        scored = {
            m: r["mean_profile_rmse"] for m, r in mode_results.items()
            if r["mean_profile_rmse"] is not None
        }
        best = min(scored, key=scored.get) if scored else None
        report.add(
            "6_dem_mode_arbitration", None,
            {
                "mean_profile_rmse_by_mode": scored,
                "recommended_dem_mode": best,
                "interpretation": (
                    "Lower RMSE against the WRF profile wins. If the two are within "
                    "noise of each other, prefer 'absolute' as the faithful reading "
                    "of the model card."
                ),
            },
        )

    print(report.render())
    if args.out:
        report.to_json(args.out)
        print(f"\n[validate] report written to {args.out}")
    return 0 if report.checks["1_upstream_parity"]["passed"] else 1


def _cmd_wrg(args) -> int:
    from .model.assets import default_cache_root
    from .wrg_export import export_wrg

    result = export_wrg(
        args.store,
        args.out,
        hub_height=args.hub_height,
        res_m=args.resolution,
        bin_size_deg=args.bin_size,
        buffer=args.buffer,
        air_density=args.air_density,
        time_block=args.time_block,
        terrain_cache=args.terrain_cache or default_cache_root(),
        tile_ids=args.tile or None,
        min_coverage=args.min_coverage,
        allow_partial=args.allow_partial,
        trim_edge_m=args.trim_edge,
        n_workers=args.workers,
    )
    print(json.dumps(result, indent=2))
    if abs(result["level_m"] - args.hub_height) > 0.01:
        print(
            f"\nNOTE: FuXi-CFD has no {args.hub_height} m level. This file uses "
            f"{result['level_m']:.5f} m and the hub_height column reports that "
            f"value, not {args.hub_height}."
        )
    return 0


def _cmd_wrg_concat(args) -> int:
    from .wrg_concat import concat_wrgs, largest_full_rect

    paths = sorted(args.directory.glob(args.pattern))
    if not paths:
        print(f"no files matching {args.pattern} in {args.directory}", file=sys.stderr)
        return 1
    if args.largest_rect:
        paths = largest_full_rect(paths)
    result = concat_wrgs(paths, args.out, air_density=args.air_density,
                         allow_holes=args.allow_holes)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_wrg_inspect(args) -> int:
    """Summarise a coarse .wrg read as a sector-transfer driving source."""
    from .sources.wrg_source import WrgSectorSource

    src = WrgSectorSource(args.path, crs=args.crs, speed_bins=args.speed_bins)
    print(src.describe())
    return 0


def _cmd_wrg_downscale(args) -> int:
    from .config import DownscaleConfig
    from .wrg_downscale import run as run_downscale

    cfg = DownscaleConfig.from_yaml(args.config)
    if args.device:
        cfg.model.device = args.device
    if args.out:
        cfg.output.path = args.out
    if args.hub_height is not None:
        cfg.output.hub_height = args.hub_height
    if args.resolution is not None:
        cfg.output.resolution_m = args.resolution
    if args.no_deflect:
        cfg.output.deflect = False
    run_downscale(cfg)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="fuxicfd-wrf",
        description="Drive FuXi-CFD with WRF output to produce 30 m 3D wind fields.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch-model", help="download upstream ONNX weights and scalers")
    f.add_argument("--assets-root", type=Path, default=None)
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=_cmd_fetch_model)

    i = sub.add_parser("inspect", help="summarise a WRF source's grid and time structure")
    i.add_argument("path", type=Path)
    i.add_argument("--kind", choices=["h5", "npz"], default="h5")
    i.add_argument("--coordinates-from", type=Path, default=None)
    i.set_defaults(func=_cmd_inspect)

    pl = sub.add_parser("plan", help="dry run: tile/time counts and output size")
    pl.add_argument("config", type=Path)
    pl.set_defaults(func=_cmd_plan)

    n = sub.add_parser(
        "init", help="create the store and cache terrain before launching an array"
    )
    n.add_argument("config", type=Path)
    n.set_defaults(func=_cmd_init)

    r = sub.add_parser("run", help="run inference and write Zarr")
    r.add_argument("config", type=Path)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--n-shards", type=int, default=1)
    r.add_argument("--limit", type=int, default=None, help="cap number of timesteps")
    r.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    r.add_argument("--resume", action="store_true",
                   help="skip blocks this shard already wrote (needs the same "
                        "--n-shards as the original launch)")
    r.set_defaults(func=_cmd_run)

    wi = sub.add_parser(
        "wrg-inspect", help="summarise a coarse .wrg as a driving source"
    )
    wi.add_argument("path", type=Path)
    wi.add_argument("--crs", required=True,
                    help="projection the coarse .wrg is written in, e.g. EPSG:32613")
    wi.add_argument("--speed-bins", type=int, default=1)
    wi.set_defaults(func=_cmd_wrg_inspect)

    wd = sub.add_parser(
        "wrg-downscale",
        help="downscale a coarse .wrg to a fine .wrg via per-sector inference",
    )
    wd.add_argument("config", type=Path)
    wd.add_argument("--out", type=Path, default=None)
    wd.add_argument("--hub-height", type=float, default=None)
    wd.add_argument("--resolution", type=float, default=None,
                    help="output grid spacing in m")
    wd.add_argument("--no-deflect", action="store_true",
                    help="keep sectors in place; downscale magnitude only")
    wd.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    wd.set_defaults(func=_cmd_wrg_downscale)

    v = sub.add_parser("validate", help="Phase 0 validation report")
    v.add_argument("--config", type=Path, default=None)
    v.add_argument("--assets-root", type=Path, default=None)
    v.add_argument("--n-tiles", type=int, default=4)
    v.add_argument("--n-times", type=int, default=24)
    v.add_argument("--compare-modes", action="store_true",
                   help="run both dem_mode settings and recommend one")
    v.add_argument("--out", type=Path, default=None, help="write JSON report here")
    v.set_defaults(func=_cmd_validate)

    w = sub.add_parser("wrg", help="export a WAsP-style .wrg from a Zarr store")
    w.add_argument("store", type=Path, help="tile or mosaic .zarr produced by `run`")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--hub-height", type=float, default=100.0,
                   help="target hub height (m); snaps to the nearest model level")
    w.add_argument("--resolution", type=float, default=TERRAIN_RES_M,
                   help=f"WRG grid spacing in m (default {TERRAIN_RES_M:g}, the "
                        "model's native output resolution). Coarser values "
                        "average model pixels away: 100 m keeps 1 point per 11.")
    w.add_argument("--bin-size", type=float, default=30.0, help="sector width, deg")
    w.add_argument("--buffer", type=float, default=0.0,
                   help="fraction trimmed from each edge (reference default was 0.2)")
    w.add_argument("--air-density", type=float, default=1.225,
                   help="kg/m3 for power density; ~1.06 at Broadview's elevation")
    w.add_argument("--time-block", type=int, default=240)
    w.add_argument("--terrain-cache", type=Path, default=None)
    w.add_argument("--tile", action="append", default=[],
                   help="restrict to this tile id; repeatable. Site-mode stores "
                        "hold scattered tiles whose bounding box is mostly empty.")
    w.add_argument("--min-coverage", type=float, default=0.98,
                   help="refuse to write if tiles cover less of the grid than this")
    w.add_argument("--workers", type=int, default=None,
                   help="parallel time-shard workers (default: Slurm CPUs, max 12). "
                        "Histograms are additive so sharding time is exact.")
    w.add_argument("--trim-edge", type=float, default=1500.0,
                   help="metres discarded from each edge of a single-tile export; "
                        "FuXi-CFD gets no buffer and its outer ~1.5 km is 1.9x "
                        "noisier. Ignored for multi-tile/mosaic exports.")
    w.add_argument("--allow-partial", action="store_true",
                   help="write anyway despite low coverage (uncovered points = 0)")
    w.set_defaults(func=_cmd_wrg)

    wc = sub.add_parser("wrg-concat",
                        help="merge per-tile WRGs into one domain WRG")
    wc.add_argument("directory", type=Path)
    wc.add_argument("--pattern", default="*.wrg")
    wc.add_argument("--out", type=Path, required=True)
    wc.add_argument("--air-density", type=float, default=1.01)
    wc.add_argument("--allow-holes", action="store_true",
                    help="write even if the tiles do not cover a full rectangle. "
                         "Uncovered points are written as zeros, which downstream "
                         "tools read as real data -- prefer --largest-rect.")
    wc.add_argument("--largest-rect", action="store_true",
                    help="trim to the biggest complete rectangle of tiles. Use "
                         "when a bbox run sits against a rotated domain edge and "
                         "comes back L-shaped; reports which tiles it drops.")
    wc.set_defaults(func=_cmd_wrg_concat)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
