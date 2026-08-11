"""A/B a terrain-input change through the model and localise its effect.

    python scripts/ab_terrain.py --sources worldcover nlcd --n-times 48
    python scripts/ab_terrain.py --sources worldcover worldcover --despeckle 3

Runs the same timesteps twice with only the static channels differing, then
reports how the wind field changed as a function of distance from WorldCover's
"built-up" pixels. Concentration near those pixels means the change is acting on
the speckle that caused the localised artifacts; a flat profile means it is a
broad reclassification instead.

Measured so far on bv_center (48 timesteps):
    despeckle(3)          33x larger change near the speckle than far from it
    worldcover -> nlcd     5.2x, peak 1.04 m/s

This is the tool for deciding whether a terrain-input change is worth adopting --
run it before switching `landcover_source` on a new domain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from fuxicfd_wrf.config import RunConfig
from fuxicfd_wrf.constants import OUTPUT_HEIGHTS
from fuxicfd_wrf.model import assets as assets_mod
from fuxicfd_wrf.model import postprocess, preprocess
from fuxicfd_wrf.model.session import FuxiCFDSession
from fuxicfd_wrf.pipeline import prepare
from fuxicfd_wrf.terrain.build import build_terrain, static_channels
from fuxicfd_wrf.winds import stencil_winds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/broadview_sites.yaml"))
    ap.add_argument("--tile", default="bv_center")
    ap.add_argument("--n-times", type=int, default=48)
    ap.add_argument("--despeckle", type=int, default=0)
    ap.add_argument("--sources", nargs=2, default=["worldcover", "nlcd"],
                    help="two land-cover sources to compare")
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    cfg.time.limit = args.n_times
    prep = prepare(cfg, build_static=False)
    tile = next(t for t in prep.tiles if t.tile_id == args.tile)

    session = FuxiCFDSession(
        assets_mod.resolve(cfg.model.assets_root).onnx, device=cfg.model.device
    )
    print(f"{session.describe()}\n")

    speed, direction = prep.source.read_wind_100m(prep.rows)
    u9, v9 = stencil_winds(prep.interpolators[tile.tile_id], speed, direction)
    winds = preprocess.normalize_winds(u9, v9, prep.in_stats)
    batch = session.max_batch(cfg.model.batch_size)
    lvl = int(np.argmin(np.abs(OUTPUT_HEIGHTS - 100.0)))

    fields = {}
    for label in args.sources:
        static = static_channels(
            tile, prep.in_stats, cfg.terrain.cache_root,
            dem_mode=cfg.terrain.dem_mode, roughness_mode=cfg.terrain.roughness_mode,
            roughness_despeckle=args.despeckle, landcover_source=label,
        )
        x = preprocess.assemble(winds, static)
        pred = np.concatenate([
            postprocess.denormalize(session.run(x[i:i + batch]), prep.out_stats)
            for i in range(0, x.shape[0], batch)
        ], axis=0)
        f = postprocess.split(pred)
        fields[label] = np.hypot(f["u"][:, lvl], f["v"][:, lvl]).mean(axis=0)
        print(f"{label:<24} mean {fields[label].mean():.4f}  std {fields[label].std():.4f}")

    a, b = list(fields.values())
    diff = b - a

    terr = build_terrain(tile, cfg.terrain.cache_root, source="worldcover")
    bu = np.rint(terr.landcover).astype(int) == 50
    bu300 = bu[:300, :300]
    dist_m = distance_transform_edt(~bu300) * (9000 / 299)

    print(f"\nchanged by the source switch: mean |diff| {np.abs(diff).mean():.4f} m/s, "
          f"max {np.abs(diff).max():.4f} m/s")
    print(f"\n{'distance to nearest built-up px':<34}{'n px':>8}{'mean |diff|':>14}")
    for lo, hi in [(0, 60), (60, 150), (150, 300), (300, 600), (600, 1500), (1500, 1e9)]:
        m = (dist_m >= lo) & (dist_m < hi)
        if m.sum():
            tag = f"{lo:.0f}-{hi:.0f} m" if hi < 1e8 else f">{lo:.0f} m"
            print(f"  {tag:<32}{m.sum():8d}{np.abs(diff)[m].mean():13.4f}")

    near = np.abs(diff)[dist_m < 150].mean()
    far = np.abs(diff)[dist_m > 1500].mean()
    print(f"\n  near (<150 m) / far (>1500 m) ratio = {near / max(far, 1e-9):.1f}x")
    print("  A high ratio means the change is concentrated exactly where"
          "\n  WorldCover's built-up speckle was -- i.e. that is what got fixed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
