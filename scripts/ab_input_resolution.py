"""What would coarser driving data do? Degrade the WRF winds and re-run.

    python scripts/ab_input_resolution.py --config configs/wsw_center.yaml --factor 3

FuXi-CFD's contract fixes the input at a 9x9 stencil on 1 km spacing spanning the
9 km tile. With 3 km driving data you would still have to fill that 9x9, so the
window is unchanged and what is lost is the mesoscale gradient *within* the tile:
81 stencil points carrying ~3x3 independent values.

This block-averages the WRF wind field by `factor` before building the stencil,
isolating that effect. u and v are averaged, never speed and direction -- see
winds.py.

**What this does and does not tell you.** It answers "what if the driving field
were this much smoother". It does NOT answer "what if WRF had been run at 3 km":
a real 3 km run also resolves terrain and PBL differently, so its 100 m winds
differ from a smoothed 1 km field by more than smoothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from fuxicfd_wrf.config import RunConfig
from fuxicfd_wrf.constants import OUTPUT_HEIGHTS
from fuxicfd_wrf.model import assets as assets_mod
from fuxicfd_wrf.model import postprocess, preprocess
from fuxicfd_wrf.model.session import FuxiCFDSession
from fuxicfd_wrf.pipeline import prepare
from fuxicfd_wrf.terrain.build import static_channels
from fuxicfd_wrf.winds import rotate_to_grid, speeddir_to_uv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/wsw_center.yaml"))
    ap.add_argument("--tile", default=None)
    ap.add_argument("--n-times", type=int, default=48)
    ap.add_argument("--factor", type=int, nargs="+", default=[3])
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    cfg.time.limit = args.n_times
    prep = prepare(cfg, build_static=False)
    tile = (next(t for t in prep.tiles if t.tile_id == args.tile)
            if args.tile else prep.tiles[0])
    interp = prep.interpolators[tile.tile_id]
    nrow, ncol = prep.geom.grid_shape

    session = FuxiCFDSession(
        assets_mod.resolve(cfg.model.assets_root).onnx, device=cfg.model.device)
    print(f"{session.describe()}\ntile: {tile.tile_id}, {len(prep.rows)} timesteps\n")
    static = static_channels(
        tile, prep.in_stats, cfg.terrain.cache_root,
        dem_mode=cfg.terrain.dem_mode, roughness_mode=cfg.terrain.roughness_mode,
        landcover_source=cfg.terrain.landcover_source)
    batch = session.max_batch(cfg.model.batch_size)
    lvl = int(np.argmin(np.abs(OUTPUT_HEIGHTS - 100.0)))

    speed, direction = prep.source.read_wind_100m(prep.rows)
    u_flat, v_flat = speeddir_to_uv(speed, direction)

    def run(fac: int):
        if fac > 1:
            # Block-average on the native WRF grid, mimicking coarser driving data.
            u = uniform_filter(u_flat.reshape(-1, nrow, ncol), size=(1, fac, fac),
                               mode="nearest").reshape(u_flat.shape)
            v = uniform_filter(v_flat.reshape(-1, nrow, ncol), size=(1, fac, fac),
                               mode="nearest").reshape(v_flat.shape)
        else:
            u, v = u_flat, v_flat
        u9, v9 = interp.apply(u), interp.apply(v)
        u9, v9 = rotate_to_grid(u9, v9, interp.convergence_deg)
        # How much horizontal structure survives inside the stencil?
        grad = float(np.mean(np.abs(np.diff(np.hypot(u9, v9), axis=-1))))
        x = preprocess.assemble(
            preprocess.normalize_winds(u9, v9, prep.in_stats), static)
        pred = np.concatenate([
            postprocess.denormalize(session.run(x[i:i + batch]), prep.out_stats)
            for i in range(0, x.shape[0], batch)], axis=0)
        f = postprocess.split(pred)
        return np.hypot(f["u"][:, lvl], f["v"][:, lvl]).mean(axis=0), grad

    base, g0 = run(1)
    print(f"{'input':<16}{'stencil grad':>13}{'out mean':>10}{'out std':>9}"
          f"{'RMS vs 1km':>12}{'corr':>8}")
    print("-" * 70)
    print(f"{'1 km (native)':<16}{g0:13.4f}{base.mean():10.3f}{base.std():9.4f}"
          f"{'--':>12}{'--':>8}")
    for fac in args.factor:
        out, g = run(fac)
        d = out - base
        print(f"{f'{fac} km (x{fac} avg)':<16}{g:13.4f}{out.mean():10.3f}{out.std():9.4f}"
              f"{np.sqrt((d**2).mean()):12.4f}"
              f"{np.corrcoef(out.ravel(), base.ravel())[0,1]:8.4f}")
    print("\nstencil grad = mean |d(speed)| between adjacent stencil points: the")
    print("within-tile mesoscale structure the coarser input removes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
