"""Compare one high-resolution tile WRG against the coarse 1 km WRG beneath it.

    python scripts/compare_tile_wrg.py <fine.wrg> <coarse.wrg>

Two separate questions, kept separate:

1. Does the fine field reproduce the coarse one when averaged back to 1 km?
   Bias here means a broken chain, not a discovery.
2. How much structure exists *below* 1 km? That is the entire product.

Power density is recomputed from A and k on both sides, so the comparison does
not depend on which convention each file's own column used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fuxicfd_wrf.wrg_read import read_wrg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fine", type=Path)
    ap.add_argument("coarse", type=Path)
    ap.add_argument("--air-density", type=float, default=1.06)
    args = ap.parse_args()

    f = read_wrg(args.fine, air_density=args.air_density)
    c = read_wrg(args.coarse, air_density=args.air_density)

    # Restrict the coarse grid to the fine tile's footprint.
    pad = c.attrs["cell_size"] / 2
    m = (
        (c["x"] >= f["x"].min() - pad) & (c["x"] <= f["x"].max() + pad)
        & (c["y"] >= f["y"].min() - pad) & (c["y"] <= f["y"].max() + pad)
    )
    c = c[m].copy()
    if c.empty:
        print("no coarse cells overlap this tile")
        return 1

    print(f"fine   : {len(f):,} pts @ {f.attrs['cell_size']:.0f} m, "
          f"{f.attrs['n_sectors']} sectors, hub {f['hub_height'].iloc[0]:.0f} m")
    print(f"coarse : {len(c):,} pts @ {c.attrs['cell_size']:.0f} m, "
          f"{c.attrs['n_sectors']} sectors, hub {c['hub_height'].iloc[0]:.0f} m")
    if f.attrs["n_sectors"] != c.attrs["n_sectors"]:
        print("  NOTE: sector counts differ; per-sector stats not comparable")

    # Bin fine points into coarse cells.
    cell = c.attrs["cell_size"]
    cx = np.sort(c["x"].unique())
    cy = np.sort(c["y"].unique())
    ix = np.round((f["x"].values - cx[0]) / cell).astype(int)
    iy = np.round((f["y"].values - cy[0]) / cell).astype(int)
    ok = (ix >= 0) & (ix < len(cx)) & (iy >= 0) & (iy < len(cy))

    key = iy[ok] * len(cx) + ix[ok]
    spd = f["mean_speed"].values[ok]
    pdn = f["power_density"].values[ok]

    stats = {}
    for k in np.unique(key):
        s = spd[key == k]
        stats[k] = (s.mean(), s.std(), s.min(), s.max(), len(s), pdn[key == k].mean())

    ck = (np.round((c["y"].values - cy[0]) / cell).astype(int) * len(cx)
          + np.round((c["x"].values - cx[0]) / cell).astype(int))
    matched = [(stats[k], c["mean_speed"].values[i], c["power_density"].values[i])
               for i, k in enumerate(ck) if k in stats]
    if not matched:
        print("no coarse cell centres fall inside the tile")
        return 1

    fine_mean = np.array([m[0][0] for m in matched])
    within_sd = np.array([m[0][1] for m in matched])
    within_rng = np.array([m[0][3] - m[0][2] for m in matched])
    npts = np.array([m[0][4] for m in matched])
    fine_pd = np.array([m[0][5] for m in matched])
    coarse_mean = np.array([m[1] for m in matched])
    coarse_pd = np.array([m[2] for m in matched])

    print(f"\nmatched {len(matched)} coarse cells, median {np.median(npts):.0f} fine points each")

    print("\n--- 1. coarse-scale agreement (fine averaged back to 1 km) ---")
    d = fine_mean - coarse_mean
    print(f"  coarse mean speed : {coarse_mean.mean():7.3f} m/s")
    print(f"  fine   mean speed : {fine_mean.mean():7.3f} m/s")
    print(f"  bias              : {d.mean():+7.3f} m/s  ({100*d.mean()/coarse_mean.mean():+.1f}%)")
    print(f"  RMSE              : {np.sqrt((d**2).mean()):7.3f} m/s")
    if len(d) > 2:
        print(f"  correlation       : {np.corrcoef(fine_mean, coarse_mean)[0,1]:7.3f}")
    dp = fine_pd - coarse_pd
    print(f"  power density     : coarse {coarse_pd.mean():6.0f} -> fine {fine_pd.mean():6.0f} W/m2"
          f"  ({100*dp.mean()/coarse_pd.mean():+.1f}%)")

    print("\n--- 2. sub-kilometre structure (what the coarse grid cannot see) ---")
    print(f"  within-cell std   : median {np.median(within_sd):.3f} m/s"
          f"  ({100*np.median(within_sd)/fine_mean.mean():.1f}% of mean)")
    print(f"  within-cell range : median {np.median(within_rng):.3f} m/s,"
          f" max {within_rng.max():.3f} m/s")
    print(f"  fine-field spread : {f['mean_speed'].min():.2f} .. {f['mean_speed'].max():.2f} m/s"
          f"  (std {f['mean_speed'].std():.3f})")
    print(f"  elevation range   : {f['elevation'].min():.0f} .. {f['elevation'].max():.0f} m")
    corr_terr = np.corrcoef(f["elevation"].values, f["mean_speed"].values)[0, 1]
    print(f"  corr(elev, speed) : {corr_terr:+.3f}"
          "   <- terrain signal the model actually used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
