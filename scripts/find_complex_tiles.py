"""Rank candidate tiles by terrain complexity, to find where FuXi-CFD has something to do.

    python scripts/find_complex_tiles.py <control.h5> [--top 6] [--turbines <csv>]

Two-stage search, because the two available elevation sources answer different
questions:

1. **Coarse scan** over the WRF ~1 km elevation field, using a sliding 9x9 window
   (one tile footprint) to get relief everywhere at once. Cheap, but WRF terrain
   is smoothed and understates real relief.
2. **Refine** the top candidates against the actual 30 m SRTM the model is fed,
   reporting relief, mean slope and roughness -- the descriptors the paper's
   Figure S1 uses to characterise the training distribution.

Complexity is what FuXi-CFD responds to: at Broadview's 70 m relief the
sub-kilometre signal was 0.4% of mean speed, which is near-nothing. A tile with
several hundred metres of relief is the honest test of whether the model earns
its keep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyproj
from scipy.ndimage import maximum_filter, minimum_filter

from fuxicfd_wrf.constants import TILE_SPAN_M
from fuxicfd_wrf.grids import DomainGeometry
from fuxicfd_wrf.sources.h5_source import H5WrfSource
from fuxicfd_wrf.terrain.build import build_terrain, sigma_offsets
from fuxicfd_wrf.terrain.roughness import class_histogram


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", type=Path)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--turbines", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=Path("/scratch/moptis/fuxicfd/cache"))
    ap.add_argument("--assets", type=Path, default=Path("/scratch/moptis/fuxicfd/assets"))
    args = ap.parse_args()

    src = H5WrfSource(args.store)
    lat, lon = src.coordinates()
    geom = DomainGeometry(lat, lon, crs="auto", grid_shape=src.domain_shape)
    nrow, ncol = geom.grid_shape
    elev = src.elevation().reshape(nrow, ncol)

    # --- stage 1: sliding-window relief on the coarse grid --------------------
    win = int(round(TILE_SPAN_M / 1000.0))          # ~9 WRF cells per tile
    relief = maximum_filter(elev, size=win) - minimum_filter(elev, size=win)
    gy, gx = np.gradient(elev, 1000.0)
    rug = np.hypot(gy, gx)

    valid = np.zeros_like(relief, dtype=bool)
    m = win // 2 + 1
    valid[m:-m, m:-m] = True
    for r in range(nrow):
        for c in range(ncol):
            if valid[r, c]:
                x, y = geom.rowcol_to_xy(r, c)
                from fuxicfd_wrf.grids import Tile
                if not geom.covers_stencil(Tile("t", float(x), float(y), geom.crs)):
                    valid[r, c] = False

    score = np.where(valid, relief, -np.inf)
    print(f"coarse relief over {win}x{win} WRF cells: "
          f"median {np.median(relief[valid]):.0f} m, max {score.max():.0f} m")

    # Pick well-separated peaks so candidates are not all the same hill.
    picks, taken = [], np.zeros_like(valid)
    sep = int(round(9000 / 1000.0))
    order = np.argsort(score.ravel())[::-1]
    for flat in order:
        r, c = divmod(int(flat), ncol)
        if not valid[r, c] or taken[r, c]:
            continue
        picks.append((r, c, float(relief[r, c]), float(rug[r, c])))
        taken[max(0, r - sep):r + sep + 1, max(0, c - sep):c + sep + 1] = True
        if len(picks) >= args.top:
            break

    # --- stage 2: refine against real 30 m SRTM -------------------------------
    tf = pyproj.Transformer.from_crs(geom.crs, "EPSG:4326", always_xy=True)
    from fuxicfd_wrf.grids import Tile
    from fuxicfd_wrf.model import assets as assets_mod

    in_stats = assets_mod.resolve(args.assets).load_input_scaler()

    turbines = None
    if args.turbines and args.turbines.exists():
        import pandas as pd
        turbines = pd.read_csv(args.turbines)
    geod = pyproj.Geod(ellps="WGS84")

    print(f"\nrefining top {len(picks)} candidates against 30 m SRTM ...\n")
    print(f"{'tile':<14}{'lon':>10}{'lat':>8}{'relief':>9}{'slope':>8}{'z0':>8}"
          f"{'dem_sig':>9}  nearest farm")
    print("-" * 88)
    rows = []
    for i, (r, c, _rel, _rug) in enumerate(picks):
        x, y = geom.rowcol_to_xy(r, c)
        tlon, tlat = tf.transform(float(x), float(y))
        tile = Tile(f"wsw_hi{i+1}", float(x), float(y), geom.crs)
        terr = build_terrain(tile, args.cache)
        st = terr.stats()
        gyy, gxx = np.gradient(terr.dem.astype(float), 30.0)
        slope_deg = float(np.degrees(np.arctan(np.hypot(gyy, gxx).mean())))
        sig = sigma_offsets(terr, in_stats)["dem_sigma_mean"]

        near = ""
        if turbines is not None:
            d = [
                (geod.inv(tlon, tlat, g.longitude.mean(), g.latitude.mean())[2] / 1000, f)
                for f, g in turbines.groupby("wind_farm")
            ]
            dist, farm = min(d)
            near = f"{farm} ({dist:.0f} km)"
        print(f"{tile.tile_id:<14}{tlon:10.4f}{tlat:8.4f}{st['dem_relief']:8.0f} m"
              f"{slope_deg:7.1f}°{st['z0_mean']:8.3f}{sig:+9.2f}  {near}")
        rows.append((tile.tile_id, tlon, tlat, st, slope_deg, terr))

    best = max(rows, key=lambda r: r[3]["dem_relief"])
    print(f"\nhighest relief: {best[0]} at lon {best[1]:.4f}, lat {best[2]:.4f}")
    print(f"  relief {best[3]['dem_relief']:.0f} m, std {best[3]['dem_std']:.0f} m, "
          f"mean slope {best[4]:.1f}°")
    print(f"  land cover: { {k: round(v,3) for k,v in list(best[3]['landcover'].items())[:5]} }")
    print(f"\n  config snippet:\n    - {{id: {best[0]}, lon: {best[1]:.4f}, lat: {best[2]:.4f}}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
