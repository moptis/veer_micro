"""Show where a run's tiles sit inside the WRF domain, and how much they cover.

    python scripts/show_tiles.py <store.zarr> [--source <control.h5>]

Answers the question "what am I actually getting?" for a site-mode run, where the
tile list is a handful of chosen points rather than full coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyproj
import zarr

from fuxicfd_wrf.constants import TILE_SPAN_M
from fuxicfd_wrf.grids import DomainGeometry
from fuxicfd_wrf.sources.h5_source import H5WrfSource

DEFAULT_H5 = "/scratch/moptis/c2wind/broadview/control/processed/broadview/control.h5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", type=Path)
    ap.add_argument("--source", type=Path, default=Path(DEFAULT_H5))
    args = ap.parse_args()

    src = H5WrfSource(args.source)
    lat, lon = src.coordinates()
    geom = DomainGeometry(lat, lon)
    minx, miny, maxx, maxy = geom.extent
    dom_km2 = (maxx - minx) * (maxy - miny) / 1e6

    root = zarr.open_group(str(args.store), mode="r")
    tiles = sorted(root.group_keys())
    if not tiles:
        print("mosaic store: covers the full tiled domain, no per-tile list")
        return 0

    print(f"{'tile':<16}{'lon':>10}{'lat':>9}   {'position in domain':<24}")
    print("-" * 62)
    rows = []
    for t in tiles:
        g = root[t]
        x, y = float(g.attrs["center_x"]), float(g.attrs["center_y"])
        tf = pyproj.Transformer.from_crs(g.attrs["crs"], "EPSG:4326", always_xy=True)
        lo, la = tf.transform(x, y)
        fx = 100 * (x - minx) / (maxx - minx)
        fy = 100 * (y - miny) / (maxy - miny)
        ew = "W" if fx < 40 else ("E" if fx > 60 else "centre")
        ns = "S" if fy < 40 else ("N" if fy > 60 else "centre")
        where = f"{fx:3.0f}% E, {fy:3.0f}% N  ({ns}-{ew})"
        print(f"{t:<16}{lo:10.3f}{la:9.3f}   {where:<24}")
        rows.append((x, y))

    cov = len(tiles) * (TILE_SPAN_M / 1000) ** 2
    print()
    print(f"domain              : {(maxx-minx)/1000:.0f} x {(maxy-miny)/1000:.0f} km"
          f" = {dom_km2:,.0f} km2")
    print(f"covered by {len(tiles)} tiles   : {cov:,.0f} km2"
          f" = {100*cov/dom_km2:.1f}% of the domain")
    print(f"full coverage needs : 57 tiles at 9 km, or 140 at 6 km stride (mosaic)")

    # Crude ASCII map so the spread is obvious at a glance.
    W, H = 46, 14
    grid = [["."] * W for _ in range(H)]
    for i, (x, y) in enumerate(rows):
        cx = int((x - minx) / (maxx - minx) * (W - 1))
        cy = int((1 - (y - miny) / (maxy - miny)) * (H - 1))
        grid[cy][cx] = str(i + 1)
    print("\ndomain map (N up, 1 char ~ 2.3 x 4.6 km):")
    print("  +" + "-" * W + "+")
    for row in grid:
        print("  |" + "".join(row) + "|")
    print("  +" + "-" * W + "+")
    for i, t in enumerate(tiles):
        print(f"   {i+1} = {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
