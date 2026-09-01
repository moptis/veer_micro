"""Summarise a WRF domain: geometry, CRS, tile capacity, centre, and turbines.

    python scripts/describe_domain.py <control.h5> [--turbines <csv>]

Run this before writing a config for a new domain. Grid shape and UTM zone are
read from the store rather than assumed -- Broadview is 78x132 at ~800 m, WSW is
96x114 at ~1000 m, and hard-coding either breaks the other.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyproj

from fuxicfd_wrf.grids import DomainGeometry
from fuxicfd_wrf.sources.h5_source import H5WrfSource
from fuxicfd_wrf.winds import build_stencil_interpolator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", type=Path)
    ap.add_argument("--turbines", type=Path, default=None)
    args = ap.parse_args()

    src = H5WrfSource(args.store)
    lat, lon = src.coordinates()
    try:
        geom = DomainGeometry(lat, lon, crs="auto", grid_shape=src.domain_shape)
    except ValueError as exc:
        if "not affine" not in str(exc):
            raise
        print(f"[geometry] {exc}\n[geometry] describing with affine_fit='local'")
        geom = DomainGeometry(lat, lon, crs="auto", grid_shape=src.domain_shape,
                              affine_fit="local")
    minx, miny, maxx, maxy = geom.extent

    print("=== geometry ===")
    print(f"  grid_shape      : {geom.grid_shape}  ({src.n_points:,} points)")
    print(f"  CRS             : {geom.crs}")
    print(f"  lat/lon         : {lat.min():.4f}..{lat.max():.4f}, "
          f"{lon.min():.4f}..{lon.max():.4f}")
    print(f"  extent          : {(maxx-minx)/1000:.0f} x {(maxy-miny)/1000:.0f} km")
    print(f"  affine residual : {geom.affine_residual_m:.2f} m global"
          + (f", {geom.local_residual_m:.2f} m local" if geom.local_residual_m is not None else ""))
    print(f"  affine_fit      : {geom.affine_fit}")
    print(f"  tiles @ 9 km    : {len(geom.make_tiles())}")
    print(f"  tiles @ 6 km    : {len(geom.make_tiles(stride_m=6000.0))}")

    print("\n=== time ===")
    print("  " + src.time_report().replace("\n", "\n  "))

    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    tf = pyproj.Transformer.from_crs(geom.crs, "EPSG:4326", always_xy=True)
    clon, clat = tf.transform(cx, cy)
    print("\n=== domain centre ===")
    print(f"  lon {clon:.4f}   lat {clat:.4f}")

    tile = geom.tile_at_lonlat(clon, clat, "centre")
    print(f"  stencil inside domain: {geom.covers_stencil(tile)}")
    elev = build_stencil_interpolator(geom, tile).apply(src.elevation())
    print(f"  WRF elevation over the centre stencil: {elev.min():.0f}..{elev.max():.0f} m "
          f"(relief {elev.max()-elev.min():.0f} m)")

    all_elev = src.elevation()
    print(f"  domain elevation: {all_elev.min():.0f}..{all_elev.max():.0f} m")

    if args.turbines and args.turbines.exists():
        import pandas as pd

        tb = pd.read_csv(args.turbines)
        print("\n=== turbines ===")
        print(f"  {len(tb)} turbines, hub heights {sorted(tb.hub_height.unique().tolist())}")
        g = pyproj.Geod(ellps="WGS84")
        for farm, grp in tb.groupby("wind_farm"):
            flon, flat = grp.longitude.mean(), grp.latitude.mean()
            _, _, dist = g.inv(clon, clat, flon, flat)
            t = geom.tile_at_lonlat(flon, flat, farm)
            print(f"  {farm:16s} n={len(grp):4d}  lon {flon:8.3f} lat {flat:7.3f}  "
                  f"{dist/1000:5.1f} km from centre  in-domain={geom.covers_stencil(t)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
