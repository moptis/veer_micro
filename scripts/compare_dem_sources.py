"""Compare SRTM GL1 against USGS 3DEP over our tiles.

    python scripts/compare_dem_sources.py

SRTM (Feb 2000, C-band radar, 1 arc-second) is what FuXi-CFD was trained on.
3DEP 1/3 arc-second is lidar-derived bare earth, an order of magnitude better in
vertical accuracy and ~10 m native.

The physically interesting difference is not accuracy but *what surface each one
describes*. C-band radar scatters from within the canopy, so SRTM over forest sits
above the ground; 3DEP bare earth does not. If that is what separates them, the
difference should correlate with tree cover -- which is testable, and matters,
because for wind the flow arguably sees the canopy top rather than the soil.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling

from fuxicfd_wrf.grids import Tile
from fuxicfd_wrf.terrain._fetch import lonlat_bounds, resample_to_tile
from fuxicfd_wrf.terrain.build import build_terrain
from fuxicfd_wrf.terrain.roughness import NLCD_CLASS_NAMES

TDEP = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/"
        "current/n{n:02d}w{w:03d}/USGS_13_n{n:02d}w{w:03d}.tif")

TILES = {
    "bv_center":  (-103.035, 34.817),
    "wsw_center": (-105.4868, 34.5350),
    "wsw_hi5":    (-106.0163, 34.5112),
    "wsw_hi1":    (-105.7535, 34.2872),
}
#: NLCD forest + woody-wetland classes, for the canopy test.
FOREST = {41, 42, 43, 90}


def tdep_urls(tile: Tile) -> list[str]:
    """3DEP tiles are 1x1 degree, named by their NORTH-WEST corner."""
    minlon, minlat, maxlon, maxlat = lonlat_bounds(tile)
    out = []
    for lat in range(int(np.floor(minlat)), int(np.floor(maxlat)) + 1):
        for lon in range(int(np.floor(minlon)), int(np.floor(maxlon)) + 1):
            out.append(f"/vsicurl/{TDEP.format(n=lat + 1, w=abs(lon))}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("/scratch/moptis/fuxicfd/cache"))
    args = ap.parse_args()
    tf = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32613", always_xy=True)

    print(f"{'tile':<12}{'SRTM relief':>12}{'3DEP relief':>12}{'bias':>9}"
          f"{'RMS diff':>10}{'forest%':>9}{'bias|forest':>12}{'bias|open':>11}")
    print("-" * 88)
    for name, (lon, lat) in TILES.items():
        x, y = tf.transform(lon, lat)
        tile = Tile(name, x=float(x), y=float(y), crs="EPSG:32613")
        srtm_dem = build_terrain(tile, args.cache, source="worldcover").dem.astype(float)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            tdep = resample_to_tile(
                tdep_urls(tile), tile, resampling=Resampling.bilinear, dtype="float64"
            ).astype(float)

        d = tdep - srtm_dem
        nl = np.rint(build_terrain(tile, args.cache, source="nlcd").landcover).astype(int)
        is_forest = np.isin(nl, list(FOREST))
        fb = d[is_forest].mean() if is_forest.any() else float("nan")
        ob = d[~is_forest].mean() if (~is_forest).any() else float("nan")
        print(f"{name:<12}{srtm_dem.max()-srtm_dem.min():10.0f} m"
              f"{tdep.max()-tdep.min():10.0f} m{d.mean():+9.2f}{np.sqrt((d**2).mean()):10.2f}"
              f"{100*is_forest.mean():8.1f}%{fb:+12.2f}{ob:+11.2f}")

    print("\nbias = 3DEP - SRTM (m). C-band radar scatters inside the canopy, so a")
    print("more negative bias over forest than open ground is SRTM sitting on the")
    print("canopy while 3DEP sits on the ground.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
