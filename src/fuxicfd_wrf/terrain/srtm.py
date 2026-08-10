"""SRTM GL1 (30 m) elevation from the OpenTopography public S3 bucket.

FuXi-CFD was trained on SRTM, so this is the matching source. Tiles are 1x1 degree
GeoTIFFs named by their south-west corner, e.g. ``N34W104.tif``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from rasterio.enums import Resampling

from ..grids import Tile
from ._fetch import download, lonlat_bounds, resample_to_tile

BASE_URL = "https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm"


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def covering_tiles(bounds: tuple[float, float, float, float]) -> list[str]:
    """1-degree SRTM tile names covering (minlon, minlat, maxlon, maxlat)."""
    minlon, minlat, maxlon, maxlat = bounds
    names = []
    for lat in range(math.floor(minlat), math.floor(maxlat) + 1):
        for lon in range(math.floor(minlon), math.floor(maxlon) + 1):
            names.append(tile_name(lat, lon))
    return names


def ensure_tiles(bounds, cache_root: Path) -> list[Path]:
    cache = Path(cache_root) / "srtm"
    paths = []
    for name in covering_tiles(bounds):
        try:
            paths.append(download(f"{BASE_URL}/{name}.tif", cache / f"{name}.tif"))
        except FileNotFoundError:
            # SRTM has no tiles over open ocean; a partial set is fine as long as
            # resample_to_tile ends up fully covered, which it checks.
            print(f"[terrain] SRTM tile {name} not published (ocean?), skipping")
    if not paths:
        raise FileNotFoundError(f"no SRTM tiles available for bounds {bounds}")
    return paths


def dem_for_tile(tile: Tile, cache_root: Path) -> np.ndarray:
    """(301, 301) float32 elevation in metres on the tile's grid."""
    paths = ensure_tiles(lonlat_bounds(tile), cache_root)
    return resample_to_tile(paths, tile, resampling=Resampling.bilinear)
