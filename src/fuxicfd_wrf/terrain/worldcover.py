"""ESA WorldCover 10 m land cover from its public S3 bucket.

Tiles are 3x3 degrees, named by the south-west corner snapped down to a multiple
of 3, e.g. ``ESA_WorldCover_10m_2021_v200_N33W105_Map.tif``.

Land cover is resampled with nearest neighbour, not bilinear -- averaging class
*codes* would produce meaningless intermediates (the mean of "water" 80 and
"herbaceous wetland" 90 is 85, which is not a class). The z0 lookup is applied
after resampling for the same reason.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from rasterio.enums import Resampling

from ..grids import Tile
from ._fetch import download, lonlat_bounds, resample_to_tile

VERSION = "v200"
YEAR = 2021
BASE_URL = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/{VERSION}/{YEAR}/map"


def _snap3(value: float) -> int:
    return int(math.floor(value / 3.0) * 3)


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def covering_tiles(bounds: tuple[float, float, float, float]) -> list[str]:
    """3-degree WorldCover tile names covering (minlon, minlat, maxlon, maxlat)."""
    minlon, minlat, maxlon, maxlat = bounds
    names = []
    for lat in range(_snap3(minlat), _snap3(maxlat) + 1, 3):
        for lon in range(_snap3(minlon), _snap3(maxlon) + 1, 3):
            names.append(tile_name(lat, lon))
    return names


def ensure_tiles(bounds, cache_root: Path) -> list[Path]:
    cache = Path(cache_root) / "worldcover"
    paths = []
    for name in covering_tiles(bounds):
        fname = f"ESA_WorldCover_10m_{YEAR}_{VERSION}_{name}_Map.tif"
        try:
            paths.append(download(f"{BASE_URL}/{fname}", cache / fname))
        except FileNotFoundError:
            print(f"[terrain] WorldCover tile {name} not published, skipping")
    if not paths:
        raise FileNotFoundError(f"no WorldCover tiles available for bounds {bounds}")
    return paths


def landcover_for_tile(tile: Tile, cache_root: Path) -> np.ndarray:
    """(301, 301) WorldCover class codes on the tile's grid, nearest-neighbour."""
    paths = ensure_tiles(lonlat_bounds(tile), cache_root)
    return resample_to_tile(
        paths, tile, resampling=Resampling.nearest, dtype="float32", nodata=0
    )
