"""Shared remote-raster download and reprojection helpers.

Both terrain sources follow the same shape: figure out which global tiles cover a
bounding box, cache them under $FUXICFD_CACHE, then resample onto a tile's exact
301x301 projected grid.

Resampling uses `WarpedVRT`, which reprojects lazily and reads only the source
window that intersects the destination. That matters for ESA WorldCover, where one
3-degree tile is 36000x36000 -- a naive full read would pull 1.3 GB to fill a 9 km
box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from ..constants import TERRAIN_RES_M, TERRAIN_SHAPE
from ..grids import Tile


def download(url: str, dest: Path, *, timeout: int = 300) -> Path:
    """Download `url` to `dest` unless already cached. Atomic via .part rename."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[terrain] fetching {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        if r.status_code == 404:
            raise FileNotFoundError(f"tile not available upstream: {url}")
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 22):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def tile_transform(tile: Tile):
    """Affine transform for the tile's 301x301 north-up grid.

    Grid nodes are pixel *centres*, so the raster origin sits half a cell
    (15 m) outside the node extent -- the usual off-by-half that otherwise shifts
    terrain by one pixel relative to the wind stencil.
    """
    xs, ys = tile.terrain_axes()
    half = TERRAIN_RES_M / 2
    return from_origin(xs[0] - half, ys[0] + half, TERRAIN_RES_M, TERRAIN_RES_M)


def resample_to_tile(
    sources: list[Path],
    tile: Tile,
    *,
    resampling: Resampling,
    dtype: str = "float32",
    nodata: float | None = None,
) -> np.ndarray:
    """Reproject and mosaic `sources` onto the tile's 301x301 grid.

    Sources are composited in order; later sources only fill pixels still unset,
    so overlapping global tiles do not fight over seam pixels.
    """
    if not sources:
        raise ValueError("no source rasters given")

    dst_transform = tile_transform(tile)
    out = np.full(TERRAIN_SHAPE, np.nan, dtype="float64")

    for path in sources:
        with rasterio.open(path) as src:
            src_nodata = src.nodata if nodata is None else nodata
            with WarpedVRT(
                src,
                crs=tile.crs,
                transform=dst_transform,
                width=TERRAIN_SHAPE[1],
                height=TERRAIN_SHAPE[0],
                resampling=resampling,
                src_nodata=src_nodata,
                nodata=src_nodata,
            ) as vrt:
                block = vrt.read(1).astype("float64")
        valid = np.isfinite(block)
        if src_nodata is not None:
            valid &= block != src_nodata
        fill = valid & ~np.isfinite(out)
        out[fill] = block[fill]

    if not np.isfinite(out).all():
        n_bad = int((~np.isfinite(out)).sum())
        # `sources` may hold Paths or plain strings (an S3 URI, for one), so take
        # the basename textually rather than assuming Path.name -- this raise is
        # the only report of a genuine coverage hole, and an AttributeError here
        # replaces that diagnostic with a traceback into the error handler.
        names = [str(s).rstrip("/").rsplit("/", 1)[-1] for s in sources]
        raise ValueError(
            f"tile {tile.tile_id}: {n_bad}/{out.size} terrain pixels unfilled. "
            f"Sources {names} do not fully cover "
            f"{tile.bounds()} in {tile.crs}."
        )
    return out.astype(dtype)


def lonlat_bounds(tile: Tile, *, pad_deg: float = 0.02) -> tuple[float, float, float, float]:
    """Geographic bounding box of a tile, padded so resampling has neighbours."""
    lon, lat = tile.terrain_lonlat()
    return (
        float(lon.min()) - pad_deg,
        float(lat.min()) - pad_deg,
        float(lon.max()) + pad_deg,
        float(lat.max()) + pad_deg,
    )
