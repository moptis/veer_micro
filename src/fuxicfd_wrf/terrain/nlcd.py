"""NLCD (National Land Cover Database) land cover for US domains.

Annual NLCD Collection 1, served as Cloud-Optimized GeoTIFFs from USGS's S3
bucket. Preferred over ESA WorldCover wherever it has coverage, for four reasons:

1. **Four developed-intensity classes** instead of WorldCover's single "built-up".
   That single class covers everything from a farm track to a city centre, which
   is what put rural roads at an urban z0 of 0.8 m and produced 3.8-sigma
   single-pixel spikes in the roughness field. Measured on the Broadview tile,
   75% of the pixels WorldCover called built-up are NLCD "developed, open space"
   (verges and lawns, z0 ~ 0.05 m) and only 3 of 83 are medium-intensity
   development. A further 12% are not developed at all.
2. **30 m native**, matching FuXi-CFD's terrain grid exactly; WorldCover is 10 m
   and has to be downsampled.
3. **Better vegetation discrimination** -- deciduous/evergreen/mixed forest,
   pasture vs cultivated crops -- which carry genuinely different z0.
4. **Annual products**, so land cover can be matched to the WRF period rather
   than pinned to WorldCover's 2021 snapshot.

Coverage is CONUS + Alaska + Hawaii only. Outside that the reader reports what
fraction of the tile is nodata so the caller can fall back to WorldCover.

The bucket is **requester-pays**: reads are billed to the caller's AWS account.
Because these are COGs we window-read a single 9 km tile -- a few hundred KB, and
fractions of a cent -- rather than the 1.4 GB CONUS mosaic. The result is then
cached per tile, so S3 is touched once per tile ever.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from ..grids import Tile
from ._fetch import resample_to_tile

BUCKET = "usgs-landcover"
PREFIX = "annual-nlcd/c1/v0/cu/mosaic"

#: Latest year in Collection 1 at time of writing.
DEFAULT_YEAR = 2023

#: Value marking "outside the mapped area" in the NLCD mosaic.
NODATA = 250


def url_for(year: int = DEFAULT_YEAR) -> str:
    return f"/vsis3/{BUCKET}/{PREFIX}/Annual_NLCD_LndCov_{year}_CU_C1V0.tif"


def landcover_for_tile(
    tile: Tile, cache_root: Path | None = None, *, year: int = DEFAULT_YEAR
) -> tuple[np.ndarray, float]:
    """(301, 301) NLCD class codes on the tile grid, plus the covered fraction.

    Returns `(classes, covered)` where `covered` is the fraction of pixels with
    real data. A tile outside CONUS comes back all-nodata with covered == 0.
    """
    # Requester-pays and the COG hints are set here rather than relied upon from
    # the environment, so Slurm jobs behave the same as an interactive shell.
    with rasterio.Env(
        AWS_REQUEST_PAYER="requester",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    ):
        arr = resample_to_tile(
            [url_for(year)], tile, resampling=Resampling.nearest,
            dtype="float32", nodata=NODATA,
        )
    codes = np.rint(arr).astype(np.int32)
    covered = float((codes != NODATA).mean())
    return codes, covered


def covers(tile: Tile, *, year: int = DEFAULT_YEAR, min_fraction: float = 0.99) -> bool:
    """True if NLCD has data over essentially the whole tile."""
    try:
        _, frac = landcover_for_tile(tile, year=year)
    except Exception:
        return False
    return frac >= min_fraction
