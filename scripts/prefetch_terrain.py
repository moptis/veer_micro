#!/usr/bin/env python
"""Warm a project's terrain cache without running the model.

    python scripts/prefetch_terrain.py configs/sweep/ph1_dabs_rabs.yaml

Phase 1 of the sweep. Downloads each tile's SRTM and land-cover rasters and
builds every dem_mode x roughness_mode static tensor the sweep will ask for, so
phase 2's 60 cells can run concurrently against a warm cache instead of racing to
fetch the same files.

Terrain only -- deliberately no ONNX session and no inference. The obvious way to
warm the cache is to run one cell of the project and let it populate as a side
effect, but on CPU that means ~1,900 inferences for ph1 (117 tiles x 16 sectors
at the measured 0.91 tile-steps/s), roughly 35 minutes of compute to produce an
output the sweep is about to regenerate on a GPU in three. Fetching the rasters
directly takes minutes and is almost entirely network wait, which also means the
job can ask for a short wall time and get backfilled quickly.

Both cache layers are warmed:

* the **raw** SRTM/land-cover npz, which depends only on tile geometry, so it is
  fetched once and shared by all four mode cells;
* the four **normalized** static tensors, which additionally depend on
  dem_mode/roughness_mode -- cheap to derive once the raw data is local, and
  doing it here keeps phase 2 purely GPU-bound.
"""

from __future__ import annotations

import argparse
import io
import contextlib
import time
from pathlib import Path

from fuxicfd_wrf.config import DownscaleConfig
from fuxicfd_wrf.grids import DomainGeometry
from fuxicfd_wrf.model import assets as assets_mod
from fuxicfd_wrf.pipeline import build_tiles
from fuxicfd_wrf.sources.wrg_source import WrgSectorSource
from fuxicfd_wrf.terrain.build import static_channels

#: Every (dem_mode, roughness_mode) pair the sweep runs. Matches CELLS in
#: scripts/gen_downscale_configs.py.
MODES = [
    ("absolute", "absolute"),
    ("absolute", "recentered"),
    ("recentered", "absolute"),
    ("recentered", "recentered"),
    ("absolute", "constant"),
    ("recentered", "constant"),
    ("absolute", "coarse"),
    ("recentered", "coarse"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="any one cell of the project")
    args = ap.parse_args()

    t0 = time.time()
    cfg = DownscaleConfig.from_yaml(args.config)
    print(f"[prefetch] {cfg.name}: {cfg.source.path}", flush=True)

    # The source summary is noisy and irrelevant here; only the geometry matters.
    with contextlib.redirect_stdout(io.StringIO()):
        src = WrgSectorSource(cfg.source.path, crs=cfg.source.crs, speed_bins=1)
        lat, lon = src.coordinates()
        geom = DomainGeometry(
            lat, lon, crs=src.crs, grid_shape=src.domain_shape,
            affine_fit=cfg.tiling.affine_fit,
        )
        tiles = build_tiles(cfg, geom)

    in_stats = assets_mod.resolve(cfg.model.assets_root).load_input_scaler()
    print(f"[prefetch] {len(tiles)} tiles x {len(MODES)} modes, "
          f"landcover={cfg.terrain.landcover_source}", flush=True)

    for i, tile in enumerate(tiles, 1):
        for dem_mode, rough_mode in MODES:
            static_channels(
                tile, in_stats, cfg.terrain.cache_root,
                dem_mode=dem_mode, roughness_mode=rough_mode,
                roughness_despeckle=cfg.terrain.roughness_despeckle,
                landcover_source=cfg.terrain.landcover_source,
            )
        if i % 10 == 0 or i == len(tiles):
            print(f"[prefetch]   {i}/{len(tiles)} tiles "
                  f"({time.time() - t0:.0f} s)", flush=True)

    print(f"[prefetch] {cfg.name} warm in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
