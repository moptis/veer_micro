"""Assemble the per-tile static channels and cache them.

Two levels of cache, because they invalidate on different things:

* **Raw** ``dem`` / ``roughness`` / ``landcover`` at (301, 301) in physical units.
  Depends only on tile geometry. Written as ``<tile_id>_raw.npz``.
* **Normalized** (2, 300, 300) tensor ready to concatenate with the wind channels.
  Depends additionally on the scalers and on dem_mode/roughness_mode, so it is
  keyed by those. Written as ``<tile_id>_static_<key>.npy``.

The second cache is the one that pays: it removes a normalize + zoom from every
single inference, and the static channels are identical across all 52560 timesteps.

Distribution shift
------------------
Broadview sits at 1183-1503 m and is grassland/cropland (z0 ~ 0.03-0.05 m).
FuXi-CFD trained on dem mean 351.9 +/- 227.6 m and roughness mean 0.817 +/- 0.202 m.
Both static channels therefore standardize to roughly +/-4 sigma -- far outside
what the model saw in training.

`dem_mode`/`roughness_mode` expose the two defensible responses:

* ``absolute``   - feed true physical values, as the model card instructs.
* ``recentered`` - keep the per-tile *anomaly* (relief, which is what actually
  drives terrain-induced flow) but shift the tile mean onto the training mean, so
  the model operates in the input range it was fitted on.
* ``constant``   - replace the field with the training mean everywhere, so the
  channel normalizes to exactly 0 and carries no signal at all. A diagnostic, not
  a production setting: whatever spatial structure survives came from the *other*
  static channel and the wind stencil, which is the only clean way to attribute
  the output pattern between terrain and roughness. Note a zero-variance field is
  itself mildly off-distribution -- the model was fitted on varying inputs -- so
  read it as an attribution experiment rather than a well-posed forward run. Do
  not use 0.0 for this: z0=0 is -4.05 sigma, further out than any real surface,
  and a singular log-law besides.

Neither is obviously right. Phase 0 decides between them empirically by comparing
tile-mean vertical profiles against the WRF profile the site already provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import TERRAIN_SHAPE
from ..grids import Tile
from ..model.preprocess import normalize_static
from . import roughness as rough_mod
from . import nlcd, srtm, worldcover

STATIC_MODES = ("absolute", "recentered", "constant")
LANDCOVER_SOURCES = ("auto", "nlcd", "worldcover")


@dataclass
class TerrainTile:
    """Physical-unit static fields for one tile."""

    tile_id: str
    dem: np.ndarray          # (301,301) m
    roughness: np.ndarray    # (301,301) m
    landcover: np.ndarray    # (301,301) class codes, meaning depends on `source`
    source: str = "worldcover"

    def stats(self) -> dict:
        return {
            "dem_min": float(self.dem.min()),
            "dem_max": float(self.dem.max()),
            "dem_mean": float(self.dem.mean()),
            "dem_relief": float(self.dem.max() - self.dem.min()),
            "dem_std": float(self.dem.std()),
            "z0_mean": float(self.roughness.mean()),
            "z0_min": float(self.roughness.min()),
            "z0_max": float(self.roughness.max()),
            "landcover_source": self.source,
            "landcover": rough_mod.class_histogram(self.landcover, self.source),
        }


def build_terrain(
    tile: Tile,
    cache_root: Path,
    *,
    source: str = "auto",
    force: bool = False,
) -> TerrainTile:
    """Fetch/resample raw terrain for a tile, using the raw cache when present.

    `source` picks the land-cover dataset:

    * ``nlcd``       - Annual NLCD Collection 1 (CONUS/AK/HI, 30 m). Preferred
      where available: it separates four development intensities, so rural roads
      and verges no longer inherit an urban z0.
    * ``worldcover`` - ESA WorldCover (global, 10 m). The fallback, and what
      FuXi-CFD was trained on.
    * ``auto``       - try NLCD, fall back to WorldCover if the tile is not fully
      covered (outside the US).

    The cache is keyed by source, so switching does not silently return a tile
    built from the other dataset.
    """
    if source not in LANDCOVER_SOURCES:
        raise ValueError(f"source={source!r} not in {LANDCOVER_SOURCES}")
    cache_dir = Path(cache_root) / "tiles"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache(src: str) -> Path:
        # Keyed by geo_key, not tile_id alone: ids repeat across domains and a
        # collision serves another site's terrain with no error. See Tile.geo_key.
        stem = f"{tile.tile_id}-{tile.geo_key}"
        return cache_dir / (f"{stem}_raw.npz" if src == "worldcover"
                            else f"{stem}_raw_{src}.npz")

    for src in (("nlcd", "worldcover") if source == "auto" else (source,)):
        path = _cache(src)
        if path.exists() and not force:
            with np.load(path) as z:
                # z0 is DERIVED from the cached classes, never read back from
                # the cache. The classes are the downloaded data; z0 is a
                # function of a table that can change under them, so a cached z0
                # can be stale while the classes beside it are still correct.
                return TerrainTile(
                    tile_id=tile.tile_id, dem=z["dem"],
                    roughness=rough_mod.classes_to_z0(
                        z["landcover"], src),
                    landcover=z["landcover"], source=src,
                )

    dem = srtm.dem_for_tile(tile, cache_root)

    used, landcover = None, None
    if source in ("auto", "nlcd"):
        codes, covered = nlcd.landcover_for_tile(tile, cache_root)
        if covered >= 0.99:
            used, landcover = "nlcd", codes
        elif source == "nlcd":
            raise ValueError(
                f"NLCD covers only {covered:.1%} of {tile.tile_id}; it is "
                "CONUS/AK/HI only. Use source='auto' or 'worldcover'."
            )
        else:
            print(f"[terrain] NLCD covers {covered:.1%} of {tile.tile_id}; "
                  "falling back to WorldCover")
    if used is None:
        used, landcover = "worldcover", worldcover.landcover_for_tile(tile, cache_root)

    z0 = rough_mod.classes_to_z0(landcover, used)
    for name, arr in (("dem", dem), ("roughness", z0), ("landcover", landcover)):
        if arr.shape != TERRAIN_SHAPE:
            raise ValueError(f"{name} for {tile.tile_id} is {arr.shape}, expected {TERRAIN_SHAPE}")

    np.savez_compressed(_cache(used), dem=dem, roughness=z0, landcover=landcover)
    return TerrainTile(tile_id=tile.tile_id, dem=dem, roughness=z0,
                       landcover=landcover, source=used)


def _recenter(field: np.ndarray, target_mean: float) -> np.ndarray:
    """Shift a field's mean to `target_mean`, preserving all spatial structure."""
    return field - float(field.mean()) + target_mean


def apply_modes(
    terrain: TerrainTile,
    in_stats: dict,
    *,
    dem_mode: str = "absolute",
    roughness_mode: str = "absolute",
    roughness_despeckle: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the distribution-shift mitigation, returning physical-unit fields."""
    for name, mode in (("dem_mode", dem_mode), ("roughness_mode", roughness_mode)):
        if mode not in STATIC_MODES:
            raise ValueError(f"{name}={mode!r} not in {STATIC_MODES}")

    dem = terrain.dem
    z0 = terrain.roughness
    if roughness_despeckle:
        z0 = rough_mod.despeckle(z0, size=roughness_despeckle)
    if dem_mode == "recentered":
        dem = _recenter(dem, float(np.asarray(in_stats["high_mean"])[0]))
    elif dem_mode == "constant":
        dem = np.full_like(dem, float(np.asarray(in_stats["high_mean"])[0]))
    if roughness_mode == "recentered":
        z0 = _recenter(z0, float(np.asarray(in_stats["high_mean"])[1]))
    elif roughness_mode == "constant":
        z0 = np.full_like(z0, float(np.asarray(in_stats["high_mean"])[1]))
    return dem, z0


def _z0_source(landcover_source: str) -> str:
    """Which z0 table a landcover_source will end up using, for cache keying.

    "auto" resolves per tile, so key it on NLCD -- the table it takes wherever it
    has coverage. A WorldCover fallback tile is then keyed slightly pessimistically,
    which costs a rebuild and never a wrong answer.
    """
    return "worldcover" if landcover_source == "worldcover" else "nlcd"


def static_channels(
    tile: Tile,
    in_stats: dict,
    cache_root: Path,
    *,
    dem_mode: str = "absolute",
    roughness_mode: str = "absolute",
    roughness_despeckle: int = 0,
    landcover_source: str = "auto",
    force: bool = False,
) -> np.ndarray:
    """Normalized, zoomed (2, 300, 300) static tensor, cached per (tile, mode)."""
    cache_dir = Path(cache_root) / "tiles"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{dem_mode[:3]}-{roughness_mode[:3]}"
    if roughness_despeckle:
        key += f"-ds{roughness_despeckle}"
    if landcover_source != "worldcover":
        key += f"-{landcover_source}"
    # The z0 values are part of this tensor's identity, so the key carries a hash
    # of the table actually used. A key naming only the source cannot tell that
    # the values behind it changed, and would have handed back tensors built with
    # the old table when UL's values replaced them.
    key += f"-z{rough_mod.table_fingerprint(_z0_source(landcover_source))}"
    path = cache_dir / f"{tile.tile_id}-{tile.geo_key}_static_{key}.npy"

    if path.exists() and not force:
        return np.load(path)

    terrain = build_terrain(tile, cache_root, source=landcover_source, force=force)
    dem, z0 = apply_modes(
        terrain, in_stats, dem_mode=dem_mode, roughness_mode=roughness_mode,
        roughness_despeckle=roughness_despeckle,
    )
    static = normalize_static(dem, z0, in_stats)
    np.save(path, static)
    return static


def sigma_offsets(terrain: TerrainTile, in_stats: dict) -> dict:
    """How far this tile's static channels sit from the training distribution.

    Reported in standard deviations of the training data. Values beyond about
    +/-3 mean the model is extrapolating and its output deserves suspicion.
    """
    high_mean = np.asarray(in_stats["high_mean"], dtype=float)
    high_std = np.asarray(in_stats["high_std"], dtype=float)
    return {
        "dem_sigma_mean": float((terrain.dem.mean() - high_mean[0]) / high_std[0]),
        "dem_sigma_min": float((terrain.dem.min() - high_mean[0]) / high_std[0]),
        "dem_sigma_max": float((terrain.dem.max() - high_mean[0]) / high_std[0]),
        "z0_sigma_mean": float((terrain.roughness.mean() - high_mean[1]) / high_std[1]),
        "z0_sigma_min": float((terrain.roughness.min() - high_mean[1]) / high_std[1]),
        "z0_sigma_max": float((terrain.roughness.max() - high_mean[1]) / high_std[1]),
    }
