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
from . import srtm, worldcover

STATIC_MODES = ("absolute", "recentered")


@dataclass
class TerrainTile:
    """Physical-unit static fields for one tile."""

    tile_id: str
    dem: np.ndarray          # (301,301) m
    roughness: np.ndarray    # (301,301) m
    landcover: np.ndarray    # (301,301) WorldCover class codes

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
            "landcover": rough_mod.class_histogram(self.landcover),
        }


def build_terrain(tile: Tile, cache_root: Path, *, force: bool = False) -> TerrainTile:
    """Fetch/resample raw terrain for a tile, using the raw cache when present."""
    cache_dir = Path(cache_root) / "tiles"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / f"{tile.tile_id}_raw.npz"

    if raw_path.exists() and not force:
        with np.load(raw_path) as z:
            return TerrainTile(
                tile_id=tile.tile_id,
                dem=z["dem"],
                roughness=z["roughness"],
                landcover=z["landcover"],
            )

    dem = srtm.dem_for_tile(tile, cache_root)
    landcover = worldcover.landcover_for_tile(tile, cache_root)
    z0 = rough_mod.classes_to_z0(landcover)

    for name, arr in (("dem", dem), ("roughness", z0), ("landcover", landcover)):
        if arr.shape != TERRAIN_SHAPE:
            raise ValueError(f"{name} for {tile.tile_id} is {arr.shape}, expected {TERRAIN_SHAPE}")

    np.savez_compressed(raw_path, dem=dem, roughness=z0, landcover=landcover)
    return TerrainTile(tile_id=tile.tile_id, dem=dem, roughness=z0, landcover=landcover)


def _recenter(field: np.ndarray, target_mean: float) -> np.ndarray:
    """Shift a field's mean to `target_mean`, preserving all spatial structure."""
    return field - float(field.mean()) + target_mean


def apply_modes(
    terrain: TerrainTile,
    in_stats: dict,
    *,
    dem_mode: str = "absolute",
    roughness_mode: str = "absolute",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the distribution-shift mitigation, returning physical-unit fields."""
    for name, mode in (("dem_mode", dem_mode), ("roughness_mode", roughness_mode)):
        if mode not in STATIC_MODES:
            raise ValueError(f"{name}={mode!r} not in {STATIC_MODES}")

    dem = terrain.dem
    z0 = terrain.roughness
    if dem_mode == "recentered":
        dem = _recenter(dem, float(np.asarray(in_stats["high_mean"])[0]))
    if roughness_mode == "recentered":
        z0 = _recenter(z0, float(np.asarray(in_stats["high_mean"])[1]))
    return dem, z0


def static_channels(
    tile: Tile,
    in_stats: dict,
    cache_root: Path,
    *,
    dem_mode: str = "absolute",
    roughness_mode: str = "absolute",
    force: bool = False,
) -> np.ndarray:
    """Normalized, zoomed (2, 300, 300) static tensor, cached per (tile, mode)."""
    cache_dir = Path(cache_root) / "tiles"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{dem_mode[:3]}-{roughness_mode[:3]}"
    path = cache_dir / f"{tile.tile_id}_static_{key}.npy"

    if path.exists() and not force:
        return np.load(path)

    terrain = build_terrain(tile, cache_root, force=force)
    dem, z0 = apply_modes(
        terrain, in_stats, dem_mode=dem_mode, roughness_mode=roughness_mode
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
