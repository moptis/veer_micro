"""YAML-backed run configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .constants import TILE_SPAN_M
from .model.assets import default_assets_root, default_cache_root


class SourceConfig(BaseModel):
    kind: Literal["h5", "npz"] = "h5"
    path: Path
    #: npz files carry no coordinates; point this at a control.h5 to borrow them.
    coordinates_from: Path | None = None


class TimeConfig(BaseModel):
    start: str | None = None
    end: str | None = None
    #: Subsamples the *chronologically sorted* selection, not the file rows.
    stride: int = 1
    limit: int | None = None


class SiteTile(BaseModel):
    id: str
    lon: float
    lat: float


class BBox(BaseModel):
    """A geographic bounding box, in degrees.

    Used to run only the part of a WRF domain you actually need at 30 m. The
    coarse domain is usually far larger than the area of interest, and inference
    cost is linear in tile count, so this is normally the single biggest lever on
    run cost.
    """

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @model_validator(mode="after")
    def _check(self):
        if self.min_lon >= self.max_lon:
            raise ValueError(
                f"bbox min_lon ({self.min_lon}) must be < max_lon ({self.max_lon})"
            )
        if self.min_lat >= self.max_lat:
            raise ValueError(
                f"bbox min_lat ({self.min_lat}) must be < max_lat ({self.max_lat})"
            )
        # Catches the classic swapped-argument mistake before it silently returns
        # zero tiles half an hour into a job.
        for name, v, lim in (("min_lon", self.min_lon, 180), ("max_lon", self.max_lon, 180),
                             ("min_lat", self.min_lat, 90), ("max_lat", self.max_lat, 90)):
            if not -lim <= v <= lim:
                raise ValueError(f"bbox {name}={v} is out of range for degrees "
                                 f"(expected -{lim}..{lim}); lon/lat swapped?")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


class TilingConfig(BaseModel):
    mode: Literal["sites", "domain"] = "sites"
    #: Projected CRS tiles are laid out in. "auto" picks the UTM zone containing
    #: the domain centre, which is what you want unless comparing across domains.
    working_crs: str = "auto"
    #: mode="sites"
    sites: list[SiteTile] = Field(default_factory=list)
    #: mode="domain": tile centre spacing. Less than 9000 m gives overlap for blending.
    stride_m: float = TILE_SPAN_M
    #: mode="domain": restrict to tiles whose core intersects this lon/lat box.
    #: The tile lattice is unchanged -- tiles are filtered, not re-originated --
    #: so a subset run keeps the same centres and ids as the full-domain run and
    #: the two stores stay directly comparable.
    bbox: BBox | None = None
    #: Refuse to run if the selected tiles cover less than this fraction of the
    #: requested bbox. A box overhanging the domain edge is legitimate and only
    #: costs a few tiles, but a box mostly outside it is a typo -- and without
    #: this the run proceeds happily on whatever fraction did intersect, which is
    #: only visible by comparing two lon/lat ranges in the log. Set 0.0 to allow
    #: any partial coverage.
    min_bbox_coverage: float = 0.9
    max_tiles: int | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.mode == "sites" and not self.sites:
            raise ValueError("tiling.mode='sites' requires a non-empty tiling.sites list")
        if self.mode == "sites" and self.bbox is not None:
            raise ValueError(
                "tiling.bbox applies to mode='domain'; in mode='sites' the tiles "
                "are already given explicitly by tiling.sites"
            )
        if self.stride_m <= 0 or self.stride_m > TILE_SPAN_M:
            raise ValueError(f"tiling.stride_m must be in (0, {TILE_SPAN_M}]")
        if not 0.0 <= self.min_bbox_coverage <= 1.0:
            raise ValueError(
                f"tiling.min_bbox_coverage must be a fraction in [0, 1], got "
                f"{self.min_bbox_coverage}"
            )
        return self

    @property
    def overlaps(self) -> bool:
        return self.mode == "domain" and self.stride_m < TILE_SPAN_M


class TerrainConfig(BaseModel):
    #: Land-cover dataset behind the roughness channel.
    #:
    #: "nlcd" (US only, 30 m) separates four development intensities, so rural
    #: roads and verges no longer inherit an urban z0 -- the cause of the
    #: single-pixel roughness spikes that produced localised wind artifacts.
    #: "worldcover" is global and is what FuXi-CFD was trained on. "auto" (the
    #: default) uses NLCD where it has coverage and falls back to WorldCover
    #: elsewhere -- so US domains get the better data and non-US ones still run.
    landcover_source: Literal["auto", "nlcd", "worldcover"] = "auto"
    #: Median-filter width (px) applied to z0 to remove isolated spikes. 0 = off.
    #: Largely unnecessary with NLCD, which fixes the values at source.
    roughness_despeckle: int = 0

    #: See terrain/build.py for why these exist -- Broadview is ~4 sigma off the
    #: training distribution on both static channels.
    dem_mode: Literal["absolute", "recentered"] = "absolute"
    roughness_mode: Literal["absolute", "recentered"] = "absolute"
    cache_root: Path = Field(default_factory=default_cache_root)


class ModelConfig(BaseModel):
    assets_root: Path = Field(default_factory=default_assets_root)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 8
    #: Timesteps of WRF wind read from the source per I/O call.
    #:
    #: control.h5 is chunked (411, 81), so one timestep costs the same read as
    #: hundreds. Reading per output chunk left the first production run I/O bound
    #: at 4.8 tile-steps/s versus ~42 on the GPU. 822 = two h5 row-chunks; each
    #: block holds ~68 MB of both wind variables.
    read_block: int = 822


class OutputConfig(BaseModel):
    path: Path
    dtype: Literal["float16", "float32"] = "float16"
    variables: list[str] = Field(default_factory=lambda: ["u", "v", "w", "k"])
    #: Subset of the 27 heights to keep, by height in metres. None keeps all.
    levels: list[float] | None = None
    #: Chunk length along time. Slurm shards are aligned to this so array tasks
    #: never write to the same chunk concurrently.
    time_chunk: int = 6
    mosaic: bool = False

    @model_validator(mode="after")
    def _check(self):
        known = {"u", "v", "w", "k"}
        bad = set(self.variables) - known
        if bad:
            raise ValueError(f"output.variables has unknown entries {sorted(bad)}")
        if self.time_chunk < 1:
            raise ValueError("output.time_chunk must be >= 1")
        return self


class RunConfig(BaseModel):
    name: str = "run"
    source: SourceConfig
    time: TimeConfig = Field(default_factory=TimeConfig)
    tiling: TilingConfig
    terrain: TerrainConfig = Field(default_factory=TerrainConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RunConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        cfg = cls.model_validate(raw)
        if cfg.output.mosaic and not cfg.tiling.overlaps:
            print(
                "[config] WARNING: output.mosaic is on but tiles do not overlap "
                f"(stride_m={cfg.tiling.stride_m}). FuXi-CFD sees only the 9 km "
                "window with no buffer, so tile edges will show seams. Consider "
                "tiling.stride_m: 6000."
            )
        return cfg
