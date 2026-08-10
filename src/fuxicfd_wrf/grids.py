"""Tile geometry and the WRF-domain <-> projected-grid mapping.

Array orientation convention (used everywhere downstream)
---------------------------------------------------------
All 2-D tile arrays -- ``dem``, ``roughness``, and the model's (y, x) output axes --
are **north-up raster order**: axis -2 is y with row 0 at the *northern* edge
(y decreasing), axis -1 is x increasing eastward. This is what rasterio/SRTM give
natively and is the overwhelmingly likely convention for a geospatial ML dataset,
but upstream never states it.

It is an assumption, and Phase 0 tests it rather than trusting it: the physical
closure check compares the tile-mean FuXi-CFD wind at ~100 m against the WRF wind
that was fed in, reporting u and v bias *separately*. A flipped y axis shows up
immediately as a sign error in v while u stays fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyproj

from .constants import (
    COARSE_RES_M,
    COARSE_SHAPE,
    GEOGRAPHIC_CRS,
    MODEL_SHAPE,
    TERRAIN_RES_M,
    TERRAIN_SHAPE,
    TILE_SPAN_M,
    WORKING_CRS,
    WRF_GRID_SHAPE,
    WRF_N_POINTS,
)

#: Half-width of a tile's coarse stencil: outermost cell centres sit +/- 4 km.
COARSE_HALF_EXTENT_M = (COARSE_SHAPE[0] - 1) / 2 * COARSE_RES_M  # 4000.0
#: Half-width of the terrain footprint: +/- 4.5 km, i.e. the coarse cell *edges*.
TERRAIN_HALF_EXTENT_M = TILE_SPAN_M / 2  # 4500.0


def _check_grid_consistency() -> None:
    """The two input grids must describe the same 9 km x 9 km box."""
    span = (TERRAIN_SHAPE[0] - 1) * TERRAIN_RES_M
    if span != TILE_SPAN_M:
        raise AssertionError(f"terrain grid spans {span} m, expected {TILE_SPAN_M}")
    if COARSE_SHAPE[0] * COARSE_RES_M != TILE_SPAN_M:
        raise AssertionError("coarse grid footprint disagrees with tile span")


_check_grid_consistency()


@dataclass(frozen=True)
class Tile:
    """A single 9 km x 9 km inference tile, defined by its centre in `crs`."""

    tile_id: str
    x: float
    y: float
    crs: str = WORKING_CRS

    def coarse_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """1-D x (east) and y (north-up, descending) axes of the 9x9 wind stencil."""
        off = np.arange(COARSE_SHAPE[0]) * COARSE_RES_M - COARSE_HALF_EXTENT_M
        return self.x + off, self.y - off  # y descends: row 0 is north

    def terrain_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """1-D x and y axes of the 301x301 terrain grid, north-up."""
        off = np.arange(TERRAIN_SHAPE[0]) * TERRAIN_RES_M - TERRAIN_HALF_EXTENT_M
        return self.x + off, self.y - off

    def model_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """1-D x and y axes of the model's 300x300 output grid, north-up.

        This is **not** the first 300 of the 301 terrain nodes. Upstream's
        preprocessing resamples 301 -> 300 with `zoom(..., order=1)`, which under
        scipy's default `grid_mode=False` aligns the corner pixel *centres*. The
        300 output pixels therefore span the full 9000 m, at 9000/299 = 30.1003 m
        spacing -- not 30 m.

        Verified empirically: pushing a coordinate ramp through the same zoom puts
        output pixel 0 at node 0.0 and pixel 299 at node 300.0.

        Treating the output as 30 m puts the tile edge off by a full pixel and
        misaligns every mosaic.
        """
        off = np.linspace(-TERRAIN_HALF_EXTENT_M, TERRAIN_HALF_EXTENT_M, MODEL_SHAPE[0])
        return self.x + off, self.y - off

    @property
    def model_res_m(self) -> float:
        return TILE_SPAN_M / (MODEL_SHAPE[0] - 1)

    def coarse_points(self) -> tuple[np.ndarray, np.ndarray]:
        """(9,9) meshed x, y of the coarse stencil."""
        xs, ys = self.coarse_axes()
        return np.meshgrid(xs, ys)

    def terrain_points(self) -> tuple[np.ndarray, np.ndarray]:
        """(301,301) meshed x, y of the terrain grid."""
        xs, ys = self.terrain_axes()
        return np.meshgrid(xs, ys)

    def bounds(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) of the 9 km terrain footprint."""
        h = TERRAIN_HALF_EXTENT_M
        return (self.x - h, self.y - h, self.x + h, self.y + h)

    def stencil_bounds(self) -> tuple[float, float, float, float]:
        """Bounds the coarse stencil *reaches*, i.e. +/- 4 km of centre.

        Slightly tighter than `bounds`, and it is the one that matters for asking
        whether the WRF domain covers this tile.
        """
        h = COARSE_HALF_EXTENT_M
        return (self.x - h, self.y - h, self.x + h, self.y + h)

    def lonlat_center(self) -> tuple[float, float]:
        tf = pyproj.Transformer.from_crs(self.crs, GEOGRAPHIC_CRS, always_xy=True)
        return tf.transform(self.x, self.y)

    def terrain_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """(301,301) lon, lat of the terrain grid, for writing CF-style coords."""
        gx, gy = self.terrain_points()
        tf = pyproj.Transformer.from_crs(self.crs, GEOGRAPHIC_CRS, always_xy=True)
        return tf.transform(gx, gy)


class DomainGeometry:
    """The WRF domain projected into the working CRS, plus tile generation.

    The WRF grid is curvilinear in UTM, but only barely (it is a Lambert grid
    rotated -0.34 deg from true east). We fit an affine map from (row, col) index
    space to (x, y) and invert it, which makes every later interpolation a cheap
    index-space bilinear rather than a triangulation search. The affine residual is
    checked at construction -- if this domain were ever curved enough for that to
    matter, you get an error instead of quiet interpolation bias.
    """

    #: Refuse the affine shortcut if any grid node is off by more than this.
    MAX_AFFINE_RESIDUAL_M = 25.0

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        crs: str = WORKING_CRS,
        grid_shape: tuple[int, int] = WRF_GRID_SHAPE,
    ):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if lat.size != WRF_N_POINTS or lon.size != WRF_N_POINTS:
            raise ValueError(
                f"expected {WRF_N_POINTS} points, got lat={lat.size} lon={lon.size}"
            )
        self.grid_shape = grid_shape
        self.lat = lat.reshape(grid_shape)
        self.lon = lon.reshape(grid_shape)
        self.crs = crs

        tf = pyproj.Transformer.from_crs(GEOGRAPHIC_CRS, crs, always_xy=True)
        self.x, self.y = tf.transform(self.lon, self.lat)

        self._fit_affine()

    def _fit_affine(self) -> None:
        nrow, ncol = self.grid_shape
        rr, cc = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
        # Design matrix for [row, col, 1] -> x and -> y.
        A = np.column_stack(
            [rr.ravel(), cc.ravel(), np.ones(rr.size)]
        )
        coef_x, *_ = np.linalg.lstsq(A, self.x.ravel(), rcond=None)
        coef_y, *_ = np.linalg.lstsq(A, self.y.ravel(), rcond=None)
        self._fwd = np.array([coef_x, coef_y])  # (2,3)

        pred_x = A @ coef_x
        pred_y = A @ coef_y
        resid = np.hypot(pred_x - self.x.ravel(), pred_y - self.y.ravel())
        self.affine_residual_m = float(resid.max())
        if self.affine_residual_m > self.MAX_AFFINE_RESIDUAL_M:
            raise ValueError(
                f"WRF grid is not affine in {self.crs} to within "
                f"{self.MAX_AFFINE_RESIDUAL_M} m (max residual "
                f"{self.affine_residual_m:.1f} m). Use a triangulated interpolator "
                f"instead of the index-space shortcut."
            )

        # Invert: [x, y] -> [row, col].
        M = self._fwd[:, :2]  # d(x,y)/d(row,col)
        self._inv_M = np.linalg.inv(M)
        self._offset = self._fwd[:, 2]

    def xy_to_rowcol(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map projected coordinates to fractional (row, col) index space."""
        d = np.stack([np.ravel(x) - self._offset[0], np.ravel(y) - self._offset[1]])
        rc = self._inv_M @ d
        shape = np.shape(x)
        return rc[0].reshape(shape), rc[1].reshape(shape)

    def rowcol_to_xy(self, row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        row = np.asarray(row, dtype=np.float64)
        col = np.asarray(col, dtype=np.float64)
        x = self._fwd[0, 0] * row + self._fwd[0, 1] * col + self._fwd[0, 2]
        y = self._fwd[1, 0] * row + self._fwd[1, 1] * col + self._fwd[1, 2]
        return x, y

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.x.min(), self.y.min(), self.x.max(), self.y.max())

    def covers_stencil(self, tile: Tile) -> bool:
        """True if the tile's 9x9 stencil lies wholly inside the WRF domain.

        The test runs in **index space**, not in projected coordinates. The WRF
        domain is a rotated rectangle in UTM (-0.34 deg here), so its axis-aligned
        bounding box is strictly larger than the domain itself: a tile can sit
        inside `self.extent` and still fall off the grid near a corner. Comparing
        against the bounding box accepts such tiles, and they then fail deeper in
        with an extrapolation error from the interpolator.

        Because the index map is affine, checking the stencil's four corners is
        sufficient -- the image of a rectangle is a parallelogram, and containment
        of its corners implies containment of the whole.
        """
        minx, miny, maxx, maxy = tile.stencil_bounds()
        xs = np.array([minx, minx, maxx, maxx])
        ys = np.array([miny, maxy, miny, maxy])
        row, col = self.xy_to_rowcol(xs, ys)
        nrow, ncol = self.grid_shape
        eps = 1e-6
        return bool(
            (row >= -eps).all() and (row <= nrow - 1 + eps).all()
            and (col >= -eps).all() and (col <= ncol - 1 + eps).all()
        )

    def make_tiles(
        self,
        *,
        stride_m: float = TILE_SPAN_M,
        prefix: str = "t",
    ) -> list[Tile]:
        """Lay out a regular tile grid covering the domain.

        `stride_m` < 9000 produces overlapping tiles for mosaic blending; the
        default is edge-to-edge with no overlap.
        """
        dminx, dminy, dmaxx, dmaxy = self.extent
        h = COARSE_HALF_EXTENT_M
        xs = np.arange(dminx + h, dmaxx - h + 1e-6, stride_m)
        ys = np.arange(dminy + h, dmaxy - h + 1e-6, stride_m)
        tiles = []
        for j, ty in enumerate(ys[::-1]):  # north-to-south, so tile row 0 is north
            for i, tx in enumerate(xs):
                tile = Tile(tile_id=f"{prefix}_{j:03d}_{i:03d}", x=float(tx), y=float(ty), crs=self.crs)
                if self.covers_stencil(tile):
                    tiles.append(tile)
        return tiles

    def tile_at_lonlat(self, lon: float, lat: float, tile_id: str = "site") -> Tile:
        """A tile centred on a geographic point (e.g. a mast or turbine)."""
        tf = pyproj.Transformer.from_crs(GEOGRAPHIC_CRS, self.crs, always_xy=True)
        x, y = tf.transform(lon, lat)
        return Tile(tile_id=tile_id, x=float(x), y=float(y), crs=self.crs)
