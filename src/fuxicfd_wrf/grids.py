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

from .constants import (  # noqa: F401
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
    utm_crs_for,
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

    @property
    def geo_key(self) -> str:
        """Short hash of what the tile's terrain actually depends on: where it is.

        Tile ids are lattice indices (`t_008_010`) and are NOT unique across
        domains -- two different WRF domains generate the same ids for entirely
        different places. Keying the terrain cache on the id alone therefore lets
        one domain serve another domain's terrain, silently: the Panhandle run
        reused 23 tiles of New Mexico DEM cached weeks earlier by the WSW run,
        putting ~1900 m terrain into a ~1000 m site. That corrupts the model's dem
        channel, not just the WRG elevation column, and nothing downstream checks.

        Rounded to the metre so float noise in the lattice cannot split the key.
        """
        import hashlib

        raw = f"{self.crs}|{round(self.x)}|{round(self.y)}"
        return hashlib.sha1(raw.encode()).hexdigest()[:8]

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

    `affine_fit` decides *where* that map is fitted:

    * ``"global"`` -- one fit over the whole domain. Exact enough for a domain a
      hundred km across (Broadview 10.3 m, WSW 13.9 m) and the cheapest thing that
      works, so it stays the default.
    * ``"local"``  -- refit around each query point over a
      ``2*LOCAL_FIT_HALF+1`` window. Lambert-vs-UTM misfit is a smooth bowl that
      grows with domain size, so a global fit degrades quadratically: Pattern
      ph1 is 180 x 153 km and misses by 42.7 m, past the guard. Locally the same
      grid fits to 2.6 m, which is the floor set by the float32 lon/lat in the
      store. Use this for any domain the global fit rejects.

    The two agree to within the global residual, so a store built one way is not
    bit-comparable with one built the other. That is why this is an explicit
    setting and not an automatic fallback.
    """

    #: Refuse the affine shortcut if any grid node is off by more than this.
    MAX_AFFINE_RESIDUAL_M = 25.0

    #: Half-width, in grid cells, of the window a local fit is made over. 8 gives
    #: a 17 x 17 cell window -- comfortably larger than the 9 km stencil that is
    #: the usual query, and small enough that the curvature across it is ~0.
    LOCAL_FIT_HALF = 8

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        crs: str = WORKING_CRS,
        grid_shape: tuple[int, int] = WRF_GRID_SHAPE,
        affine_fit: str = "global",
    ):
        if affine_fit not in ("global", "local"):
            raise ValueError(f"affine_fit={affine_fit!r} not in ('global', 'local')")
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        n_expected = grid_shape[0] * grid_shape[1]
        if lat.size != n_expected or lon.size != n_expected:
            raise ValueError(
                f"grid_shape {grid_shape} implies {n_expected} points, but got "
                f"lat={lat.size} lon={lon.size}. Pass the store's `domain_shape`."
            )
        if crs == "auto":
            crs = utm_crs_for(lon, lat)
        self.grid_shape = grid_shape
        self.lat = lat.reshape(grid_shape)
        self.lon = lon.reshape(grid_shape)
        self.crs = crs
        self.affine_fit = affine_fit
        self._local_cache: dict[tuple[int, int], tuple] = {}

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

        # Invert: [x, y] -> [row, col].
        M = self._fwd[:, :2]  # d(x,y)/d(row,col)
        self._inv_M = np.linalg.inv(M)
        self._offset = self._fwd[:, 2]

        # The global map is always built: in local mode it is still the first
        # guess that says which window to refit around, and rowcol_to_xy uses it.
        if self.affine_fit == "global":
            self.local_residual_m = None
            if self.affine_residual_m > self.MAX_AFFINE_RESIDUAL_M:
                raise ValueError(
                    f"WRF grid is not affine in {self.crs} to within "
                    f"{self.MAX_AFFINE_RESIDUAL_M} m (max residual "
                    f"{self.affine_residual_m:.1f} m). Set tiling.affine_fit: local "
                    f"to refit per window, or use a triangulated interpolator."
                )
            return

        self.local_residual_m = self._probe_local_residual()
        if self.local_residual_m > self.MAX_AFFINE_RESIDUAL_M:
            raise ValueError(
                f"WRF grid is not affine in {self.crs} to within "
                f"{self.MAX_AFFINE_RESIDUAL_M} m even over "
                f"{2 * self.LOCAL_FIT_HALF + 1} x {2 * self.LOCAL_FIT_HALF + 1} cell "
                f"windows (max residual {self.local_residual_m:.1f} m). This grid "
                f"needs a triangulated interpolator, not an index-space shortcut."
            )

    def _local_affine(self, row0: int, col0: int) -> tuple:
        """Affine map fitted over a window of the grid centred on (row0, col0).

        Returns `(inv_M, offset, residual_m)` in *absolute* index coordinates, so
        the result is a drop-in replacement for the global map. Cached per window:
        one tile's 9x9 stencil spans ~9 cells and so reuses a handful of windows,
        and successive tiles overlap.
        """
        nrow, ncol = self.grid_shape
        h = self.LOCAL_FIT_HALF
        span_r = min(2 * h + 1, nrow)
        span_c = min(2 * h + 1, ncol)
        # Slide the window fully inside the grid rather than truncating it at an
        # edge -- a lopsided window fits the corner worse than a shifted one.
        rs = int(np.clip(row0 - h, 0, nrow - span_r))
        cs = int(np.clip(col0 - h, 0, ncol - span_c))
        key = (rs, cs)
        hit = self._local_cache.get(key)
        if hit is not None:
            return hit

        rr, cc = np.meshgrid(
            np.arange(rs, rs + span_r), np.arange(cs, cs + span_c), indexing="ij"
        )
        A = np.column_stack([rr.ravel(), cc.ravel(), np.ones(rr.size)])
        x = self.x[rs:rs + span_r, cs:cs + span_c].ravel()
        y = self.y[rs:rs + span_r, cs:cs + span_c].ravel()
        coef_x, *_ = np.linalg.lstsq(A, x, rcond=None)
        coef_y, *_ = np.linalg.lstsq(A, y, rcond=None)
        fwd = np.array([coef_x, coef_y])
        resid = float(np.hypot(A @ coef_x - x, A @ coef_y - y).max())
        out = (np.linalg.inv(fwd[:, :2]), fwd[:, 2], resid)
        self._local_cache[key] = out
        return out

    def _probe_local_residual(self) -> float:
        """Worst local-window misfit anywhere on the grid.

        Windows are stepped by their half-width so every cell falls inside at
        least one probed window; the misfit varies smoothly, so this is the same
        answer a per-cell probe gives at a fraction of the cost.
        """
        nrow, ncol = self.grid_shape
        h = max(self.LOCAL_FIT_HALF, 1)
        worst = 0.0
        for r in range(0, nrow, h):
            for c in range(0, ncol, h):
                worst = max(worst, self._local_affine(r, c)[2])
        return worst

    def xy_to_rowcol(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map projected coordinates to fractional (row, col) index space.

        In `affine_fit="local"` mode the global map only picks the window; the
        answer comes from an affine refitted around each point's own cell. The
        refit is per point rather than per call so that a caller passing scattered
        points (a whole grid, say) gets the same accuracy as one passing a single
        tile's stencil.
        """
        shape = np.shape(x)
        d = np.stack([np.ravel(x) - self._offset[0], np.ravel(y) - self._offset[1]])
        rc = self._inv_M @ d
        if self.affine_fit == "global":
            return rc[0].reshape(shape), rc[1].reshape(shape)

        fx, fy = np.ravel(x), np.ravel(y)
        out = np.empty_like(rc)
        for i, (r_est, c_est) in enumerate(zip(np.rint(rc[0]), np.rint(rc[1]))):
            inv_M, offset, _ = self._local_affine(int(r_est), int(c_est))
            out[:, i] = inv_M @ np.array([fx[i] - offset[0], fy[i] - offset[1]])
        return out[0].reshape(shape), out[1].reshape(shape)

    def rowcol_to_xy(self, row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of `xy_to_rowcol`, fitted the same way so the pair round-trips."""
        row = np.asarray(row, dtype=np.float64)
        col = np.asarray(col, dtype=np.float64)
        if self.affine_fit == "global":
            x = self._fwd[0, 0] * row + self._fwd[0, 1] * col + self._fwd[0, 2]
            y = self._fwd[1, 0] * row + self._fwd[1, 1] * col + self._fwd[1, 2]
            return x, y

        shape = row.shape
        fr, fc = row.ravel(), col.ravel()
        out = np.empty((2, fr.size))
        for i in range(fr.size):
            inv_M, offset, _ = self._local_affine(int(round(fr[i])), int(round(fc[i])))
            out[:, i] = np.linalg.inv(inv_M) @ np.array([fr[i], fc[i]]) + offset
        return out[0].reshape(shape), out[1].reshape(shape)

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

    def core_bounds(self, tile: Tile, stride_m: float) -> tuple[float, float, float, float]:
        """The part of `tile` that survives into a mosaic or WRG, in projected m.

        FuXi-CFD gets no buffer around its 9 km window and its outer ~1.5 km is
        1.9x noisier, so overlapping runs keep only a `stride_m` square about the
        centre -- which at stride < 9000 tessellates exactly. With no overlap the
        core is the whole footprint.
        """
        h = min(stride_m, TILE_SPAN_M) / 2
        return (tile.x - h, tile.y - h, tile.x + h, tile.y + h)

    def core_lonlat_bounds(self, tile: Tile, stride_m: float):
        """Geographic bbox of a tile's core.

        The core is a rectangle in the projected CRS, so in lon/lat it is a
        slightly rotated quad; its axis-aligned bbox is marginally larger. That
        over-inclusion is deliberate -- erring toward one extra tile is safe,
        while erring the other way leaves a hole in the middle of a deliverable.
        """
        minx, miny, maxx, maxy = self.core_bounds(tile, stride_m)
        tf = pyproj.Transformer.from_crs(self.crs, GEOGRAPHIC_CRS, always_xy=True)
        lon, lat = tf.transform(
            np.array([minx, minx, maxx, maxx]), np.array([miny, maxy, miny, maxy])
        )
        return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())

    def make_tiles(
        self,
        *,
        stride_m: float = TILE_SPAN_M,
        prefix: str = "t",
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[Tile]:
        """Lay out a regular tile grid covering the domain.

        `stride_m` < 9000 produces overlapping tiles for mosaic blending; the
        default is edge-to-edge with no overlap.

        `bbox` is `(min_lon, min_lat, max_lon, max_lat)` and restricts the result
        to tiles whose **core** intersects that box -- i.e. the minimal set that
        still fully covers the requested area once edges are trimmed. Two
        properties matter here:

        * The lattice is **not** re-originated on the bbox. Tiles are generated
          over the whole domain and then filtered, so a subset run lands on the
          same centres and keeps the same tile ids as the full-domain run. A
          bbox store and a full store are therefore directly comparable, and a
          bbox run can be dropped into an existing store without realigning.
        * Selection uses the core rather than the 9 km footprint. Using the
          footprint would pull in a ring of tiles whose contribution is entirely
          discarded by the 1.5 km trim.
        """
        dminx, dminy, dmaxx, dmaxy = self.extent
        h = COARSE_HALF_EXTENT_M
        xs = np.arange(dminx + h, dmaxx - h + 1e-6, stride_m)
        ys = np.arange(dminy + h, dmaxy - h + 1e-6, stride_m)
        tiles = []
        for j, ty in enumerate(ys[::-1]):  # north-to-south, so tile row 0 is north
            for i, tx in enumerate(xs):
                tile = Tile(tile_id=f"{prefix}_{j:03d}_{i:03d}", x=float(tx), y=float(ty), crs=self.crs)
                if not self.covers_stencil(tile):
                    continue
                if bbox is not None:
                    tminlon, tminlat, tmaxlon, tmaxlat = self.core_lonlat_bounds(tile, stride_m)
                    minlon, minlat, maxlon, maxlat = bbox
                    if (tmaxlon < minlon or tminlon > maxlon
                            or tmaxlat < minlat or tminlat > maxlat):
                        continue
                tiles.append(tile)
        return tiles

    def tile_at_lonlat(self, lon: float, lat: float, tile_id: str = "site") -> Tile:
        """A tile centred on a geographic point (e.g. a mast or turbine)."""
        tf = pyproj.Transformer.from_crs(GEOGRAPHIC_CRS, self.crs, always_xy=True)
        x, y = tf.transform(lon, lat)
        return Tile(tile_id=tile_id, x=float(x), y=float(y), crs=self.crs)
