"""Wind conversion, frame rotation, and the WRF -> 9x9 stencil interpolator.

Three transformations happen here, in order:

1. **speed/direction -> u, v.** WRF's ``wind_direction_*`` is meteorological: the
   direction the wind blows *from*, degrees clockwise from north. So a 270 deg wind
   is westerly and has u > 0::

       u = -ws * sin(pi/180 * wd)
       v = -ws * cos(pi/180 * wd)

2. **Earth-relative -> grid-relative.** u/v from step 1 point along true east/north.
   The model consumes them on a projected grid whose "north" differs from true north
   by the grid convergence. For Broadview in UTM 13N that is well under a degree, but
   it is cheap and correct to apply it rather than to assume it away.

3. **~800 m WRF grid -> 1 km stencil.** The WRF grid is 802 x 798 m, not the 1 km
   the model expects, so the 9x9 stencil is bilinearly interpolated. The weights
   depend only on tile geometry, so they are baked once into a sparse (81, 10296)
   matrix and reapplied as a matmul for every timestep.

Interpolate u and v, never speed and direction. Averaging direction across a
discontinuity at 0/360 is a classic way to manufacture nonsense; the components
have no such branch cut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyproj
from scipy.sparse import coo_matrix

from .constants import COARSE_SHAPE, GEOGRAPHIC_CRS
from .grids import DomainGeometry, Tile


def speeddir_to_uv(
    speed: np.ndarray, direction_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological speed/direction -> eastward u, northward v."""
    theta = np.deg2rad(np.asarray(direction_deg, dtype=np.float64))
    speed = np.asarray(speed, dtype=np.float64)
    return -speed * np.sin(theta), -speed * np.cos(theta)


def uv_to_speeddir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `speeddir_to_uv`; direction in [0, 360)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    speed = np.hypot(u, v)
    direction = (np.rad2deg(np.arctan2(-u, -v))) % 360.0
    return speed, direction


def grid_convergence_deg(
    lon: np.ndarray, lat: np.ndarray, crs: str, *, delta_deg: float = 1e-4
) -> np.ndarray:
    """Angle from projected-grid north to true north, degrees, positive counterclockwise.

    Computed numerically by projecting a short due-north step, which works for any
    CRS without needing a closed-form convergence formula.
    """
    tf = pyproj.Transformer.from_crs(GEOGRAPHIC_CRS, crs, always_xy=True)
    x0, y0 = tf.transform(lon, lat)
    x1, y1 = tf.transform(lon, np.asarray(lat) + delta_deg)
    return np.rad2deg(np.arctan2(x1 - x0, y1 - y0))


def rotate_to_grid(
    u: np.ndarray, v: np.ndarray, convergence_deg: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate earth-relative (u, v) into the projected grid frame."""
    a = np.deg2rad(convergence_deg)
    ca, sa = np.cos(a), np.sin(a)
    return u * ca - v * sa, u * sa + v * ca


@dataclass
class StencilInterpolator:
    """Sparse (81, 10296) bilinear operator: flat WRF field -> tile's 9x9 stencil.

    Built once per tile from geometry alone, then applied to every timestep. For a
    full-domain sweep this turns the per-timestep resampling cost from a
    triangulation search into one sparse matmul.
    """

    tile_id: str
    matrix: coo_matrix
    convergence_deg: float

    def apply(self, flat: np.ndarray) -> np.ndarray:
        """(n, n_points) or (n_points,) -> (n, 9, 9) or (9, 9).

        Pass float64 to skip the cast. In a domain sweep this is called once per
        tile against the same domain-wide block, so a cast here is paid ~n_tiles
        times over -- see `stencil_uv`.
        """
        if flat.dtype != np.float64:
            flat = np.asarray(flat, dtype=np.float64)
        single = flat.ndim == 1
        if single:
            flat = flat[None, :]
        out = flat @ self.matrix.T  # (n, 81)
        out = out.reshape(-1, *COARSE_SHAPE)
        return out[0] if single else out


def build_stencil_interpolator(
    geom: DomainGeometry, tile: Tile
) -> StencilInterpolator:
    """Bake the bilinear weights mapping the WRF grid onto `tile`'s 9x9 stencil."""
    gx, gy = tile.coarse_points()  # (9,9) projected coords
    row, col = geom.xy_to_rowcol(gx, gy)

    nrow, ncol = geom.grid_shape
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)

    # Clamp the *base* index so r0+1 stays in range; covers_stencil() already
    # guarantees we are inside the domain, so this only handles the top edge.
    r0 = np.clip(r0, 0, nrow - 2)
    c0 = np.clip(c0, 0, ncol - 2)
    fr = row - r0
    fc = col - c0

    if np.any(fr < -1e-6) or np.any(fr > 1 + 1e-6) or np.any(fc < -1e-6) or np.any(fc > 1 + 1e-6):
        raise ValueError(
            f"tile {tile.tile_id} needs extrapolation beyond the WRF grid "
            f"(row frac {fr.min():.3f}..{fr.max():.3f}, "
            f"col frac {fc.min():.3f}..{fc.max():.3f}). "
            "It should have been rejected by DomainGeometry.covers_stencil()."
        )

    n_out = fr.size
    out_idx = np.repeat(np.arange(n_out), 4)
    corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
    flat_idx = np.empty(n_out * 4, dtype=np.int64)
    weights = np.empty(n_out * 4, dtype=np.float64)
    frf, fcf, r0f, c0f = fr.ravel(), fc.ravel(), r0.ravel(), c0.ravel()
    for i, (dr, dc) in enumerate(corners):
        wr = frf if dr else 1.0 - frf
        wc = fcf if dc else 1.0 - fcf
        flat_idx[i::4] = (r0f + dr) * ncol + (c0f + dc)
        weights[i::4] = wr * wc

    matrix = coo_matrix(
        (weights, (out_idx, flat_idx)), shape=(n_out, nrow * ncol)
    ).tocsr()

    lon, lat = tile.lonlat_center()
    conv = float(grid_convergence_deg(np.array(lon), np.array(lat), tile.crs))
    return StencilInterpolator(tile_id=tile.tile_id, matrix=matrix, convergence_deg=conv)


def stencil_winds(
    interp: StencilInterpolator,
    speed_flat: np.ndarray,
    direction_flat: np.ndarray,
    *,
    rotate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Full path: flat WRF speed/direction -> (n, 9, 9) grid-relative u, v.

    Conversion to components happens on the native WRF grid, *before*
    interpolation, so no direction averaging ever occurs.
    """
    u, v = speeddir_to_uv(speed_flat, direction_flat)
    return stencil_uv(interp, u, v, rotate=rotate)


def stencil_uv(
    interp: StencilInterpolator,
    u_flat: np.ndarray,
    v_flat: np.ndarray,
    *,
    rotate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """`stencil_winds` for winds already converted to components.

    Split out because speed/direction -> u,v depends only on the domain-wide
    field, not on the tile. Calling it per tile redoes a sin/cos over every WRF
    point once for each tile in the sweep: at 174 tiles and 26550 points that is
    ~62x the cost of the sparse matmuls it feeds, and it is what made the first
    Panhandle launch run at 15 tile-steps/s instead of the GPU's ~42. Convert
    once per read block, then call this per tile.
    """
    u9 = interp.apply(u_flat)
    v9 = interp.apply(v_flat)
    if rotate:
        u9, v9 = rotate_to_grid(u9, v9, interp.convergence_deg)
    return u9, v9
