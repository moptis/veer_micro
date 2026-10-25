"""Coarse ``.wrg`` as a driving source: one inflow state per direction sector.

Why this exists
---------------
FuXi-CFD's labels are steady-state RANS solutions -- ensemble-mean flow fields --
and its 9x9 coarse input was derived by coarsening those same mean fields.
Driving it with instantaneous WRF snapshots hands a mean-field operator a
turbulent realisation. A sector-conditional mean drawn from a wind resource grid
is much closer to what the network was trained on, and there are ~12 of them
instead of ~52,000 timesteps.

What a state is
---------------
One state per sector ``s``, built per coarse cell from that cell's sector Weibull::

    U_s     = A_s * Gamma(1 + 1/k_s)     # sector-conditional mean speed
    theta_s = sector centre               # meteorological, deg from true north

so the speed field varies cell to cell while the direction field is uniform
across the tile. That uniformity is a real approximation and it is forced: a WRG
records a frequency per sector, never a mean direction *within* a sector. Over a
9 km tile the directional spread of a mesoscale field is small, but this is
exactly the assumption the timestep-vs-sector speedup comparison exists to test.

Optional speed binning
----------------------
With ``speed_bins > 1`` each sector is additionally split into equal-probability
speed bins drawn from its own Weibull, giving ``n_sectors * speed_bins`` states.
In neutral RANS the speedup is very nearly independent of inflow magnitude --
which is what makes the whole statistical shortcut work -- but FuXi-CFD never
sees inflow speed as an explicit input, so that invariance is a property to
measure, not to assume. One extra run with ``speed_bins: 3`` settles it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyproj
from scipy.special import gamma

from ..constants import GEOGRAPHIC_CRS
from ..wrg_read import read_wrg
from .base import CoarseSource

#: The model's wind channels are 100 m. A coarse WRG cut at another hub height
#: can still be driven through, but it is a different field from the one the
#: network was trained to accept, so it is worth saying out loud.
MODEL_INPUT_HEIGHT_M = 100.0


def sector_centers(n_sectors: int) -> np.ndarray:
    """Sector centre directions, first sector centred on north.

    Matches ``wrg.sector_edges``: for 12 sectors the centres are 0, 30, ..., 330.
    """
    return np.arange(n_sectors, dtype=np.float64) * (360.0 / n_sectors)


def weibull_mean(A: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Mean of a Weibull(A, k): ``A * Gamma(1 + 1/k)``."""
    A = np.asarray(A, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    return A * gamma(1.0 + 1.0 / k)


def weibull_quantile(A: np.ndarray, k: np.ndarray, p: float) -> np.ndarray:
    """Inverse CDF of a Weibull(A, k) at probability ``p``."""
    A = np.asarray(A, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    return A * (-np.log(1.0 - p)) ** (1.0 / k)


def _check_regular(v: np.ndarray, name: str, *, tol: float = 1e-3) -> None:
    d = np.diff(v)
    if d.size and (d.max() - d.min()) > tol * abs(np.median(d)):
        raise ValueError(
            f"{name} axis of the coarse WRG is not regularly spaced "
            f"(spacing {d.min():g}..{d.max():g}). DomainGeometry assumes a "
            "regular lattice."
        )


class WrgSectorSource(CoarseSource):
    """A coarse ``.wrg`` presented as one inflow state per sector (x speed bin).

    The WRG's x/y are already a regular grid in a projected CRS, so unlike the
    WRF backend there is no Lambert-in-UTM affine to fit -- ``DomainGeometry``
    gets a lattice it can represent exactly, provided it is told the right CRS.
    There is no CRS in the ``.wrg`` format, so ``crs`` must be supplied and is
    not guessable.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        crs: str,
        speed_bins: int = 1,
        expected_height_m: float | None = MODEL_INPUT_HEIGHT_M,
    ):
        self.path = Path(path)
        self.crs = str(crs)
        self.speed_bins = int(speed_bins)
        if self.speed_bins < 1:
            raise ValueError(f"speed_bins must be >= 1, got {speed_bins}")

        df = read_wrg(self.path)
        self.n_sectors = int(df.attrs["n_sectors"])
        self.bin_size_deg = 360.0 / self.n_sectors
        self.centers = sector_centers(self.n_sectors)

        xs = np.sort(df["x"].unique())
        ys = np.sort(df["y"].unique())[::-1]          # north-up
        nx, ny = len(xs), len(ys)
        if nx * ny != len(df):
            raise ValueError(
                f"{self.path} is not a complete rectangular grid: {len(df)} rows "
                f"for a {ny} x {nx} lattice ({ny * nx} expected). A WRG with "
                "holes cannot be tiled -- run `wrg-concat --largest-rect` first."
            )
        _check_regular(xs, "x")
        _check_regular(ys[::-1], "y")

        self.x, self.y = xs, ys
        self.domain_shape = (ny, nx)
        self.res_m = float(np.median(np.diff(xs)))

        # Pivot every column onto the (ny, nx) north-up lattice once. Index maps
        # rather than reshaping the frame: row order in a .wrg is column-major,
        # so a bare reshape would transpose every field.
        xi = {v: i for i, v in enumerate(xs)}
        yi = {v: i for i, v in enumerate(ys)}
        r = df["y"].map(yi).to_numpy()
        c = df["x"].map(xi).to_numpy()

        def grid(col: str) -> np.ndarray:
            out = np.full((ny, nx), np.nan, dtype=np.float64)
            out[r, c] = df[col].to_numpy()
            return out

        self.A = np.stack([grid(f"A_{s}") for s in range(self.n_sectors)])
        self.k = np.stack([grid(f"k_{s}") for s in range(self.n_sectors)])
        self.freq = np.stack([grid(f"freq_{s}") for s in range(self.n_sectors)])
        self.elevation = grid("elevation")
        self.hub_height = float(np.nanmedian(grid("hub_height")))

        # Frequencies are stored as freq*1000 rounded to an integer, so they do
        # not sum to exactly 1. Renormalise rather than carrying that rounding
        # through into every downstream weighting.
        fsum = self.freq.sum(axis=0)
        bad = ~np.isfinite(fsum) | (fsum <= 0)
        if bad.any():
            raise ValueError(
                f"{self.path}: {int(bad.sum())} grid points carry no sector "
                "frequency at all; the file is malformed or truncated."
            )
        self.freq = self.freq / fsum

        if expected_height_m is not None and abs(self.hub_height - expected_height_m) > 1.0:
            print(
                f"[wrg-source] WARNING: {self.path.name} is cut at "
                f"{self.hub_height:g} m, but FuXi-CFD's wind channels are "
                f"{expected_height_m:g} m. Driving it with a "
                f"{self.hub_height:g} m field feeds the model something other "
                "than what it was trained on. Prefer the 100 m coarse deliverable."
            )

        tf = pyproj.Transformer.from_crs(self.crs, GEOGRAPHIC_CRS, always_xy=True)
        gx, gy = np.meshgrid(xs, ys)
        self._lon, self._lat = tf.transform(gx, gy)

    # --- CoarseSource --------------------------------------------------------

    @property
    def n_states(self) -> int:
        return self.n_sectors * self.speed_bins

    @property
    def state_labels(self) -> list[str]:
        if self.speed_bins == 1:
            return [
                f"sec{s:02d}_{self.centers[s]:.0f}deg" for s in range(self.n_sectors)
            ]
        return [
            f"sec{s:02d}_{self.centers[s]:.0f}deg_q{b}"
            for s in range(self.n_sectors)
            for b in range(self.speed_bins)
        ]

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lat.ravel(), self._lon.ravel()

    def state_sector(self, states) -> np.ndarray:
        """Sector index for each state index."""
        return np.asarray(states, dtype=int) // self.speed_bins

    def state_speed_field(self, state: int) -> np.ndarray:
        """(ny, nx) inflow speed for one state.

        With ``speed_bins == 1`` this is the sector-conditional mean. Otherwise
        the sector Weibull is sampled at bin-midpoint quantiles, so bin ``b`` of
        ``n`` is the ``(b + 0.5)/n`` quantile -- equal-probability bins, which
        keeps every state's frequency weight identical and the reconstruction
        unbiased.
        """
        s = int(state) // self.speed_bins
        b = int(state) % self.speed_bins
        if self.speed_bins == 1:
            return weibull_mean(self.A[s], self.k[s])
        return weibull_quantile(self.A[s], self.k[s], (b + 0.5) / self.speed_bins)

    def read_uv(self, states) -> tuple[np.ndarray, np.ndarray]:
        states = np.atleast_1d(np.asarray(states, dtype=int))
        n_pts = self.domain_shape[0] * self.domain_shape[1]
        u = np.empty((states.size, n_pts), dtype=np.float64)
        v = np.empty((states.size, n_pts), dtype=np.float64)
        for i, st in enumerate(states):
            speed = self.state_speed_field(st).ravel()
            theta = np.deg2rad(self.centers[int(st) // self.speed_bins])
            # Meteorological: the direction the wind blows *from*.
            u[i] = -speed * np.sin(theta)
            v[i] = -speed * np.cos(theta)
        return u, v

    def describe(self) -> str:
        ny, nx = self.domain_shape
        mean_speed = np.nansum(self.freq * weibull_mean(self.A, self.k), axis=0)
        return "\n".join([
            f"path          : {self.path}",
            f"grid          : {ny} x {nx} @ {self.res_m:g} m ({self.crs})",
            f"hub height    : {self.hub_height:g} m",
            f"sectors       : {self.n_sectors} ({self.bin_size_deg:g} deg)",
            f"speed bins    : {self.speed_bins}",
            f"states        : {self.n_states}",
            f"mean speed    : {np.nanmin(mean_speed):.2f} .. "
            f"{np.nanmax(mean_speed):.2f} m/s "
            f"(domain mean {np.nanmean(mean_speed):.2f})",
            f"elevation     : {np.nanmin(self.elevation):.0f} .. "
            f"{np.nanmax(self.elevation):.0f} m",
        ])
