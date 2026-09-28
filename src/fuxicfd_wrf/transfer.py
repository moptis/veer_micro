"""Sector transfer functions: coarse wind climate -> fine wind climate.

The operation
-------------
For each direction sector the model is run once, and its output is reduced to two
fields on the tile's 300x300 grid:

    S(x, y)      = |U_fine| / |U_coarse_upsampled|     speedup
    dtheta(x, y) = theta_fine - theta_coarse_upsampled  deflection

Those, applied to the coarse WRG's sector Weibulls, are the whole product.

Why the reference is the *upsampled* coarse field
-------------------------------------------------
The obvious denominator for a speedup is the parent cell's value. The right one
is the 9x9 stencil bilinearly zoomed to 300x300 -- because that is literally what
the model ingested (``model.preprocess`` normalises then zooms, and both
operations are affine, so zooming the raw winds gives exactly the de-normalised
field the network saw).

Using the parent cell instead would fold the coarse-scale gradient across the
tile into S, so S would carry structure the mesoscale model already resolved and
applying it would double-count. Against the upsampled field, S isolates precisely
what FuXi-CFD added.

How A and k are carried across
------------------------------
Not ``A_fine = A_coarse * S``. The model hands back a *speed*, so::

    k_fine = k_coarse                       (carried over; see below)
    A_fine = U_fine / Gamma(1 + 1/k_fine)   (exact, no reliance on S)

which stays exact when k varies across the tile -- ``zoom(A*Gamma(1+1/k))`` is
not ``Gamma(1+1/k)*zoom(A)`` unless k is constant. S is still computed and
returned, because it is the quantity the timestep-vs-sector comparison tests, but
it is a diagnostic here rather than the mechanism.

**k is carried over unchanged, and that is an assumption, not a result.** Terrain
reshapes the speed distribution as well as scaling it, and a neutral steady RANS
emulator driven by one inflow speed per sector has no way to report that. If a
``speed_bins > 1`` run shows S varying with inflow magnitude, that dependence
*is* the k adjustment and it belongs here.

Power density convention
------------------------
This module works analytically throughout: the mixture's third moment is
``sum_s f_s A_s^3 Gamma(1+3/k_s)``, and the all-sector Weibull is fitted to match
the first and third moments exactly, so ``0.5 rho A^3 Gamma(1+3/k)`` reproduces
it by construction. That matches ``wrg_read.weibull_power_density`` and
deliberately differs from ``wrg.power_density``, which uses an accumulated third
moment from a streamed time series. The two agree on well-fitted data and diverge
where the Weibull fit is poor, so a timestep-derived WRG and a WRG-path WRG must
be compared on one convention -- use this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.ndimage import zoom
from scipy.special import gamma

from .constants import COARSE_SHAPE, MODEL_SHAPE

#: Same clip range as the streaming MLE in `wrg.py`, so both estimators refuse
#: the same degenerate fits rather than disagreeing at the tails.
K_BOUNDS = (0.2, 20.0)

_COARSE_ZOOM = (MODEL_SHAPE[0] / COARSE_SHAPE[0], MODEL_SHAPE[1] / COARSE_SHAPE[1])


def upsample_coarse(field: np.ndarray) -> np.ndarray:
    """Zoom a (..., 9, 9) field to (..., 300, 300) exactly as preprocessing does.

    ``scipy.ndimage.zoom`` defaults to ``grid_mode=False``, which aligns the
    corner pixel *centres* of input and output, so the 300 output pixels span the
    full 9 km between the outer coarse cell centres. Any hand-rolled replacement
    would have to match that, which is why this calls scipy -- and why it zooms
    one 2-D plane at a time, provably identical to the single-sample path that
    ``tests/test_upstream_parity.py`` pins.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.shape[-2:] != COARSE_SHAPE:
        raise ValueError(f"expected a field ending in {COARSE_SHAPE}, got {field.shape}")
    if field.ndim == 2:
        return zoom(field, _COARSE_ZOOM, order=1)
    flat = field.reshape(-1, *COARSE_SHAPE)
    out = np.stack([zoom(p, _COARSE_ZOOM, order=1) for p in flat], axis=0)
    return out.reshape(*field.shape[:-2], *MODEL_SHAPE)


def wrap180(deg: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180).

    Note the half-open end: exactly 180 comes back as -180. That only arises for
    a complete flow reversal, where the sign carries no meaning anyway, so this
    does not special-case it.
    """
    return (np.asarray(deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def met_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological direction (deg, blowing *from*) of components u, v."""
    return np.rad2deg(np.arctan2(-np.asarray(u), -np.asarray(v))) % 360.0


@dataclass
class SectorTransfer:
    """Per-sector transfer fields on one tile's 300x300 model grid.

    ``u_fine``/``v_fine`` are in the *projected grid* frame, as the model emits
    them. ``dtheta`` is a difference of two directions in that same frame, so the
    grid convergence cancels and it is directly usable as an earth-relative
    deflection -- which is why nothing here ever rotates back.
    """

    speed_fine: np.ndarray     # (n_state, 300, 300) m/s
    speed_coarse: np.ndarray   # (n_state, 300, 300) m/s, the model's own input
    dtheta: np.ndarray         # (n_state, 300, 300) deg, wrapped to (-180, 180]

    @property
    def speedup(self) -> np.ndarray:
        """S = |U_fine| / |U_coarse|, with calm coarse cells masked to NaN."""
        den = np.where(self.speed_coarse > 1e-6, self.speed_coarse, np.nan)
        return self.speed_fine / den


def sector_transfer(
    u_fine: np.ndarray,
    v_fine: np.ndarray,
    u9: np.ndarray,
    v9: np.ndarray,
) -> SectorTransfer:
    """Build the transfer fields for a batch of states on one tile.

    `u_fine`, `v_fine` are (n_state, 300, 300) model output at the target height.
    `u9`, `v9` are the (n_state, 9, 9) grid-frame winds that drove them -- the
    same arrays handed to ``preprocess.normalize_winds``, before normalisation.
    """
    u_fine = np.asarray(u_fine, dtype=np.float64)
    v_fine = np.asarray(v_fine, dtype=np.float64)
    if u_fine.shape != v_fine.shape:
        raise ValueError(f"fine u/v shape mismatch: {u_fine.shape} vs {v_fine.shape}")
    if u_fine.shape[-2:] != MODEL_SHAPE:
        raise ValueError(f"fine fields must end in {MODEL_SHAPE}, got {u_fine.shape}")

    u_up = upsample_coarse(u9)
    v_up = upsample_coarse(v9)
    if u_up.shape != u_fine.shape:
        raise ValueError(
            f"coarse states ({u_up.shape[0]}) and fine states ({u_fine.shape[0]}) "
            "disagree -- they must be the same batch"
        )

    return SectorTransfer(
        speed_fine=np.hypot(u_fine, v_fine),
        speed_coarse=np.hypot(u_up, v_up),
        dtheta=wrap180(met_direction(u_fine, v_fine) - met_direction(u_up, v_up)),
    )


# --- Weibull moment algebra --------------------------------------------------


def weibull_moments(A: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and third raw moments of a Weibull(A, k)."""
    A = np.asarray(A, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    return A * gamma(1.0 + 1.0 / k), A**3 * gamma(1.0 + 3.0 / k)


def _shape_ratio(k: np.ndarray) -> np.ndarray:
    """``E[U^3] / E[U]^3`` for a Weibull of shape k. Strictly decreasing in k."""
    return gamma(1.0 + 3.0 / k) / gamma(1.0 + 1.0 / k) ** 3


#: Table resolution for the inverse of `_shape_ratio`. 2^20 points log-spaced
#: over `K_BOUNDS` puts adjacent k within 4.4e-6 relative of each other, and
#: linear interpolation between them is second-order, so the inversion error
#: lands ~1e-11 -- far below the 2 decimals k is written to in a .wrg, and below
#: the bisection's own convergence on any realistic array.
_K_TABLE_N = 1 << 20


@lru_cache(maxsize=1)
def _ratio_inverse_table() -> tuple[np.ndarray, np.ndarray]:
    """Ascending (ratio, k) pair for inverting `_shape_ratio` by interpolation.

    `_shape_ratio` is strictly *decreasing* in k, so the table is reversed to
    give `np.interp` the ascending abscissa it requires. Built once per process;
    the cost is ~1e6 gamma evaluations against the tens of millions the caller
    would otherwise repeat sixty times over.
    """
    k = np.geomspace(K_BOUNDS[0], K_BOUNDS[1], _K_TABLE_N)
    with np.errstate(over="ignore", invalid="ignore"):
        r = _shape_ratio(k)
    return r[::-1].copy(), k[::-1].copy()


def fit_weibull_moments(
    m1: np.ndarray, m3: np.ndarray, *, iters: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Weibull (A, k) matching a given first and third moment.

    Used to collapse a mixture of sector Weibulls into one distribution -- for the
    WRG's all-sector A and k columns, and for any sector that receives flow
    deflected out of more than one source sector.

    ``E[U^3]/E[U]^3 = Gamma(1+3/k)/Gamma(1+1/k)^3`` is strictly decreasing in k,
    so inverting it is a 1-D root find. It is done by interpolating a precomputed
    table rather than by bisection, because the function being inverted depends
    on *k alone*: a vectorised bisection re-evaluates the identical 1-D curve at
    every grid point on every iteration. Measured on one sww-sized array
    (16 x 1500 x 1301), 60 iterations cost 92 s of a 106 s ``apply_transfer`` --
    the single largest cost in the whole downscale run, larger than the ASCII
    write and the GPU combined. Tabulating the inverse once makes it ~2 s.

    Points whose ratio falls outside the range reachable on `K_BOUNDS` clip to
    the bound -- ``np.interp`` does that at the table ends by construction, which
    is the same convention the streaming MLE uses.

    ``iters`` is accepted and ignored; it named the old bisection's iteration
    count and is kept so existing callers do not break.
    """
    m1 = np.asarray(m1, dtype=np.float64)
    m3 = np.asarray(m3, dtype=np.float64)
    ok = np.isfinite(m1) & np.isfinite(m3) & (m1 > 0) & (m3 > 0)

    ratio = np.full(m1.shape, np.nan)
    np.divide(m3, m1**3, out=ratio, where=ok)

    r_asc, k_desc = _ratio_inverse_table()
    k = np.interp(ratio, r_asc, k_desc)

    A = np.full(m1.shape, np.nan)
    np.divide(m1, gamma(1.0 + 1.0 / k), out=A, where=ok)
    k = np.where(ok, k, np.nan)
    return A, k


def combine_sectors(
    freq: np.ndarray, A: np.ndarray, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """All-sector Weibull of a frequency-weighted mixture, by moment matching.

    `freq`, `A`, `k` are (n_sectors, ...). Returns (A, k) of shape (...).
    """
    freq = np.asarray(freq, dtype=np.float64)
    m1_s, m3_s = weibull_moments(A, k)
    # A NaN anywhere in the triple has to zero the weight, not propagate: a
    # coarse grid point outside the source WRG carries NaN frequency, and
    # `np.where(cond, nan, 0)` would keep it and poison the whole mixture.
    ok = np.isfinite(m1_s) & np.isfinite(m3_s) & np.isfinite(freq)
    w = np.where(ok, freq, 0.0)
    tot = w.sum(axis=0)
    tot = np.where(tot > 0, tot, np.nan)
    m1 = np.nansum(w * m1_s, axis=0) / tot
    m3 = np.nansum(w * m3_s, axis=0) / tot
    return fit_weibull_moments(m1, m3)


def power_density(A: np.ndarray, k: np.ndarray, air_density: float) -> np.ndarray:
    """``0.5 * rho * E[U^3]`` in W/m^2 from Weibull parameters."""
    _, m3 = weibull_moments(A, k)
    return 0.5 * air_density * m3


# --- applying a transfer to a coarse wind climate ----------------------------


def apply_transfer(
    A_coarse: np.ndarray,
    k_coarse: np.ndarray,
    freq_coarse: np.ndarray,
    speed_fine: np.ndarray,
    dtheta: np.ndarray,
    *,
    deflect: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a coarse sector wind climate onto the fine grid.

    All inputs are (n_sectors, ny, nx) on the *same* fine grid: the coarse
    Weibulls already interpolated up, and the model's fine speed and deflection.
    Returns fine (A, k, freq) with the same shape.

    With ``deflect=True`` each source sector's frequency is reassigned to whatever
    sector contains ``theta_s + dtheta``, and sectors that collect flow from more
    than one source are recombined by moment matching. That is a nearest-bin
    reassignment, not a continuous rotation of the distribution: a deflection of
    less than half a bin moves nothing, and one of more than half a bin moves
    everything. At 30 deg bins that is a coarse instrument, and it is the honest
    limit of what a sector-binned representation can express.

    With ``deflect=False`` sectors stay put and only the magnitude is downscaled.
    Use it to isolate the speedup when comparing against a timestep-derived
    transfer, where the two paths' deflection handling would otherwise confound
    the speedup comparison.
    """
    A_coarse = np.asarray(A_coarse, dtype=np.float64)
    k_coarse = np.asarray(k_coarse, dtype=np.float64)
    freq_coarse = np.asarray(freq_coarse, dtype=np.float64)
    speed_fine = np.asarray(speed_fine, dtype=np.float64)
    n_sec = A_coarse.shape[0]

    shapes = {a.shape for a in (A_coarse, k_coarse, freq_coarse, speed_fine)}
    if len(shapes) != 1:
        raise ValueError(f"all fields must share one shape, got {sorted(shapes)}")

    # k rides along unchanged; A is recovered exactly from the model's own speed.
    k_fine = k_coarse
    A_fine = speed_fine / gamma(1.0 + 1.0 / k_fine)

    if not deflect:
        return A_fine, k_fine, freq_coarse.copy()

    dtheta = np.asarray(dtheta, dtype=np.float64)
    bin_size = 360.0 / n_sec
    centers = np.arange(n_sec) * bin_size

    # Points outside tile coverage carry NaN deflection. `np.rint(nan).astype(int)`
    # is undefined behaviour -- on x86-64 it yields INT64_MIN, which then survives
    # the modulo as a valid-looking sector index and scatters that point's weight
    # into an arbitrary bin. Treat a missing deflection as zero; the point's A and
    # k are NaN regardless, so it contributes nothing either way.
    dtheta = np.where(np.isfinite(dtheta), dtheta, 0.0)

    # Target sector per (source sector, point). Rounding theta/bin is exactly the
    # "first sector centred on north" convention in `wrg.assign_sector`.
    target = np.rint(
        (centers[:, None, None] + dtheta) / bin_size
    ).astype(np.int64) % n_sec

    m1_s, m3_s = weibull_moments(A_fine, k_fine)
    valid = np.isfinite(m1_s) & np.isfinite(m3_s) & np.isfinite(freq_coarse)
    w = np.where(valid, freq_coarse, 0.0)

    acc_f = np.zeros_like(w)
    acc_m1 = np.zeros_like(w)
    acc_m3 = np.zeros_like(w)

    # Scatter each source sector's weight into whichever target sector it was
    # deflected to. One source sector can land anywhere, so this is a scatter
    # rather than a reshuffle -- but for a *fixed* source sector every point has
    # exactly one target, so the flat indices `target[s]*npix + point` are unique
    # and plain fancy-index `+=` is correct. No np.add.at (slow) and no dense
    # one-hot.
    #
    # This is a memory change, not a speed one: it avoids building an
    # (n_sec, ny, nx) bool mask and three full-size float64 temporaries on each
    # of n_sec iterations -- ~12 GB of allocation churn per sww-sized cell, and
    # 2.6x that on ph1 -- but measured runtime is unchanged (110.8 s vs 106.2 s,
    # bitwise-identical output). The time in this function was never the one-hot;
    # it was `fit_weibull_moments` below. Left in place for the smaller footprint
    # on the larger domains, where the old version crowded a 64 GB node.
    npix = w[0].size
    pix = np.arange(npix)
    # Views onto the accumulators, so the scatter writes straight through.
    flat_f, flat_m1, flat_m3 = acc_f.reshape(-1), acc_m1.reshape(-1), acc_m3.reshape(-1)
    for s in range(n_sec):
        dst = target[s].ravel() * npix + pix
        ws = w[s].ravel()
        flat_f[dst] += ws
        flat_m1[dst] += ws * np.nan_to_num(m1_s[s]).ravel()
        flat_m3[dst] += ws * np.nan_to_num(m3_s[s]).ravel()

    tot = np.where(acc_f > 0, acc_f, np.nan)
    A_out, k_out = fit_weibull_moments(acc_m1 / tot, acc_m3 / tot)

    # An empty target sector has no distribution to describe. Emitting its zero
    # frequency alongside NaN A/k would put "nan" in the file, so fall back to
    # the un-deflected parameters -- carrying zero weight, they change nothing.
    A_out = np.where(np.isfinite(A_out), A_out, A_fine)
    k_out = np.where(np.isfinite(k_out), k_out, k_fine)
    return A_out, k_out, acc_f
