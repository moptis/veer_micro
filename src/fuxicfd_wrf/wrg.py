"""WRG (Wind Resource Grid) export from FuXi-CFD output.

Produces a WAsP-style ``.wrg`` matching the layout in
``~/wakemap_processing/wakemap_processor/create_wrg.py``, so the result drops into
the existing toolchain::

    <nx> <ny> <xmin> <ymin> <cellsize>
    GridPoint X Y elev hub_height A k power_density n_sectors [freq*1000 A*10 k*100]*n

Sector convention matches that code: the first sector is centred on 0 deg (north),
edges at -15, 15, 45, ... for the default 30 deg bins, giving 12 sectors.

Two deliberate deviations from the reference implementation
------------------------------------------------------------
**1. Power density is the standard W/m^2**, ``0.5 * rho * mean(U^3)``, computed
from the exact cubed-speed moment rather than from the fitted Weibull. The
reference instead computes ``0.5 * rho * pi*(H/2)^2 * mean(U)^3 * 0.33 / 1000``,
a turbine power estimate in kW using rotor area and a power coefficient -- not a
power density, and ``mean(U)^3`` understates ``mean(U^3)`` by 50-100% for a
typical Weibull (Jensen's inequality). Existing coarse WRG deliverables may use
the other convention; recompute rather than compare those columns directly.

**2. No regridding interpolation.** The reference scatter-interpolates pixels onto
a uniform grid with `griddata`. FuXi-CFD output is already on a regular projected
grid, so this samples it directly -- no interpolation error, and no need for the
20% edge buffer the reference applies by default.

Why histograms
--------------
A WRG needs a Weibull fit per grid point per sector. Holding the full time series
for a 100 m grid over this domain would be ~285 GB, so instead the time axis is
streamed once and accumulated into per-point, per-sector speed histograms
(0.5 m/s bins by default). Fitting Weibull from a binned histogram is exactly what
WAsP itself does, and it makes memory independent of series length.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import OUTPUT_HEIGHTS
from .grids import Tile

#: Sea-level standard density. Broadview sits at ~1340 m where rho ~ 1.06, so the
#: default overstates power density by ~13%; pass air_density explicitly.
DEFAULT_AIR_DENSITY = 1.225

DEFAULT_BIN_SIZE_DEG = 30.0
DEFAULT_SPEED_BIN_MS = 0.5
DEFAULT_MAX_SPEED_MS = 50.0


# --- Weibull ----------------------------------------------------------------

#: Shape parameter is clipped to this range during iteration. Real wind climates
#: sit near k ~ 1.5-3; hitting a bound means the input distribution is degenerate
#: (usually too short a record), not that the wind is exotic.
K_BOUNDS = (0.2, 20.0)


def weibull_mle_from_hist(
    counts: np.ndarray,
    centers: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    block: int = 200_000,
    warn_clipped: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised 2-parameter Weibull MLE (loc fixed at 0) from histograms.

    `counts` is (n_points, n_bins); `centers` is (n_bins,). Returns (A, k), each
    (n_points,), with NaN where a point has no data.

    Solves the standard MLE condition by fixed-point iteration on the shape
    parameter::

        1/k = sum(w x^k ln x) / sum(w x^k) - sum(w ln x)/sum(w)
        A   = (sum(w x^k) / sum(w))^(1/k)

    This is the same estimator `scipy.stats.weibull_min.fit(x, floc=0)` computes,
    but solved for every grid point at once. Looping scipy over a few hundred
    thousand points x 13 sectors would take hours; this takes seconds.

    Points are processed in blocks because the `x^k` term depends on each point's
    own k, so the working array is (block, n_bins).
    """
    counts = np.asarray(counts, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != centers.size:
        raise ValueError(f"counts {counts.shape} incompatible with centers {centers.shape}")
    if (centers <= 0).any():
        raise ValueError("Weibull requires strictly positive bin centres")

    n_pts = counts.shape[0]
    A_out = np.full(n_pts, np.nan)
    k_out = np.full(n_pts, np.nan)
    ln_x = np.log(centers)

    for s in range(0, n_pts, block):
        w = counts[s : s + block]
        n = w.sum(axis=1)
        ok = n > 1
        if not ok.any():
            continue
        wo = w[ok]
        no = n[ok]
        mean_ln = (wo * ln_x).sum(axis=1) / no

        # Method-of-moments style seed; robust for wind speed distributions.
        k = np.full(wo.shape[0], 2.0)
        for _ in range(max_iter):
            # exp(k*ln x) rather than x**k: one exp, and no overflow for large k.
            xk = np.exp(k[:, None] * ln_x[None, :])
            wxk = wo * xk
            denom = wxk.sum(axis=1)
            denom = np.where(denom > 0, denom, np.nan)
            num = (wxk * ln_x).sum(axis=1)
            k_new = 1.0 / np.maximum(num / denom - mean_ln, 1e-12)
            k_new = np.clip(k_new, *K_BOUNDS)
            if np.nanmax(np.abs(k_new - k)) < tol:
                k = k_new
                break
            k = k_new

        xk = np.exp(k[:, None] * ln_x[None, :])
        A = ((wo * xk).sum(axis=1) / no) ** (1.0 / k)
        idx = np.flatnonzero(ok) + s
        A_out[idx] = A
        k_out[idx] = k

    if warn_clipped:
        fitted = np.isfinite(k_out)
        n_fit = int(fitted.sum())
        if n_fit:
            at_bound = fitted & (
                (k_out <= K_BOUNDS[0] * 1.001) | (k_out >= K_BOUNDS[1] * 0.999)
            )
            n_bad = int(at_bound.sum())
            frac = n_bad / n_fit
            # A low threshold on purpose: over a full record essentially nothing
            # should clip, so even a small population is worth surfacing. A
            # handful of isolated pixels is not worth a line of output.
            if frac > 0.001:
                print(
                    f"[wrg] WARNING: {n_bad:,}/{n_fit:,} ({frac:.2%}) of "
                    f"{warn_clipped} fits hit the k bound {K_BOUNDS}. Those points "
                    f"have a near-degenerate speed distribution -- usually too "
                    f"short a record -- and their A/k are not meaningful."
                )
    return A_out, k_out


def sector_edges(bin_size_deg: float = DEFAULT_BIN_SIZE_DEG) -> np.ndarray:
    """Sector edges with the first sector centred on north.

    Matches the reference: for 30 deg bins the edges are -15, 15, ..., 345, so
    sector 0 spans (-15, 15] and is centred on 0 deg.
    """
    center = 360.0 - bin_size_deg / 2.0
    return np.arange(-bin_size_deg / 2.0, center + 1e-9, bin_size_deg)


def assign_sector(direction_deg: np.ndarray, bin_size_deg: float) -> np.ndarray:
    """Sector index per sample, wrapping directions above the last edge."""
    d = np.asarray(direction_deg, dtype=np.float32).copy()
    center = 360.0 - bin_size_deg / 2.0
    d[d > center] -= 360.0
    idx = np.floor((d + bin_size_deg / 2.0) / bin_size_deg).astype(np.int16)
    n = len(sector_edges(bin_size_deg)) - 1
    return np.clip(idx, 0, n - 1)


# --- accumulation ------------------------------------------------------------

@dataclass
class SectorHistograms:
    """Per-point, per-sector wind speed histograms, filled by streaming time."""

    n_points: int
    n_sectors: int
    speed_bin: float = DEFAULT_SPEED_BIN_MS
    max_speed: float = DEFAULT_MAX_SPEED_MS
    counts: np.ndarray = field(init=False)
    sum_u3: np.ndarray = field(init=False)
    sum_u: np.ndarray = field(init=False)
    n_samples: np.ndarray = field(init=False)

    def __post_init__(self):
        self.n_bins = int(round(self.max_speed / self.speed_bin))
        self.counts = np.zeros(
            (self.n_points, self.n_sectors, self.n_bins), dtype=np.uint32
        )
        # Exact moments alongside the histogram: power density should not inherit
        # binning error, and mean(U^3) is what it actually needs.
        self.sum_u3 = np.zeros(self.n_points, dtype=np.float64)
        self.sum_u = np.zeros(self.n_points, dtype=np.float64)
        self.n_samples = np.zeros(self.n_points, dtype=np.int64)

    @property
    def centers(self) -> np.ndarray:
        return (np.arange(self.n_bins) + 0.5) * self.speed_bin

    def nbytes(self) -> int:
        return self.counts.nbytes

    #: Points per bincount pass. The temporary is
    #: `block x n_sectors x n_bins x 8` bytes, so this caps it regardless of grid
    #: size. Doing all points at once needs `n_points x 16 x 100 x 8` -- 11.7 GB
    #: on a 913k-point domain grid, which OOM-killed a full-domain export.
    POINT_BLOCK = 50_000

    def add(self, speed: np.ndarray, direction: np.ndarray) -> None:
        """Accumulate a time block. `speed`/`direction` are (n_time, n_points).

        Points are processed in blocks so the bincount temporary stays bounded.
        `np.bincount(..., minlength=counts.size)` allocates an int64 array of the
        FULL histogram size on every call; at domain scale that is 11.7 GB per
        call per worker, which is what killed the first full-domain attempt.
        Blocking makes peak memory independent of how many grid points there are.
        """
        speed = np.asarray(speed, dtype=np.float32)
        direction = np.asarray(direction, dtype=np.float32)
        nt, npts = speed.shape
        if npts != self.n_points:
            raise ValueError(f"expected {self.n_points} points, got {npts}")

        bin_deg = 360.0 / self.n_sectors
        stride = self.n_sectors * self.n_bins

        for p0 in range(0, npts, self.POINT_BLOCK):
            p1 = min(p0 + self.POINT_BLOCK, npts)
            sp = speed[:, p0:p1]
            di = direction[:, p0:p1]
            valid = np.isfinite(sp) & np.isfinite(di) & (sp > 0)
            if not valid.any():
                continue
            # Zero non-finite entries before casting: NaN -> int is undefined.
            safe_sp = np.where(valid, sp, 0.0)
            safe_di = np.where(valid, di, 0.0)
            sec = assign_sector(safe_di, bin_deg)
            sbin = np.clip((safe_sp / self.speed_bin).astype(np.int32), 0, self.n_bins - 1)

            nblock = p1 - p0
            col = np.broadcast_to(np.arange(nblock, dtype=np.int64), (nt, nblock))
            flat = (
                col[valid] * stride
                + sec[valid].astype(np.int64) * self.n_bins
                + sbin[valid].astype(np.int64)
            )
            bc = np.bincount(flat, minlength=nblock * stride)
            self.counts[p0:p1] += bc.reshape(nblock, self.n_sectors, self.n_bins).astype(
                np.uint32
            )
            del bc, flat, col, sec, sbin

            spv = np.where(valid, sp, 0.0).astype(np.float64)
            self.sum_u[p0:p1] += spv.sum(axis=0)
            self.sum_u3[p0:p1] += (spv**3).sum(axis=0)
            self.n_samples[p0:p1] += valid.sum(axis=0)

    def estimate_peak_gb(self, n_time_block: int, grid_points: int) -> float:
        """Rough per-worker peak, so a doomed run fails before it starts."""
        hist = self.counts.nbytes
        temp = self.POINT_BLOCK * self.n_sectors * self.n_bins * 8
        acc = 2 * n_time_block * grid_points * 4 * 2   # u,v accumulators + copies
        return (hist + temp + acc) / 1e9


# --- record assembly ---------------------------------------------------------

def power_density(
    hist: SectorHistograms,
    *,
    air_density: float = DEFAULT_AIR_DENSITY,
) -> np.ndarray:
    """Power density per point: ``0.5 * rho * mean(U^3)`` in W/m^2.

    Uses the exact accumulated third moment, not the fitted Weibull, so binning
    and fit error do not propagate into it.
    """
    n = np.maximum(hist.n_samples, 1)
    return 0.5 * air_density * (hist.sum_u3 / n)


def build_records(
    hist: SectorHistograms,
    *,
    elevation: np.ndarray,
    hub_height: float,
    air_density: float = DEFAULT_AIR_DENSITY,
) -> np.ndarray:
    """Assemble the per-point WRG value array.

    Column layout matches the reference:
        [elev, hub_height, A, k, power_density, n_sectors,
         (freq*1000, A*10, k*100) * n_sectors]
    """
    centers = hist.centers
    total = hist.counts.sum(axis=1)                     # (n_pts, n_bins)
    A_all, k_all = weibull_mle_from_hist(total, centers, warn_clipped="all-sector")

    n_pts, n_sec = hist.n_points, hist.n_sectors
    rec = np.zeros((n_pts, 6 + 3 * n_sec), dtype=np.float64)
    rec[:, 0] = elevation
    rec[:, 1] = hub_height
    rec[:, 2] = A_all
    rec[:, 3] = k_all
    rec[:, 4] = power_density(hist, air_density=air_density)
    rec[:, 5] = n_sec

    n_tot = np.maximum(total.sum(axis=1), 1)
    for s in range(n_sec):
        c = hist.counts[:, s, :]
        A_s, k_s = weibull_mle_from_hist(c, centers)
        freq = c.sum(axis=1) / n_tot
        # A sector with too little data has no meaningful fit; fall back to the
        # all-sector parameters rather than emitting NaN into the file.
        A_s = np.where(np.isfinite(A_s), A_s, A_all)
        k_s = np.where(np.isfinite(k_s), k_s, k_all)
        rec[:, 6 + 3 * s + 0] = freq * 1000.0
        rec[:, 6 + 3 * s + 1] = A_s * 10.0
        rec[:, 6 + 3 * s + 2] = k_s * 100.0
    return rec


def write_wrg(
    path: Path | str,
    x: np.ndarray,
    y: np.ndarray,
    records: np.ndarray,
    *,
    resolution_m: float,
) -> Path:
    """Write a ``.wrg`` file.

    `x`, `y` are per-point coordinates (flattened); `records` is the array from
    `build_records`. Formatting follows the reference: the all-sector A and k keep
    2 decimals, power density 0 decimals, everything else is written as integers
    (sector triplets are pre-scaled by 1000/10/100 so they carry their precision).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if not (len(x) == len(y) == records.shape[0]):
        raise ValueError(
            f"length mismatch: x={len(x)} y={len(y)} records={records.shape[0]}"
        )

    n_nan = int((~np.isfinite(records[:, 2])).sum())
    if n_nan:
        print(
            f"[wrg] WARNING: {n_nan:,}/{records.shape[0]:,} grid points have no "
            "Weibull fit and are written as zeros. A consumer cannot distinguish "
            "those from genuine calm; prefer a fully covered grid."
        )

    df = pd.DataFrame({"X": np.round(x).astype(np.int64), "Y": np.round(y).astype(np.int64)})
    for i in range(records.shape[1]):
        col = records[:, i]
        if i in (2, 3):
            # nan_to_num here too: these are float columns, so NaN would otherwise
            # be written as the literal text "nan" and break any WAsP-style reader.
            df[i] = np.round(np.nan_to_num(col), 2)
        elif i == 4:
            # Power density is a float column like 2 and 3, and it needs the same
            # treatment for a worse reason. `to_csv` renders NaN as the *empty
            # string*, so an uncovered point wrote a line with 56 fields instead
            # of 57 -- every sector value after it shifted left by one, silently,
            # in a file that still parses. Measured on sww: 1,500 of 1,951,500
            # lines, the whole uncovered edge column.
            df[i] = np.round(np.nan_to_num(col), 0)
        else:
            df[i] = np.round(np.nan_to_num(col), 0).astype(np.int64)
    df.insert(0, "GridPoint", "GridPoint")

    header = " ".join(
        str(v) for v in [
            len(np.unique(df["X"])), len(np.unique(df["Y"])),
            int(df["X"].min()), int(df["Y"].min()), int(round(resolution_m)),
        ]
    )
    # Write to a sibling temp file and rename into place. os.replace is atomic
    # within a filesystem, so `path` only ever exists complete: a reader either
    # sees the previous version or the new one, never a truncated prefix.
    #
    # This is not hypothetical. Two overlapping sweeps were once launched with
    # intersecting cell sets, and the second wrote a complete file while the
    # first was still running the same cell; when the first reached its write it
    # truncated the finished file back to a 28-byte header. A validation pass
    # that had counted the file as present would have read nothing, and the
    # resume guard (`test -s`) would have called the placeholder "done".
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "w") as fh:
            fh.write(header + "\n")
        df.to_csv(tmp, sep=" ", header=False, mode="a", index=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def nearest_model_level(hub_height: float) -> tuple[int, float]:
    """Index and exact height of the model level nearest `hub_height`.

    FuXi-CFD has no 100 m level -- the nearest is 102.17944 m. This returns what
    the file will actually represent so the caller can report it rather than
    quietly mislabelling the hub height.
    """
    j = int(np.argmin(np.abs(OUTPUT_HEIGHTS - hub_height)))
    return j, float(OUTPUT_HEIGHTS[j])
