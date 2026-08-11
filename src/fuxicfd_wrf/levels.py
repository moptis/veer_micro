"""Vertical level selection and interpolation.

FuXi-CFD emits 27 fixed heights (10 m to 300.795 m). None of them is a round
number above 50 m -- the stack goes 75.5255, 83.57805, 92.435855, 102.17944,
112.897385, 124.687123, 137.655835, 151.921419, 167.613561 -- so any deliverable
at 80/100/120/160 m has to be interpolated.

Selection and interpolation are handled by one object: a (n_target, n_source)
weight matrix. Exact hits get a one-hot row, interpolated heights get a two-point
row, and applying it is a single einsum either way. That keeps the "did we
interpolate?" question out of every call site.

Interpolation is linear in log(z)
---------------------------------
Wind speed near the surface follows a log profile, so log(z) is the physically
motivated abscissa. In practice it barely matters here: model levels are 8-11%
apart, and interpolating to 80 m gives a weight of 0.567 in log-z versus 0.556
linear -- about 0.002 m/s on a typical 0.15 m/s inter-level difference. The choice
is defensible rather than consequential, so it is not worth agonising over.

Extrapolation is refused. A request below 10 m or above 300.795 m raises rather
than silently clamping to the end of the stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A target within this distance of a source level is treated as an exact hit.
EXACT_TOL_M = 0.01


def interp_weights(
    src_heights: np.ndarray, targets: np.ndarray, *, method: str = "log"
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (n_target, n_source) interpolation matrix.

    Returns (weights, exact_mask).
    """
    src = np.asarray(src_heights, dtype=np.float64)
    tgt = np.atleast_1d(np.asarray(targets, dtype=np.float64))
    if np.any(np.diff(src) <= 0):
        raise ValueError("source heights must be strictly ascending")
    if method not in ("log", "linear"):
        raise ValueError(f"method must be 'log' or 'linear', got {method!r}")

    lo, hi = src[0], src[-1]
    out_of_range = (tgt < lo - EXACT_TOL_M) | (tgt > hi + EXACT_TOL_M)
    if out_of_range.any():
        raise ValueError(
            f"requested heights {tgt[out_of_range].tolist()} m fall outside the "
            f"available stack [{lo:.5f}, {hi:.5f}] m. Extrapolating a wind profile "
            "beyond the model's top or bottom level is not supported."
        )

    W = np.zeros((tgt.size, src.size), dtype=np.float64)
    exact = np.zeros(tgt.size, dtype=bool)

    x_src = np.log(src) if method == "log" else src
    for i, h in enumerate(tgt):
        d = np.abs(src - h)
        j = int(np.argmin(d))
        if d[j] <= EXACT_TOL_M:
            W[i, j] = 1.0
            exact[i] = True
            continue
        upper = int(np.searchsorted(src, h))
        upper = min(max(upper, 1), src.size - 1)
        lower = upper - 1
        x_h = np.log(h) if method == "log" else h
        span = x_src[upper] - x_src[lower]
        f = 0.0 if span == 0 else (x_h - x_src[lower]) / span
        W[i, lower] = 1.0 - f
        W[i, upper] = f
    return W, exact


@dataclass
class LevelSelection:
    """Maps a source level axis onto requested output heights."""

    heights: np.ndarray       # (n_out,) requested heights, ascending
    weights: np.ndarray       # (n_out, n_src)
    exact: np.ndarray         # (n_out,) bool
    source_heights: np.ndarray
    method: str = "log"

    @classmethod
    def build(
        cls,
        source_heights: np.ndarray,
        requested: list[float] | np.ndarray | None,
        *,
        method: str = "log",
    ) -> "LevelSelection":
        src = np.asarray(source_heights, dtype=np.float64)
        if requested is None:
            return cls(
                heights=src.copy(),
                weights=np.eye(src.size),
                exact=np.ones(src.size, dtype=bool),
                source_heights=src,
                method=method,
            )
        tgt = np.sort(np.asarray(requested, dtype=np.float64))
        W, exact = interp_weights(src, tgt, method=method)
        return cls(
            heights=tgt, weights=W, exact=exact, source_heights=src, method=method
        )

    @property
    def n_out(self) -> int:
        return len(self.heights)

    @property
    def any_interpolated(self) -> bool:
        return bool((~self.exact).any())

    def describe(self) -> str:
        parts = []
        for h, ex in zip(self.heights, self.exact):
            if ex:
                parts.append(f"{h:g} m (exact)")
            else:
                w = self.weights[list(self.heights).index(h)]
                nz = np.flatnonzero(w)
                src = self.source_heights
                parts.append(
                    f"{h:g} m (interp {src[nz[0]]:.3f}/{src[nz[-1]]:.3f} m, "
                    f"w={w[nz[-1]]:.3f})"
                )
        return "; ".join(parts)

    @property
    def support(self) -> tuple[int, int]:
        """Contiguous source-level range [lo, hi) that the weights actually touch.

        Interpolating one height needs two *adjacent* source levels, so reading
        the whole level axis to produce it wastes most of the I/O -- 27 levels
        read for 2 used. Callers that read from disk should slice to this range
        and use `weights_within` instead.
        """
        nz = np.flatnonzero(self.weights.any(axis=0))
        return int(nz[0]), int(nz[-1]) + 1

    def weights_within(self, lo: int, hi: int) -> np.ndarray:
        """Weight matrix restricted to source levels [lo, hi)."""
        return self.weights[:, lo:hi]

    def apply_sliced(self, field: np.ndarray, lo: int, hi: int, *, level_axis: int = 1) -> np.ndarray:
        """Apply to a field already sliced to source levels [lo, hi)."""
        moved = np.moveaxis(np.asarray(field), level_axis, 0)
        out = np.tensordot(self.weights_within(lo, hi), moved, axes=([1], [0]))
        return np.moveaxis(out, 0, level_axis)

    def apply(self, field: np.ndarray, *, level_axis: int = 1) -> np.ndarray:
        """Apply the selection to `field`, whose level axis is `level_axis`.

        Typical shape is (n_time, n_level, y, x) -> (n_time, n_out, y, x).
        """
        field = np.asarray(field)
        if field.shape[level_axis] != self.weights.shape[1]:
            raise ValueError(
                f"field has {field.shape[level_axis]} levels on axis {level_axis}, "
                f"selection expects {self.weights.shape[1]}"
            )
        moved = np.moveaxis(field, level_axis, 0)          # (n_src, ...)
        out = np.tensordot(self.weights, moved, axes=([1], [0]))  # (n_out, ...)
        return np.moveaxis(out, 0, level_axis).astype(field.dtype, copy=False)
