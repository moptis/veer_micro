"""Common interface for the coarse driving fields FuXi-CFD consumes.

The model is a **steady-state RANS emulator**: every training label is a
converged solution for one (terrain, inflow) pair, and nothing in the network
carries a time dimension. So the axis the pipeline sweeps is not intrinsically
time -- it is a list of independent *inflow states*, each of which the model maps
to a 3-D field.

Two things fill that axis:

* `WrfSource` -- one state per WRF timestep. What the original pipeline does.
* `WrgSectorSource` -- one state per wind-direction sector, built from a coarse
  `.wrg`. Statistics in, statistics out; ~12 states instead of ~52,000.

`CoarseSource` is the seam they share. It deliberately says nothing about time:
`WrfSource` adds the timestamp machinery on top, because non-chronological file
order and timestamp selection are real problems for that backend and meaningless
for the other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class CoarseSource(ABC):
    """A set of coarse 100 m wind states on one flat lat/lon grid.

    Implementations expose `n_states` inflow cases; `read_uv` returns the
    earth-relative components for a subset of them. Components, never speed and
    direction: averaging or interpolating direction across the 0/360 branch cut
    is a reliable way to manufacture nonsense, so the conversion happens once at
    the source and every downstream stage works in u/v.
    """

    #: (nrow, ncol) of the coarse grid the flat arrays unravel to, C order.
    domain_shape: tuple[int, int]

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of independent inflow states this source can supply."""

    @property
    @abstractmethod
    def state_labels(self) -> list[str]:
        """Human-readable label per state, for logs and store metadata."""

    @abstractmethod
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """(lat, lon), each flat of length `prod(domain_shape)`, C order."""

    @abstractmethod
    def read_uv(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Earth-relative u, v at 100 m for `states` -> two (n, n_points) arrays."""


class WrfSource(CoarseSource):
    """A time series of WRF fields on the flat coarse grid.

    Implementations expose time as a `pandas.DatetimeIndex` in *file row order*,
    which for control.h5 is emphatically not chronological order. Callers select
    rows by timestamp via `select`, never by assuming order.
    """

    @property
    @abstractmethod
    def times(self) -> pd.DatetimeIndex:
        """Timestamps in file row order. May be unsorted and non-contiguous."""

    @abstractmethod
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """(lat, lon), each flat of length 10296."""

    @abstractmethod
    def read(self, variables: list[str], rows: np.ndarray) -> dict[str, np.ndarray]:
        """Read `variables` at the given file row indices -> {name: (n, 10296)}."""

    # --- CoarseSource ---------------------------------------------------------
    #
    # A WRF state *is* a timestep, so these are thin views over the time axis.
    # `read_uv` is expressed via `read_wind_100m`, which every concrete backend
    # provides; the conversion to components lives in `winds.speeddir_to_uv`.

    @property
    def n_states(self) -> int:
        return len(self.times)

    @property
    def state_labels(self) -> list[str]:
        return [str(t) for t in self.times]

    def read_uv(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from ..winds import speeddir_to_uv

        return speeddir_to_uv(*self.read_wind_100m(np.asarray(states)))

    def select(
        self,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stride: int = 1,
        limit: int | None = None,
    ) -> np.ndarray:
        """Row indices whose timestamps fall in [start, end], in chronological order.

        `stride` subsamples the chronologically-sorted selection, so it means
        "every Nth timestep in time", not "every Nth row in the file".
        """
        t = self.times
        mask = np.ones(len(t), dtype=bool)
        if start is not None:
            mask &= t >= pd.Timestamp(start, tz="UTC")
        if end is not None:
            mask &= t <= pd.Timestamp(end, tz="UTC")
        rows = np.flatnonzero(mask)
        rows = rows[np.argsort(t[rows].values, kind="stable")]
        if stride > 1:
            rows = rows[::stride]
        if limit is not None:
            rows = rows[:limit]
        return rows

    def time_report(self) -> str:
        """Human-readable summary of the (surprising) time structure."""
        t = self.times
        by_year = t.year.value_counts().sort_index()
        gaps = np.diff(t.values).astype("timedelta64[m]").astype(np.int64)
        n_back = int((gaps < 0).sum())
        lines = [
            f"n_times          : {len(t)}",
            f"range            : {t.min()} .. {t.max()}",
            f"monotonic        : {t.is_monotonic_increasing}",
            f"backward jumps   : {n_back}",
            f"unique           : {t.nunique()}",
            "per-year counts  :",
        ]
        lines += [f"    {y}: {n}" for y, n in by_year.items()]
        return "\n".join(lines)
