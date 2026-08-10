"""Common interface for WRF input backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class WrfSource(ABC):
    """A time series of WRF fields on the flat 10296-point Broadview grid.

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
