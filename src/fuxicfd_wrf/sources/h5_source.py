"""Reader for the consolidated control.h5 store.

Layout (verified on /scratch/moptis/c2wind/broadview/control/processed/broadview):

    <var>            (52560, 10296) float32, uncompressed, chunks (411, 81)
    coordinates      (10296, 2)  float32  -- column 0 is lat, column 1 is lon
    elevation        (10296,)    float32
    time_index       (52560,)    |S25     -- ISO strings with +00:00 offset
    root attr domain_shape = [78, 132]

Two properties of this file drive the whole design:

**Time is not chronological.** The 52560 rows are 10-min cadence but stitched from
~12 month-long chunks scattered over 2000-2018, and the index is not monotonic --
row 8759 is 2013-03-02 sitting between 2002 rows. Never use row order as time
order. (`control_status.csv` alongside starts at 2019-01-01 and matches neither;
it is ignored.)

**Chunking punishes row-at-a-time access.** With chunks of (411, 81), pulling one
timestep across all 10296 columns touches 127 chunks and reads ~17 MB to deliver
41 KB. Reading a full 411-row chunk block costs the same 17 MB but yields 411
timesteps. `read` therefore groups requested rows into chunk-aligned blocks.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..constants import WRF_N_POINTS
from .base import WrfSource

#: Row-chunk height of the variable datasets. Read blocks are aligned to this.
DEFAULT_CHUNK_ROWS = 411


class H5WrfSource(WrfSource):
    def __init__(self, path: Path | str, *, max_block_rows: int = DEFAULT_CHUNK_ROWS * 2):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.max_block_rows = max_block_rows
        self._file: h5py.File | None = None
        self._times: pd.DatetimeIndex | None = None
        self._chunk_rows = DEFAULT_CHUNK_ROWS

    # --- lifecycle ---------------------------------------------------------

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            for name in self._file:
                ds = self._file[name]
                if isinstance(ds, h5py.Dataset) and ds.ndim == 2 and ds.chunks:
                    self._chunk_rows = ds.chunks[0]
                    break
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- metadata ----------------------------------------------------------

    @property
    def times(self) -> pd.DatetimeIndex:
        if self._times is None:
            raw = self.file["time_index"][:]
            self._times = pd.DatetimeIndex(
                pd.to_datetime([b.decode() for b in raw], utc=True)
            )
        return self._times

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.file["coordinates"][:]
        if c.shape != (WRF_N_POINTS, 2):
            raise ValueError(f"coordinates has shape {c.shape}, expected ({WRF_N_POINTS}, 2)")
        return c[:, 0].astype(np.float64), c[:, 1].astype(np.float64)

    def elevation(self) -> np.ndarray:
        return self.file["elevation"][:].astype(np.float64)

    @property
    def variables(self) -> list[str]:
        return [
            n for n in self.file
            if isinstance(self.file[n], h5py.Dataset) and self.file[n].ndim == 2
        ]

    # --- reading -----------------------------------------------------------

    def read(self, variables: list[str], rows: np.ndarray) -> dict[str, np.ndarray]:
        """Read variables at `rows`, returning arrays in the order `rows` was given.

        Rows are internally regrouped into chunk-aligned contiguous blocks so the
        HDF5 chunk cache is not thrashed, then scattered back to the caller's order.
        """
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return {v: np.empty((0, WRF_N_POINTS), dtype=np.float32) for v in variables}

        missing = [v for v in variables if v not in self.file]
        if missing:
            raise KeyError(f"variables not in {self.path.name}: {missing}")

        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        out = {
            v: np.empty((rows.size, WRF_N_POINTS), dtype=np.float32) for v in variables
        }

        for block_start, block_end, sel in self._blocks(sorted_rows):
            local = sorted_rows[sel] - block_start
            for v in variables:
                block = self.file[v][block_start:block_end, :]
                out[v][order[sel]] = block[local]
        return out

    def _blocks(self, sorted_rows: np.ndarray):
        """Yield (block_start, block_end, slice_into_sorted_rows) chunk-aligned groups."""
        i = 0
        n = sorted_rows.size
        while i < n:
            start = (sorted_rows[i] // self._chunk_rows) * self._chunk_rows
            end = start + self.max_block_rows
            j = i
            while j < n and sorted_rows[j] < end:
                j += 1
            yield int(start), int(min(end, self.file["time_index"].shape[0])), slice(i, j)
            i = j

    def read_wind_100m(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: (wind_speed_100m, wind_direction_100m) at `rows`."""
        d = self.read(["wind_speed_100m", "wind_direction_100m"], rows)
        return d["wind_speed_100m"], d["wind_direction_100m"]
