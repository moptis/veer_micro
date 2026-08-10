"""Reader for the per-timestep npz directory.

Files are named ``wrfout_d03_YYYY-MM-DD_HH:MM:SS.npz`` and hold 52 variables as
flat (10296,) arrays -- the same grid as control.h5, one timestep per file.

**The npz files carry no coordinates.** There is no lat/lon or elevation array in
them, so this backend cannot geolocate itself. Pass `coordinates_from=` pointing at
control.h5 (or supply lat/lon directly); geometry is identical between the two.

Prefer the HDF5 backend for long runs -- 52560 file opens is far slower than
chunk-blocked reads from one file. This backend exists for spot checks, for
timesteps not present in the h5, and as a cross-check that the two agree.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import WRF_N_POINTS
from .base import WrfSource

FILENAME_RE = re.compile(r"wrfout_d03_(\d{4}-\d{2}-\d{2})_(\d{2}):(\d{2}):(\d{2})\.npz$")


class NpzWrfSource(WrfSource):
    def __init__(
        self,
        directory: Path | str,
        *,
        coordinates_from: Path | str | None = None,
        lat: np.ndarray | None = None,
        lon: np.ndarray | None = None,
    ):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise NotADirectoryError(self.directory)

        paths, stamps = [], []
        for p in sorted(self.directory.glob("wrfout_d03_*.npz")):
            m = FILENAME_RE.search(p.name)
            if m:
                date, hh, mm, ss = m.groups()
                paths.append(p)
                stamps.append(f"{date} {hh}:{mm}:{ss}")
        if not paths:
            raise FileNotFoundError(f"no wrfout_d03_*.npz files under {self.directory}")

        self.paths = np.array(paths, dtype=object)
        self._times = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True))

        self._coordinates_from = Path(coordinates_from) if coordinates_from else None
        self._lat = np.asarray(lat, dtype=np.float64) if lat is not None else None
        self._lon = np.asarray(lon, dtype=np.float64) if lon is not None else None

    @property
    def times(self) -> pd.DatetimeIndex:
        return self._times

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        if self._lat is not None and self._lon is not None:
            return self._lat, self._lon
        if self._coordinates_from is None:
            raise ValueError(
                "npz files contain no coordinates. Construct NpzWrfSource with "
                "coordinates_from=<path to control.h5>, or pass lat=/lon= explicitly."
            )
        import h5py

        with h5py.File(self._coordinates_from, "r") as f:
            c = f["coordinates"][:]
        self._lat = c[:, 0].astype(np.float64)
        self._lon = c[:, 1].astype(np.float64)
        return self._lat, self._lon

    @property
    def variables(self) -> list[str]:
        with np.load(self.paths[0]) as z:
            return list(z.files)

    def read(self, variables: list[str], rows: np.ndarray) -> dict[str, np.ndarray]:
        rows = np.asarray(rows, dtype=np.int64)
        out = {
            v: np.empty((rows.size, WRF_N_POINTS), dtype=np.float32) for v in variables
        }
        for i, r in enumerate(rows):
            with np.load(self.paths[r]) as z:
                for v in variables:
                    if v not in z.files:
                        raise KeyError(f"{v} not in {self.paths[r].name}")
                    arr = z[v]
                    if arr.shape != (WRF_N_POINTS,):
                        raise ValueError(
                            f"{v} in {self.paths[r].name} has shape {arr.shape}, "
                            f"expected ({WRF_N_POINTS},)"
                        )
                    out[v][i] = arr.astype(np.float32)
        return out

    def read_wind_100m(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.read(["wind_speed_100m", "wind_direction_100m"], rows)
        return d["wind_speed_100m"], d["wind_direction_100m"]
