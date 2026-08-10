"""Read a ``.wrg`` file back into a DataFrame.

Used by the comparison notebook to load both the coarse 1 km deliverables and the
FuXi-CFD output on equal footing. Sector count is read from each row rather than
assumed -- the existing Broadview coarse files use **16** sectors (22.5 deg) while
`create_wrg.py` defaults to 12, so hard-coding either would silently mis-parse.

Power density is deliberately **recomputed** from the fitted A and k rather than
taken from the file's column, because different producers write different things
there (turbine kW vs W/m^2). Recomputing puts every file on one convention:

    P = 0.5 * rho * A^3 * Gamma(1 + 3/k)     [W/m^2]

which is the analytic mean of U^3 for a Weibull distribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gamma

DEFAULT_AIR_DENSITY = 1.225


def weibull_mean_speed(A: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Mean wind speed of a Weibull(A, k): ``A * Gamma(1 + 1/k)``."""
    return np.asarray(A) * gamma(1.0 + 1.0 / np.asarray(k))


def weibull_power_density(
    A: np.ndarray, k: np.ndarray, air_density: float = DEFAULT_AIR_DENSITY
) -> np.ndarray:
    """Power density in W/m^2 from Weibull parameters: ``0.5*rho*A^3*Gamma(1+3/k)``."""
    A = np.asarray(A, dtype=float)
    k = np.asarray(k, dtype=float)
    return 0.5 * air_density * A**3 * gamma(1.0 + 3.0 / k)


def read_wrg(path: Path | str, *, air_density: float = DEFAULT_AIR_DENSITY) -> pd.DataFrame:
    """Parse a .wrg into a DataFrame with one row per grid point.

    Columns: x, y, elevation, hub_height, A, k, power_density_file, n_sectors,
    mean_speed, power_density (recomputed), plus per-sector freq_i / A_i / k_i.
    """
    path = Path(path)
    with open(path) as fh:
        header = fh.readline().split()
        rows = [line.split() for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"{path} has no data rows")

    nx, ny, xmin, ymin, cell = (float(v) for v in header[:5])
    n_sec = int(rows[0][8])
    base = {
        "x": [], "y": [], "elevation": [], "hub_height": [],
        "A": [], "k": [], "power_density_file": [],
    }
    sect = {f"{q}_{i}": [] for i in range(n_sec) for q in ("freq", "A", "k")}

    for r in rows:
        if int(r[8]) != n_sec:
            raise ValueError(
                f"{path}: inconsistent sector count ({r[8]} vs {n_sec}) -- "
                "the file mixes record layouts"
            )
        base["x"].append(float(r[1]))
        base["y"].append(float(r[2]))
        base["elevation"].append(float(r[3]))
        base["hub_height"].append(float(r[4]))
        base["A"].append(float(r[5]))
        base["k"].append(float(r[6]))
        base["power_density_file"].append(float(r[7]))
        for i in range(n_sec):
            # Stored scaled: freq*1000, A*10, k*100.
            sect[f"freq_{i}"].append(float(r[9 + 3 * i]) / 1000.0)
            sect[f"A_{i}"].append(float(r[10 + 3 * i]) / 10.0)
            sect[f"k_{i}"].append(float(r[11 + 3 * i]) / 100.0)

    df = pd.DataFrame({**base, **sect})
    df["n_sectors"] = n_sec
    df["mean_speed"] = weibull_mean_speed(df["A"].values, df["k"].values)
    df["power_density"] = weibull_power_density(
        df["A"].values, df["k"].values, air_density
    )
    df.attrs.update(
        path=str(path), nx=int(nx), ny=int(ny),
        xmin=xmin, ymin=ymin, cell_size=cell, n_sectors=n_sec,
        air_density=air_density,
    )
    return df


def to_grid(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot a column onto its (y, x) grid. Returns (x, y, values) north-up."""
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())[::-1]           # north-up
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    grid = np.full((len(ys), len(xs)), np.nan)
    grid[df["y"].map(yi).values, df["x"].map(xi).values] = df[column].values
    return xs, ys, grid
