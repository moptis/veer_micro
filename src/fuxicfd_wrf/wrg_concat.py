"""Concatenate per-tile WRG files into one domain WRG.

    fuxicfd-wrf wrg-concat <dir-of-wrgs> --out domain.wrg

Exporting one WRG per tile makes the job embarrassingly parallel, and it is also
what makes the model's native resolution affordable. A 6x6 km trimmed core at
30 m is 200x200 points, so its histogram is 256 MB; the same 30 m grid spanning
the whole domain would be 12M points and 77 GB, which is why the first
(single-job) version of this export had to coarsen to 100 m. Per tile it does
not, and each task reads only its own tile -- a ~9 hour sequential job becomes a
Slurm array.

The catch is overlap. Tiles are laid out at 6 km stride with a 9 km footprint, so
neighbours share 3 km. Exported independently, those points appear twice with
slightly different values, which a rectangular WRG cannot represent.

Two things resolve it:

* Export each tile with ``--trim-edge 1500``. At 6 km stride that leaves a
  6x6 km core, which tessellates exactly -- and it discards the outer margin we
  measured at 1.9x the residual noise, since FuXi-CFD gets no buffer.
* Whatever duplicates remain (grid rounding at the joins) are resolved here by
  **keeping the sample from the tile whose centre is nearest**, i.e. the most
  interior estimate.

The result is checked for completeness before writing: a WRG with holes is worse
than no WRG, because nothing downstream will tell you they are there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .wrg_read import read_wrg


def _tile_centre(df: pd.DataFrame) -> tuple[float, float]:
    return float(df["x"].mean()), float(df["y"].mean())


def concat_wrgs(
    paths: list[Path],
    out_path: Path,
    *,
    air_density: float = 1.225,
    allow_holes: bool = False,
) -> dict:
    """Merge per-tile WRGs into one, resolving overlaps by interiority."""
    if not paths:
        raise ValueError("no WRG files given")

    frames, cell, n_sec = [], None, None
    for p in sorted(paths):
        df = read_wrg(p, air_density=air_density)
        if cell is None:
            cell, n_sec = df.attrs["cell_size"], df.attrs["n_sectors"]
        if df.attrs["cell_size"] != cell:
            raise ValueError(
                f"{p.name} has cell size {df.attrs['cell_size']}, expected {cell}. "
                "All tiles must share one grid resolution."
            )
        if df.attrs["n_sectors"] != n_sec:
            raise ValueError(
                f"{p.name} has {df.attrs['n_sectors']} sectors, expected {n_sec}."
            )
        cx, cy = _tile_centre(df)
        # Chebyshev distance from the tile centre: larger means closer to that
        # tile's edge, so it loses to a neighbour that holds the point interior.
        df["_edge_dist"] = np.maximum((df["x"] - cx).abs(), (df["y"] - cy).abs())
        df["_src"] = p.name
        frames.append(df)

    allpts = pd.concat(frames, ignore_index=True)
    before = len(allpts)
    allpts = (allpts.sort_values("_edge_dist")
                    .drop_duplicates(subset=["x", "y"], keep="first")
                    .sort_values(["x", "y"], kind="stable")
                    .reset_index(drop=True))
    dupes = before - len(allpts)

    xs = np.sort(allpts["x"].unique())
    ys = np.sort(allpts["y"].unique())
    expected = len(xs) * len(ys)
    holes = expected - len(allpts)
    gaps_x = np.unique(np.round(np.diff(xs), 6))
    gaps_y = np.unique(np.round(np.diff(ys), 6))

    print(f"[concat] {len(paths)} tiles -> {len(allpts):,} points "
          f"({dupes:,} overlapping duplicates resolved)")
    print(f"[concat] grid {len(xs)} x {len(ys)} @ {cell:.0f} m, {n_sec} sectors")
    if len(gaps_x) > 1 or len(gaps_y) > 1:
        print(f"[concat] WARNING: irregular spacing -- x gaps {gaps_x}, y gaps {gaps_y}")
    if holes:
        msg = (f"{holes:,} of {expected:,} grid points are missing "
               f"({100*holes/expected:.2f}%). The tiles do not cover a full "
               "rectangle -- check the tiling stride and trim.")
        if not allow_holes:
            raise ValueError(msg + "\nPass allow_holes=True to write anyway.")
        print(f"[concat] WARNING: {msg}")

    n_cols = 6 + 3 * n_sec
    rec = np.zeros((len(allpts), n_cols))
    rec[:, 0] = allpts["elevation"].values
    rec[:, 1] = allpts["hub_height"].values
    rec[:, 2] = allpts["A"].values
    rec[:, 3] = allpts["k"].values
    rec[:, 4] = allpts["power_density_file"].values
    rec[:, 5] = n_sec
    for i in range(n_sec):
        rec[:, 6 + 3 * i + 0] = allpts[f"freq_{i}"].values * 1000.0
        rec[:, 6 + 3 * i + 1] = allpts[f"A_{i}"].values * 10.0
        rec[:, 6 + 3 * i + 2] = allpts[f"k_{i}"].values * 100.0

    from .wrg import write_wrg

    out = write_wrg(out_path, allpts["x"].values, allpts["y"].values, rec,
                    resolution_m=cell)
    print(f"[concat] wrote {out}")
    return {
        "path": str(out),
        "n_tiles": len(paths),
        "n_points": len(allpts),
        "grid_shape": [len(ys), len(xs)],
        "resolution_m": cell,
        "n_sectors": n_sec,
        "duplicates_resolved": dupes,
        "missing_points": holes,
    }
