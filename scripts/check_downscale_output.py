#!/usr/bin/env python
"""Post-check a downscaled .wrg. Exits non-zero on anything that failed quietly.

    python scripts/check_downscale_output.py /scratch/.../wrg_sweep/*.wrg

Every check here corresponds to a failure mode that has already happened once and
produced a *complete, well-formed, wrong* file. A downscale run writes its .wrg
whatever the inputs did, so "the job exited 0" is not evidence of a good product
and across a 60-run sweep nobody reads 60 logs.

What it checks and why:

* **Elevation column is not zero.** `_elevation_on_grid` skips any tile whose raw
  terrain cache it cannot find, and the cache filename has been got wrong twice
  -- once on the `_{source}` suffix, once on the `-{geo_key}` infix. Both times
  every run silently wrote a zeroed elevation column. The code now prints a
  warning; a warning is not enough at this scale.
* **Coarse-WRG coverage.** Points outside the source WRG are written as zeros,
  which looks like a legitimate calm cell. A wrong `source.crs` lands the whole
  domain somewhere else and shows up here as near-total non-coverage, which is
  the one error mode with no other symptom at all.
* **A, k, power density in physical range.** Catches a broken moment fit or a
  Weibull clipped to `K_BOUNDS`.
* **Sector frequencies sum to 1000.** The deflection step redistributes frequency
  between sectors; if it ever loses or duplicates weight, this is where it shows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

#: A, k and power-density bounds wide enough to pass any real site and narrow
#: enough to catch a broken fit. k is bounded by transfer.K_BOUNDS = (0.2, 20).
A_RANGE = (0.5, 25.0)
K_RANGE = (0.5, 10.0)
PD_MAX = 5000.0


def check(path: Path) -> list[str]:
    fails = []
    with open(path) as fh:
        header = fh.readline().split()
        rows = [ln.split() for ln in fh if ln.strip()]
    if not rows:
        return [f"{path.name}: no grid points"]

    nx, ny = int(header[0]), int(header[1])
    if len(rows) != nx * ny:
        fails.append(f"{len(rows)} rows for a {ny}x{nx} header ({nx*ny} expected)")

    # GridPoint X Y elev hub A k power_density n_sectors [freq*1000 A*10 k*100]*n
    a = np.array([[float(v) for v in r[1:]] for r in rows])
    elev, _hub, A, k, pd, nsec = (a[:, i] for i in range(2, 8))
    n = len(a)
    n_sec = int(nsec[0])
    sect = a[:, 8:8 + 3 * n_sec].reshape(n, n_sec, 3)
    freq = sect[:, :, 0]

    live = (A > 0) & np.isfinite(A)

    # A missed terrain cache leaves the elevation column flat -- zeros with no
    # relief anywhere. A coastal site legitimately has many points at exactly
    # 0 m: gwr runs -22..34 m with 29% of live points at sea level, and an
    # earlier version of this check failed it on that alone, twice. So test for
    # the degenerate case (no relief at all) rather than for the count of zeros,
    # which cannot distinguish a cache miss from a coastline.
    live_elev = elev[live] if live.any() else elev
    relief = float(live_elev.max() - live_elev.min()) if live_elev.size else 0.0
    zero_elev = float(np.mean(live_elev == 0.0)) if live_elev.size else 0.0
    if relief < 1.0 or zero_elev > 0.90:
        fails.append(
            f"elevation column is degenerate ({zero_elev:.1%} exactly 0, "
            f"{relief:.1f} m of relief) -- the terrain cache was very likely "
            "missed (see _elevation_on_grid)")

    dead = np.mean(~live)
    if dead > 0.01:
        fails.append(
            f"{dead:.1%} of points have no wind climate -- either the tile "
            "lattice does not fill its bounding rectangle (a corner clipped by "
            "the rotated domain boundary; see the [tiles] NOTE in the run log) "
            "or the grid falls outside the coarse WRG, so check source.crs")

    for name, v, (lo, hi) in (("A", A, A_RANGE), ("k", k, K_RANGE)):
        vals = v[live & np.isfinite(v)]
        if vals.size and (vals.min() < lo or vals.max() > hi):
            fails.append(f"{name} out of range: {vals.min():.2f}..{vals.max():.2f} "
                         f"(expected {lo}..{hi})")

    # Percentile, not max. Complex terrain produces a genuine extreme tail: mkw
    # has 170 ridge-top points of 2.47M above 5000 W/m2 at ~1970 m, and the same
    # points appear whatever the roughness mode. Testing the maximum failed the
    # file on 0.007% of its points; testing p99.9 still catches a field that is
    # broadly wrong while tolerating a summit.
    live_pd = pd[live & np.isfinite(pd)]
    if live_pd.size:
        p999 = float(np.percentile(live_pd, 99.9))
        if p999 > PD_MAX:
            fails.append(f"power density p99.9 = {p999:.0f} W/m2 "
                         f"(expected < {PD_MAX:.0f}; max {live_pd.max():.0f})")

    # Frequencies are stored as round(freq*1000) per sector, so each carries up
    # to 0.5 of rounding error and the sum can miss 1000 by n_sec/2 -- +/-8 at 16
    # sectors. An earlier version of this check used a flat +/-5 and failed every
    # 16-sector file at 994..1006, which is pure quantisation, not lost weight.
    # Allow the format's own error plus one, so a real leak still trips it.
    tol = n_sec / 2.0 + 1.0
    fsum = freq[live].sum(axis=1)
    if fsum.size and np.abs(fsum - 1000.0).max() > tol:
        fails.append(f"sector frequencies sum to {fsum.min():.0f}..{fsum.max():.0f}, "
                     f"expected 1000 +/- {tol:.0f} (deflection may be losing weight)")

    if not fails:
        print(f"  OK  {path.name:34s} {ny}x{nx}  {n_sec} sec  "
              f"elev {elev.min():.0f}-{elev.max():.0f} m  "
              f"A {A[live].mean():.2f}  k {k[live].mean():.2f}  "
              f"PD {pd[live].mean():.0f} W/m2")
    return [f"{path.name}: {f}" for f in fails]


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv]
    if not paths:
        raise SystemExit(__doc__)
    all_fails = []
    for p in paths:
        if not p.exists():
            all_fails.append(f"{p}: missing")
            continue
        all_fails += check(p)
    if all_fails:
        print(f"\nFAILED ({len(all_fails)} problems):", file=sys.stderr)
        for f in all_fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"\nall {len(paths)} file(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
