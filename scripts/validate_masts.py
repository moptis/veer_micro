#!/usr/bin/env python
"""Validate downscaled WRGs against met-mast long-term mean wind speeds.

    python scripts/validate_masts.py /scratch/moptis/fuxicfd/validation/masts_100m.csv

The first check in this project that uses anything other than the model's own
inputs. Everything else -- upstream parity, the timestep-vs-sector comparison,
the post-check -- tests internal consistency. This tests whether the answer is
right.

What it reports, and why the second column is the one that matters
------------------------------------------------------------------
For every mast inside a project's terrain box, at the WRG's 100 m hub height:

* **fine**: the downscaled 30 m WRG sampled at the mast, per mode cell.
* **coarse**: the 1 km deliverable that drove it, sampled at the same point.

Bias against the mast is interesting; bias *relative to the coarse input* is the
product. FuXi-CFD is only worth running if it moves the answer toward the
observation. A mode cell that is closer to the masts than the coarse field is
adding value; one that is further away is subtracting it, however small its
absolute error. So the summary leads with the change in |error|, not the error.

Sampling is nearest-node. At 30 m the nearest node is within 21 m of the mast,
which is far inside the model's own resolution, so interpolating would imply a
precision the field does not have. The coarse file is sampled the same way, at
1 km, where the same argument does not hold -- but using one convention for both
keeps the comparison honest, and the coarse field varies slowly by construction.

Caveats this script prints rather than hides
--------------------------------------------
* Nearly every 100 m mast value is **extrapolated** from lower sensors, not
  measured at 100 m. The comparison therefore tests the model against a sheared
  observation, and shear is exactly where FuXi-CFD is known to differ from WRF
  (README: biased low through 20-200 m, high at 250-300 m).
* Masts carrying a data-quality flag are dropped, not down-weighted. Two are
  known bad: one recorded in mph and labelled m/s, one extrapolated from a 57 m
  top sensor at shear 0.57.
* A mast is assigned to a project only if it falls inside that project's terrain
  box. Some masts fall in two overlapping projects (ph1/ph2); both are reported.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pyproj

REPO = Path(__file__).resolve().parent.parent
SWEEP = Path("/scratch/moptis/fuxicfd/out/wrg_sweep")
C2WIND = Path("/scratch/moptis/c2wind")
CELLS = ["dabs_rabs", "dabs_rrec", "drec_rabs", "drec_rrec",
         "dabs_rconst", "drec_rconst", "dabs_rcoarse", "drec_rcoarse"]


def read_wrg_speed(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, mean speed) from a .wrg. Speed is A*Gamma(1+1/k), the Weibull mean."""
    from scipy.special import gamma
    xs, ys, A, k = [], [], [], []
    with open(path) as fh:
        next(fh)
        for ln in fh:
            f = ln.split()
            if len(f) < 8:
                continue
            xs.append(float(f[1])); ys.append(float(f[2]))
            A.append(float(f[5])); k.append(float(f[6]))
    A = np.array(A); k = np.array(k)
    spd = np.where((A > 0) & (k > 0), A * gamma(1.0 + 1.0 / np.where(k > 0, k, 1)), np.nan)
    return np.array(xs), np.array(ys), spd


def sample_nearest(x, y, spd, px, py, max_dist_m):
    """Nearest-node sample, or NaN if the closest node is further than max_dist_m."""
    d2 = (x - px) ** 2 + (y - py) ** 2
    i = int(np.argmin(d2))
    if math.sqrt(d2[i]) > max_dist_m:
        return np.nan, math.sqrt(d2[i])
    return spd[i], math.sqrt(d2[i])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masts", type=Path)
    ap.add_argument("--projects", type=Path, default=REPO / "configs" / "projects.csv")
    ap.add_argument("--out", type=Path, default=None, help="write per-mast CSV here")
    args = ap.parse_args()

    projects = [r for r in csv.DictReader(
        l for l in open(args.projects) if not l.startswith("#"))]
    masts = list(csv.DictReader(open(args.masts)))

    dropped = [m for m in masts if m["flag"]]
    masts = [m for m in masts if not m["flag"]]
    print(f"[validate] {len(masts)} masts, {len(dropped)} dropped for data-quality flags"
          f"{': ' + ', '.join(m['mast'] for m in dropped) if dropped else ''}")
    n_extrap = sum(1 for m in masts if m["extrapolated"].strip().lower() == "true")
    print(f"[validate] {n_extrap}/{len(masts)} of the 100 m values are EXTRAPOLATED "
          "from lower sensors, not measured at 100 m")

    rows = []
    for p in projects:
        code = p["project"].lower()
        fine_paths = {c: SWEEP / f"{code}_{c}_30m.wrg" for c in CELLS}
        if not all(q.exists() for q in fine_paths.values()):
            continue
        coarse = (C2WIND / "pattern" / code / "control" / "deliverables" / "wrg"
                  / "pattern" / code / "control_100m.wrg")
        w, s, e, n = (float(p["west_lon"]), float(p["south_lat"]),
                      float(p["east_lon"]), float(p["north_lat"]))
        inside = [m for m in masts
                  if w <= float(m["longitude"]) <= e and s <= float(m["latitude"]) <= n]
        if not inside:
            continue

        tf = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{p['epsg']}", always_xy=True)
        grids = {c: read_wrg_speed(q) for c, q in fine_paths.items()}
        cg = read_wrg_speed(coarse) if coarse.exists() else None

        for m in inside:
            px, py = tf.transform(float(m["longitude"]), float(m["latitude"]))
            obs = float(m["ws_lt"])
            rec = {"project": code, "mast": m["mast"], "obs": obs}
            for c in CELLS:
                v, _ = sample_nearest(*grids[c], px, py, max_dist_m=100.0)
                rec[c] = v
            if cg is not None:
                v, _ = sample_nearest(*cg, px, py, max_dist_m=1500.0)
                rec["coarse"] = v
            rows.append(rec)

    if not rows:
        print("[validate] no masts fell inside any completed project")
        return 1

    print(f"\n[validate] {len(rows)} mast-project matches\n")
    hdr = f"{'project':7s} {'n':>3s} {'coarse':>17s}  " + "  ".join(
        f"{c:>17s}" for c in CELLS)
    print(hdr)
    print(f"{'':7s} {'':>3s} " + " ".join(f"{'bias   |err|':>18s}" for _ in range(5)))

    by_proj: dict[str, list] = {}
    for r in rows:
        by_proj.setdefault(r["project"], []).append(r)

    for proj in sorted(by_proj):
        rs = by_proj[proj]
        obs = np.array([r["obs"] for r in rs])
        line = f"{proj:7s} {len(rs):3d} "
        for key in ["coarse"] + CELLS:
            v = np.array([r.get(key, np.nan) for r in rs], dtype=float)
            ok = np.isfinite(v)
            if not ok.any():
                line += f"{'--':>18s} "
                continue
            bias = float(np.mean(v[ok] - obs[ok]))
            mae = float(np.mean(np.abs(v[ok] - obs[ok])))
            line += f"{bias:+8.2f}{mae:9.2f} "
        print(line)

    # Overall, and the question that decides whether this is worth doing.
    obs = np.array([r["obs"] for r in rows])
    print()
    base = np.array([r.get("coarse", np.nan) for r in rows], dtype=float)
    ok_b = np.isfinite(base)
    print(f"{'ALL':7s} {len(rows):3d}  coarse 1 km : bias {np.mean(base[ok_b]-obs[ok_b]):+.3f}  "
          f"MAE {np.mean(np.abs(base[ok_b]-obs[ok_b])):.3f}  "
          f"RMSE {np.sqrt(np.mean((base[ok_b]-obs[ok_b])**2)):.3f} m/s")
    for c in CELLS:
        v = np.array([r.get(c, np.nan) for r in rows], dtype=float)
        ok = np.isfinite(v) & ok_b
        d_fine = np.abs(v[ok] - obs[ok])
        d_coarse = np.abs(base[ok] - obs[ok])
        better = float(np.mean(d_fine < d_coarse)) * 100.0
        print(f"{'':7s} {ok.sum():3d}  {c:12s}: bias {np.mean(v[ok]-obs[ok]):+.3f}  "
              f"MAE {d_fine.mean():.3f}  RMSE {np.sqrt(np.mean((v[ok]-obs[ok])**2)):.3f}"
              f"  | dMAE vs coarse {d_fine.mean()-d_coarse.mean():+.3f}"
              f"  closer at {better:.0f}% of masts")

    print("\nThe column that matters is dMAE vs coarse: negative means the downscaling\n"
          "moved the answer toward the observations, positive means away from them.")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["project", "mast", "obs", "coarse"] + CELLS)
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n[validate] per-mast values -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
