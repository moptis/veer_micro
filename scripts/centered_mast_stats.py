#!/usr/bin/env python
"""Bias-removed (centered) mast statistics: does downscaling add spatial skill?

    python scripts/centered_mast_stats.py /scratch/moptis/fuxicfd/validation/mast_comparison.csv

Raw MAE conflates two different failures:

* a **mean offset** -- the whole field is a bit fast or slow. Downscaling is not
  supposed to fix this; the mean comes from the mesoscale run it was handed.
* a **spatial error** -- the field ranks and spaces the masts wrongly relative to
  each other. This is precisely what a terrain-aware downscaler is for.

Centering removes the first so the second can be seen. Two centerings, because
they answer different questions:

* **global** -- subtract one offset over all masts. Tests whether the method has
  a single systematic bias.
* **per project** -- subtract each project's own offset. Each project is a
  separate WRF run with its own mean-state error, so this isolates *within-domain*
  spatial skill, which is the honest test of the downscaling itself.

Degrees of freedom are reported and enforced, not hidden: removing a per-project
mean from n masts eats one degree of freedom, so at n=2 the centered error is
driven almost to zero by construction and means nothing. Projects below
`--min-n` are excluded from the per-project summary for that reason, and the
excluded set is printed rather than silently dropped.

Also reported per project: the correlation between modelled and observed
across masts, and the ratio of modelled to observed spatial spread. A method
that adds real structure should raise the correlation; one that adds noise
raises the spread without raising the correlation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

CELLS = ["dabs_rabs", "dabs_rrec", "drec_rabs", "drec_rrec"]
LABEL = {"dabs_rabs": "DEM abs · z0 abs", "dabs_rrec": "DEM abs · z0 rec",
         "drec_rabs": "DEM rec · z0 abs", "drec_rrec": "DEM rec · z0 rec"}


def centered_mae(pred, obs, groups=None):
    """MAE after removing the mean error, globally or per group."""
    err = pred - obs
    if groups is None:
        return float(np.mean(np.abs(err - err.mean())))
    out = []
    for g in np.unique(groups):
        m = groups == g
        out.append(np.abs(err[m] - err[m].mean()))
    return float(np.mean(np.concatenate(out)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--min-n", type=int, default=4,
                    help="min masts per project for the per-project centered summary")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    obs = np.array([float(r["obs"]) for r in rows])
    coarse = np.array([float(r["coarse"]) if r["coarse"] else np.nan for r in rows])
    fine = {c: np.array([float(r[c]) if r[c] else np.nan for r in rows]) for c in CELLS}
    proj = np.array([r["project"] for r in rows])

    ok = np.isfinite(coarse) & np.all([np.isfinite(fine[c]) for c in CELLS], axis=0)
    obs, coarse, proj = obs[ok], coarse[ok], proj[ok]
    fine = {c: v[ok] for c, v in fine.items()}

    counts = {p: int((proj == p).sum()) for p in np.unique(proj)}
    keep = np.array([counts[p] >= args.min_n for p in proj])
    small = sorted(p for p, n in counts.items() if n < args.min_n)

    print(f"{len(obs)} comparisons across {len(counts)} projects\n")

    # --- 1. global ----------------------------------------------------------
    print("RAW and GLOBALLY-CENTERED (one offset removed over all masts)")
    print(f"{'':22s} {'bias':>7s} {'MAE':>7s} {'cMAE':>7s} {'dcMAE':>8s}")
    base_c = centered_mae(coarse, obs)
    print(f"{'Raw WakeMap 1 km':22s} {np.mean(coarse-obs):+7.3f} "
          f"{np.mean(np.abs(coarse-obs)):7.3f} {base_c:7.3f} {'--':>8s}")
    for c in CELLS:
        v = fine[c]; cm = centered_mae(v, obs)
        print(f"{LABEL[c]:22s} {np.mean(v-obs):+7.3f} {np.mean(np.abs(v-obs)):7.3f} "
              f"{cm:7.3f} {cm-base_c:+8.3f}")

    # --- 2. per project -----------------------------------------------------
    print(f"\nPER-PROJECT-CENTERED (each project's own offset removed)")
    print(f"  projects with n >= {args.min_n} only; excluded (n too small to center): "
          f"{', '.join(f'{p}(n={counts[p]})' for p in small) or 'none'}")
    o, p_, cs = obs[keep], proj[keep], coarse[keep]
    print(f"  {int(keep.sum())} comparisons across {len(set(p_))} projects\n")
    print(f"{'':22s} {'cMAE':>7s} {'dcMAE':>8s}   {'verdict':<28s}")
    base_p = centered_mae(cs, o, p_)
    print(f"{'Raw WakeMap 1 km':22s} {base_p:7.3f} {'--':>8s}   {'baseline':<28s}")
    for c in CELLS:
        v = fine[c][keep]; cm = centered_mae(v, o, p_)
        d = cm - base_p
        verdict = "adds spatial skill" if d < -0.005 else (
            "no change" if abs(d) <= 0.005 else "degrades spatial pattern")
        print(f"{LABEL[c]:22s} {cm:7.3f} {d:+8.3f}   {verdict:<28s}")

    # --- 3. per-project correlation and spread ------------------------------
    print(f"\nPER-PROJECT SPATIAL AGREEMENT (n >= {args.min_n})")
    print(f"{'proj':6s} {'n':>3s}  {'corr: 1km':>9s} {'best cell':>10s}   "
          f"{'spread ratio (model/obs)':>26s}")
    for p in sorted(set(p_)):
        m = p_ == p
        if m.sum() < 3:
            continue
        so = o[m].std()
        rc = np.corrcoef(cs[m], o[m])[0, 1] if so > 0 else np.nan
        best_r, best_c = -9, None
        for c in CELLS:
            r = np.corrcoef(fine[c][keep][m], o[m])[0, 1]
            if np.isfinite(r) and r > best_r:
                best_r, best_c = r, c
        ratio_c = cs[m].std() / so if so > 0 else np.nan
        ratio_f = fine[best_c][keep][m].std() / so if so > 0 else np.nan
        print(f"{p.upper():6s} {int(m.sum()):3d}  {rc:9.2f} {best_r:10.2f}   "
              f"1km {ratio_c:5.2f}   fine {ratio_f:5.2f}")

    print("\nReading this: dcMAE < 0 means the downscaling resolves the mast-to-mast\n"
          "pattern better than the 1 km field, independent of any mean offset.\n"
          "A spread ratio far above 1 with no correlation gain is added noise,\n"
          "not added structure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
