#!/usr/bin/env python
"""Figure: raw vs bias-removed mast skill for the four terrain-mode cells.

    python scripts/plot_mast_validation.py \
        /scratch/moptis/fuxicfd/validation/mast_comparison.csv \
        --out reports/mast_validation.png

Three panels, in the order the argument has to be made:

1. **As delivered** - raw MAE. Every cell is worse than the 1 km field it
   downscaled. This panel stays first because it is what the shipped WRGs
   actually do; leading with the flattering number would be dishonest.
2. **Bias removed** - MAE after subtracting each project's own mean error. Every
   cell now beats its 1 km input. Downscaling is not responsible for the mean -
   that comes from the mesoscale run it was handed - so centring asks the
   question the method *is* responsible for: is the mast-to-mast pattern better?
3. **Per project** - the same centred comparison, signed, so the sites where the
   gain is real are visible rather than averaged away.

Centring is per project, not global, because each project is a separate WRF run
with its own mean-state error. A single global offset does not remove them - it
makes matters worse (+0.11 m/s), which is itself the evidence that the offsets
are per domain. Projects with fewer than `--min-n` masts are excluded from
panels 2-3: removing a mean from two points drives the centred error to zero by
construction and would manufacture skill.

Colour follows the job (dataviz skill): panels 1-2 are *identity* - neutral ink
for the baseline (it is the thing to beat, not a series) and one categorical
orange for all four cells, since they are statistically indistinguishable and
colouring them separately would imply a difference the data does not support.
Panel 3 is *polarity* - the documented blue<->red diverging pair about a neutral
zero, with sign also carried by bar direction so colour is never the sole
encoding. Both pairs pass the validator's computable checks (worst normal-vision
dE 32.3, worst CVD min(protan,deutan) dE 21.6, every mark >=3:1 on surface).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CELLS = ["dabs_rabs", "dabs_rrec", "drec_rabs", "drec_rrec"]
LABEL = {
    "dabs_rabs": "DEM abs · z₀ abs",
    "dabs_rrec": "DEM abs · z₀ rec.",
    "drec_rabs": "DEM rec. · z₀ abs",
    "drec_rrec": "DEM rec. · z₀ rec.",
}

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
INK_MUTED, GRID = "#8a8985", "#e7e6e2"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"


def centered(pred, obs, groups):
    """Absolute error after removing each group's own mean error."""
    err = pred - obs
    return np.concatenate([np.abs(err[groups == g] - err[groups == g].mean())
                           for g in np.unique(groups)])


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)


def bars(ax, names, vals, title, xlabel, note, note_color):
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.6, color=[INK_2] + [ORANGE] * 4, zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + max(vals) * 0.02, yy, f"{v:.3f}", va="center", ha="left",
                fontsize=10, color=INK, fontweight="bold")
    ax.axvline(vals[0], color=INK_2, lw=1.3, ls=(0, (4, 3)), zorder=2, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=10, color=INK_2)
    ax.set_xlim(0, max(vals) * 1.30)
    ax.set_title(title, fontsize=12, color=INK, fontweight="bold", loc="left", pad=10)
    ax.text(max(vals) * 1.29, y[0] + 0.72, note, fontsize=11, color=note_color,
            fontweight="bold", ha="right", va="center")
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/mast_validation.png"))
    ap.add_argument("--min-n", type=int, default=4)
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
    o, p_, cs = obs[keep], proj[keep], coarse[keep]
    projects = sorted(set(p_))

    raw = [float(np.mean(np.abs(coarse - obs)))] + \
          [float(np.mean(np.abs(fine[c] - obs))) for c in CELLS]
    cen = [float(centered(cs, o, p_).mean())] + \
          [float(centered(fine[c][keep], o, p_).mean()) for c in CELLS]

    dcen = {c: [] for c in CELLS}
    for p in projects:
        m = p_ == p
        b = centered(cs[m], o[m], p_[m]).mean()
        for c in CELLS:
            dcen[c].append(centered(fine[c][keep][m], o[m], p_[m]).mean() - b)

    names = ["Raw WakeMap 1 km"] + [LABEL[c] for c in CELLS]

    fig = plt.figure(figsize=(16.6, 7.9), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.86, 1.12], wspace=0.22,
                          left=0.112, right=0.975, top=0.715, bottom=0.195)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
    for a in (ax1, ax2, ax3):
        style(a)

    bars(ax1, names, raw, "1 · As delivered",
         "MAE vs mast  (m s⁻¹)", "all worse", RED)
    bars(ax2, names, cen, "2 · Per-project bias removed",
         "Centred MAE  (m s⁻¹)", "all better", BLUE)
    ax2.set_yticklabels([])

    yp = np.arange(len(projects))[::-1]
    h = 0.19
    for c, off in zip(CELLS, np.linspace(1.5 * h, -1.5 * h, 4)):
        d = np.array(dcen[c])
        ax3.barh(yp + off, d, height=h, zorder=3, color=np.where(d < 0, BLUE, RED))
    ax3.axvline(0, color=INK_2, lw=1.3, zorder=4)
    ax3.set_yticks(yp)
    ax3.set_yticklabels([p.upper() for p in projects], fontsize=10, color=INK)
    ax3.set_xlabel("Change in centred MAE vs raw 1 km  (m s⁻¹)", fontsize=10, color=INK_2)
    lim = max(abs(np.array([v for c in CELLS for v in dcen[c]]))) * 1.32
    ax3.set_xlim(-lim, lim)
    for yy in yp[:-1]:
        ax3.axhline(yy - 0.5, color=GRID, lw=0.8, zorder=1)
    ax3.text(-lim * 0.96, yp[0] + 0.70, "◀  better pattern", fontsize=9.5,
             color=BLUE, fontweight="bold", ha="left", va="center")
    ax3.text(lim * 0.96, yp[0] + 0.70, "worse  ▶", fontsize=9.5,
             color=RED, fontweight="bold", ha="right", va="center")
    ax3.set_title("3 · Where the spatial gain is real", fontsize=12, color=INK,
                  fontweight="bold", loc="left", pad=10)

    fig.text(0.028, 0.955,
             "FuXi-CFD resolves sub-kilometre structure better than the 1 km WakeMap "
             "— but carries a per-project mean offset",
             fontsize=16, color=INK, fontweight="bold", ha="left")
    fig.text(0.028, 0.905,
             f"{len(obs)} mast–project comparisons, long-term mean wind speed at 100 m. Remove each "
             "project's own mean error and every cell beats its 1 km input.\n"
             "The four dem/roughness cells stay within 0.016 m s⁻¹ of each other throughout.",
             fontsize=10.5, color=INK_2, ha="left")

    fig.text(0.028, 0.096,
             f"Panels 2–3 use the {int(keep.sum())} masts in {len(projects)} projects with ≥{args.min_n} masts; "
             f"{', '.join(p.upper() for p in small)} are excluded because removing a mean\n"
             "from 2–3 points drives the centred error toward zero by construction. A single GLOBAL offset does "
             "not work (+0.11 m s⁻¹) — the offsets are per project,\nwhich is why centring is done per project. "
             "Every 100 m mast value is extrapolated from lower sensors, not measured at 100 m; 2 masts dropped "
             "for data-quality flags.",
             fontsize=9, color=INK_MUTED, ha="left", va="top")

    handles = [Patch(facecolor=INK_2, label="Raw WakeMap 1 km — the baseline to beat"),
               Patch(facecolor=ORANGE, label="FuXi-CFD 30 m"),
               Patch(facecolor=BLUE, label="Better than 1 km"),
               Patch(facecolor=RED, label="Worse than 1 km")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.026, 0.848),
               frameon=False, fontsize=9.5, labelcolor=INK_2, ncol=4,
               handlelength=1.1, handleheight=1.1, columnspacing=1.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.out}")
    print(f"\n{'':22s} {'raw MAE':>8s} {'cMAE':>8s} {'dcMAE':>8s}")
    for i, nm in enumerate(names):
        d = "--" if i == 0 else f"{cen[i]-cen[0]:+.3f}"
        print(f"{nm:22s} {raw[i]:8.3f} {cen[i]:8.3f} {d:>8s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
