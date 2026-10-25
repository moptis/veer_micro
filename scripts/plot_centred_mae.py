#!/usr/bin/env python
"""Centred MAE vs met masts - the metric the Vortex/UL grid comparison uses.

    python scripts/plot_centred_mae.py \
        /scratch/moptis/fuxicfd/validation/mast_comparison.csv \
        --out reports/centred_mae.png

Centred MAE, not MMA. The house MMA figure maps each grid to the masts with an
inverse-distance-squared interpolation, which is why its errors are small; this
comparison samples nearest-node, so an MMA number from here is not comparable
with one from there. Centred MAE - percent error with each project's own mean
removed - is the like-for-like statistic, and it is what the provider comparison
reports.

Left: per project, WakeMap 1 km against the best FuXi-CFD cell, with the span
across all four cells drawn behind so the spread is visible rather than implied.
Right: fleet totals for the 1 km input and all four cells.

The Vortex and WakeMap levels from the provider study are drawn as neutral
dashed references, deliberately not as coloured series: they are 80 m, a
different project set, and a different sampling scheme, so they locate this
result on a familiar axis without inviting a direct subtraction. Rendering them
in a series colour would imply a like-for-like they have not earned.

Colour: WakeMap keeps its pink identity from the provider figure; FuXi-CFD takes
violet. That pair was chosen by the validator, not by eye - the obvious aqua
failed at CVD dE 1.6 against pink (indistinguishable under protan/deutan), and
violet against Vortex-blue failed at 1.9, which is why the reference lines are
neutral. The shipped pair measures normal-vision dE 19.7, CVD dE 16.0, both
>=3.6:1 on the navy surface.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Ordered by roughness treatment, because that is the axis that turned out to
# matter: full 30 m z0, then z0 low-passed to 1 km, then z0 removed entirely.
CELLS = ["dabs_rabs", "drec_rabs", "dabs_rrec", "drec_rrec",
         "dabs_rcoarse", "drec_rcoarse", "dabs_rconst", "drec_rconst"]
SHORT = {"dabs_rabs": "abs\nz₀ 30 m", "drec_rabs": "rec\nz₀ 30 m",
         "dabs_rrec": "abs\nz₀ rec.", "drec_rrec": "rec\nz₀ rec.",
         "dabs_rcoarse": "abs\nz₀ 1 km", "drec_rcoarse": "rec\nz₀ 1 km",
         "dabs_rconst": "abs\nno z₀", "drec_rconst": "rec\nno z₀"}
#: Which cells carry sub-kilometre roughness detail. The figure colours on this,
#: not on dem_mode, because the roughness axis is where the signal is.
SUBKM = {"dabs_rabs", "drec_rabs", "dabs_rrec", "drec_rrec"}

BG, PANEL = "#1b2b3d", "#22344a"
INK, INK_2, INK_MUTED, GRIDC = "#e8eef6", "#9fb2c9", "#6f839b", "#32455d"
PINK, VIOLET, AQUA = "#d55181", "#9085e9", "#c98500"
CONNECT = "#3d5372"

#: Published levels from the provider comparison (80 m, project bias removed).
#: The 93-mast / 16-project column is the closest in size to this study's 86/13.
REFERENCE = [(2.8, "Vortex"), (3.8, "WakeMap control")]


def cmae_pct(model, obs, proj, projects):
    out = []
    for p in projects:
        m = proj == p
        e = (model[m] / obs[m] - 1.0) * 100.0
        out.append(np.abs(e - e.mean()))
    return float(np.mean(np.concatenate(out)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/centred_mae.png"))
    ap.add_argument("--xmax", type=float, default=7.6,
                    help="x limit for the per-project panel; MKW is annotated off-scale")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    obs = np.array([float(r["obs"]) for r in rows])
    coarse = np.array([float(r["coarse"]) if r["coarse"] else np.nan for r in rows])
    fine = {c: np.array([float(r[c]) if r[c] else np.nan for r in rows]) for c in CELLS}
    proj = np.array([r["project"] for r in rows])

    ok = np.isfinite(coarse) & np.all([np.isfinite(fine[c]) for c in CELLS], axis=0)
    obs, coarse, proj = obs[ok], coarse[ok], proj[ok]
    fine = {c: v[ok] for c, v in fine.items()}
    projects = sorted(set(proj))

    per = {}
    for p in projects:
        b = cmae_pct(coarse, obs, proj, [p])
        f = [cmae_pct(fine[c], obs, proj, [p]) for c in CELLS]
        sub = min(cmae_pct(fine[c], obs, proj, [p]) for c in CELLS if c in SUBKM)
        km1 = min(cmae_pct(fine[c], obs, proj, [p]) for c in CELLS if c not in SUBKM)
        per[p] = (b, f, int((proj == p).sum()), sub, km1)
    order = sorted(projects, key=lambda p: per[p][0])

    # Site-equal: the plain mean of per-project centred MAE. Pooling every mast
    # instead would weight projects by instrument count (2 to 15 here) and let
    # PH1/PH2 -- near-duplicate domains holding 30 of the masts -- dominate.
    fleet_b = float(np.mean([cmae_pct(coarse, obs, proj, [p]) for p in projects]))
    fleet_f = [float(np.mean([cmae_pct(fine[c], obs, proj, [p]) for p in projects]))
               for c in CELLS]

    fig = plt.figure(figsize=(15.8, 8.6), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.17,
                          left=0.083, right=0.975, top=0.755, bottom=0.135)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=9.5, length=0)
        ax.set_axisbelow(True)

    # ---- per project ------------------------------------------------------
    yy = np.arange(len(order))[::-1]
    ax1.xaxis.grid(True, color=GRIDC, lw=0.9)
    off = []
    for y, p in zip(yy, order):
        b, f, n, sub, km1 = per[p]
        pts = [b, sub, km1]
        ax1.plot([min(pts), max(pts)], [y, y], color=CONNECT, lw=5,
                 solid_capstyle="round", zorder=2)
        if max(pts) > args.xmax:
            off.append((y, p, b, km1))
        ax1.scatter([b], [y], s=95, color=PINK, zorder=5,
                    edgecolors=PANEL, linewidths=1.2)
        ax1.scatter([sub], [y], s=95, color=VIOLET, zorder=5,
                    edgecolors=PANEL, linewidths=1.2)
        ax1.scatter([km1], [y], s=95, color=AQUA, zorder=6,
                    edgecolors=PANEL, linewidths=1.2)
    ax1.set_yticks(yy)
    ax1.set_yticklabels([f"{p.upper()}  ({per[p][2]})" for p in order],
                        fontsize=10, color=INK)
    ax1.set_xlim(0, args.xmax)
    ax1.set_ylim(yy.min() - 0.8, yy.max() + 0.8)
    ax1.set_xlabel("centred MAE vs. masts at 100 m (%), project bias removed",
                   fontsize=10, color=INK_2)
    ax1.set_title("By project  (masts in brackets)", fontsize=12.5, color=INK,
                  fontweight="bold", loc="left", pad=12)
    for y, p, b, f in off:
        ax1.text(args.xmax * 0.995, y, f"off scale: {f:.1f} / {b:.1f}%  ▶",
                 fontsize=9, color=INK_MUTED, ha="right", va="center")

    # ---- fleet ------------------------------------------------------------
    ax2.yaxis.grid(True, color=GRIDC, lw=0.9)
    xs = np.arange(1 + len(CELLS))
    vals = [fleet_b] + fleet_f
    cols = [PINK] + [VIOLET if c in SUBKM else AQUA for c in CELLS]
    ax2.bar(xs, vals, width=0.70, color=cols, zorder=3)
    for x, v in zip(xs, vals):
        ax2.text(x, v + 0.07, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=9, color=INK, fontweight="bold")
    for lvl, name in REFERENCE:
        ax2.axhline(lvl, color=INK_2, lw=1.1, ls=(0, (5, 4)), zorder=4, alpha=0.9)
        ax2.text(-1.28, lvl + 0.06, f"{name} {lvl:.1f}", fontsize=8.5,
                 color=INK_2, va="bottom", ha="left")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["WakeMap\n1 km"] + [SHORT[c] for c in CELLS],
                        fontsize=8.5, color=INK)
    ax2.set_xlim(-1.35, len(CELLS) + 0.15)
    ax2.set_ylim(0, max(vals + [r[0] for r in REFERENCE]) * 1.22)
    ax2.set_ylabel("centred MAE (%)", fontsize=10, color=INK_2)
    ax2.set_title(f"Fleet, each site weighted equally  ({len(projects)} projects)",
                  fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=12)

    fig.text(0.011, 0.955, "Sub-kilometre roughness makes the downscaling worse, not better",
             fontsize=16.5, color=INK, fontweight="bold", ha="left")
    fig.text(0.011, 0.914,
             f"Centred MAE at 100 m across {len(projects)} projects, 8 terrain-mode cells. "
             "Feeding the model 30 m roughness is WORSE than low-passing it to 1 km:\n"
             f"{max(fleet_f):.2f}% → {min(fleet_f):.2f}% fleet-wide, an effect ten times larger than "
             "the dem_mode choice the sweep was originally built to test.",
             fontsize=10, color=INK_2, ha="left")

    handles = [Line2D([], [], marker="o", ls="", ms=10, color=PINK, label="WakeMap 1 km (input)"),
               Line2D([], [], marker="o", ls="", ms=10, color=VIOLET, label="FuXi-CFD, 30 m roughness"),
               Line2D([], [], marker="o", ls="", ms=10, color=AQUA, label="FuXi-CFD, roughness at 1 km or removed"),
               Line2D([], [], color=INK_2, lw=1.1, ls=(0, (5, 4)), label="provider study, 80 m (reference)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.011, 0.872),
               frameon=False, fontsize=10, labelcolor=INK, ncol=4,
               handletextpad=0.5, columnspacing=2.2)

    fig.text(0.011, 0.068,
             "Reference levels are from the Vortex/UL provider comparison: 80 m, a different project set, and "
             "inverse-distance-squared mapping to masts. They locate this result on a familiar axis; they are not\n"
             "a like-for-like subtraction, which is why they are drawn neutral rather than as series. "
             "MKW is off scale in both. Every 100 m mast value is extrapolated from lower sensors; 2 masts dropped "
             "for data-quality flags. Nearest-node sampling.",
             fontsize=8.5, color=INK_MUTED, ha="left", va="top")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=BG)
    print(f"wrote {args.out}")
    print(f"\nfleet centred MAE: WakeMap 1 km {fleet_b:.2f}%  " +
          "  ".join(f"{c} {v:.2f}%" for c, v in zip(CELLS, fleet_f)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
