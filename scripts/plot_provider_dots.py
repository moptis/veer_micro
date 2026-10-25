#!/usr/bin/env python
"""Per-mast dot plot, in the house Vortex-vs-UL style, for the FuXi-CFD grids.

    python scripts/plot_provider_dots.py \
        /scratch/moptis/fuxicfd/validation/mast_comparison.csv \
        --out reports/provider_dots.png

Mirrors the existing provider-comparison figure: one row per project, a small
dot per mast, a large dot for the project mean, x = grid / mast - 1 in percent.

Two panels rather than three. The reference figure's panel 3 is ops gross
against SCADA; there is no SCADA here, and inventing a third panel to match the
layout would be decoration. Its panels 1 and 2 map exactly:

* **1 Raw grid vs met masts** - as delivered.
* **2 MMA grid vs met masts** - mean-mast-adjusted. That is the same operation
  as the per-project bias removal in `centered_mast_stats.py`: subtract each
  project's own mean error and ask whether the mast-to-mast *pattern* is right.
  Using the house term keeps this comparable with the Vortex/UL work.

Two series, not five. The four dem/roughness cells sit within 0.016 m/s of each
other overall - plotting them as four dot clouds would put four indistinguishable
colours on top of each other and imply a difference the data does not support.
The best cell by centred MAE is shown against the 1 km input it downscaled, and
the spread across all four is printed in the subtitle instead. Pass
`--all-cells` to draw all four anyway.

Percent rather than m/s, because the projects run from 4 to 10 m/s and an
absolute error is not comparable across them - the reference figure uses percent
for the same reason.

Colour: dark-mode categorical slots 1-2 validated against this navy surface
(contrast 3.95 / 3.70, normal-vision dE 31.8, CVD min(protan,deutan) dE 26.8).
Series identity is also carried by the legend and by row position, never colour
alone.
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

CELLS = ["dabs_rabs", "dabs_rrec", "drec_rabs", "drec_rrec"]
LABEL = {"dabs_rabs": "DEM abs · z₀ abs", "dabs_rrec": "DEM abs · z₀ rec.",
         "drec_rabs": "DEM rec. · z₀ abs", "drec_rrec": "DEM rec. · z₀ rec."}

BG        = "#1b2b3d"
PANEL     = "#22344a"
INK       = "#e8eef6"
INK_2     = "#9fb2c9"
INK_MUTED = "#6f839b"
GRIDC     = "#32455d"
BLUE      = "#3987e5"    # dark categorical slot 1 - the 1 km input
ORANGE    = "#d95926"    # dark categorical slot 2 - FuXi-CFD
CELLCOLS  = ["#d95926", "#199e70", "#c98500", "#d55181"]


def pct(model, obs):
    return (model / obs - 1.0) * 100.0


def style(ax, xlabel):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0)
    ax.xaxis.grid(True, color=GRIDC, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK_2, lw=1.1, ls=(0, (5, 4)), zorder=2, alpha=0.85)
    ax.set_xlabel(xlabel, fontsize=10, color=INK_2)


def draw(ax, projects, data, series, yy):
    """data[name][project] -> array of per-mast percent errors."""
    n = len(series)
    offs = np.linspace(0.18, -0.18, n) if n > 1 else [0.0]
    for (name, col), off in zip(series, offs):
        for i, p in enumerate(projects):
            v = data[name].get(p)
            if v is None or not len(v):
                continue
            y = yy[i] + off
            ax.scatter(v, np.full(len(v), y), s=13, color=col, alpha=0.55,
                       linewidths=0, zorder=3)
            ax.scatter([np.mean(v)], [y], s=86, color=col, zorder=4,
                       edgecolors=PANEL, linewidths=1.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/provider_dots.png"))
    ap.add_argument("--all-cells", action="store_true")
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

    # Best cell by per-project centred MAE - the same statistic panel 2 shows.
    def cmae(v):
        e = v - obs
        return np.mean(np.concatenate(
            [np.abs(e[proj == p] - e[proj == p].mean()) for p in projects]))
    best = min(CELLS, key=lambda c: cmae(fine[c]))
    spread = max(cmae(fine[c]) for c in CELLS) - min(cmae(fine[c]) for c in CELLS)

    chosen = CELLS if args.all_cells else [best]
    series = [("WakeMap 1 km", BLUE)] + [
        (f"FuXi-CFD 30 m — {LABEL[c]}" if not args.all_cells else LABEL[c],
         ORANGE if not args.all_cells else CELLCOLS[CELLS.index(c)])
        for c in chosen]

    raw, mma = {}, {}
    for name, _ in series:
        raw[name], mma[name] = {}, {}
    def fill(name, v):
        for p in projects:
            m = proj == p
            raw[name][p] = pct(v[m], obs[m])
            k = obs[m].mean() / v[m].mean()      # scale grid onto the mast mean
            mma[name][p] = pct(v[m] * k, obs[m])
    fill("WakeMap 1 km", coarse)
    for (name, _), c in zip(series[1:], chosen):
        fill(name, fine[c])

    yy = np.arange(len(projects))[::-1]
    fig = plt.figure(figsize=(15.5, 8.6), facecolor=BG)
    gs = fig.add_gridspec(1, 2, wspace=0.10, left=0.085, right=0.977,
                          top=0.755, bottom=0.125)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    style(ax1, "grid ÷ mast − 1 at 100 m  (%)")
    style(ax2, "MMA grid ÷ mast − 1 at 100 m  (%)")
    draw(ax1, projects, raw, series, yy)
    draw(ax2, projects, mma, series, yy)

    for ax in (ax1, ax2):
        ax.set_yticks(yy)
        ax.set_ylim(yy.min() - 0.8, yy.max() + 0.8)
    ax1.set_yticklabels([p.upper() for p in projects], fontsize=10, color=INK)
    ax2.set_yticklabels([])

    lim1 = np.nanmax([np.abs(v).max() for d in raw.values() for v in d.values()]) * 1.12
    lim2 = np.nanmax([np.abs(v).max() for d in mma.values() for v in d.values()]) * 1.12
    ax1.set_xlim(-lim1, lim1)
    ax2.set_xlim(-lim2, lim2)

    def stats(d):
        allv = np.concatenate([v for p in projects for v in [d[p]]])
        return np.mean(np.abs(allv))
    m_raw = {n: stats(raw[n]) for n, _ in series}
    m_mma = {n: stats(mma[n]) for n, _ in series}

    ax1.set_title("1   Raw grid vs. met masts", fontsize=12.5, color=INK,
                  fontweight="bold", loc="left", pad=22)
    ax2.set_title("2   MMA grid vs. met masts", fontsize=12.5, color=INK,
                  fontweight="bold", loc="left", pad=22)
    ax1.text(0, 1.012, "  ·  ".join(f"MAE {n.split('—')[0].strip()} {v:.1f}%"
                                    for n, v in m_raw.items()),
             transform=ax1.transAxes, fontsize=9.5, color=INK_MUTED, va="bottom")
    ax2.text(0, 1.012, "  ·  ".join(f"MAE {n.split('—')[0].strip()} {v:.1f}%"
                                    for n, v in m_mma.items()),
             transform=ax2.transAxes, fontsize=9.5, color=INK_MUTED, va="bottom")

    fig.text(0.011, 0.958, "FuXi-CFD 30 m vs. WakeMap 1 km wind resource grids",
             fontsize=16.5, color=INK, fontweight="bold", ha="left")
    fig.text(0.011, 0.918,
             f"{len(obs)} mast–project comparisons at 100 m across {len(projects)} projects. "
             f"Large dot = project mean, small dots = individual masts.\n"
             f"All four dem/roughness cells agree to within {spread:.3f} m/s of centred MAE, "
             f"so one is shown; --all-cells draws the rest.",
             fontsize=10, color=INK_2, ha="left")

    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=n)
               for n, c in series]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.011, 0.878),
               frameon=False, fontsize=10.5, labelcolor=INK, ncol=len(series),
               handletextpad=0.4, columnspacing=2.0)

    fig.text(0.011, 0.062,
             "Panel 2 removes each project's own mean error (mean-mast-adjusted), isolating the "
             "mast-to-mast pattern — the part downscaling is responsible for; the mean comes from "
             "the mesoscale run.\n"
             "LCW and LFW have no masts in the file and are absent. Every 100 m mast value is "
             "extrapolated from lower sensors, not measured at 100 m; 2 masts dropped for "
             "data-quality flags. Nearest-node sampling.",
             fontsize=8.5, color=INK_MUTED, ha="left", va="top")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=BG)
    print(f"wrote {args.out}")
    for n, _ in series:
        print(f"  {n:42s} raw MAE {m_raw[n]:5.2f}%   MMA MAE {m_mma[n]:5.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
