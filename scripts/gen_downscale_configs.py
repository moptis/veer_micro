#!/usr/bin/env python
"""Generate sector-transfer downscale configs: one per project x terrain-mode cell.

    python scripts/gen_downscale_configs.py --out-dir configs/sweep

Reads `configs/projects.csv` -- the analysis CRS and high-resolution terrain
bounds from the Wakemap order sheet -- locates each project's coarse 100 m
deliverable, and writes the 2x2 of dem_mode x roughness_mode.

Three things are derived rather than typed, because typing them 60 times is how
transcription errors get in:

* **CRS** comes from the order sheet's `analysis_epsg`. A `.wrg` records no
  projection and it is not guessable, so getting it wrong lands the whole domain
  somewhere else silently. This is the one field with no fallback.
* **Air density** from the coarse WRG's own elevation column via the ISA
  barometric formula. The 1.225 default is sea-level; these sites run from 30 m
  (GWR, on the Gulf coast) to 1900 m (SWW, PRW), where it overstates power
  density by up to ~20%.
* **Land cover** is pinned per project, never left on `auto`. `nlcd.covers()`
  swallows every exception, so `auto` cannot distinguish "outside CONUS" from a
  credentials failure -- and a sweep whose terrain source varied unrecorded
  between projects would confound the very mode comparison it exists to make.

The four cells share one raw terrain cache (it is keyed on tile geometry, not on
mode), so the SRTM/NLCD fetch is paid once per project and the other three cells
are nearly free.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
C2WIND = Path("/scratch/moptis/c2wind")
SCRATCH = Path("/scratch/moptis/fuxicfd")

#: dem_mode x roughness_mode, named as in the existing wsw_nlcdwrf_year_* configs.
CELLS = {
    "dabs_rabs": ("absolute", "absolute"),
    "dabs_rrec": ("absolute", "recentered"),
    "drec_rabs": ("recentered", "absolute"),
    "drec_rrec": ("recentered", "recentered"),
    # Roughness held at the TRAINING mean (0.817 m), so the z0 channel
    # normalises to exactly 0 and carries no signal. Elevation is then the only
    # high-resolution forcing, and whatever sub-kilometre structure survives is
    # terrain-driven. Roughness still reaches the answer through the coarse WRG,
    # which already embeds WRF's own surface drag at 1 km.
    "dabs_rconst": ("absolute", "constant"),
    "drec_rconst": ("recentered", "constant"),
    # Roughness low-passed to 1 km: the site's own z0 level and its >1 km
    # variation are kept, only the sub-kilometre detail is removed. Against
    # "absolute" this is a one-variable contrast -- the normalised mean the
    # network sees is unchanged to 6e-4 m, where "constant" moves it by 3.5
    # sigma. This is the controlled version of the elevation-only experiment.
    "dabs_rcoarse": ("absolute", "coarse"),
    "drec_rcoarse": ("recentered", "coarse"),
}

#: Projects with a `pattern/<name>/control` domain on this filesystem. The order
#: sheet also lists CCW/Broadview/GWE/WSW (other clients) and HIW/HRW/SIW/SJW/SVW
#: (no WRF domain here), which is why the sweep set is not simply every CSV row.
PATTERN = {
    "arw", "azw", "grw", "gwr", "k2w", "lcw", "lfw", "lgw",
    "mkw", "msm", "oew", "ph1", "ph2", "prw", "sww",
    # Not a Pattern project -- its deliverable lives at c2wind/wsw/control/...
    # rather than under pattern/. See coarse_wrg().
    "wsw",
}


def isa_density(elev_m: float) -> float:
    """Air density from the ISA barometric formula, kg/m^3."""
    return 1.225 * (1.0 - 2.25577e-5 * elev_m) ** 4.25588


def mean_elevation(wrg: Path) -> float:
    """Mean of a coarse .wrg's elevation column (4th field of each GridPoint)."""
    tot = n = 0
    with open(wrg) as fh:
        next(fh)                                  # header: nx ny xmin ymin cell
        for line in fh:
            f = line.split()
            if len(f) < 4:
                continue
            tot += float(f[3])
            n += 1
    if not n:
        raise ValueError(f"{wrg} has no grid points")
    return tot / n


def coarse_wrg(project: str) -> Path:
    """The 100 m *hub height* control deliverable for a project.

    Note 100m here is the cut height, not the grid spacing -- these files are on
    the 1 km WRF lattice, which is exactly the stencil FuXi-CFD wants. Driving
    the model with the 80/120/160 m files would feed its 100 m wind channels a
    field they were never trained to accept.

    The deliverables are not all under `pattern/`: WSW and the other non-Pattern
    clients sit one level up, with no client directory in the wrg path. Search
    the known layouts rather than hardcoding one -- WSW was missed from the first
    sweep precisely because the pattern/ layout was assumed to be universal.
    """
    candidates = [
        C2WIND / "pattern" / project / "control" / "deliverables" / "wrg" / "pattern" / project / "control_100m.wrg",
        C2WIND / project / "control" / "deliverables" / "wrg" / project / "control_100m.wrg",
        C2WIND / "wakemap_deep" / project / "control" / "deliverables" / "wrg" / "wakemap_deep" / project / "control_100m.wrg",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"no coarse deliverable for {project}; tried:\n  " +
        "\n  ".join(str(c) for c in candidates))


TEMPLATE = """\
# {proj_u} sector-transfer downscale -- dem_mode={dem}, roughness_mode={rough}.
#
# GENERATED by scripts/gen_downscale_configs.py from configs/projects.csv.
# Edit the CSV or the generator, not this file.
#
#   fuxicfd-wrf wrg-downscale {cfg_path}

name: {proj}_{cell}

source:
  # 100 m is the hub height, not the grid spacing: this file is on the 1 km WRF
  # lattice, which is the stencil the model's wind channels expect.
  path: {wrg}
  # From the order sheet's analysis_epsg. A .wrg records no projection and it is
  # not guessable -- wrong here means the whole domain lands somewhere else, with
  # no error anywhere downstream.
  crs: EPSG:{epsg}
  speed_bins: {speed_bins}

tiling:
  mode: domain
  # 3 km of overlap for the cosine taper. FuXi-CFD sees only its own 9 km window
  # with none of the 1.5 km buffer its training CFD had, so tile edges are its
  # least reliable region.
  stride_m: 6000
  # "auto" tiles in the coarse WRG's own CRS, making the affine fit exact.
  working_crs: auto
  affine_fit: global
  # The high-resolution terrain box from the order sheet, NOT the full wakemap
  # domain. Area is the only lever that trades both cost and storage, and the
  # full {full_km} domain would be ~{full_x}x the points for terrain nobody ordered.
  bbox:
    min_lon: {west}
    min_lat: {south}
    max_lon: {east}
    max_lat: {north}

terrain:
  # Pinned, never "auto": nlcd.covers() catches every exception, so "auto" cannot
  # tell "outside CONUS" from an AWS credentials failure, and a sweep whose
  # terrain source varied silently between projects would confound the mode
  # comparison this sweep exists to make.
  landcover_source: {landcover}
  cache_root: {cache}
  dem_mode: {dem}
  roughness_mode: {rough}

model:
  device: auto
  batch_size: 8

output:
  path: {out}
  # FuXi-CFD's native lattice is 30.1003 m; this is the resolution the whole
  # pipeline exists to produce, so it is not coarsened for convenience.
  resolution_m: 30.0
  hub_height: 100.0
  # ISA at the coarse WRG's mean elevation of {elev:.0f} m. The 1.225 default is
  # sea level and would overstate power density by {rho_err:.0f}% here.
  air_density: {rho:.4f}
  deflect: true
  save_transfer: true
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "configs" / "sweep")
    ap.add_argument("--csv", type=Path, default=REPO / "configs" / "projects.csv")
    ap.add_argument("--projects", nargs="*", default=None,
                    help="subset of project codes (default: all with a pattern/ domain)")
    ap.add_argument("--speed-bins", type=int, default=1,
                    help="3 measures whether the speedup is speed-invariant")
    ap.add_argument("--out-root", type=Path, default=SCRATCH / "out" / "wrg_sweep")
    ap.add_argument("--cache-root", type=Path, default=SCRATCH / "cache")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(
        l for l in open(args.csv) if not l.startswith("#"))]
    want = {p.lower() for p in args.projects} if args.projects else PATTERN
    rows = [r for r in rows if r["project"].lower() in want]
    missing = want - {r["project"].lower() for r in rows}
    if missing:
        raise SystemExit(f"not in {args.csv}: {sorted(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    print(f"{'proj':6s} {'epsg':6s} {'landcover':11s} {'elev m':>7s} {'rho':>7s} {'err%':>5s}")
    for r in sorted(rows, key=lambda r: r["project"]):
        proj = r["project"].lower()
        wrg = coarse_wrg(proj)
        elev = mean_elevation(wrg)
        rho = isa_density(elev)
        err = (1.225 / rho - 1.0) * 100.0
        full_km = f"{float(r['w_km']):.0f}x{float(r['h_km']):.0f} km terrain box inside a ~120x120 km"
        print(f"{proj:6s} {r['epsg']:6s} {r['landcover']:11s} {elev:7.0f} {rho:7.4f} {err:5.1f}")

        for cell, (dem, rough) in CELLS.items():
            cfg_path = args.out_dir / f"{proj}_{cell}.yaml"
            cfg_path.write_text(TEMPLATE.format(
                proj=proj, proj_u=proj.upper(), cell=cell, dem=dem, rough=rough,
                wrg=wrg, epsg=r["epsg"], speed_bins=args.speed_bins,
                west=r["west_lon"], south=r["south_lat"],
                east=r["east_lon"], north=r["north_lat"],
                landcover=r["landcover"], cache=args.cache_root,
                out=args.out_root / f"{proj}_{cell}_30m.wrg",
                elev=elev, rho=rho, rho_err=err,
                cfg_path=cfg_path.relative_to(REPO),
                full_km=full_km, full_x="6",
            ))
            n += 1

    print(f"\nwrote {n} configs ({len(rows)} projects x {len(CELLS)} cells) "
          f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
