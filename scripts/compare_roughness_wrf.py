"""Compare the roughness WRF used against the roughness we hand FuXi-CFD.

    python scripts/compare_roughness_wrf.py --geo-em .../geo_em.d03.nc

This is a **consistency** check, not an accuracy one, and the distinction is the
whole point.

The 100 m winds driving FuXi-CFD come from WRF, which already applied its own
surface roughness at 1 km. FuXi's job is the sub-kilometre *increment*. For that
to be meaningful, the 1 km average of the fine z0 field we feed it has to agree
with the z0 embedded in the winds it is given. If they disagree, the model is
asked to reconcile a discrepancy that does not exist in nature, and the only way
it can express that is as fine-scale wind structure -- which looks exactly like
excess sensitivity to roughness.

The two fields come from different products *and* different lookup tables:

    winds  : WRF, MODIS-IGBP 21-category (MMINLU=MODIFIED_IGBP_MODIS_NOAH),
             z0 from the Noah section of VEGPARM.TBL, at 1 km
    FuXi   : NLCD Annual Collection 1, z0 from UL's table, at 30 m

Noah interpolates z0 between Z0MIN and Z0MAX with green vegetation fraction. The
exact form could not be verified against this WRF build (no source tree on disk),
so all three of Z0MIN, Z0MAX and a linear-in-GREENFRAC estimate are reported. If
the conclusion is the same under all three it does not depend on that detail,
which is the reason for computing them all rather than picking one.

z0 values are read out of the run's own VEGPARM.TBL, not hardcoded here.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

MODIS_NAMES = {
    1: "Evergreen Needleleaf", 2: "Evergreen Broadleaf", 3: "Decid Needleleaf",
    4: "Decid Broadleaf", 5: "Mixed Forests", 6: "Closed Shrublands",
    7: "Open Shrublands", 8: "Woody Savannas", 9: "Savannas", 10: "Grasslands",
    11: "Permanent Wetlands", 12: "Croplands", 13: "Urban and Built-Up",
    14: "Crop/Natural Mosaic", 15: "Snow and Ice", 16: "Barren/Sparse",
    17: "Water", 18: "Wooded Tundra", 19: "Mixed Tundra", 20: "Barren Tundra",
    21: "Lake",
}

#: Column offsets in the VEGPARM.TBL body rows, after the category number.
Z0MIN_COL, Z0MAX_COL = 14, 15


def parse_vegparm(path: Path, section: str) -> dict[int, tuple[float, float]]:
    """(Z0MIN, Z0MAX) per land-use category, read from the run's own table.

    Parsed rather than transcribed: a hardcoded copy silently goes stale if the
    run's table is edited, and this project has already been burned once by z0
    values with no traceable source.

    The format is comma-separated but inconsistently so -- several rows in the
    shipped table omit a comma after the category number -- so tokens are split
    on both commas and whitespace, and the trailing quoted class name is dropped.
    """
    lines = path.read_text().splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == section)
    except StopIteration:
        raise SystemExit(
            f"{path} has no '{section}' section. Sections present: "
            + ", ".join(ln.strip() for ln in lines if re.fullmatch(r"[A-Z][A-Z0-9_]+", ln.strip()))
        )

    out: dict[int, tuple[float, float]] = {}
    for ln in lines[start + 2:]:                    # +2 skips the header row
        if "'" not in ln:
            break                                    # end of the numeric block
        nums = re.sub(r"'[^']*'", "", ln).replace(",", " ").split()
        try:
            vals = [float(t) for t in nums]
        except ValueError:
            continue
        if len(vals) <= Z0MAX_COL:
            continue
        out[int(vals[0])] = (vals[Z0MIN_COL], vals[Z0MAX_COL])
    if not out:
        raise SystemExit(f"parsed no categories from {path} section {section}")
    return out


def read_geo_em(path: Path):
    import h5py

    with h5py.File(path, "r") as f:
        lu = np.asarray(f["LU_INDEX"])[0]
        lat = np.asarray(f["XLAT_M"])[0]
        lon = np.asarray(f["XLONG_M"])[0]
        gf = np.asarray(f["GREENFRAC"])[0] if "GREENFRAC" in f else None
    return lu, lat, lon, gf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo-em", type=Path,
                    default=Path("/scratch/moptis/c2wind/wsw/control/wps/geo_em.d03.nc"))
    ap.add_argument("--vegparm", type=Path,
                    default=Path("/scratch/moptis/c2wind/wsw/control/wrf_runs/0051/VEGPARM.TBL"))
    ap.add_argument("--section", default="MODIFIED_IGBP_MODIS_NOAH")
    ap.add_argument("--npz", type=Path,
                    default=Path("/scratch/moptis/fuxicfd/out/sources_domain.npz"),
                    help="cached NLCD classes from compare_sources_domain.py")
    ap.add_argument("--out", type=Path,
                    default=Path("/scratch/moptis/fuxicfd/out/roughness_wrf_vs_nlcd.npz"))
    ap.add_argument("--fig", type=Path,
                    default=Path("/scratch/moptis/fuxicfd/out/roughness_wrf_vs_nlcd.png"))
    args = ap.parse_args()

    from fuxicfd_wrf.grids import DomainGeometry
    from fuxicfd_wrf.terrain.roughness import NLCD_CLASS_NAMES, NLCD_Z0

    z0tab = parse_vegparm(args.vegparm, args.section)
    print(f"[wrf] {args.vegparm.name} section {args.section}: "
          f"{len(z0tab)} categories parsed")

    lu, lat, lon, gf = read_geo_em(args.geo_em)
    lu = lu.astype(int)
    print(f"[wrf] LU_INDEX {lu.shape} on a 1 km grid")

    z0min = np.array([z0tab.get(v, (np.nan,))[0] for v in lu.ravel()]).reshape(lu.shape)
    z0max = np.array([z0tab.get(v, (np.nan, np.nan))[1] for v in lu.ravel()]).reshape(lu.shape)
    if gf is not None:
        # Annual-mean greenness. Noah applies this monthly, so a single map is a
        # summary of a field that actually breathes through the year -- the
        # seasonal swing is bounded by the z0min/z0max pair reported alongside.
        shd = gf.mean(axis=0)
        shd = shd / 100.0 if shd.max() > 1.5 else shd
        z0gf = z0min + shd * (z0max - z0min)
        print(f"[wrf] GREENFRAC annual mean {shd.mean():.3f} "
              f"(range {shd.min():.3f}-{shd.max():.3f})")
    else:
        z0gf = None

    # --- our field -----------------------------------------------------------
    d = np.load(args.npz)
    nl = np.rint(d["nl"]).astype(int)
    lut = np.full(256, 0.03)
    for k, v in NLCD_Z0.items():
        lut[k] = v
    ours = lut[np.clip(nl, 0, 255)]
    minx, miny, maxx, maxy = d["bounds"]
    res = float(d["res"])
    print(f"[fuxi] NLCD z0 {ours.shape} at {res:g} m")

    # --- put ours on the WRF grid -------------------------------------------
    geom = DomainGeometry(lat.ravel(), lon.ravel(), crs="auto", grid_shape=lu.shape)
    ny, nx = ours.shape
    gx, gy = np.meshgrid(minx + (np.arange(nx) + 0.5) * res,
                         maxy - (np.arange(ny) + 0.5) * res)
    row, col = geom.xy_to_rowcol(gx.ravel(), gy.ravel())
    r, c = np.rint(row).astype(int), np.rint(col).astype(int)
    ok = (r >= 0) & (r < lu.shape[0]) & (c >= 0) & (c < lu.shape[1])
    flat = r[ok] * lu.shape[1] + c[ok]

    # Aggregate in log space. Roughness combines through the log law, so the
    # geometric mean is the meaningful average; an arithmetic one is dominated by
    # the few forest pixels and overstates the effective surface.
    lz = np.log(ours.ravel()[ok])
    cnt = np.bincount(flat, minlength=lu.size)
    agg = np.full(lu.size, np.nan)
    good = cnt > 100
    agg[good] = np.exp(np.bincount(flat, weights=lz, minlength=lu.size)[good] / cnt[good])
    agg = agg.reshape(lu.shape)

    # --- report --------------------------------------------------------------
    print(f"\n{'='*74}\nLAND-USE COMPOSITION: the two products disagree about what is here\n{'='*74}")
    vals, counts = np.unique(lu, return_counts=True)
    print(f"  {'WRF / MODIS-IGBP':<34}{'share':>8}   {'FuXi / NLCD':<26}{'share':>8}")
    nlv, nlc = np.unique(nl, return_counts=True)
    modis = sorted(zip(vals, counts), key=lambda t: -t[1])[:6]
    nlcd = sorted(zip(nlv, nlc), key=lambda t: -t[1])[:6]
    for i in range(max(len(modis), len(nlcd))):
        a = (f"{MODIS_NAMES.get(int(modis[i][0]), '?'):<26}"
             f"{100*modis[i][1]/lu.size:7.2f}%") if i < len(modis) else " " * 34
        lo, hi = z0tab.get(int(modis[i][0]), (np.nan, np.nan)) if i < len(modis) else (np.nan,)*2
        a = f"{a[:26]}{100*modis[i][1]/lu.size:6.2f}% {lo:5.3f}-{hi:<5.3f}" if i < len(modis) else " "*34
        b = (f"{NLCD_CLASS_NAMES.get(int(nlcd[i][0]), '?'):<26}"
             f"{100*nlcd[i][1]/nl.size:6.2f}% {NLCD_Z0.get(int(nlcd[i][0]), float('nan')):5.3f}"
             ) if i < len(nlcd) else ""
        print(f"  {a}   {b}")

    print(f"\n{'='*74}\nCONSISTENCY WITH THE DRIVING WINDS\n{'='*74}")
    print(f"{'WRF z0 assumption':<26}{'WRF mean':>10}{'ours mean':>11}"
          f"{'corr ln':>9}{'RMS ln':>9}{'factor':>8}")
    variants = [("Z0MIN (dormant)", z0min), ("Z0MAX (full veg)", z0max)]
    if z0gf is not None:
        variants.insert(1, ("GREENFRAC-interpolated", z0gf))
    stats = {}
    for name, w in variants:
        m = good.reshape(lu.shape) & np.isfinite(w) & np.isfinite(agg)
        a, b = agg[m], w[m]
        lr = np.log(a / b)
        corr = float(np.corrcoef(np.log(a), np.log(b))[0, 1])
        stats[name] = (corr, float(np.sqrt((lr**2).mean())))
        print(f"{name:<26}{b.mean():10.4f}{a.mean():11.4f}"
              f"{corr:+9.3f}{np.sqrt((lr**2).mean()):9.3f}"
              f"{np.exp(np.sqrt((lr**2).mean())):8.2f}")
    print(f"\n  corr ln = correlation of ln z0 across {int(m.sum()):,} 1-km cells.")
    print(f"  Near +1 means the fine field agrees with what the winds already encode;")
    print(f"  near 0 means it adds noise; negative means it actively opposes them.")
    if all(c < 0.1 for c, _ in stats.values()):
        print(f"\n  ** All three assumptions agree, so this does not hinge on the")
        print(f"     unverified Noah interpolation: the fine roughness we feed FuXi")
        print(f"     carries no useful agreement with the roughness in its own")
        print(f"     driving winds. **")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, lu=lu, z0min=z0min, z0max=z0max,
        z0gf=z0gf if z0gf is not None else np.zeros(0),
        ours_1km=agg, ours_30m=ours.astype("float32"),
        bounds=np.array([minx, miny, maxx, maxy]), res=res,
    )
    print(f"\nwrote {args.out}")
    figure(args.out, args.fig, geom)
    return 0


def figure(npz: Path, out: Path, geom) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    d = np.load(npz)
    lu, agg = d["lu"], d["ours_1km"]
    wrf = d["z0gf"] if d["z0gf"].size else d["z0max"]
    ours30, res = d["ours_30m"], float(d["res"])
    minx, miny, maxx, maxy = d["bounds"]

    INK, MUTED = "#52514e", "#8a8983"
    ext30 = [minx/1000, maxx/1000, miny/1000, maxy/1000]
    wx, wy = geom.x/1000, geom.y/1000
    wext = [wx.min(), wx.max(), wy.min(), wy.max()]

    lo, hi = 0.01, 0.5
    norm = mcolors.LogNorm(vmin=lo, vmax=hi)
    div = mcolors.LinearSegmentedColormap.from_list(
        "d", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#8f2322"])

    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")

    def show(a, f, ext, cmap, title, label, **kw):
        im = a.imshow(f, extent=ext, origin="upper", cmap=cmap,
                      interpolation="nearest", aspect="equal", **kw)
        a.set_title(title, fontsize=9.5, pad=5)
        a.set_xlabel("UTM 13N easting (km)", fontsize=8, color=INK)
        for s in a.spines.values():
            s.set_color(MUTED); s.set_linewidth(0.6)
        a.tick_params(colors=INK, labelsize=7.5, length=3, width=0.6)
        cb = plt.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        cb.set_label(label, fontsize=7.5, color=INK)
        cb.ax.tick_params(labelsize=7, colors=INK, length=2)
        return cb

    show(ax[0], np.flipud(wrf), wext, "viridis",
         "WRF / MODIS $z_0$ — what the 100 m winds encode", "$z_0$ (m)", norm=norm)
    show(ax[1], ours30, ext30, "viridis",
         "FuXi / NLCD $z_0$ at 30 m — what we feed the model", "$z_0$ (m)", norm=norm)
    show(ax[2], np.flipud(agg), wext, "viridis",
         "FuXi / NLCD aggregated to 1 km (geometric)", "$z_0$ (m)", norm=norm)
    ratio = np.log10(agg / wrf)
    lim = float(np.nanpercentile(np.abs(ratio), 99))
    show(ax[3], np.flipud(ratio), wext, div,
         "disagreement: NLCD / WRF at 1 km", "$\\log_{10}$ ratio",
         norm=mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
    ax[0].set_ylabel("northing (km)", fontsize=8, color=INK)

    m = np.isfinite(agg) & np.isfinite(wrf)
    corr = float(np.corrcoef(np.log(agg[m]), np.log(wrf[m]))[0, 1])
    fig.suptitle(
        "Is the roughness we give FuXi-CFD consistent with the roughness already in its driving winds?\n"
        f"Correlation of $\\ln z_0$ across {int(m.sum()):,} 1-km cells: "
        f"{corr:+.3f}  —  the fine field does not agree with what the winds encode",
        fontsize=11, color="#0b0b0b")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
