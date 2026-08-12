"""Domain-wide comparison of the terrain sources actually used for a run.

    python scripts/compare_sources_domain.py --store /scratch/.../wsw_domain.zarr

The tile-scale scripts (`plot_roughness_sources.py`, `compare_dem_sources.py`)
answered "what is wrong with this 9 km box". This one answers the prior
question: **across the whole domain we ran, how much do the source choices
matter?** Broadview's built-up speckle was found in a single tile; whether WSW
has the same problem, more of it, or none, is not something a tile tells you.

Two comparisons, run on one common grid so every panel is directly subtractable:

  * **roughness** -- ESA WorldCover (what WSW was run with) vs NLCD (the new
    default). The pixels that matter are the ones WorldCover calls built-up.
  * **elevation** -- SRTM GL1 (what FuXi-CFD was trained on, and what we use)
    vs USGS 3DEP lidar bare earth.

The grid is the model's own 30 m by default, not a plotting-convenient coarse
grid: downsampling before comparing would average away exactly the isolated
single-pixel spikes that motivated the whole question.

Fields are cached to an .npz so the notebook can plot without re-reading ~2 GB
of remote raster.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from fuxicfd_wrf.terrain import nlcd, srtm, worldcover
from fuxicfd_wrf.terrain.roughness import (
    CLASS_NAMES, NLCD_CLASS_NAMES, NLCD_Z0, WORLDCOVER_Z0, classes_to_z0,
)

TDEP = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/"
        "current/n{n:02d}w{w:03d}/USGS_13_n{n:02d}w{w:03d}.tif")

#: NLCD forest + woody-wetland classes, for the canopy-height test.
FOREST = {41, 42, 43, 90}

#: WorldCover's single built-up class -- the source of the Broadview spikes.
WC_BUILTUP = 50


def store_bounds(store: Path) -> tuple[str, float, float, float, float]:
    """(crs, minx, miny, maxx, maxy) covered by every tile in an inference store.

    Taken from the store rather than recomputed from the config so the maps
    describe the area we actually ran, including whatever the tiler decided
    about edge tiles.
    """
    import zarr

    root = zarr.open_group(str(store), mode="r")
    tiles = sorted(root.group_keys())
    if not tiles:
        raise SystemExit(f"{store} has no tile groups")
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for t in tiles:
        g = root[t]
        x = np.asarray(g["x"][:])
        y = np.asarray(g["y"][:])
        xmin, xmax = min(xmin, x.min()), max(xmax, x.max())
        ymin, ymax = min(ymin, y.min()), max(ymax, y.max())
    crs = root.attrs.get("crs") or root[tiles[0]].attrs.get("crs", "EPSG:32613")
    print(f"[domain] {len(tiles)} tiles, {crs}, "
          f"{(xmax - xmin) / 1000:.1f} x {(ymax - ymin) / 1000:.1f} km")
    return crs, float(xmin), float(ymin), float(xmax), float(ymax)


def read_grid(sources, crs, bounds, res_m, resampling, nodata=None) -> np.ndarray:
    """Reproject and mosaic `sources` onto a north-up grid over `bounds`.

    WarpedVRT reads only the intersecting source window, which is what makes a
    114 x 95 km read off a 36000^2 WorldCover tile tractable. Sources composite
    in order: later ones fill only what is still unset, so global tile seams do
    not fight.
    """
    minx, miny, maxx, maxy = bounds
    w = int(np.ceil((maxx - minx) / res_m))
    h = int(np.ceil((maxy - miny) / res_m))
    transform = from_origin(minx, maxy, res_m, res_m)
    out = np.full((h, w), np.nan)

    for src_path in sources:
        with rasterio.open(src_path) as src:
            nd = src.nodata if nodata is None else nodata
            with WarpedVRT(src, crs=crs, transform=transform, width=w, height=h,
                           resampling=resampling, src_nodata=nd, nodata=nd) as vrt:
                block = vrt.read(1).astype("float64")
        ok = np.isfinite(block)
        if nd is not None:
            ok &= block != nd
        fill = ok & ~np.isfinite(out)
        out[fill] = block[fill]
    return out


def lonlat_box(crs, bounds, pad_deg=0.05):
    import pyproj

    minx, miny, maxx, maxy = bounds
    tf = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs = [minx, minx, maxx, maxx]
    ys = [miny, maxy, miny, maxy]
    lon, lat = tf.transform(xs, ys)
    return (min(lon) - pad_deg, min(lat) - pad_deg,
            max(lon) + pad_deg, max(lat) + pad_deg)


def tdep_urls(lonlat) -> list[str]:
    """3DEP tiles are 1x1 degree, named by their NORTH-WEST corner."""
    minlon, minlat, maxlon, maxlat = lonlat
    out = []
    for lat in range(int(np.floor(minlat)), int(np.floor(maxlat)) + 1):
        for lon in range(int(np.floor(minlon)), int(np.floor(maxlon)) + 1):
            out.append(f"/vsicurl/{TDEP.format(n=lat + 1, w=abs(lon))}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path,
                    default=Path("/scratch/moptis/fuxicfd/out/wsw_domain.zarr"))
    ap.add_argument("--res", type=float, default=30.0,
                    help="comparison grid spacing (m). Default is the model's "
                         "own 30 m; coarsening averages away the single-pixel "
                         "spikes this is meant to find.")
    ap.add_argument("--cache", type=Path,
                    default=Path("/scratch/moptis/fuxicfd/cache"))
    ap.add_argument("--out", type=Path,
                    default=Path("/scratch/moptis/fuxicfd/out/sources_domain.npz"))
    ap.add_argument("--fig", type=Path,
                    default=Path("/scratch/moptis/fuxicfd/out/sources_domain.png"),
                    help="six-panel figure rendered from the cached fields")
    ap.add_argument("--reuse", action="store_true",
                    help="re-print the report from a cached .npz instead of "
                         "re-reading the sources (3DEP alone is ~30 s of "
                         "network for an unchanged answer)")
    args = ap.parse_args()

    if args.reuse:
        if not args.out.exists():
            raise SystemExit(f"--reuse given but {args.out} does not exist")
        d = np.load(args.out)
        print(f"[reuse] {args.out}")
        report(d["wc"], d["nl"], d["z_wc"], d["z_nl"],
               d["dem_srtm"], d["dem_tdep"], float(d["res"]))
        if args.fig:
            figure(d, args.fig)
        return 0

    crs, *bnds = store_bounds(args.store)
    bounds = tuple(bnds)
    lonlat = lonlat_box(crs, bounds)
    print(f"[domain] lon/lat box {lonlat[0]:.3f} {lonlat[1]:.3f} "
          f"{lonlat[2]:.3f} {lonlat[3]:.3f}")

    def timed(label, fn):
        t0 = time.time()
        arr = fn()
        print(f"[read] {label:<22} {arr.shape}  {time.time() - t0:5.1f}s  "
              f"nodata {100 * np.isnan(arr).mean():.2f}%")
        return arr

    wc_src = worldcover.ensure_tiles(lonlat, args.cache)
    srtm_src = srtm.ensure_tiles(lonlat, args.cache)

    wc = timed("WorldCover 10 m", lambda: read_grid(
        wc_src, crs, bounds, args.res, Resampling.nearest, nodata=0))
    with rasterio.Env(AWS_REQUEST_PAYER="requester",
                      GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        nl = timed("NLCD 30 m", lambda: read_grid(
            [nlcd.url_for()], crs, bounds, args.res, Resampling.nearest,
            nodata=nlcd.NODATA))
    dem_srtm = timed("SRTM GL1", lambda: read_grid(
        srtm_src, crs, bounds, args.res, Resampling.bilinear))
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        dem_tdep = timed("USGS 3DEP 1/3 arcsec", lambda: read_grid(
            tdep_urls(lonlat), crs, bounds, args.res, Resampling.bilinear))

    z_wc = classes_to_z0(np.nan_to_num(wc, nan=30), "worldcover")
    z_nl = classes_to_z0(np.nan_to_num(nl, nan=71), "nlcd")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, wc=wc.astype("float32"), nl=nl.astype("float32"),
        z_wc=z_wc.astype("float32"), z_nl=z_nl.astype("float32"),
        dem_srtm=dem_srtm.astype("float32"), dem_tdep=dem_tdep.astype("float32"),
        bounds=np.array(bounds), res=args.res, crs=np.array(crs),
    )
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")

    report(wc, nl, z_wc, z_nl, dem_srtm, dem_tdep, args.res)
    if args.fig:
        figure(np.load(args.out), args.fig)
    return 0


def report(wc, nl, z_wc, z_nl, dem_srtm, dem_tdep, res) -> None:
    n = wc.size
    print(f"\n{'=' * 74}\nROUGHNESS: ESA WorldCover vs NLCD  ({n:,} px at {res:g} m)\n{'=' * 74}")
    print(f"{'':>4}{'WorldCover class':<26}{'share':>8}   {'NLCD class':<26}{'share':>8}")
    wc_top = sorted(zip(*np.unique(np.rint(wc[np.isfinite(wc)]).astype(int),
                                   return_counts=True)), key=lambda t: -t[1])[:8]
    nl_top = sorted(zip(*np.unique(np.rint(nl[np.isfinite(nl)]).astype(int),
                                   return_counts=True)), key=lambda t: -t[1])[:8]
    for i in range(max(len(wc_top), len(nl_top))):
        a = (f"{CLASS_NAMES.get(wc_top[i][0], wc_top[i][0]):<26}"
             f"{100 * wc_top[i][1] / n:7.2f}%") if i < len(wc_top) else " " * 34
        b = (f"{NLCD_CLASS_NAMES.get(nl_top[i][0], nl_top[i][0]):<26}"
             f"{100 * nl_top[i][1] / n:7.2f}%") if i < len(nl_top) else ""
        print(f"{'':>4}{a}   {b}")

    print(f"\n  z0 mean      WorldCover {z_wc.mean():.4f} m     NLCD {z_nl.mean():.4f} m")
    print(f"  z0 max       WorldCover {z_wc.max():.3f} m      NLCD {z_nl.max():.3f} m")

    bu = np.rint(wc) == WC_BUILTUP
    print(f"\n  WorldCover 'built-up' -> z0 {WORLDCOVER_Z0[WC_BUILTUP]} m: "
          f"{bu.sum():,} px ({100 * bu.mean():.3f}% of domain)")
    if bu.any():
        codes, counts = np.unique(np.rint(nl[bu]).astype(int), return_counts=True)
        print(f"    what NLCD calls those same pixels:")
        for c, k in sorted(zip(codes, counts), key=lambda t: -t[1])[:6]:
            print(f"      {NLCD_CLASS_NAMES.get(c, c):<28}{k:>9,}  "
                  f"{100 * k / bu.sum():5.1f}%   z0 {NLCD_Z0.get(c, float('nan')):.2f} m")
        print(f"    median z0 NLCD assigns them: {np.median(z_nl[bu]):.3f} m "
              f"(vs {WORLDCOVER_Z0[WC_BUILTUP]} m from WorldCover)")

    # An isolated high-z0 pixel is what produced the Broadview dipoles: a big
    # step over one 30 m cell with no terrain behind it. Count them explicitly
    # rather than inferring from the class share.
    from scipy.ndimage import uniform_filter
    for name, z in (("WorldCover", z_wc), ("NLCD", z_nl)):
        lz = np.log10(z)
        local = uniform_filter(lz, size=5, mode="nearest")
        spike = (lz - local) > 0.7        # >5x its own 150 m neighbourhood
        print(f"  isolated z0 spikes ({name}): {spike.sum():,} px "
              f"({100 * spike.mean():.4f}%)  -- >5x the 150 m local mean")

    print(f"\n{'=' * 74}\nELEVATION: SRTM GL1 vs USGS 3DEP\n{'=' * 74}")
    ok = np.isfinite(dem_srtm) & np.isfinite(dem_tdep)
    d = dem_tdep[ok] - dem_srtm[ok]
    print(f"  valid overlap      {100 * ok.mean():.2f}% of domain")
    print(f"  SRTM   range       {dem_srtm[ok].min():.0f} to {dem_srtm[ok].max():.0f} m"
          f"   (relief {np.ptp(dem_srtm[ok]):.0f} m)")
    print(f"  3DEP   range       {dem_tdep[ok].min():.0f} to {dem_tdep[ok].max():.0f} m"
          f"   (relief {np.ptp(dem_tdep[ok]):.0f} m)")
    print(f"  bias (3DEP-SRTM)   {d.mean():+.2f} m")
    print(f"  RMS difference     {np.sqrt((d ** 2).mean()):.2f} m")
    print(f"  |diff| p50/p95/max {np.percentile(np.abs(d), 50):.2f} / "
          f"{np.percentile(np.abs(d), 95):.2f} / {np.abs(d).max():.1f} m")

    nlf = np.rint(nl)
    is_forest = np.isin(nlf, list(FOREST)) & ok
    is_open = ~np.isin(nlf, list(FOREST)) & ok
    if is_forest.any():
        fb = (dem_tdep - dem_srtm)[is_forest].mean()
        ob = (dem_tdep - dem_srtm)[is_open].mean()
        print(f"  forest cover       {100 * is_forest.mean():.2f}% of domain")
        print(f"  bias over forest   {fb:+.2f} m")
        print(f"  bias over open     {ob:+.2f} m")
        print(f"  canopy signal      {ob - fb:+.2f} m  (positive = SRTM sits "
              f"higher over forest, i.e. on the canopy)")

    # Slope is what drives the flow, so agreement in slope matters more than
    # agreement in absolute height.
    gs = np.hypot(*np.gradient(np.nan_to_num(dem_srtm), res))
    gt = np.hypot(*np.gradient(np.nan_to_num(dem_tdep), res))
    print(f"  mean |slope|       SRTM {100 * gs[ok].mean():.2f}%   "
          f"3DEP {100 * gt[ok].mean():.2f}%")
    print(f"  slope correlation  {np.corrcoef(gs[ok], gt[ok])[0, 1]:.3f}")


AQUA, BLUE, RED = "#1baf7a", "#2a78d6", "#e34948"
INK, MUTED = "#52514e", "#8a8983"


def figure(d, out: Path) -> None:
    """Six panels on one grid: both source pairs plus their differences."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    minx, miny, maxx, maxy = d["bounds"]
    ext = [minx / 1000, maxx / 1000, miny / 1000, maxy / 1000]
    z_wc, z_nl = d["z_wc"], d["z_nl"]
    ds, dt = d["dem_srtm"], d["dem_tdep"]

    def show(ax, f, cmap, title, label, norm=None, **kw):
        im = ax.imshow(f, extent=ext, origin="upper", cmap=cmap, norm=norm,
                       interpolation="nearest", aspect="equal", **kw)
        ax.set_title(title, fontsize=9.5, pad=5)
        for s in ax.spines.values():
            s.set_color(MUTED); s.set_linewidth(0.6)
        ax.tick_params(colors=INK, labelsize=7.5, length=3, width=0.6)
        cb = plt.colorbar(im, ax=ax, fraction=0.040, pad=0.03)
        cb.set_label(label, fontsize=7.5, color=INK)
        cb.ax.tick_params(labelsize=7, colors=INK, length=2)
        cb.outline.set_edgecolor(MUTED); cb.outline.set_linewidth(0.6)
        return cb

    # z0 is a class lookup, so it takes a handful of discrete values. A
    # continuous ramp spends most of its range on gaps that hold no data and
    # renders 0.03 vs 0.10 -- 92% of this domain -- indistinguishable.
    levels = np.array(sorted(set(np.round(
        np.concatenate([np.unique(z_wc), np.unique(z_nl)]), 6).tolist())))
    base = mcolors.LinearSegmentedColormap.from_list(
        "z0", [(1 - (1 - np.array(mcolors.to_rgb(AQUA))) * 0.16),
               mcolors.to_rgb(AQUA), np.array(mcolors.to_rgb(AQUA)) * 0.42])
    zcmap = mcolors.ListedColormap(base(np.linspace(0.12, 1.0, len(levels))))
    edges = np.concatenate([[levels[0] * 0.5],
                            np.sqrt(levels[:-1] * levels[1:]), [levels[-1] * 2]])
    znorm = mcolors.BoundaryNorm(edges, zcmap.N)
    div = mcolors.LinearSegmentedColormap.from_list(
        "d", ["#184f95", BLUE, "#f0efec", RED, "#8f2322"])

    fig, ax = plt.subplots(2, 3, figsize=(16, 9.6), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    for a in ax.ravel():
        a.set_facecolor("#fcfcfb")

    for col, (z, name) in enumerate([(z_wc, "ESA WorldCover (used for this run)"),
                                     (z_nl, "NLCD 2023")]):
        cb = show(ax[0, col], z, zcmap, f"$z_0$ — {name}", "$z_0$ (m)", znorm)
        cb.set_ticks(levels); cb.set_ticklabels([f"{t:g}" for t in levels])
    r = np.log10(z_nl) - np.log10(z_wc)
    lim = float(np.abs(r).max()) or 1.0
    show(ax[0, 2], r, div, "$z_0$: NLCD / WorldCover", "$\\log_{10}$ ratio",
         mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))

    vmin, vmax = float(min(ds.min(), dt.min())), float(max(ds.max(), dt.max()))
    for col, (e, name) in enumerate([(ds, "SRTM GL1 (used for this run)"),
                                     (dt, "USGS 3DEP")]):
        show(ax[1, col], e, "terrain", f"elevation — {name}", "m",
             vmin=vmin, vmax=vmax)
    diff = dt - ds
    dlim = float(np.percentile(np.abs(diff), 99))
    show(ax[1, 2], diff, div, "elevation: 3DEP − SRTM", "m",
         mcolors.TwoSlopeNorm(vmin=-dlim, vcenter=0, vmax=dlim))

    for a in ax[1, :]:
        a.set_xlabel("UTM 13N easting (km)", fontsize=8, color=INK)
    for a in ax[:, 0]:
        a.set_ylabel("northing (km)", fontsize=8, color=INK)

    bu = np.rint(d["wc"]) == WC_BUILTUP
    rms = float(np.sqrt(((dt - ds) ** 2).mean()))
    fig.suptitle(
        "WSW domain: how much do the terrain source choices matter?\n"
        f"Roughness — WorldCover calls {bu.sum():,} px built-up ($z_0$=0.8 m); "
        f"NLCD gives those a median of {np.median(z_nl[bu]):.2f} m.   "
        f"Elevation — 3DEP−SRTM RMS {rms:.2f} m over {np.ptp(ds):.0f} m of relief.",
        fontsize=11, color="#0b0b0b")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
