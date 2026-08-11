"""Compare WorldCover vs NLCD roughness around the Broadview built-up speckle.

    python scripts/plot_roughness_sources.py --out fig.png

Two rows at different scales, because the problem only makes sense with both:

  * **context** (27 km) -- shows that the two datasets broadly agree on the
    landscape, so this is not a wholesale disagreement.
  * **zoom** (3 km) -- shows the isolated pixels WorldCover puts at the urban
    z0 of 0.8 m, which is what produced the localised wind dipoles.

Both z0 panels share one colour scale and one ramp. Comparing two fields on
independent scales is the fastest way to make a real difference invisible or an
imaginary one look dramatic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from fuxicfd_wrf.terrain import nlcd, srtm, worldcover
from fuxicfd_wrf.terrain.roughness import (
    CLASS_NAMES, NLCD_CLASS_NAMES, NLCD_Z0, WORLDCOVER_Z0, classes_to_z0,
)

AQUA, BLUE, RED = "#1baf7a", "#2a78d6", "#e34948"
INK, MUTED = "#52514e", "#8a8983"
CRS = "EPSG:32613"


def one_hue(hex_hue, name):
    base = np.array(mcolors.to_rgb(hex_hue))
    return mcolors.LinearSegmentedColormap.from_list(
        name, [1 - (1 - base) * 0.16, base, base * 0.42])


def read_box(sources, cx, cy, half_m, res_m=30.0, nodata=None):
    """Read raster(s) onto a square grid centred on (cx, cy), north-up."""
    n = int(2 * half_m / res_m)
    transform = from_origin(cx - half_m, cy + half_m, res_m, res_m)
    out = np.full((n, n), np.nan)
    for src_path in sources:
        with rasterio.open(src_path) as src:
            nd = src.nodata if nodata is None else nodata
            with WarpedVRT(src, crs=CRS, transform=transform, width=n, height=n,
                           resampling=Resampling.nearest, src_nodata=nd, nodata=nd) as vrt:
                block = vrt.read(1).astype("float64")
        ok = np.isfinite(block) & (block != nd if nd is not None else True)
        fill = ok & ~np.isfinite(out)
        out[fill] = block[fill]
    return out


def panel(ax, x, y, f, cmap, title, norm=None, label=""):
    im = ax.pcolormesh(x / 1000, y / 1000, f, cmap=cmap, norm=norm,
                       shading="auto", rasterized=True)
    ax.set_title(title, fontsize=9.5, pad=5)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=7.5, length=3, width=0.6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, fontsize=7.5, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK, length=2)
    cb.outline.set_edgecolor(MUTED); cb.outline.set_linewidth(0.6)
    return cb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon", type=float, default=-103.035)
    ap.add_argument("--lat", type=float, default=34.817)
    ap.add_argument("--context-km", type=float, default=13.5)
    ap.add_argument("--zoom-km", type=float, default=1.5)
    ap.add_argument("--zoom-on", choices=["speckle", "centre"], default="speckle",
                    help="centre the zoom on the densest WorldCover built-up "
                         "cluster (the thing under investigation) or on the tile")
    ap.add_argument("--cache", type=Path, default=Path("/scratch/moptis/fuxicfd/cache"))
    ap.add_argument("--out", type=Path,
                    default=Path("/scratch/moptis/fuxicfd/out/roughness_sources.png"))
    args = ap.parse_args()

    tf = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    cx, cy = tf.transform(args.lon, args.lat)

    lonlat = (args.lon - 0.25, args.lat - 0.25, args.lon + 0.25, args.lat + 0.25)
    wc_src = worldcover.ensure_tiles(lonlat, args.cache)

    def fields(half_m):
        wc = read_box(wc_src, cx, cy, half_m, nodata=0)
        with rasterio.Env(AWS_REQUEST_PAYER="requester",
                          GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            nl = read_box([nlcd.url_for()], cx, cy, half_m, nodata=nlcd.NODATA)
        z_wc = classes_to_z0(np.nan_to_num(wc, nan=30), "worldcover")
        z_nl = classes_to_z0(np.nan_to_num(nl, nan=71), "nlcd")
        n = z_wc.shape[0]
        x = cx - half_m + (np.arange(n) + 0.5) * 30.0
        y = cy + half_m - (np.arange(n) + 0.5) * 30.0
        return x, y, z_wc, z_nl, wc, nl

    ctx = fields(args.context_km * 1000)

    if args.zoom_on == "speckle":
        # Centre the zoom on the densest built-up cluster: zooming on the tile
        # centre shows a box with almost none of the pixels under investigation.
        from scipy.ndimage import uniform_filter
        bu = (np.rint(ctx[5 - 1]) == 50).astype(float)   # ctx[4] = worldcover codes
        win = max(3, int(2 * args.zoom_km * 1000 / 30))
        dens = uniform_filter(bu, size=win, mode="constant")
        r, c = np.unravel_index(int(np.argmax(dens)), dens.shape)
        cx, cy = float(ctx[0][c]), float(ctx[1][r])
        n_here = int(bu[max(0, r - win // 2):r + win // 2,
                        max(0, c - win // 2):c + win // 2].sum())
        print(f"[zoom] densest built-up cluster at ({cx:.0f}, {cy:.0f}) "
              f"-- {n_here} built-up pixels in the box")
    zm = fields(args.zoom_km * 1000)

    # One scale for every z0 panel: comparing on independent scales is how a real
    # difference becomes invisible.
    allz = np.concatenate([ctx[2].ravel(), ctx[3].ravel(), zm[2].ravel(), zm[3].ravel()])
    # z0 is ordinal, not continuous: it comes from a class lookup and takes a
    # handful of distinct values. A continuous log ramp spends most of its range
    # on gaps that contain no data, rendering 0.03 and 0.05 -- the bulk of the
    # field -- indistinguishable. Discrete bands give every level its own colour.
    levels = np.array(sorted(set(np.round(allz, 6).tolist())))
    if len(levels) > 12:
        levels = np.array(sorted(set(np.round(np.quantile(allz, np.linspace(0, 1, 12)), 4))))
    base = one_hue(AQUA, "z0")
    cmap = mcolors.ListedColormap(base(np.linspace(0.12, 1.0, len(levels))))
    edges = np.concatenate([[levels[0] * 0.5],
                            np.sqrt(levels[:-1] * levels[1:]),
                            [levels[-1] * 2]])
    znorm = mcolors.BoundaryNorm(edges, cmap.N)
    div = mcolors.LinearSegmentedColormap.from_list(
        "d", ["#184f95", BLUE, "#f0efec", RED, "#8f2322"])

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")

    for row, (dat, tag) in enumerate([(ctx, f"context {2*args.context_km:.0f} km"),
                                      (zm, f"zoom {2*args.zoom_km:.0f} km")]):
        x, y, z_wc, z_nl, _, _ = dat
        for col, (z, name) in enumerate([(z_wc, "ESA WorldCover"), (z_nl, "NLCD 2023")]):
            cb = panel(axes[row, col], x, y, z, cmap,
                       f"{name} — {tag}", znorm, "$z_0$ (m)")
            cb.set_ticks(levels)
            cb.set_ticklabels([f"{t:g}" for t in levels])
        d = np.log10(z_nl) - np.log10(z_wc)
        lim = float(np.abs(d).max()) or 1.0
        panel(axes[row, 2], x, y, d, div,
              f"NLCD / WorldCover — {tag}", mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
              "$\\log_{10}$ ratio")

    for ax in axes[1, :]:
        ax.set_xlabel("UTM 13N easting (km)", fontsize=8, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("northing (km)", fontsize=8, color=INK)

    wc_bu = (np.rint(zm[4]) == 50)
    fig.suptitle(
        "Broadview: surface roughness from two land-cover datasets\n"
        f"WorldCover puts {wc_bu.sum()} pixels in the zoom box at $z_0$=0.8 m "
        "(one 'built-up' class for everything from a farm track to a city centre); "
        "NLCD resolves development intensity",
        fontsize=11, color="#0b0b0b")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=125, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")

    for tag, dat in [("context", ctx), ("zoom", zm)]:
        _, _, z_wc, z_nl, wc, nl = dat
        print(f"\n--- {tag} ---")
        print(f"  WorldCover z0: mean {z_wc.mean():.4f}  max {z_wc.max():.3f}  "
              f"pixels at 0.8 m: {(z_wc > 0.7).sum()}")
        print(f"  NLCD       z0: mean {z_nl.mean():.4f}  max {z_nl.max():.3f}  "
              f"pixels above 0.7: {(z_nl > 0.7).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
