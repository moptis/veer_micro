"""Diagnose spatial structure in a tile WRG: is it terrain, roughness, or artifact?

    python scripts/diagnose_tile.py <tile.wrg> --tile bv_center

Separates three things that look similar on a colour map:

* **elevation-driven** structure -- correlates with the DEM.
* **roughness-driven** structure -- correlates with log(z0). Land-cover patches
  produce real, sharp-edged speed deficits with no terrain signature at all,
  which is exactly what "variability where the ground is flat" looks like.
* **coarse-input imprint** -- the 9x9 km wind stencil is bilinearly upsampled to
  300x300, putting kinks every 1 km. Structure periodic at 1 km is the input grid
  showing through the output, i.e. a genuine artifact rather than physics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fuxicfd_wrf.grids import Tile
from fuxicfd_wrf.terrain.build import build_terrain
from fuxicfd_wrf.terrain.roughness import class_histogram
from fuxicfd_wrf.wrg_read import read_wrg, to_grid


def resample(src: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Block-average a (301,301) terrain field onto the (ny,nx) WRG grid."""
    yi = np.linspace(0, src.shape[0] - 1, ny).round().astype(int)
    xi = np.linspace(0, src.shape[1] - 1, nx).round().astype(int)
    return src[np.ix_(yi, xi)]


def radial_period_power(field: np.ndarray, res_m: float) -> dict[float, float]:
    """Power at a few spatial wavelengths, to spot input-grid periodicity."""
    f = field - field.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    ny, nx = f.shape
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=res_m))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=res_m))
    KX, KY = np.meshgrid(fx, fy)
    K = np.hypot(KX, KY)
    out = {}
    total = F.sum()
    for wl in (3000.0, 2000.0, 1500.0, 1000.0, 750.0, 500.0, 300.0):
        k = 1.0 / wl
        band = (K > k * 0.9) & (K < k * 1.1)
        out[wl] = 100.0 * F[band].sum() / total if band.any() else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wrg", type=Path)
    ap.add_argument("--tile", required=True)
    ap.add_argument("--cache", type=Path, default=Path("/scratch/moptis/fuxicfd/cache"))
    args = ap.parse_args()

    df = read_wrg(args.wrg, air_density=1.06)
    gx, gy, speed = to_grid(df, "mean_speed")
    _, _, elev_wrg = to_grid(df, "elevation")
    ny, nx = speed.shape
    res = df.attrs["cell_size"]

    # Rebuild the exact static inputs the model saw.
    cx, cy = float(np.mean(gx)), float(np.mean(gy))
    terr = build_terrain(Tile(args.tile, x=cx, y=cy), args.cache)
    dem = resample(terr.dem, ny, nx)
    z0 = resample(terr.roughness, ny, nx)
    logz0 = np.log10(z0)

    print(f"grid {ny}x{nx} @ {res:.0f} m   speed {speed.min():.2f}..{speed.max():.2f} m/s")
    print(f"land cover: { {k: round(v,4) for k,v in class_histogram(terr.landcover).items()} }")

    print("\n--- what explains the spatial pattern? ---")
    for name, field in (("elevation", dem), ("log10(z0)", logz0)):
        r = np.corrcoef(field.ravel(), speed.ravel())[0, 1]
        print(f"  corr(speed, {name:<10}) = {r:+.3f}")
    # Partial correlation with roughness, controlling for elevation.
    A = np.column_stack([dem.ravel(), np.ones(dem.size)])
    resid_s = speed.ravel() - A @ np.linalg.lstsq(A, speed.ravel(), rcond=None)[0]
    resid_z = logz0.ravel() - A @ np.linalg.lstsq(A, logz0.ravel(), rcond=None)[0]
    print(f"  partial corr(speed, log z0 | elevation) = "
          f"{np.corrcoef(resid_s, resid_z)[0,1]:+.3f}")

    print("\n--- quadrants (is the NW corner special?) ---")
    hy, hx = ny // 2, nx // 2
    quads = {
        "NW": (slice(0, hy), slice(0, hx)), "NE": (slice(0, hy), slice(hx, nx)),
        "SW": (slice(hy, ny), slice(0, hx)), "SE": (slice(hy, ny), slice(hx, nx)),
    }
    print(f"  {'quad':<5}{'speed mean':>12}{'speed std':>11}{'elev range':>12}{'z0 max':>9}{'%rough':>8}")
    for q, sl in quads.items():
        s, e, z = speed[sl], dem[sl], z0[sl]
        rough_frac = 100.0 * (z > 0.3).mean()
        print(f"  {q:<5}{s.mean():12.3f}{s.std():11.4f}"
              f"{e.max()-e.min():11.0f} m{z.max():9.3f}{rough_frac:7.1f}%")

    print("\n--- spatial periodicity (share of variance per wavelength) ---")
    print("     the coarse wind stencil is 1 km; a spike there is input imprint")
    for wl, p in radial_period_power(speed, res).items():
        bar = "#" * int(p * 2)
        flag = "  <-- 1 km input grid" if wl == 1000.0 else ""
        print(f"  {wl:6.0f} m : {p:5.2f}%  {bar}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
