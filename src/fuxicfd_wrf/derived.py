"""Turbine-relevant quantities derived from the FuXi-CFD 3D fields.

These are small compared to the 3D stores (a few numbers per point instead of 27
levels), so they are the right thing to compute once and share for downstream
energy analysis.
"""

from __future__ import annotations

import numpy as np

from .constants import OUTPUT_HEIGHTS
from .winds import uv_to_speeddir


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.hypot(u, v)


def wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological direction (degrees the wind blows *from*)."""
    return uv_to_speeddir(u, v)[1]


def at_height(field: np.ndarray, heights: np.ndarray, target_m: float, *, axis: int = -3) -> np.ndarray:
    """Log-interpolate a field to an arbitrary height between model levels.

    Wind speed near the surface follows a log profile far better than a linear
    one, so interpolating in log(z) is materially more accurate than np.interp
    across the widening gaps higher in the stack.
    """
    heights = np.asarray(heights, dtype=float)
    if target_m < heights[0] or target_m > heights[-1]:
        raise ValueError(
            f"target height {target_m} m outside the model stack "
            f"[{heights[0]}, {heights[-1]}]"
        )
    j = int(np.searchsorted(heights, target_m, side="right")) - 1
    j = min(max(j, 0), len(heights) - 2)
    z0, z1 = heights[j], heights[j + 1]
    if z1 == z0:
        w = 0.0
    else:
        w = (np.log(target_m) - np.log(z0)) / (np.log(z1) - np.log(z0))
    lo = np.take(field, j, axis=axis)
    hi = np.take(field, j + 1, axis=axis)
    return lo * (1 - w) + hi * w


def shear_exponent(
    speed: np.ndarray, heights: np.ndarray, z_lo: float, z_hi: float, *, axis: int = -3
) -> np.ndarray:
    """Power-law shear exponent alpha between two heights: U(z) ~ z^alpha."""
    s_lo = at_height(speed, heights, z_lo, axis=axis)
    s_hi = at_height(speed, heights, z_hi, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.log(s_hi / s_lo) / np.log(z_hi / z_lo)
    return np.where(np.isfinite(alpha), alpha, np.nan)


def turbulence_intensity(k: np.ndarray, speed: np.ndarray, *, floor: float = 0.5) -> np.ndarray:
    """TI from TKE, assuming isotropy: sigma = sqrt(2k/3), TI = sigma / U.

    The isotropy assumption is wrong in the surface layer (streamwise variance
    dominates), so this underestimates streamwise TI -- it is a consistent index,
    not a substitute for a measured TI. `floor` suppresses the divide-by-zero
    blow-up at near-calm points rather than emitting spurious huge values.
    """
    sigma = np.sqrt(np.maximum(2.0 * k / 3.0, 0.0))
    return np.where(speed >= floor, sigma / np.maximum(speed, 1e-9), np.nan)


def hub_height_summary(
    u: np.ndarray, v: np.ndarray, k: np.ndarray, hub_m: float,
    *, heights: np.ndarray = OUTPUT_HEIGHTS, shear_span: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Speed, direction, TI and shear at a hub height, from 3D fields.

    `u`, `v`, `k` have the level axis at -3, i.e. (..., level, y, x).
    """
    speed = wind_speed(u, v)
    out = {
        "wind_speed": at_height(speed, heights, hub_m),
        "wind_direction": wind_direction(
            at_height(u, heights, hub_m), at_height(v, heights, hub_m)
        ),
        "tke": at_height(k, heights, hub_m),
    }
    out["turbulence_intensity"] = turbulence_intensity(out["tke"], out["wind_speed"])
    lo, hi = shear_span or (max(heights[0], hub_m / 2), min(heights[-1], hub_m * 2))
    out["shear_exponent"] = shear_exponent(speed, heights, lo, hi)
    return out
