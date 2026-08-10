"""Surface roughness length from ESA WorldCover land cover.

FuXi-CFD wants z0 in metres on the 30 m grid. WorldCover gives 10 m land-cover
*classes*, so a lookup table bridges them. The values below are the conventional
wind-energy z0 assignments (WAsP-style roughness classes).

Caveat worth carrying: a class -> z0 table is a coarse instrument. Real z0 varies
with crop stage, canopy height and season, none of which WorldCover encodes. It is
what the FuXi-CFD authors describe using, so it is the right choice for matching
their input distribution -- but do not read the resulting z0 field as ground truth.
"""

from __future__ import annotations

import numpy as np

#: ESA WorldCover v200 class code -> roughness length z0 (m).
WORLDCOVER_Z0 = {
    10: 0.75,    # Tree cover
    20: 0.10,    # Shrubland
    30: 0.03,    # Grassland
    40: 0.05,    # Cropland
    50: 0.80,    # Built-up
    60: 0.005,   # Bare / sparse vegetation
    70: 0.001,   # Snow and ice
    80: 0.0002,  # Permanent water bodies
    90: 0.02,    # Herbaceous wetland
    95: 0.50,    # Mangroves
    100: 0.01,   # Moss and lichen
}

#: Used where a pixel carries a code outside the table.
FALLBACK_Z0 = 0.03

CLASS_NAMES = {
    10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built-up", 60: "bare/sparse", 70: "snow/ice", 80: "water",
    90: "herbaceous wetland", 95: "mangroves", 100: "moss/lichen",
}


def classes_to_z0(landcover: np.ndarray) -> np.ndarray:
    """Map a WorldCover class raster to z0 in metres."""
    codes = np.rint(np.asarray(landcover)).astype(np.int32)
    out = np.full(codes.shape, FALLBACK_Z0, dtype=np.float32)
    seen = set(np.unique(codes).tolist())
    for code, z0 in WORLDCOVER_Z0.items():
        out[codes == code] = z0
    unknown = seen - set(WORLDCOVER_Z0)
    if unknown:
        print(
            f"[terrain] unmapped WorldCover codes {sorted(unknown)} -> "
            f"z0={FALLBACK_Z0} m fallback"
        )
    return out


def class_histogram(landcover: np.ndarray) -> dict[str, float]:
    """Fractional land-cover composition, for the Phase 0 report."""
    codes = np.rint(np.asarray(landcover)).astype(np.int32)
    total = codes.size
    vals, counts = np.unique(codes, return_counts=True)
    return {
        CLASS_NAMES.get(int(v), f"code {int(v)}"): float(c) / total
        for v, c in sorted(zip(vals, counts), key=lambda p: -p[1])
    }
