"""Surface roughness length from land-cover classes.

FuXi-CFD wants z0 in metres on the 30 m grid; the land-cover products give
classes, so a lookup table bridges them.

**Provenance is uneven between the two tables -- read this before trusting a
number.**

NLCD_Z0 is UL's table: external reference values, not assignments made here. The
repo originally shipped a set assigned by analogy with WAsP-style roughness
classes, with no source recorded. That set has been removed. It mattered: it put
grassland at 0.03 against UL's 0.06 while both agree on shrub/scrub at 0.10, and
those two classes cover 92% of the WSW domain -- so it imposed roughly double the
roughness *contrast* across nearly the whole area, ln(0.10/0.03)=1.20 against
ln(0.10/0.06)=0.51. Since flow response scales with the log of z0, that alone
changed how sensitive a run appeared to land cover, independently of the model.

WORLDCOVER_Z0 has NOT had the same treatment and is still the original
analogy-assigned set. It is kept because WorldCover is the global fallback and
what FuXi-CFD was trained on, but it carries no more provenance than the NLCD
table that was just removed. Do not read it as a reference.

Caveat applying to any such table: a class -> z0 lookup is a coarse instrument.
Real z0 varies with crop stage, canopy height and season, none of which the class
encodes. Do not read the resulting field as ground truth.
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

#: NLCD class code -> roughness length z0 (m). Sourced from UL's table.
#:
#: These are external reference values, not assignments made here -- which is the
#: whole reason they replaced the repo's original set. Relative to those, this
#: table is floored higher on vegetated land (0.05-0.06 rather than 0.03-0.04),
#: rates developed open space far rougher (0.2 vs 0.05), and is somewhat rougher
#: in forest and wetland: a narrower dynamic range, and a much smaller
#: grass/shrub contrast.
NLCD_Z0 = {
    11: 0.0002,  # open water
    12: 0.003,   # perennial ice/snow
    21: 0.20,    # developed, open space
    22: 0.30,    # developed, low intensity
    23: 0.55,    # developed, medium intensity
    24: 0.75,    # developed, high intensity
    31: 0.01,    # barren land
    41: 0.90,    # deciduous forest
    42: 1.125,   # evergreen forest
    43: 0.90,    # mixed forest
    51: 0.05,    # dwarf scrub (AK)
    52: 0.10,    # shrub/scrub
    71: 0.06,    # grassland/herbaceous
    72: 0.06,    # sedge/herbaceous (AK)
    73: 0.01,    # lichens
    74: 0.01,    # moss
    81: 0.06,    # pasture/hay
    82: 0.06,    # cultivated crops
    90: 0.66,    # woody wetlands
    95: 0.15,    # emergent herbaceous wetlands
}

NLCD_CLASS_NAMES = {
    11: "open water", 12: "ice/snow",
    21: "developed, open space", 22: "developed, low intensity",
    23: "developed, medium intensity", 24: "developed, high intensity",
    31: "barren", 41: "deciduous forest", 42: "evergreen forest",
    43: "mixed forest", 51: "dwarf scrub", 52: "shrub/scrub",
    71: "grassland/herbaceous", 72: "sedge", 73: "lichens", 74: "moss",
    81: "pasture/hay", 82: "cultivated crops",
    90: "woody wetlands", 95: "emergent wetlands",
}

#: Registry so callers can stay source-agnostic.
Z0_TABLES = {"worldcover": None, "nlcd": NLCD_Z0}      # filled below
CLASS_NAME_TABLES = {"worldcover": None, "nlcd": NLCD_CLASS_NAMES}

CLASS_NAMES = {
    10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built-up", 60: "bare/sparse", 70: "snow/ice", 80: "water",
    90: "herbaceous wetland", 95: "mangroves", 100: "moss/lichen",
}


Z0_TABLES["worldcover"] = WORLDCOVER_Z0
CLASS_NAME_TABLES["worldcover"] = CLASS_NAMES


def table_fingerprint(source: str) -> str:
    """Short hash of a source's z0 values, for cache keys.

    Derived terrain (the normalized static tensor) is cached on disk, so editing
    any z0 value silently invalidates every cached tile -- and a key naming only
    the *source* cannot see that. Hashing the values themselves means a table
    edit auto-invalidates without anyone remembering to bump a version, which is
    exactly the failure that swapping in UL's table would otherwise have caused.
    """
    import hashlib

    table = Z0_TABLES.get(source)
    if table is None:
        raise ValueError(f"unknown land-cover source {source!r}")
    blob = ";".join(f"{k}={table[k]!r}" for k in sorted(table))
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def classes_to_z0(landcover: np.ndarray, source: str = "worldcover") -> np.ndarray:
    """Map a land-cover class raster to z0 in metres.

    `source` selects the lookup table: "worldcover" or "nlcd". The model consumes
    z0 in metres, not class codes, so mixing sources across tiles is physically
    coherent as long as each tile's mapping is right.
    """
    table = Z0_TABLES.get(source)
    if table is None:
        raise ValueError(f"unknown land-cover source {source!r}; expected {sorted(Z0_TABLES)}")
    codes = np.rint(np.asarray(landcover)).astype(np.int32)
    out = np.full(codes.shape, FALLBACK_Z0, dtype=np.float32)
    seen = set(np.unique(codes).tolist())
    for code, z0 in table.items():
        out[codes == code] = z0
    unknown = seen - set(table)
    if unknown:
        print(
            f"[terrain] unmapped {source} codes {sorted(unknown)} -> "
            f"z0={FALLBACK_Z0} m fallback"
        )
    return out


def class_histogram(landcover: np.ndarray, source: str = "worldcover") -> dict[str, float]:
    """Fractional land-cover composition, for the Phase 0 report."""
    names = CLASS_NAME_TABLES.get(source) or CLASS_NAMES
    codes = np.rint(np.asarray(landcover)).astype(np.int32)
    total = codes.size
    vals, counts = np.unique(codes, return_counts=True)
    return {
        names.get(int(v), f"code {int(v)}"): float(c) / total
        for v, c in sorted(zip(vals, counts), key=lambda p: -p[1])
    }


def despeckle(z0: np.ndarray, size: int = 3) -> np.ndarray:
    """Remove isolated single-pixel roughness spikes with a median filter.

    ESA WorldCover's "built-up" class (50) covers roads, farmsteads and towns
    alike. In rural farmland that produces *scattered isolated 30 m pixels* --
    83 of them across the Broadview tile, many in one-pixel-wide lines that are
    roads. Mapping them to the urban z0 of 0.8 m creates a 3.8-sigma jump across
    a single pixel against a grassland background sitting at -3.9 sigma.

    Two things are wrong with that input:

    * **The value.** A paved rural road is *smoother* than grassland (z0 ~ 0.005),
      not 27x rougher; a 30 m pixel holding one farmstead is not urban canopy.
    * **The geometry.** An isolated one-pixel spike is a spatial pattern the
      training set essentially never contains -- its built-up areas are
      contiguous towns, not speckle in an otherwise uniform field.

    A median filter removes lone spikes while leaving contiguous built-up areas
    intact, which is the standard treatment for roughness maps in microscale wind
    modelling. `size=3` clears single pixels and one-pixel-wide lines; larger
    sizes start eroding genuine small settlements.

    Off by default -- turn it on only with the A/B evidence to justify it.
    """
    from scipy.ndimage import median_filter

    return median_filter(np.asarray(z0), size=size, mode="nearest")
