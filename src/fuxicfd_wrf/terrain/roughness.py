"""Surface roughness length from land-cover classes.

FuXi-CFD wants z0 in metres on the 30 m grid; the land-cover products give
classes, so a lookup table bridges them.

**Provenance is uneven between the two tables -- read this before trusting a
number.**

NLCD_Z0 is now derived from WRF's own VEGPARM.TBL, so the static channel and the
WRF winds driving the model describe the same surface. It previously held UL's
table (kept as NLCD_Z0_UL), which was a sourced external reference but disagreed
sharply with WRF: 2.25x on evergreen forest, and more than 2x on domain-mean z0
across WSW (0.124 m against WRF's 0.055 m). Since flow response scales with the
log of z0, a table disagreement of that size changes how sensitive a run appears
to land cover independently of the model, which is exactly what it did.

Note this is a choice, not a correction. Matching WRF buys pipeline consistency
and costs distance from FuXi-CFD's training convention -- the model learned
roughness from WorldCover, where tree cover is 0.75 m, between WRF's 0.50 and
UL's 1.125. Validate against observations rather than assuming.

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

#: NLCD class code -> roughness length z0 (m), derived from WRF's own land-use
#: table so the static channel and the driving winds agree on the surface.
#:
#: Each value is the midpoint of Z0MIN/Z0MAX in the NLCD40 section of WRF's
#: VEGPARM.TBL, read through the NLCD -> NLCD40 category mapping used to build
#: the WPS geogrid data set. Two details of what WRF *actually* runs are folded in:
#:
#:   * Open water and perennial ice map onto the IGBP slots (NLCD40 17 and 15),
#:     matching the geogrid data set, so they take those categories' values.
#:   * Noah remaps NLCD40 categories 23 and 25 to 24 at init
#:     (module_physics_init.F:3268), so developed open space and medium intensity
#:     both run at developed-low's 0.70 m rather than their own 0.50 and 1.50.
#:
#: WRF interpolates between Z0MIN and Z0MAX by green vegetation fraction, which a
#: static channel cannot express. The midpoint is within ~10% of the true annual
#: mean (measured f_bar 0.41-0.48 over the WSW domain, so WRF runs slightly
#: smoother than the midpoint), and it is exact for the 14 classes where
#: Z0MIN == Z0MAX -- including all three forest classes, which is where the
#: disagreement mattered most.
#:
#: This replaces NLCD_Z0_UL (kept below). That table rated evergreen forest at
#: 1.125 m against WRF's 0.500 and FuXi-CFD's own WorldCover training value of
#: 0.750 -- an outlier against both references, which put a 5.1 sigma shrub->forest
#: contrast inside a single tile and drove visibly excessive sensitivity at
#: forested sites. Under this table that contrast falls to 2.3 sigma.
NLCD_Z0 = {
    11: 0.0001,  # open water                  <- NLCD40 17 (IGBP water)
    12: 0.001,   # perennial ice/snow          <- NLCD40 15 (permanent snow/ice)
    21: 0.70,    # developed, open space       <- 23, remapped to 24 by Noah
    22: 0.70,    # developed, low intensity    <- 24
    23: 0.70,    # developed, medium intensity <- 25, remapped to 24 by Noah
    24: 2.00,    # developed, high intensity   <- 26
    31: 0.01,    # barren land
    41: 0.50,    # deciduous forest
    42: 0.50,    # evergreen forest
    43: 0.35,    # mixed forest                <- midpoint of 0.20/0.50
    51: 0.025,   # dwarf scrub (AK)            <- midpoint of 0.01/0.04
    52: 0.03,    # shrub/scrub                 <- midpoint of 0.01/0.05
    71: 0.11,    # grassland/herbaceous        <- midpoint of 0.10/0.12
    72: 0.20,    # sedge/herbaceous (AK)
    73: 0.01,    # lichens
    74: 0.01,    # moss
    81: 0.10,    # pasture/hay                 <- midpoint of 0.05/0.15
    82: 0.06,    # cultivated crops            <- midpoint of 0.02/0.10
    90: 0.40,    # woody wetlands
    95: 0.20,    # emergent herbaceous wetlands
}

#: The previous table, from UL. Retained so the change is reversible and A/B-able
#: -- it is a sourced external reference, not an invented set, and departing from
#: it was a judgement call about pipeline consistency rather than a correction.
NLCD_Z0_UL = {
    11: 0.0002, 12: 0.003, 21: 0.20, 22: 0.30, 23: 0.55, 24: 0.75,
    31: 0.01, 41: 0.90, 42: 1.125, 43: 0.90, 51: 0.05, 52: 0.10,
    71: 0.06, 72: 0.06, 73: 0.01, 74: 0.01, 81: 0.06, 82: 0.06,
    90: 0.66, 95: 0.15,
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
