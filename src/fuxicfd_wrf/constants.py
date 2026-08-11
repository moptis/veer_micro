"""Fixed properties of the FuXi-CFD model contract and the Broadview WRF domain.

Verified against the shipped inference code at
https://huggingface.co/linchensen/FuXi-CFD-model AND the paper's supplementary
information. Where they appear to disagree, check carefully which is actually
authoritative -- OUTPUT_HEIGHTS documents a case where trusting an example
notebook over the supplementary produced an 86 m error.
"""

from __future__ import annotations

import numpy as np

# --- FuXi-CFD input contract -------------------------------------------------

#: Coarse wind grid: 9x9 points at 1 km spacing.
COARSE_SHAPE = (9, 9)
COARSE_RES_M = 1000.0

#: Terrain grid: 301x301 nodes at 30 m spacing (300 intervals -> exactly 9000 m).
TERRAIN_SHAPE = (301, 301)
TERRAIN_RES_M = 30.0

#: Both grids span the same 9 km x 9 km footprint.
TILE_SPAN_M = 9000.0

#: Shape the model actually consumes, after the order-1 zoom in preprocessing.
MODEL_SHAPE = (300, 300)

#: ONNX input channel order. The model is not permutation invariant -- this order
#: is mandatory (normalization/README.md says "must").
INPUT_CHANNELS = ("u_100m", "v_100m", "dem", "roughness")

#: ONNX output variable order within the level dimension.
OUTPUT_VARS = ("u", "v", "w", "k")

N_LEVELS = 27

# --- Vertical levels ---------------------------------------------------------

#: The 27 output heights in m AGL.
#:
#: Source of truth is the paper's Supplementary Note 1.1, which states them
#: explicitly: "The first 20 levels are uniformly spaced every 5 m from 5 m to
#: 100 m... The exact height levels (in meters) are: 5, 10, ..., 100, 106.5,
#: 114.95, 125.94, 140.22, 158.78, 182.91, 214.29." The dataset README for
#: outputs.npz lists the identical values, and the paper body's "extending to
#: 214 m" agrees.
#:
#: DO NOT derive these from the shipped ``utils/height.npy``. That is the CFD
#: solver's full 56-level grid reaching 4 km, not the 27 saved output levels. An
#: earlier version of this file used ``height.npy[2:29]``, giving 10 -> 300.795 m,
#: wrong by up to 86 m: it labelled the true 100 m level as 151.9 m and the true
#: 80 m level as 102.2 m. That came from ``inference_example/scripts/check.ipynb``,
#: which slices height.npy that way and looks up 45.0 m -- a value sitting at
#: index 7 under the bad slice and index 8 under the true grid, so the notebook
#: silently plots the wrong level and nothing in the shipped example reveals it.
OUTPUT_HEIGHTS = np.array(
    [
        5.0, 10.0, 15.0, 20.0, 25.0,
        30.0, 35.0, 40.0, 45.0, 50.0,
        55.0, 60.0, 65.0, 70.0, 75.0,
        80.0, 85.0, 90.0, 95.0, 100.0,
        106.5, 114.95, 125.94, 140.22,
        158.78, 182.91, 214.29,
    ],
    dtype=np.float64,
)

#: Hub heights that are exact model levels and need no interpolation:
#: 80 m is index 15, 100 m is index 19. 120 m and 160 m still interpolate.
EXACT_HUB_HEIGHTS = (80.0, 100.0)


def validate_heights(height_npy_path=None) -> np.ndarray:
    """Sanity-check OUTPUT_HEIGHTS against the documented grid.

    `height_npy_path` is accepted only to assert that the asset is still the
    56-level solver grid, i.e. that nobody has gone back to deriving output
    heights from it.
    """
    if OUTPUT_HEIGHTS.shape != (N_LEVELS,):
        raise ValueError(f"OUTPUT_HEIGHTS is {OUTPUT_HEIGHTS.shape}, expected ({N_LEVELS},)")
    if not np.allclose(OUTPUT_HEIGHTS[:20], np.arange(1, 21) * 5.0):
        raise ValueError("first 20 levels must be a uniform 5 m ladder from 5 to 100 m")
    if OUTPUT_HEIGHTS[-1] != 214.29:
        raise ValueError(f"top level is {OUTPUT_HEIGHTS[-1]}, expected 214.29 m")
    if height_npy_path is not None:
        full = np.load(height_npy_path)
        if full.size == N_LEVELS:
            raise ValueError(
                "height.npy now has 27 entries; re-check whether it is the output grid."
            )
    return OUTPUT_HEIGHTS


#: Deprecated alias kept so existing callers keep working.
validate_heights_against_asset = validate_heights


# --- Broadview WRF domain ----------------------------------------------------

#: Default grid shape (Broadview d03). Other domains differ -- WSW is (96, 114) --
#: so the real shape is always read from the store's `domain_shape` root attr.
#: These remain only as fallbacks and for tests.
WRF_GRID_SHAPE = (78, 132)
WRF_N_POINTS = WRF_GRID_SHAPE[0] * WRF_GRID_SHAPE[1]  # 10296

#: Measured spacing (geodesic, mid-domain). Note this is ~800 m, NOT the 1 km the
#: model expects -- winds must be resampled onto a true 1 km stencil.
WRF_DX_M = 802.0
WRF_DY_M = 798.4

#: Default working CRS (UTM 13N). `utm_crs_for` picks the right zone for any
#: domain; hard-coding one zone silently distorts geometry elsewhere.
WORKING_CRS = "EPSG:32613"
GEOGRAPHIC_CRS = "EPSG:4326"


def utm_crs_for(lon, lat=None) -> str:
    """EPSG code of the UTM zone containing `lon` (mean if an array).

    Tiles are laid out in this CRS, so using a zone far from the domain would
    stretch the 9 km footprint. Broadview and WSW both land in 13N, but that is
    luck rather than design.
    """
    import numpy as np

    lon_c = float(np.mean(lon))
    zone = int((lon_c + 180.0) // 6.0) + 1
    north = True if lat is None else float(np.mean(lat)) >= 0
    return f"EPSG:{326 if north else 327}{zone:02d}"

#: WRF heights (m) with wind speed/direction available in control.h5. Used by the
#: Phase 0 profile check as an independent reference for the FuXi-CFD stack.
WRF_WIND_HEIGHTS = (20, 60, 80, 100, 120, 160, 200, 250, 300)

# --- Training distribution (from scaler_input.npy) ---------------------------

#: Recorded here only so the distribution-shift report can run without loading the
#: model assets. The pipeline always reads the real scalers from disk.
TRAIN_DEM_MEAN, TRAIN_DEM_STD = 351.9218, 227.5572
TRAIN_ROUGHNESS_MEAN, TRAIN_ROUGHNESS_STD = 0.8173, 0.2019
