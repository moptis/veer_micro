"""Fixed properties of the FuXi-CFD model contract and the Broadview WRF domain.

Everything here was verified against the shipped inference code at
https://huggingface.co/linchensen/FuXi-CFD-model, not against the model card or
the paper -- those disagree with the code in places (see OUTPUT_HEIGHTS).
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
#: Derived as ``height.npy[2:29]``. This slice is not documented anywhere in prose;
#: it comes from the shipped ``inference_example/scripts/check.ipynb``, which does::
#:
#:     heights = list(np.load("../utils/height.npy")[2:27 + 2])
#:     index_height = heights.index(height)
#:
#: The paper states 27 levels "extending to 214 m", which contradicts this (the
#: stack reaches 300.8 m). The code is authoritative because it is what produced
#: the shipped predictions. ``validate_heights_against_asset`` re-derives this from
#: the downloaded height.npy so the hard-coded copy can never silently drift.
#: Below ~50 m the spacing is a uniform 5 m; above it the grid stretches
#: geometrically by ~1.1x per level, reaching 300.8 m.
OUTPUT_HEIGHTS = np.array(
    [
        10.0, 15.0, 20.0,
        25.0, 30.0, 35.0,
        40.0, 45.0, 50.0,
        55.5, 61.55, 68.205,
        75.5255, 83.57805, 92.435855,
        102.17944, 112.897385, 124.687123,
        137.655835, 151.921419, 167.613561,
        184.874917, 203.862408, 224.748649,
        247.723514, 272.995866, 300.795452,
    ],
    dtype=np.float64,
)

#: Slice into the full 56-level CFD height stack that yields OUTPUT_HEIGHTS.
HEIGHT_SLICE = slice(2, 2 + N_LEVELS)


def validate_heights_against_asset(height_npy_path) -> np.ndarray:
    """Re-derive the output heights from the shipped height.npy and check our copy.

    Guards against the hard-coded OUTPUT_HEIGHTS drifting from the asset if
    upstream ever republishes it.
    """
    full = np.load(height_npy_path)
    derived = np.asarray(full[HEIGHT_SLICE], dtype=np.float64)
    if derived.shape != (N_LEVELS,):
        raise ValueError(
            f"height.npy[{HEIGHT_SLICE}] gave {derived.shape}, expected ({N_LEVELS},). "
            f"Upstream asset changed shape (full stack is {full.shape})."
        )
    if not np.allclose(derived, OUTPUT_HEIGHTS):
        raise ValueError(
            "OUTPUT_HEIGHTS disagrees with height.npy asset.\n"
            f"  asset:  {derived}\n"
            f"  ours:   {OUTPUT_HEIGHTS}"
        )
    return derived


# --- Broadview WRF domain ----------------------------------------------------

#: Grid shape of the WRF d03 domain, from the control.h5 root attr `domain_shape`.
#: Flat variable arrays are (n_time, 10296) and reshape to this as (row, col).
WRF_GRID_SHAPE = (78, 132)
WRF_N_POINTS = WRF_GRID_SHAPE[0] * WRF_GRID_SHAPE[1]  # 10296

#: Measured spacing (geodesic, mid-domain). Note this is ~800 m, NOT the 1 km the
#: model expects -- winds must be resampled onto a true 1 km stencil.
WRF_DX_M = 802.0
WRF_DY_M = 798.4

#: UTM zone 13N. Broadview (~103.0 W) sits near the zone central meridian, so
#: grid convergence stays small across the domain.
WORKING_CRS = "EPSG:32613"
GEOGRAPHIC_CRS = "EPSG:4326"

#: WRF heights (m) with wind speed/direction available in control.h5. Used by the
#: Phase 0 profile check as an independent reference for the FuXi-CFD stack.
WRF_WIND_HEIGHTS = (20, 60, 80, 100, 120, 160, 200, 250, 300)

# --- Training distribution (from scaler_input.npy) ---------------------------

#: Recorded here only so the distribution-shift report can run without loading the
#: model assets. The pipeline always reads the real scalers from disk.
TRAIN_DEM_MEAN, TRAIN_DEM_STD = 351.9218, 227.5572
TRAIN_ROUGHNESS_MEAN, TRAIN_ROUGHNESS_STD = 0.8173, 0.2019
