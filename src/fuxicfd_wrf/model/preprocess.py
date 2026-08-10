"""Build the (B, 4, 300, 300) ONNX input tensor.

This is a batched port of the upstream ``inference_example/utils/preprocessing.py``.
It must stay numerically identical to it -- ``tests/test_upstream_parity.py``
asserts that by replaying the shipped example.

Upstream, verbatim:

    dem_rough = np.stack([dem, rough], axis=0)                    # (2,301,301)
    uv_100m   = np.stack([u100, v100], axis=0)                    # (2,9,9)
    dem_rough = (dem_rough - high_mean) / high_std
    uv_100m   = (uv_100m - low_mean) / low_std
    dem_rough = zoom(dem_rough, (1, 300/301, 300/301), order=1)
    uv_100m   = zoom(uv_100m,   (1, 300/9,   300/9),   order=1)
    x = np.concatenate([uv_100m, dem_rough], axis=0).astype(np.float32)

Two things worth knowing about that ``zoom`` call:

* Normalization happens *before* resampling. Since both are affine and zoom is
  linear, order does not change the result -- but we keep upstream's order so any
  future non-affine change upstream stays easy to mirror.
* ``scipy.ndimage.zoom`` defaults to ``grid_mode=False``, which aligns the *corner
  pixel centres* of input and output (an "align_corners=True" style mapping), not
  the cell edges. The 9 -> 300 upsample therefore spans exactly the 9 km between
  the outer coarse cell centres. Any hand-rolled replacement must match that, which
  is why we call scipy rather than reimplementing.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from ..constants import COARSE_SHAPE, MODEL_SHAPE, TERRAIN_SHAPE

_H, _W = MODEL_SHAPE
_TERRAIN_ZOOM = (1, _H / TERRAIN_SHAPE[0], _W / TERRAIN_SHAPE[1])
_COARSE_ZOOM = (1, _H / COARSE_SHAPE[0], _W / COARSE_SHAPE[1])


def normalize_static(dem: np.ndarray, roughness: np.ndarray, in_stats: dict) -> np.ndarray:
    """Normalize and resample the static channels to (2, 300, 300).

    These do not vary with time, so the pipeline computes this once per tile and
    caches the result -- it is the single biggest saving across a long time series.
    """
    if dem.shape != TERRAIN_SHAPE or roughness.shape != TERRAIN_SHAPE:
        raise ValueError(
            f"dem/roughness must be {TERRAIN_SHAPE}, got {dem.shape}/{roughness.shape}"
        )
    stacked = np.stack([dem, roughness], axis=0)
    mean = np.asarray(in_stats["high_mean"])[:, None, None]
    std = np.asarray(in_stats["high_std"])[:, None, None]
    stacked = (stacked - mean) / std
    return zoom(stacked, _TERRAIN_ZOOM, order=1)


def normalize_winds(u100: np.ndarray, v100: np.ndarray, in_stats: dict) -> np.ndarray:
    """Normalize and resample the coarse wind channels to (..., 2, 300, 300).

    Accepts either a single (9, 9) pair or a batch of (B, 9, 9).
    """
    u100 = np.asarray(u100)
    v100 = np.asarray(v100)
    if u100.shape != v100.shape:
        raise ValueError(f"u/v shape mismatch: {u100.shape} vs {v100.shape}")

    batched = u100.ndim == 3
    if u100.shape[-2:] != COARSE_SHAPE:
        raise ValueError(f"winds must end in {COARSE_SHAPE}, got {u100.shape}")

    mean = np.asarray(in_stats["low_mean"])[:, None, None]
    std = np.asarray(in_stats["low_std"])[:, None, None]

    if not batched:
        stacked = (np.stack([u100, v100], axis=0) - mean) / std
        return zoom(stacked, _COARSE_ZOOM, order=1)

    # Zoom per-sample rather than adding a batch axis to the zoom factors. A
    # factor-1 spline axis is identity for order=1, but doing it per-sample keeps
    # this provably identical to the single-sample path we validate against.
    stacked = (np.stack([u100, v100], axis=1) - mean) / std  # (B,2,9,9)
    return np.stack([zoom(s, _COARSE_ZOOM, order=1) for s in stacked], axis=0)


def assemble(winds: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Concatenate to the mandatory channel order [u_100m, v_100m, dem, roughness].

    `winds` is (2,300,300) or (B,2,300,300); `static` is (2,300,300) and is
    broadcast across the batch.
    """
    if winds.ndim == 3:
        return np.concatenate([winds, static], axis=0).astype(np.float32)
    static_b = np.broadcast_to(static, (winds.shape[0], *static.shape))
    return np.concatenate([winds, static_b], axis=1).astype(np.float32)


def build_model_input(input_dict: dict, in_stats: dict) -> np.ndarray:
    """Single-sample path matching upstream ``build_model_input`` exactly.

    Kept as the reference implementation that parity tests compare against.
    """
    static = normalize_static(input_dict["dem"], input_dict["roughness"], in_stats)
    winds = normalize_winds(input_dict["u_100m"], input_dict["v_100m"], in_stats)
    return assemble(winds, static)
