"""De-normalize raw ONNX output back to physical units.

Batched port of upstream ``inference_example/utils/postprocessing.py``:

    if pred.ndim == 5: pred = pred[0]
    mean = out_stats["mean"][:, :, None, None]   # (27,4,1,1)
    std  = out_stats["std"][:, :, None, None]
    pred = pred * std + mean                     # (27,4,300,300)
    u, v, w, k = pred[:,0], pred[:,1], pred[:,2], pred[:,3]

Note the axis order inside the model output: it is (level, var, y, x), *not*
(var, level, y, x). Getting this backwards silently produces plausible-looking
garbage, since both axes are being indexed with small integers.
"""

from __future__ import annotations

import numpy as np

from ..constants import N_LEVELS, OUTPUT_VARS


def denormalize(pred: np.ndarray, out_stats: dict) -> np.ndarray:
    """Scale raw output to physical units.

    Accepts (27,4,H,W) or (B,27,4,H,W); returns the same rank.
    """
    pred = np.asarray(pred)
    mean = np.asarray(out_stats["mean"])
    std = np.asarray(out_stats["std"])
    if mean.shape != (N_LEVELS, len(OUTPUT_VARS)):
        raise ValueError(
            f"output scaler mean is {mean.shape}, expected "
            f"({N_LEVELS}, {len(OUTPUT_VARS)})"
        )

    if pred.ndim == 4:
        return pred * std[:, :, None, None] + mean[:, :, None, None]
    if pred.ndim == 5:
        return pred * std[None, :, :, None, None] + mean[None, :, :, None, None]
    raise ValueError(f"expected 4-D or 5-D prediction, got shape {pred.shape}")


def denormalize_levels(
    pred: np.ndarray, out_stats: dict, lo: int, hi: int
) -> np.ndarray:
    """De-normalize only source levels ``[lo, hi)``.

    Identical to ``denormalize(pred, out_stats)[..., lo:hi, :, :, :]`` but without
    ever materialising the other 25-odd levels in float64.

    That matters because the model's head emits all 27 levels unconditionally
    while a WRG needs one, and de-normalising promotes float32 to float64: a
    batch of 8 is 311 MB raw and 622 MB de-normalised, against 23 MB for the two
    levels an interpolated hub height actually touches. Pair it with
    ``LevelSelection.support``, which reports exactly that range.
    """
    pred = np.asarray(pred)
    mean = np.asarray(out_stats["mean"])
    std = np.asarray(out_stats["std"])
    if mean.shape != (N_LEVELS, len(OUTPUT_VARS)):
        raise ValueError(
            f"output scaler mean is {mean.shape}, expected "
            f"({N_LEVELS}, {len(OUTPUT_VARS)})"
        )
    if not 0 <= lo < hi <= N_LEVELS:
        raise ValueError(f"level slice [{lo}, {hi}) is outside [0, {N_LEVELS})")

    m, s = mean[lo:hi], std[lo:hi]
    if pred.ndim == 4:
        return pred[lo:hi] * s[:, :, None, None] + m[:, :, None, None]
    if pred.ndim == 5:
        return pred[:, lo:hi] * s[None, :, :, None, None] + m[None, :, :, None, None]
    raise ValueError(f"expected 4-D or 5-D prediction, got shape {pred.shape}")


def split(pred: np.ndarray) -> dict[str, np.ndarray]:
    """Split the variable axis into a dict of named fields.

    (27,4,H,W)   -> each value (27,H,W)
    (B,27,4,H,W) -> each value (B,27,H,W)
    """
    pred = np.asarray(pred)
    var_axis = -3
    if pred.shape[var_axis] != len(OUTPUT_VARS):
        raise ValueError(
            f"expected {len(OUTPUT_VARS)} variables on axis {var_axis}, "
            f"got shape {pred.shape}"
        )
    return {name: np.take(pred, i, axis=var_axis) for i, name in enumerate(OUTPUT_VARS)}


def denormalize_and_split(pred: np.ndarray, out_stats: dict) -> dict[str, np.ndarray]:
    """De-normalize then split. Drops a leading singleton batch, like upstream."""
    pred = np.asarray(pred)
    if pred.ndim == 5 and pred.shape[0] == 1:
        pred = pred[0]
    return split(denormalize(pred, out_stats))
