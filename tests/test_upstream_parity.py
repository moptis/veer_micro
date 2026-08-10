"""Assert our port of the FuXi-CFD contract reproduces the shipped example.

This is the gate that everything else rests on: if preprocessing, channel order,
or de-normalization drifted from upstream, every downstream number is wrong in a
way no physical check would obviously catch.

On tolerance
------------
Exact bit equality is not achievable -- the shipped ``prediction.npz`` was produced
on different hardware with a different onnxruntime build, and float32 conv kernels
are not bit-reproducible across either. What matters is that any discrepancy is
negligible *relative to the model's own error*, so the tolerance is expressed
against the measured FuXi-CFD-vs-CFD error floor rather than an arbitrary epsilon.

Measured on this cluster (onnxruntime 1.28, CPU):

    var   max|ours - theirs|   FuXi-CFD RMSE vs CFD truth
    u          1.7e-4                    0.152 m/s
    v          3.2e-4                    0.155 m/s
    w          5.4e-5                    0.061 m/s
    k          7.8e-5                    0.058 m2/s2

i.e. our reproduction error is ~500x below the model's own error floor.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuxicfd_wrf import constants
from fuxicfd_wrf.model import assets, postprocess, preprocess
from fuxicfd_wrf.model.session import FuxiCFDSession

#: FuXi-CFD RMSE against its CFD label, measured on the shipped example.
ERROR_FLOOR = {"u": 0.152, "v": 0.155, "w": 0.061, "k": 0.058}

#: Our discrepancy must be at most this fraction of FuXi-CFD's own RMSE.
#:
#: The CPU budget is tight because float32 CPU kernels are near-deterministic.
#: The CUDA budget is looser because H100 runs convolutions on TF32 tensor cores
#: by default, which carries ~10 mantissa bits instead of 23. Measured GPU
#: discrepancy is ~0.013 m/s for v, about 8% of the 0.155 error floor -- adding
#: in quadrature that takes total error from 0.1550 to 0.1555, i.e. nowhere.
#: Worth knowing about, not worth chasing.
MAX_FRACTION_OF_ERROR_FLOOR = {"CPUExecutionProvider": 0.01}
DEFAULT_FRACTION_OF_ERROR_FLOOR = 0.20


@pytest.fixture(scope="module")
def upstream():
    a = assets.resolve()
    missing = a.missing()
    if missing:
        pytest.skip(f"upstream assets not fetched: {missing}. Run: fuxicfd-wrf fetch-model")
    return a


def test_heights_match_shipped_asset(upstream):
    """Our hard-coded OUTPUT_HEIGHTS must equal height.npy[2:29].

    The 2:29 slice is undocumented in prose -- it comes from upstream's
    check.ipynb. This guards against both our transcription and an upstream change.
    """
    derived = constants.validate_heights_against_asset(upstream.height)
    assert derived.shape == (constants.N_LEVELS,)
    assert derived[0] == 10.0
    assert derived[-1] == pytest.approx(300.795452)


def test_paper_height_claim_is_wrong(upstream):
    """The paper says the stack reaches 214 m; the shipped code says 300.8 m.

    Documented as a test so the discrepancy stays visible rather than becoming
    folklore. If a future release makes the paper right, this fails loudly.
    """
    assert constants.OUTPUT_HEIGHTS[-1] > 300.0


@pytest.fixture(scope="module")
def replayed(upstream):
    in_stats = upstream.load_input_scaler()
    out_stats = upstream.load_output_scaler()
    z = np.load(upstream.example_inputs)
    x = preprocess.build_model_input({k: z[k] for k in z.files}, in_stats)
    session = FuxiCFDSession(upstream.onnx, device="cpu")
    raw = session.run(x[None])
    return postprocess.denormalize_and_split(raw, out_stats), session


def test_model_input_shape(upstream):
    in_stats = upstream.load_input_scaler()
    z = np.load(upstream.example_inputs)
    x = preprocess.build_model_input({k: z[k] for k in z.files}, in_stats)
    assert x.shape == (4, *constants.MODEL_SHAPE)
    assert x.dtype == np.float32


def test_batch_axis_is_dynamic(replayed):
    """Batching on the GPU depends on this; the shipped config implies it is fixed."""
    _, session = replayed
    assert session.dynamic_batch, (
        "ONNX batch axis is pinned -- pipeline will fall back to a slow 1-at-a-time loop"
    )


def test_matches_shipped_prediction(replayed, upstream):
    ours, session = replayed
    theirs = np.load(upstream.example_prediction)
    fraction = MAX_FRACTION_OF_ERROR_FLOOR.get(
        session.active_provider, DEFAULT_FRACTION_OF_ERROR_FLOOR
    )
    for var in constants.OUTPUT_VARS:
        a, b = ours[var], theirs[var]
        assert a.shape == b.shape == (constants.N_LEVELS, *constants.MODEL_SHAPE)
        max_diff = float(np.abs(a - b).max())
        budget = fraction * ERROR_FLOOR[var]
        assert max_diff < budget, (
            f"{var}: max|diff|={max_diff:.2e} exceeds {budget:.2e} "
            f"({fraction:.0%} of the {ERROR_FLOOR[var]} error floor, "
            f"provider={session.active_provider}). That is too large to be float "
            "nondeterminism -- suspect a contract change."
        )


def test_error_floor_against_cfd_label(replayed, upstream):
    """Pin FuXi-CFD's own accuracy so downstream errors can be attributed.

    The shipped label is (300, 300, 27) and needs transposing to (27, 300, 300),
    which upstream's notebook does as `np.transpose(label, (2,0,1))[:, :300, :300]`.
    """
    ours, _ = replayed
    label = np.load(upstream.example_outputs)
    for var in constants.OUTPUT_VARS:
        truth = np.transpose(label[var], (2, 0, 1))[:, :300, :300]
        rmse = float(np.sqrt(np.mean((ours[var] - truth) ** 2)))
        assert rmse == pytest.approx(ERROR_FLOOR[var], abs=0.02), (
            f"{var}: RMSE vs CFD truth is {rmse:.4f}, expected ~{ERROR_FLOOR[var]}. "
            "The model or assets changed."
        )


def test_postprocess_axis_order_is_level_var(upstream):
    """Output is (level, var, y, x). Swapping those two silently yields nonsense."""
    out_stats = upstream.load_output_scaler()
    fake = np.zeros((constants.N_LEVELS, len(constants.OUTPUT_VARS), 4, 4), dtype=np.float32)
    # Mark var index 2 (w) at every level.
    fake[:, 2] = 1.0
    split = postprocess.split(postprocess.denormalize(fake, out_stats))
    mean = np.asarray(out_stats["mean"])
    std = np.asarray(out_stats["std"])
    assert np.allclose(split["w"][:, 0, 0], std[:, 2] + mean[:, 2])
    assert np.allclose(split["u"][:, 0, 0], mean[:, 0])
