"""Tests for the sector transfer algebra.

These cover the parts that are pure maths and can be pinned exactly. The
end-to-end path needs the ONNX asset and is exercised by the smoke config.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gamma
from scipy.stats import weibull_min

from fuxicfd_wrf.constants import COARSE_SHAPE, MODEL_SHAPE
from fuxicfd_wrf.model import preprocess
from fuxicfd_wrf.transfer import (
    apply_transfer, combine_sectors, fit_weibull_moments, met_direction,
    power_density, sector_transfer, upsample_coarse, weibull_moments, wrap180,
)


def test_upsample_matches_preprocessing():
    """`upsample_coarse` must reproduce what the model actually ingested.

    Preprocessing normalises then zooms; both are affine, so zooming the raw
    winds gives exactly the de-normalised field the network saw. If that ever
    stops being true the speedup denominator is wrong and every S is wrong with
    it, so pin it against the real preprocessing path rather than asserting the
    algebra by eye.
    """
    rng = np.random.default_rng(0)
    u9 = rng.normal(size=COARSE_SHAPE) * 3 + 8
    v9 = rng.normal(size=COARSE_SHAPE) * 3 - 2
    stats = {"low_mean": np.array([1.5, -0.5]), "low_std": np.array([4.0, 3.0])}

    normed = preprocess.normalize_winds(u9, v9, stats)     # (2, 300, 300)
    recovered = normed * np.asarray(stats["low_std"])[:, None, None] + \
        np.asarray(stats["low_mean"])[:, None, None]

    assert np.allclose(recovered[0], upsample_coarse(u9), atol=1e-9)
    assert np.allclose(recovered[1], upsample_coarse(v9), atol=1e-9)


def test_upsample_batched_matches_single():
    rng = np.random.default_rng(1)
    batch = rng.normal(size=(5, *COARSE_SHAPE))
    out = upsample_coarse(batch)
    assert out.shape == (5, *MODEL_SHAPE)
    for i in range(5):
        assert np.allclose(out[i], upsample_coarse(batch[i]))


def test_uniform_inflow_gives_unit_speedup():
    """A spatially uniform coarse field upsamples to itself, so S == 1."""
    u9 = np.full(COARSE_SHAPE, -7.0 * np.sin(np.deg2rad(270.0)))
    v9 = np.full(COARSE_SHAPE, -7.0 * np.cos(np.deg2rad(270.0)))
    u_fine = upsample_coarse(u9)[None]
    v_fine = upsample_coarse(v9)[None]

    tr = sector_transfer(u_fine, v_fine, u9[None], v9[None])
    assert np.allclose(tr.speedup, 1.0, atol=1e-9)
    assert np.allclose(tr.dtheta, 0.0, atol=1e-9)
    assert np.allclose(tr.speed_coarse, 7.0)


def test_speedup_and_deflection_recovered():
    """Scale and rotate a known field; S and dtheta must come back out."""
    u9 = np.full(COARSE_SHAPE, 0.0)
    v9 = np.full(COARSE_SHAPE, -10.0)          # northerly, 0 deg
    scale, turn = 1.25, 20.0

    up_u, up_v = upsample_coarse(u9), upsample_coarse(v9)
    a = np.deg2rad(turn)
    # Rotating the *direction* by +turn rotates the vector the same way.
    u_fine = scale * (up_u * np.cos(a) - up_v * np.sin(a))
    v_fine = scale * (up_u * np.sin(a) + up_v * np.cos(a))

    tr = sector_transfer(u_fine[None], v_fine[None], u9[None], v9[None])
    assert np.allclose(tr.speedup, scale, atol=1e-9)
    assert np.allclose(np.abs(tr.dtheta), turn, atol=1e-9)


def test_wrap180_branch_cut():
    assert wrap180(190.0) == pytest.approx(-170.0)
    assert wrap180(-190.0) == pytest.approx(170.0)
    assert wrap180(180.0) == pytest.approx(180.0)
    assert wrap180(0.0) == pytest.approx(0.0)


def test_met_direction_conventions():
    # Wind from the west: blows toward +x, so u > 0 and direction is 270.
    assert met_direction(np.array(5.0), np.array(0.0)) == pytest.approx(270.0)
    # From the north: v < 0, direction 0.
    assert met_direction(np.array(0.0), np.array(-5.0)) == pytest.approx(0.0)


# --- Weibull algebra ---------------------------------------------------------


@pytest.mark.parametrize("k_true", [0.8, 1.5, 2.0, 2.4, 3.5, 6.0])
def test_moment_fit_roundtrip(k_true):
    """Fitting a single Weibull's own moments returns its parameters."""
    A_true = 7.3
    m1, m3 = weibull_moments(np.array(A_true), np.array(k_true))
    A, k = fit_weibull_moments(m1, m3)
    assert float(k) == pytest.approx(k_true, rel=1e-6)
    assert float(A) == pytest.approx(A_true, rel=1e-6)


def test_moment_fit_vectorised():
    rng = np.random.default_rng(7)
    A_true = rng.uniform(4, 12, size=(3, 40))
    k_true = rng.uniform(1.2, 3.5, size=(3, 40))
    m1, m3 = weibull_moments(A_true, k_true)
    A, k = fit_weibull_moments(m1, m3)
    assert np.allclose(k, k_true, rtol=1e-5)
    assert np.allclose(A, A_true, rtol=1e-5)


def test_moment_fit_handles_invalid():
    A, k = fit_weibull_moments(
        np.array([5.0, np.nan, 0.0]), np.array([300.0, 1.0, 1.0])
    )
    assert np.isfinite(A[0]) and np.isfinite(k[0])
    assert np.isnan(A[1]) and np.isnan(k[1])
    assert np.isnan(A[2]) and np.isnan(k[2])


def test_combine_sectors_of_identical_sectors_is_identity():
    """A mixture of identical Weibulls is that Weibull."""
    A = np.full((12, 4, 4), 8.0)
    k = np.full((12, 4, 4), 2.1)
    freq = np.full((12, 4, 4), 1.0 / 12)
    A_c, k_c = combine_sectors(freq, A, k)
    assert np.allclose(A_c, 8.0, rtol=1e-6)
    assert np.allclose(k_c, 2.1, rtol=1e-6)


def test_combine_sectors_matches_sampled_mixture():
    """The combined distribution reproduces the mixture's mean and cube-mean."""
    rng = np.random.default_rng(11)
    A = np.array([6.0, 9.0, 11.0])[:, None, None]
    k = np.array([1.7, 2.3, 2.9])[:, None, None]
    freq = np.array([0.5, 0.3, 0.2])[:, None, None]

    sample = np.concatenate([
        weibull_min.rvs(k[i, 0, 0], scale=A[i, 0, 0],
                        size=int(400_000 * freq[i, 0, 0]), random_state=rng)
        for i in range(3)
    ])
    A_c, k_c = combine_sectors(freq, A, k)
    m1_fit = float(A_c) * gamma(1 + 1 / float(k_c))
    m3_fit = float(A_c) ** 3 * gamma(1 + 3 / float(k_c))

    assert m1_fit == pytest.approx(sample.mean(), rel=0.01)
    assert m3_fit == pytest.approx((sample**3).mean(), rel=0.03)


def test_power_density_is_analytic_third_moment():
    A, k, rho = np.array(8.0), np.array(2.0), 1.06
    assert float(power_density(A, k, rho)) == pytest.approx(
        0.5 * rho * 8.0**3 * gamma(1 + 3 / 2.0)
    )


# --- applying a transfer -----------------------------------------------------


def _flat_climate(n_sec=12, ny=5, nx=6, A=8.0, k=2.0):
    return (
        np.full((n_sec, ny, nx), A),
        np.full((n_sec, ny, nx), k),
        np.full((n_sec, ny, nx), 1.0 / n_sec),
    )


def test_apply_transfer_identity():
    """Fine speed equal to the coarse mean and no deflection changes nothing."""
    A_c, k_c, f_c = _flat_climate()
    speed = A_c * gamma(1 + 1 / k_c)
    A_f, k_f, f_f = apply_transfer(A_c, k_c, f_c, speed, np.zeros_like(speed))
    assert np.allclose(A_f, A_c)
    assert np.allclose(k_f, k_c)
    assert np.allclose(f_f, f_c)


def test_apply_transfer_scales_A_not_k():
    A_c, k_c, f_c = _flat_climate()
    speed = 1.3 * A_c * gamma(1 + 1 / k_c)
    A_f, k_f, _ = apply_transfer(A_c, k_c, f_c, speed, np.zeros_like(speed))
    assert np.allclose(A_f, 1.3 * A_c)
    assert np.allclose(k_f, k_c)


def test_deflection_moves_frequency_a_whole_bin():
    """A deflection of exactly one bin shifts every sector by one index."""
    n_sec = 12
    A_c, k_c, f_c = _flat_climate(n_sec=n_sec)
    # Make the sectors distinguishable by their frequency.
    f_c = np.zeros_like(f_c)
    f_c[3] = 1.0
    speed = A_c * gamma(1 + 1 / k_c)
    dth = np.full_like(speed, 30.0)             # exactly one 30 deg bin

    _, _, f_f = apply_transfer(A_c, k_c, f_c, speed, dth, deflect=True)
    assert np.allclose(f_f[4], 1.0)
    assert np.allclose(f_f[3], 0.0)


def test_deflection_below_half_bin_moves_nothing():
    A_c, k_c, f_c = _flat_climate()
    speed = A_c * gamma(1 + 1 / k_c)
    _, _, f_f = apply_transfer(
        A_c, k_c, f_c, speed, np.full_like(speed, 14.0), deflect=True
    )
    assert np.allclose(f_f, f_c)


def test_deflection_conserves_total_frequency():
    rng = np.random.default_rng(3)
    A_c, k_c, f_c = _flat_climate()
    speed = A_c * gamma(1 + 1 / k_c)
    dth = rng.uniform(-60, 60, size=speed.shape)
    _, _, f_f = apply_transfer(A_c, k_c, f_c, speed, dth, deflect=True)
    assert np.allclose(f_f.sum(axis=0), f_c.sum(axis=0))


def test_deflect_false_leaves_frequency_alone():
    A_c, k_c, f_c = _flat_climate()
    speed = A_c * gamma(1 + 1 / k_c)
    _, _, f_f = apply_transfer(
        A_c, k_c, f_c, speed, np.full_like(speed, 45.0), deflect=False
    )
    assert np.allclose(f_f, f_c)


def test_nan_deflection_does_not_scatter_weight():
    """A NaN deflection must not become a bogus sector index.

    Points outside tile coverage carry NaN dtheta. `np.rint(nan).astype(int64)`
    is undefined behaviour and yields INT64_MIN on x86-64, which survives `% 12`
    as a plausible-looking index -- silently scattering that point's frequency
    into an arbitrary sector instead of leaving it alone.
    """
    A_c, k_c, f_c = _flat_climate(ny=3, nx=3)
    speed = A_c * gamma(1 + 1 / k_c)
    dth = np.zeros_like(speed)
    dth[:, 1, 1] = np.nan

    _, _, f_f = apply_transfer(A_c, k_c, f_c, speed, dth, deflect=True)
    assert np.all(np.isfinite(f_f))
    # The NaN point is treated as zero deflection, so it keeps its own sectors.
    assert np.allclose(f_f[:, 1, 1], f_c[:, 1, 1])


def test_nan_frequency_does_not_poison_combine():
    """A coarse point outside the source WRG must not contaminate the mixture."""
    A = np.full((12, 2, 2), 8.0)
    k = np.full((12, 2, 2), 2.0)
    freq = np.full((12, 2, 2), 1.0 / 12)
    freq[3, 0, 0] = np.nan

    A_c, k_c = combine_sectors(freq, A, k)
    assert np.isfinite(A_c[0, 0]) and np.isfinite(k_c[0, 0])
    assert A_c[0, 0] == pytest.approx(8.0, rel=1e-6)


def test_apply_transfer_rejects_shape_mismatch():
    A_c, k_c, f_c = _flat_climate()
    with pytest.raises(ValueError, match="share one shape"):
        apply_transfer(A_c, k_c, f_c, np.zeros((12, 5, 7)), np.zeros((12, 5, 7)))
