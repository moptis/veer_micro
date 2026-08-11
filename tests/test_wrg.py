"""WRG export tests.

The load-bearing one is `test_weibull_matches_scipy`: the whole export rests on a
hand-written vectorised MLE replacing `scipy.stats.weibull_min.fit`, and if those
disagree every A and k in the file is wrong in a way nothing downstream would
catch.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import weibull_min

from fuxicfd_wrf.wrg import (
    DEFAULT_AIR_DENSITY, SectorHistograms, assign_sector, build_records,
    nearest_model_level, power_density, sector_edges, weibull_mle_from_hist,
    write_wrg,
)


def _hist_from_samples(x, bin_w=0.05, max_speed=50.0):
    """Fine-binned histogram, so binning error does not confound the comparison."""
    nb = int(round(max_speed / bin_w))
    idx = np.clip((x / bin_w).astype(int), 0, nb - 1)
    return np.bincount(idx, minlength=nb)[None, :].astype(float), (np.arange(nb) + 0.5) * bin_w


@pytest.mark.parametrize("true_k,true_A", [(1.8, 8.0), (2.0, 10.0), (2.6, 6.5), (1.4, 12.0)])
def test_weibull_matches_scipy(true_k, true_A):
    """Our vectorised MLE must agree with scipy's fit on the same data."""
    rng = np.random.default_rng(42)
    x = weibull_min.rvs(true_k, loc=0, scale=true_A, size=60_000, random_state=rng)
    x = x[x > 0]

    k_ref, _, A_ref = weibull_min.fit(x, floc=0)
    counts, centers = _hist_from_samples(x)
    A_ours, k_ours = weibull_mle_from_hist(counts, centers)

    assert A_ours[0] == pytest.approx(A_ref, rel=2e-3), f"A: {A_ours[0]} vs {A_ref}"
    assert k_ours[0] == pytest.approx(k_ref, rel=2e-3), f"k: {k_ours[0]} vs {k_ref}"


def test_weibull_recovers_known_parameters():
    rng = np.random.default_rng(7)
    x = weibull_min.rvs(2.1, loc=0, scale=9.0, size=200_000, random_state=rng)
    counts, centers = _hist_from_samples(x)
    A, k = weibull_mle_from_hist(counts, centers)
    assert A[0] == pytest.approx(9.0, rel=0.02)
    assert k[0] == pytest.approx(2.1, rel=0.02)


def test_weibull_handles_empty_and_single_sample_points():
    """Sparse sectors must yield NaN rather than garbage or an exception."""
    centers = (np.arange(100) + 0.5) * 0.5
    counts = np.zeros((3, 100))
    counts[1, 10] = 1          # single sample
    counts[2, 10] = 500        # degenerate: all mass in one bin
    A, k = weibull_mle_from_hist(counts, centers)
    assert np.isnan(A[0]) and np.isnan(k[0])
    assert np.isnan(A[1]) and np.isnan(k[1])
    assert np.isfinite(A[2])


def test_weibull_vectorises_over_many_points():
    """Many points at once must equal fitting each individually.

    Not bit-identical: the fixed-point iteration stops when the *worst* point in
    the block is converged, so an individual point may take one extra step when it
    is fitted alone. The gap is at the iteration tolerance (~1e-9 relative), i.e.
    the ninth decimal place of k -- physically meaningless, since k is reported to
    two decimals in the file.
    """
    centers = (np.arange(200) + 0.5) * 0.25
    counts = np.stack([
        weibull_min.pdf(centers, k, scale=A) * 10000
        for k, A in [(1.7, 7.0), (2.3, 9.5), (2.9, 11.0)]
    ])
    A_all, k_all = weibull_mle_from_hist(counts, centers)
    for i in range(3):
        A_i, k_i = weibull_mle_from_hist(counts[i : i + 1], centers)
        assert A_all[i] == pytest.approx(A_i[0], rel=1e-6)
        assert k_all[i] == pytest.approx(k_i[0], rel=1e-6)


# --- sectors -----------------------------------------------------------------

def test_sector_edges_match_reference_convention():
    """First sector centred on north, 12 sectors for 30 deg bins."""
    e = sector_edges(30.0)
    assert e[0] == -15.0 and e[-1] == 345.0
    assert len(e) - 1 == 12


@pytest.mark.parametrize(
    "direction,expected",
    [(0.0, 0), (5.0, 0), (355.0, 0), (350.0, 0), (30.0, 1), (180.0, 6), (270.0, 9)],
)
def test_assign_sector_wraps_north(direction, expected):
    assert int(assign_sector(np.array([direction]), 30.0)[0]) == expected


# --- histograms --------------------------------------------------------------

def test_histogram_accumulates_counts_and_moments():
    h = SectorHistograms(n_points=2, n_sectors=12)
    speed = np.array([[5.0, 10.0], [7.0, 3.0]])
    direction = np.array([[0.0, 90.0], [0.0, 90.0]])
    h.add(speed, direction)
    assert h.n_samples.tolist() == [2, 2]
    assert h.counts[0, 0].sum() == 2      # point 0 both samples in sector 0
    assert h.counts[1, 3].sum() == 2      # point 1 both samples in sector 3 (90 deg)
    assert h.sum_u[0] == pytest.approx(12.0)
    assert h.sum_u3[0] == pytest.approx(5.0**3 + 7.0**3)


def test_histogram_ignores_nan_and_zero_speed():
    h = SectorHistograms(n_points=1, n_sectors=12)
    h.add(np.array([[np.nan], [0.0], [8.0]]), np.array([[0.0], [0.0], [0.0]]))
    assert h.n_samples[0] == 1


# --- power density -----------------------------------------------------------

def test_standard_power_density_uses_mean_of_cubes():
    """0.5*rho*<U^3>, not 0.5*rho*<U>^3 -- they differ substantially."""
    h = SectorHistograms(n_points=1, n_sectors=12)
    speeds = np.array([[4.0], [8.0], [12.0]])
    h.add(speeds, np.zeros_like(speeds))
    pd_std = power_density(h, air_density=1.225)[0]
    expected = 0.5 * 1.225 * np.mean([4.0**3, 8.0**3, 12.0**3])
    assert pd_std == pytest.approx(expected)

    mean_cube = 0.5 * 1.225 * np.mean([4.0, 8.0, 12.0]) ** 3
    assert pd_std > mean_cube  # Jensen: <U^3> > <U>^3


# --- levels ------------------------------------------------------------------

def test_100m_is_an_exact_level():
    """100 m IS a FuXi-CFD level (index 19), so no interpolation is needed.

    This asserted the opposite until the paper's supplementary corrected the
    height grid.
    """
    j, h = nearest_model_level(100.0)
    assert j == 19
    assert h == pytest.approx(100.0)


# --- file format -------------------------------------------------------------

def test_write_wrg_format(tmp_path):
    rng = np.random.default_rng(0)
    n_sec, npts = 12, 6
    hist = SectorHistograms(npts, n_sec)
    speeds = weibull_min.rvs(2.0, scale=9.0, size=(500, npts), random_state=rng)
    dirs = rng.uniform(0, 360, size=(500, npts))
    hist.add(speeds, dirs)

    rec = build_records(hist, elevation=np.full(npts, 1340.0), hub_height=102.17944)
    x = np.repeat([0.0, 100.0, 200.0], 2)
    y = np.tile([0.0, 100.0], 3)
    out = write_wrg(tmp_path / "t.wrg", x, y, rec, resolution_m=100.0)

    lines = out.read_text().strip().split("\n")
    header = lines[0].split()
    assert len(header) == 5
    assert header[0] == "3" and header[1] == "2"     # nx, ny
    assert header[4] == "100"                        # cell size

    assert len(lines) == 1 + npts
    fields = lines[1].split()
    # GridPoint X Y elev hub A k pdens nsec + 3*12
    assert fields[0] == "GridPoint"
    assert len(fields) == 3 + 6 + 3 * n_sec
    assert int(fields[3]) == 1340                    # elevation
    assert int(fields[8]) == n_sec                   # sector count
    # Sector frequencies are scaled by 1000 and should sum to ~1000.
    freqs = [int(fields[9 + 3 * s]) for s in range(n_sec)]
    assert sum(freqs) == pytest.approx(1000, abs=15)


def test_wrg_records_have_no_nan_in_integer_columns(tmp_path):
    """A sector with no data must not write 'nan' into an integer field."""
    hist = SectorHistograms(2, 12)
    # All wind from due north -- 11 of 12 sectors stay empty.
    hist.add(np.full((300, 2), 9.0) + np.random.default_rng(1).normal(0, 1, (300, 2)),
             np.zeros((300, 2)))
    rec = build_records(hist, elevation=np.zeros(2), hub_height=102.17944)
    out = write_wrg(tmp_path / "s.wrg", np.array([0.0, 100.0]), np.array([0.0, 0.0]),
                    rec, resolution_m=100.0)
    text = out.read_text()
    assert "nan" not in text.lower()
