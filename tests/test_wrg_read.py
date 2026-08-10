"""Round-trip and real-file tests for the WRG reader.

The reader exists so the comparison notebook can load the coarse 1 km
deliverables and FuXi-CFD output on one footing. Those coarse files use **16**
sectors while `create_wrg.py` defaults to 12, so sector count must be read per
file, never assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import weibull_min

from fuxicfd_wrf.wrg import SectorHistograms, build_records, write_wrg
from fuxicfd_wrf.wrg_read import (
    read_wrg, to_grid, weibull_mean_speed, weibull_power_density,
)

COARSE_DIR = Path("/scratch/moptis/c2wind/broadview/control/deliverables/wrg/broadview")


def test_weibull_mean_matches_sampled_mean():
    rng = np.random.default_rng(0)
    x = weibull_min.rvs(2.2, scale=9.0, size=400_000, random_state=rng)
    assert weibull_mean_speed(9.0, 2.2) == pytest.approx(x.mean(), rel=5e-3)


def test_weibull_power_density_matches_sampled_cube():
    rng = np.random.default_rng(1)
    x = weibull_min.rvs(2.0, scale=8.0, size=400_000, random_state=rng)
    expected = 0.5 * 1.225 * np.mean(x**3)
    assert weibull_power_density(8.0, 2.0, 1.225) == pytest.approx(expected, rel=1e-2)


def test_power_density_is_not_mean_speed_cubed():
    """Guards the convention: <U^3> exceeds <U>^3 substantially."""
    A, k = 8.8, 2.5
    proper = weibull_power_density(A, k, 1.225)
    naive = 0.5 * 1.225 * weibull_mean_speed(A, k) ** 3
    assert proper > 1.3 * naive


def test_roundtrip_write_then_read(tmp_path):
    rng = np.random.default_rng(3)
    npts, n_sec = 12, 12
    hist = SectorHistograms(npts, n_sec)
    hist.add(
        weibull_min.rvs(2.0, scale=9.0, size=(2000, npts), random_state=rng),
        rng.uniform(0, 360, size=(2000, npts)),
    )
    rec = build_records(hist, elevation=np.full(npts, 1300.0), hub_height=100.0)
    x = np.repeat([0.0, 100.0, 200.0, 300.0], 3)
    y = np.tile([0.0, 100.0, 200.0], 4)
    path = write_wrg(tmp_path / "rt.wrg", x, y, rec, resolution_m=100.0)

    df = read_wrg(path)
    assert len(df) == npts
    assert df.attrs["n_sectors"] == n_sec
    assert df.attrs["cell_size"] == 100.0
    assert np.allclose(df["elevation"], 1300.0)
    # A and k survive their 2-decimal / scaled encodings.
    assert df["A"].mean() == pytest.approx(rec[:, 2].mean(), abs=0.05)
    assert df["k"].mean() == pytest.approx(rec[:, 3].mean(), abs=0.05)
    # Sector frequencies sum to ~1 after unscaling.
    freqs = np.stack([df[f"freq_{i}"].values for i in range(n_sec)]).sum(axis=0)
    assert np.allclose(freqs, 1.0, atol=0.02)


def test_to_grid_pivots_north_up(tmp_path):
    rng = np.random.default_rng(4)
    npts = 6
    hist = SectorHistograms(npts, 12)
    hist.add(
        weibull_min.rvs(2.0, scale=9.0, size=(500, npts), random_state=rng),
        rng.uniform(0, 360, size=(500, npts)),
    )
    rec = build_records(hist, elevation=np.zeros(npts), hub_height=100.0)
    x = np.repeat([0.0, 100.0, 200.0], 2)
    y = np.tile([0.0, 100.0], 3)
    df = read_wrg(write_wrg(tmp_path / "g.wrg", x, y, rec, resolution_m=100.0))
    gx, gy, grid = to_grid(df, "mean_speed")
    assert grid.shape == (2, 3)
    assert gy[0] > gy[-1], "y axis must descend (north-up)"
    assert np.isfinite(grid).all()


# --- real coarse deliverables -------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    not COARSE_DIR.exists(), reason="coarse WRG deliverables not present"
)


@pytestmark_real
@pytest.mark.parametrize("height", [80, 100, 120, 160])
def test_reads_real_coarse_deliverable(height):
    df = read_wrg(COARSE_DIR / f"control_{height}m.wrg", air_density=1.06)
    a = df.attrs
    assert a["cell_size"] == 1000.0
    assert a["n_sectors"] == 16, "Broadview coarse files use 16 sectors, not 12"
    assert len(df) == a["nx"] * a["ny"]
    assert (df["hub_height"] == height).all()
    assert 5.0 < df["mean_speed"].mean() < 15.0
    assert 1.0 < df["k"].mean() < 4.0
    # Sector frequencies must sum to ~1 per point.
    freqs = np.stack([df[f"freq_{i}"].values for i in range(a["n_sectors"])]).sum(axis=0)
    assert np.allclose(freqs, 1.0, atol=0.05)


@pytestmark_real
def test_coarse_speed_increases_with_height():
    """Physical sanity across the four deliverable heights."""
    means = [
        read_wrg(COARSE_DIR / f"control_{h}m.wrg")["mean_speed"].mean()
        for h in (80, 100, 120, 160)
    ]
    assert all(b > a for a, b in zip(means, means[1:])), means
