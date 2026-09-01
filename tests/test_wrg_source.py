"""Tests for driving the model from a coarse .wrg.

The orientation checks matter more than they look. A `.wrg` lists points in
column-major order while every field in this package is row-major north-up, and
a transposed pivot is silently plausible -- it produces a grid of the right shape
holding the right values in the wrong places. `wrg_export` documents a 37 m RMS
elevation error from exactly that mistake.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gamma

from fuxicfd_wrf.sources.wrg_source import (
    WrgSectorSource, sector_centers, weibull_mean, weibull_quantile,
)
from fuxicfd_wrf.wrg import write_wrg

CRS = "EPSG:32613"


def _write_coarse(
    path, *, nx=6, ny=4, n_sec=12, res=1000.0, x0=400_000.0, y0=3_800_000.0,
    A_of=None, hub=100.0,
):
    """Write a synthetic coarse .wrg in the column-major order the format uses.

    `A_of(ix, iy)` supplies the all-sector A so a test can plant a known spatial
    pattern and check it survives the round trip in the right orientation.
    """
    A_of = A_of or (lambda ix, iy: 8.0)
    xs = x0 + np.arange(nx) * res
    ys = y0 + np.arange(ny) * res

    rows, X, Y = [], [], []
    # Column-major: x outer, y inner -- what `_reorder_f` produces.
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            A = A_of(ix, iy)
            rec = [float(100 + ix + 10 * iy), hub, A, 2.0, 400.0, n_sec]
            for s in range(n_sec):
                rec += [1000.0 / n_sec, A * 10.0, 200.0]
            rows.append(rec)
            X.append(x)
            Y.append(y)
    write_wrg(path, np.array(X), np.array(Y), np.array(rows), resolution_m=res)
    return xs, ys


def test_grid_orientation_round_trip(tmp_path):
    """A planted x/y pattern must come back the right way up, not transposed."""
    p = tmp_path / "coarse.wrg"
    _write_coarse(p, nx=6, ny=4, A_of=lambda ix, iy: 5.0 + ix + 100.0 * iy)

    src = WrgSectorSource(p, crs=CRS)
    assert src.domain_shape == (4, 6)          # (ny, nx)

    # Row 0 is northmost, i.e. the largest y, i.e. the largest iy.
    A0 = src.A[0]
    assert A0[0, 0] == pytest.approx(5.0 + 0 + 100.0 * 3)
    assert A0[0, 5] == pytest.approx(5.0 + 5 + 100.0 * 3)
    assert A0[3, 0] == pytest.approx(5.0 + 0 + 100.0 * 0)

    # Coordinates must agree: y descends down the rows.
    assert src.y[0] > src.y[-1]
    assert np.all(np.diff(src.x) > 0)


def test_frequencies_renormalised(tmp_path):
    """freq is stored as an integer per mille, so it must be renormalised."""
    p = tmp_path / "coarse.wrg"
    _write_coarse(p, n_sec=7)                  # 1000/7 does not divide evenly
    src = WrgSectorSource(p, crs=CRS)
    assert np.allclose(src.freq.sum(axis=0), 1.0)


def test_states_and_labels(tmp_path):
    p = tmp_path / "coarse.wrg"
    _write_coarse(p, n_sec=12)
    src = WrgSectorSource(p, crs=CRS)
    assert src.n_states == 12
    assert src.n_sectors == 12
    assert src.bin_size_deg == pytest.approx(30.0)
    assert np.allclose(src.centers, np.arange(12) * 30.0)
    assert src.state_labels[0].startswith("sec00")

    src3 = WrgSectorSource(p, crs=CRS, speed_bins=3)
    assert src3.n_states == 36
    assert np.array_equal(src3.state_sector(np.arange(6)), [0, 0, 0, 1, 1, 1])


def test_read_uv_direction_convention(tmp_path):
    """Sector 0 is northerly: u == 0 and v < 0."""
    p = tmp_path / "coarse.wrg"
    _write_coarse(p, n_sec=12, A_of=lambda ix, iy: 8.0)
    src = WrgSectorSource(p, crs=CRS)

    u, v = src.read_uv(np.array([0]))
    expected = weibull_mean(np.array(8.0), np.array(2.0))
    assert np.allclose(u[0], 0.0, atol=1e-9)
    assert np.allclose(v[0], -expected)

    # Sector 3 of 12 is 90 deg -- easterly, blowing toward -x.
    u3, v3 = src.read_uv(np.array([3]))
    assert np.allclose(u3[0], -expected)
    assert np.allclose(v3[0], 0.0, atol=1e-9)


def test_speed_bins_are_equal_probability(tmp_path):
    """Bin midpoint quantiles must average back to roughly the Weibull mean."""
    p = tmp_path / "coarse.wrg"
    _write_coarse(p)
    src = WrgSectorSource(p, crs=CRS, speed_bins=40)

    speeds = np.stack([src.state_speed_field(b) for b in range(40)])
    mean = weibull_mean(np.array(8.0), np.array(2.0))
    assert speeds.mean(axis=0) == pytest.approx(float(mean), rel=0.01)


def test_single_bin_is_the_sector_mean(tmp_path):
    p = tmp_path / "coarse.wrg"
    _write_coarse(p)
    src = WrgSectorSource(p, crs=CRS)
    expected = 8.0 * gamma(1 + 1 / 2.0)
    assert np.allclose(src.state_speed_field(0), expected)


def test_hub_height_warning(tmp_path, capsys):
    """A coarse file cut at 80 m is not the 100 m field the model expects."""
    p = tmp_path / "coarse80.wrg"
    _write_coarse(p, hub=80.0)
    WrgSectorSource(p, crs=CRS)
    assert "100 m" in capsys.readouterr().out


def test_rejects_non_rectangular(tmp_path):
    p = tmp_path / "holey.wrg"
    _write_coarse(p, nx=4, ny=3)
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n")     # drop one point
    with pytest.raises(ValueError, match="rectangular"):
        WrgSectorSource(p, crs=CRS)


def test_weibull_quantile_matches_median():
    A, k = np.array(8.0), np.array(2.0)
    median = A * np.log(2.0) ** (1.0 / k)
    assert float(weibull_quantile(A, k, 0.5)) == pytest.approx(float(median))


def test_sector_centers():
    assert np.allclose(sector_centers(12), np.arange(12) * 30.0)
    assert np.allclose(sector_centers(16), np.arange(16) * 22.5)
