"""Tests for high-order methods and DRP schemes.

This module tests:
- Fornberg weight computation
- DRP coefficient optimization
- Dispersion analysis functions
- DRP wave equation solvers (requires Devito)
"""

# Check if optional dependencies are available
import importlib.util

import numpy as np
import pytest

from src.highorder.dispersion import (
    analytical_dispersion_relation,
    cfl_number,
    critical_dt,
    dispersion_difference,
    dispersion_error,
    dispersion_ratio,
    fornberg_weights,
    max_frequency_ricker,
    nyquist_spacing,
    ricker_wavelet,
)
from src.highorder.drp_devito import (
    DRP_COEFFICIENTS,
    drp_coefficients,
    to_full_stencil,
)

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None


class TestFornbergWeights:
    """Tests for Fornberg finite difference weight computation."""

    def test_3point_stencil(self):
        """Test 3-point stencil (M=1) gives standard coefficients."""
        weights = fornberg_weights(M=1)
        expected = np.array([-2.0, 1.0])
        np.testing.assert_allclose(weights, expected, rtol=1e-10)

    def test_5point_stencil(self):
        """Test 5-point stencil (M=2) gives standard coefficients."""
        weights = fornberg_weights(M=2)
        expected = np.array([-5/2, 4/3, -1/12])
        np.testing.assert_allclose(weights, expected, rtol=1e-10)

    def test_7point_stencil(self):
        """Test 7-point stencil (M=3) gives standard coefficients."""
        weights = fornberg_weights(M=3)
        expected = np.array([-49/18, 3/2, -3/20, 1/90])
        np.testing.assert_allclose(weights, expected, rtol=1e-10)

    def test_9point_stencil(self):
        """Test 9-point stencil (M=4) gives standard coefficients."""
        weights = fornberg_weights(M=4)
        expected = np.array([-205/72, 8/5, -1/5, 8/315, -1/560])
        np.testing.assert_allclose(weights, expected, rtol=1e-10)

    def test_consistency_constraint(self):
        """Test that weights satisfy a_0 + 2*sum(a_m) = 0."""
        for M in [1, 2, 3, 4, 5]:
            weights = fornberg_weights(M)
            total = weights[0] + 2 * np.sum(weights[1:])
            assert abs(total) < 1e-10, f"M={M}: a_0 + 2*sum(a_m) = {total}"

    def test_second_order_constraint(self):
        """Test that weights satisfy sum(a_m * m^2) = 1."""
        for M in [1, 2, 3, 4, 5]:
            weights = fornberg_weights(M)
            total = np.sum([weights[m] * m**2 for m in range(M + 1)])
            assert abs(total - 1) < 1e-10, f"M={M}: sum(a_m * m^2) = {total}"

    def test_invalid_M(self):
        """Test that invalid M raises an error."""
        with pytest.raises(ValueError):
            fornberg_weights(M=0)


class TestDRPCoefficients:
    """Tests for DRP coefficient retrieval."""

    def test_drp_coefficients_available(self):
        """Test that DRP coefficients are available for M=2,3,4,5."""
        for M in [2, 3, 4, 5]:
            weights = drp_coefficients(M, use_fornberg=False)
            assert len(weights) == M + 1

    def test_fornberg_coefficients_available(self):
        """Test that Fornberg coefficients are available for M=2,3,4,5."""
        for M in [2, 3, 4, 5]:
            weights = drp_coefficients(M, use_fornberg=True)
            assert len(weights) == M + 1

    def test_drp_consistency_constraint(self):
        """Test that DRP weights satisfy a_0 + 2*sum(a_m) = 0."""
        for M in DRP_COEFFICIENTS.keys():
            weights = drp_coefficients(M, use_fornberg=False)
            total = weights[0] + 2 * np.sum(weights[1:])
            assert abs(total) < 1e-5, f"M={M}: a_0 + 2*sum(a_m) = {total}"

    def test_drp_second_order_constraint(self):
        """Test that DRP weights satisfy sum(a_m * m^2) = 1."""
        for M in DRP_COEFFICIENTS.keys():
            weights = drp_coefficients(M, use_fornberg=False)
            total = np.sum([weights[m] * m**2 for m in range(M + 1)])
            assert abs(total - 1) < 1e-5, f"M={M}: sum(a_m * m^2) = {total}"

    def test_invalid_M(self):
        """Test that invalid M raises an error."""
        with pytest.raises(ValueError):
            drp_coefficients(M=10)


class TestFullStencil:
    """Tests for conversion to full stencil format."""

    def test_symmetric_conversion(self):
        """Test conversion from symmetric to full stencil."""
        symmetric = np.array([-2.5, 1.33, -0.08])
        full = to_full_stencil(symmetric)

        expected = np.array([-0.08, 1.33, -2.5, 1.33, -0.08])
        np.testing.assert_allclose(full, expected)

    def test_stencil_length(self):
        """Test that full stencil has correct length."""
        for M in [2, 3, 4, 5]:
            symmetric = fornberg_weights(M)
            full = to_full_stencil(symmetric)
            assert len(full) == 2 * M + 1


class TestDispersionAnalysis:
    """Tests for dispersion analysis functions."""

    def test_analytical_dispersion(self):
        """Test analytical dispersion relation omega = c*k."""
        c = 1500.0
        k = 0.1
        omega = analytical_dispersion_relation(k, c)
        assert omega == pytest.approx(c * k)

    def test_dispersion_ratio_zero_k(self):
        """Test that dispersion ratio is 1 for k=0."""
        weights = fornberg_weights(M=4)
        ratio = dispersion_ratio(weights, h=10.0, dt=0.001, v=1500.0, k=0.0)
        assert ratio == 1.0

    def test_dispersion_ratio_small_k(self):
        """Test that dispersion ratio is close to 1 for small k."""
        weights = fornberg_weights(M=4)
        ratio = dispersion_ratio(weights, h=10.0, dt=0.001, v=1500.0, k=0.01)
        assert abs(ratio - 1.0) < 0.01

    def test_dispersion_difference_zero_k(self):
        """Test that dispersion difference is 0 for k=0."""
        weights = fornberg_weights(M=4)
        diff = dispersion_difference(weights, h=10.0, dt=0.001, v=1500.0, k=0.0)
        assert diff == 0.0

    def test_dispersion_error_positive(self):
        """Test that dispersion error is non-negative."""
        weights = fornberg_weights(M=4)
        error = dispersion_error(weights, h=10.0, dt=0.001, v=1500.0, k_max=0.2)
        assert error >= 0.0


class TestCFLCondition:
    """Tests for CFL stability condition computations."""

    def test_critical_dt_positive(self):
        """Test that critical dt is positive."""
        weights = fornberg_weights(M=4)
        dt_crit = critical_dt(weights, h=10.0, v_max=4500.0)
        assert dt_crit > 0

    def test_critical_dt_scaling_with_h(self):
        """Test that critical dt scales linearly with h."""
        weights = fornberg_weights(M=4)
        dt1 = critical_dt(weights, h=10.0, v_max=4500.0)
        dt2 = critical_dt(weights, h=20.0, v_max=4500.0)
        assert dt2 == pytest.approx(2 * dt1, rel=1e-10)

    def test_critical_dt_scaling_with_v(self):
        """Test that critical dt scales inversely with v_max."""
        weights = fornberg_weights(M=4)
        dt1 = critical_dt(weights, h=10.0, v_max=4500.0)
        dt2 = critical_dt(weights, h=10.0, v_max=9000.0)
        assert dt1 == pytest.approx(2 * dt2, rel=1e-10)

    def test_cfl_number_positive(self):
        """Test that CFL number is positive."""
        weights = fornberg_weights(M=4)
        cfl = cfl_number(weights)
        assert cfl > 0

    def test_cfl_number_less_than_one(self):
        """Test that CFL number is less than 1 (typical for wave equations)."""
        weights = fornberg_weights(M=4)
        cfl = cfl_number(weights, ndim=2)
        assert cfl < 1.0


class TestRickerWavelet:
    """Tests for Ricker wavelet generation."""

    def test_peak_at_1_over_f0(self):
        """Test that wavelet peaks near t = 1/f0."""
        f0 = 30.0
        t = np.linspace(0, 0.1, 1000)
        wavelet = ricker_wavelet(t, f0=f0)

        # Find peak location
        peak_idx = np.argmax(wavelet)
        peak_time = t[peak_idx]

        # Should be close to 1/f0
        expected_peak = 1.0 / f0
        assert abs(peak_time - expected_peak) < 0.001

    def test_amplitude_scaling(self):
        """Test that amplitude parameter scales correctly."""
        t = np.linspace(0, 0.1, 100)
        w1 = ricker_wavelet(t, f0=30.0, A=1.0)
        w2 = ricker_wavelet(t, f0=30.0, A=2.0)

        np.testing.assert_allclose(w2, 2 * w1)

    def test_max_frequency_positive(self):
        """Test that max frequency estimate is positive."""
        f_max = max_frequency_ricker(f0=30.0)
        assert f_max > 0

    def test_max_frequency_greater_than_f0(self):
        """Test that max frequency is greater than peak frequency."""
        f0 = 30.0
        f_max = max_frequency_ricker(f0)
        assert f_max > f0


class TestNyquistSpacing:
    """Tests for Nyquist spacing computation."""

    def test_nyquist_positive(self):
        """Test that Nyquist spacing is positive."""
        h_max = nyquist_spacing(f_max=100.0, v_min=1500.0)
        assert h_max > 0

    def test_nyquist_formula(self):
        """Test Nyquist formula: h = v_min / (2 * f_max)."""
        f_max = 100.0
        v_min = 1500.0
        h_max = nyquist_spacing(f_max, v_min)
        expected = v_min / (2 * f_max)
        assert h_max == pytest.approx(expected)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")
class TestDRPOptimization:
    """Tests for DRP coefficient optimization."""

    def test_compute_drp_weights(self):
        """Test that DRP optimization runs successfully."""
        from src.highorder.drp_devito import compute_drp_weights

        weights = compute_drp_weights(M=4)
        assert len(weights) == 5

    def test_optimized_weights_satisfy_constraints(self):
        """Test that optimized weights satisfy required constraints."""
        from src.highorder.drp_devito import compute_drp_weights

        weights = compute_drp_weights(M=4)

        # Consistency: a_0 + 2*sum(a_m) = 0
        total = weights[0] + 2 * np.sum(weights[1:])
        assert abs(total) < 1e-5

        # Second-order: sum(a_m * m^2) = 1
        total2 = np.sum([weights[m] * m**2 for m in range(len(weights))])
        assert abs(total2 - 1) < 1e-5


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not available")
@pytest.mark.devito
class TestDRPSolvers:
    """Tests for DRP wave equation solvers (requires Devito)."""

    def test_solve_wave_drp_1d_runs(self):
        """Test that 1D DRP solver runs without error."""
        from src.highorder.drp_devito import solve_wave_drp_1d

        result = solve_wave_drp_1d(
            L=1000.0,
            Nx=101,
            velocity=1500.0,
            f0=30.0,
            t_end=0.1,
            dt=0.0005,
            use_drp=True,
        )

        assert result.u is not None
        assert len(result.u) == 101
        assert result.t_final == 0.1
        assert result.use_drp is True

    def test_solve_wave_drp_2d_runs(self):
        """Test that 2D DRP solver runs without error."""
        from src.highorder.drp_devito import solve_wave_drp

        result = solve_wave_drp(
            extent=(1000., 1000.),
            shape=(51, 51),
            velocity=1500.,
            f0=30.,
            t_end=0.1,
            dt=0.0005,
            use_drp=True,
        )

        assert result.u is not None
        assert result.u.shape == (51, 51)
        assert result.t_final == 0.1
        assert result.use_drp is True

    def test_solve_wave_drp_fornberg_vs_drp(self):
        """Test that Fornberg and DRP give different results."""
        from src.highorder.drp_devito import solve_wave_drp

        result_fornberg = solve_wave_drp(
            extent=(1000., 1000.),
            shape=(51, 51),
            velocity=1500.,
            f0=30.,
            t_end=0.1,
            dt=0.0005,
            use_drp=False,
        )

        result_drp = solve_wave_drp(
            extent=(1000., 1000.),
            shape=(51, 51),
            velocity=1500.,
            f0=30.,
            t_end=0.1,
            dt=0.0005,
            use_drp=True,
        )

        # Results should be different (different weights)
        diff = np.linalg.norm(result_drp.u - result_fornberg.u)
        assert diff > 0

    def test_compare_dispersion_wavefields(self):
        """Test comparison function returns two results."""
        from src.highorder.drp_devito import compare_dispersion_wavefields

        result_fornberg, result_drp = compare_dispersion_wavefields(
            extent=(500., 500.),
            shape=(31, 31),
            velocity=1500.,
            f0=30.,
            t_end=0.05,
            dt=0.0003,
        )

        assert result_fornberg.use_drp is False
        assert result_drp.use_drp is True
        assert result_fornberg.u.shape == result_drp.u.shape

    def test_wavefield_norm_reasonable(self):
        """Test that wavefield norm is reasonable (not NaN or Inf)."""
        from src.highorder.drp_devito import solve_wave_drp

        result = solve_wave_drp(
            extent=(1000., 1000.),
            shape=(51, 51),
            velocity=1500.,
            f0=30.,
            t_end=0.1,
            dt=0.0005,
            use_drp=True,
        )

        norm = np.linalg.norm(result.u)
        assert np.isfinite(norm)
        assert norm > 0  # Should have some wavefield


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not available")
@pytest.mark.devito
class TestCustomWeightsInDevito:
    """Tests for using custom weights in Devito."""

    def test_custom_weights_applied(self):
        """Test that custom weights can be applied to derivatives."""
        from devito import Grid, TimeFunction

        grid = Grid(shape=(11,), extent=(10.,))
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=4)

        # Custom weights (5-point stencil)
        weights = np.array([-2.5, 1.33, -0.08, 1.33, -2.5])  # Not physically correct, just for test

        # This should not raise an error
        u_xx = u.dx2(weights=weights)
        assert u_xx is not None

    def test_drp_weights_in_equation(self):
        """Test that DRP weights can be used in a Devito equation."""
        from devito import Grid, TimeFunction

        grid = Grid(shape=(21, 21), extent=(20., 20.))
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=8)

        # Get DRP weights and convert to full stencil
        weights = drp_coefficients(M=4, use_fornberg=False)
        full_weights = to_full_stencil(weights)

        # Create Laplacian with custom weights
        laplacian = u.dx2(weights=full_weights) + u.dy2(weights=full_weights)

        # This should create a valid expression
        assert laplacian is not None
