"""Tests for ADER (Arbitrary-order-accuracy via DERivatives) schemes.

This module tests:
- ADER helper functions (graddiv, gradlap, etc.)
- ADER 2D acoustic solver
- CFL advantage of ADER over leapfrog
- Wavefield properties (stability, energy bounds)
"""

import importlib.util

import numpy as np
import pytest

# Check if dependencies are available
SYMPY_AVAILABLE = importlib.util.find_spec("sympy") is not None
DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None


class TestRickerWavelet:
    """Tests for Ricker wavelet generation."""

    def test_ricker_wavelet_peak(self):
        """Test that wavelet peaks near t = 1/f0."""
        from src.highorder.ader_devito import ricker_wavelet

        f0 = 0.020  # 20 Hz in kHz, so 1/f0 = 50 ms
        t = np.linspace(0, 100., 2000)  # Time in ms to match f0 units
        wavelet = ricker_wavelet(t, f0=f0)

        # Find peak location
        peak_idx = np.argmax(wavelet)
        peak_time = t[peak_idx]

        # Should be close to 1/f0 = 50 ms
        expected_peak = 1.0 / f0
        assert abs(peak_time - expected_peak) < 1.0  # Within 1 ms

    def test_ricker_wavelet_amplitude(self):
        """Test that amplitude scaling works correctly."""
        from src.highorder.ader_devito import ricker_wavelet

        t = np.linspace(0, 0.2, 1000)
        w1 = ricker_wavelet(t, f0=0.020, A=1.0)
        w2 = ricker_wavelet(t, f0=0.020, A=2.0)

        np.testing.assert_allclose(w2, 2 * w1)

    def test_ricker_wavelet_zero_at_edges(self):
        """Test that wavelet decays to near-zero at edges."""
        from src.highorder.ader_devito import ricker_wavelet

        f0 = 0.020
        t = np.linspace(0, 0.3, 1000)
        wavelet = ricker_wavelet(t, f0=f0)

        # Wavelet should be small at early and late times
        assert abs(wavelet[-1]) < 0.01


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
class TestADERHelperFunctions:
    """Tests for ADER helper functions."""

    def test_graddiv_returns_matrix(self):
        """Test that graddiv returns a SymPy matrix."""
        pytest.importorskip("devito")
        import sympy as sp
        from devito import Grid, VectorTimeFunction

        from src.highorder.ader_devito import graddiv

        grid = Grid(shape=(11, 11), extent=(10., 10.))
        v = VectorTimeFunction(name='v', grid=grid, space_order=4)

        result = graddiv(v)

        assert isinstance(result, sp.Matrix)
        assert result.shape == (2, 1)

    def test_gradlap_returns_matrix(self):
        """Test that gradlap returns a SymPy matrix."""
        pytest.importorskip("devito")
        import sympy as sp
        from devito import Grid, TimeFunction

        from src.highorder.ader_devito import gradlap

        grid = Grid(shape=(11, 11), extent=(10., 10.))
        p = TimeFunction(name='p', grid=grid, space_order=4)

        result = gradlap(p)

        assert isinstance(result, sp.Matrix)
        assert result.shape == (2, 1)

    def test_lapdiv_returns_scalar(self):
        """Test that lapdiv returns a scalar expression."""
        pytest.importorskip("devito")
        import sympy as sp
        from devito import Grid, VectorTimeFunction

        from src.highorder.ader_devito import lapdiv

        grid = Grid(shape=(11, 11), extent=(10., 10.))
        v = VectorTimeFunction(name='v', grid=grid, space_order=4)

        result = lapdiv(v)

        # Should be a scalar SymPy expression (Add)
        assert isinstance(result, sp.Basic)
        assert not isinstance(result, sp.Matrix)

    def test_biharmonic_returns_scalar(self):
        """Test that biharmonic returns a scalar expression."""
        pytest.importorskip("devito")
        import sympy as sp
        from devito import Grid, TimeFunction

        from src.highorder.ader_devito import biharmonic

        grid = Grid(shape=(11, 11), extent=(10., 10.))
        p = TimeFunction(name='p', grid=grid, space_order=8)

        result = biharmonic(p)

        # Should be a scalar SymPy expression
        assert isinstance(result, sp.Basic)
        assert not isinstance(result, sp.Matrix)

    def test_gradlapdiv_returns_matrix(self):
        """Test that gradlapdiv returns a SymPy matrix."""
        pytest.importorskip("devito")
        import sympy as sp
        from devito import Grid, VectorTimeFunction

        from src.highorder.ader_devito import gradlapdiv

        grid = Grid(shape=(11, 11), extent=(10., 10.))
        v = VectorTimeFunction(name='v', grid=grid, space_order=8)

        result = gradlapdiv(v)

        assert isinstance(result, sp.Matrix)
        assert result.shape == (2, 1)


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not available")
@pytest.mark.devito
class TestADERSolver:
    """Tests for ADER 2D acoustic solver."""

    def test_solve_ader_2d_runs(self):
        """Test that ADER solver runs without error."""
        from src.highorder.ader_devito import solve_ader_2d

        result = solve_ader_2d(
            extent=(500., 500.),
            shape=(51, 51),
            c_value=1.5,
            t_end=100.,
            courant=0.85,
            space_order=8,
        )

        assert result.p is not None
        assert result.p.shape == (51, 51)
        assert result.vx.shape == (51, 51)
        assert result.vy.shape == (51, 51)

    def test_solve_ader_2d_wavefield_finite(self):
        """Test that wavefield values are finite (no NaN/Inf)."""
        from src.highorder.ader_devito import solve_ader_2d

        result = solve_ader_2d(
            extent=(500., 500.),
            shape=(51, 51),
            c_value=1.5,
            t_end=100.,
            courant=0.85,
            space_order=8,
        )

        assert np.all(np.isfinite(result.p))
        assert np.all(np.isfinite(result.vx))
        assert np.all(np.isfinite(result.vy))

    def test_solve_ader_2d_nonzero_wavefield(self):
        """Test that wavefield has non-zero values (source propagated)."""
        from src.highorder.ader_devito import solve_ader_2d

        result = solve_ader_2d(
            extent=(500., 500.),
            shape=(51, 51),
            c_value=1.5,
            t_end=100.,
            courant=0.85,
            space_order=8,
        )

        # Pressure field should have non-zero values
        p_norm = np.linalg.norm(result.p)
        assert p_norm > 0

    def test_solve_ader_2d_high_cfl_stable(self):
        """Test that ADER is stable at CFL = 0.85."""
        from src.highorder.ader_devito import solve_ader_2d

        result = solve_ader_2d(
            extent=(500., 500.),
            shape=(51, 51),
            c_value=1.5,
            t_end=200.,  # Longer time to test stability
            courant=0.85,
            space_order=8,
        )

        # Field should remain bounded
        p_max = np.max(np.abs(result.p))
        assert p_max < 1e10  # Should not blow up

    def test_solve_ader_2d_custom_source_location(self):
        """Test that custom source location works."""
        from src.highorder.ader_devito import solve_ader_2d

        # Source at corner
        result = solve_ader_2d(
            extent=(500., 500.),
            shape=(51, 51),
            c_value=1.5,
            t_end=100.,
            source_location=(100., 100.),
            space_order=8,
        )

        assert np.all(np.isfinite(result.p))

    def test_solve_ader_2d_metadata(self):
        """Test that result metadata is correct."""
        from src.highorder.ader_devito import solve_ader_2d

        result = solve_ader_2d(
            extent=(500., 500.),
            shape=(51, 51),
            c_value=1.5,
            t_end=100.,
            courant=0.85,
            space_order=8,
        )

        assert result.t_final == 100.
        assert result.courant == 0.85
        assert result.dt > 0
        assert result.nt > 0
        assert len(result.x) == 51
        assert len(result.y) == 51


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not available")
@pytest.mark.devito
class TestADERCFLAdvantage:
    """Tests demonstrating CFL advantage of ADER over standard schemes."""

    def test_ader_stable_at_high_cfl(self):
        """Test that ADER is stable at CFL = 0.85."""
        from src.highorder.ader_devito import solve_ader_2d

        # This CFL would be unstable for standard staggered leapfrog
        result = solve_ader_2d(
            extent=(500., 500.),
            shape=(51, 51),
            c_value=1.5,
            t_end=150.,
            courant=0.85,
            space_order=8,
        )

        # Check stability: field should remain bounded
        assert np.all(np.isfinite(result.p))
        assert np.max(np.abs(result.p)) < 1e10

    def test_ader_stable_at_low_cfl(self):
        """Test that ADER is also stable at lower CFL."""
        from src.highorder.ader_devito import solve_ader_2d

        result = solve_ader_2d(
            extent=(500., 500.),
            shape=(51, 51),
            c_value=1.5,
            t_end=150.,
            courant=0.5,
            space_order=8,
        )

        assert np.all(np.isfinite(result.p))

    def test_compare_ader_vs_staggered(self):
        """Test comparison function returns valid results."""
        from src.highorder.ader_devito import compare_ader_vs_staggered

        result_high, result_low = compare_ader_vs_staggered(
            extent=(500., 500.),
            shape=(31, 31),
            c_value=1.5,
            t_end=100.,
        )

        # Both should be stable
        assert np.all(np.isfinite(result_high.p))
        assert np.all(np.isfinite(result_low.p))

        # Different CFL should give different results
        assert result_high.courant == 0.85
        assert result_low.courant == 0.5


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not available")
@pytest.mark.devito
class TestADEREnergyBounds:
    """Tests for energy conservation/bounds in ADER schemes."""

    def test_energy_bounded(self):
        """Test that total energy remains bounded."""
        from src.highorder.ader_devito import solve_ader_2d

        result = solve_ader_2d(
            extent=(500., 500.),
            shape=(51, 51),
            c_value=1.5,
            t_end=150.,
            courant=0.85,
            space_order=8,
        )

        # Compute approximate energy (L2 norm squared)
        energy = np.sum(result.p ** 2) + np.sum(result.vx ** 2) + np.sum(result.vy ** 2)

        # Energy should be finite and bounded
        assert np.isfinite(energy)
        assert energy > 0  # Source should have injected energy
        assert energy < 1e20  # Should not blow up

    def test_field_maximum_reasonable(self):
        """Test that field maximum is reasonable (no runaway growth)."""
        from src.highorder.ader_devito import solve_ader_2d

        result = solve_ader_2d(
            extent=(500., 500.),
            shape=(51, 51),
            c_value=1.5,
            t_end=200.,
            courant=0.85,
            space_order=8,
        )

        # Maximum pressure should be reasonable
        p_max = np.max(np.abs(result.p))
        assert p_max < 100  # Reasonable bound for this problem
