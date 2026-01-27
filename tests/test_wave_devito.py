"""Tests for Devito wave equation solvers.

These tests verify that the Devito-based wave equation solvers
produce correct results and converge at the expected rates.
"""

# Check if Devito is available
import importlib.util

import numpy as np
import pytest

DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None

# Skip all tests in this file if Devito is not installed
pytestmark = pytest.mark.skipif(
    not DEVITO_AVAILABLE,
    reason="Devito not installed"
)


@pytest.mark.devito
class TestWave1DSolver:
    """Tests for the 1D wave equation solver."""

    def test_import(self):
        """Verify solver can be imported."""
        from src.wave import WaveResult, solve_wave_1d
        assert solve_wave_1d is not None
        assert WaveResult is not None

    def test_basic_run(self):
        """Verify solver runs without errors."""
        from src.wave import solve_wave_1d

        result = solve_wave_1d(
            L=1.0,
            c=1.0,
            Nx=50,
            T=0.1,
            C=0.9,
        )

        assert result.u is not None
        assert result.x is not None
        assert len(result.u) == 51
        assert len(result.x) == 51

    def test_initial_condition_preserved_at_t0(self):
        """Initial condition should be exact at t=0."""
        from src.wave import solve_wave_1d

        def I(x):
            return np.sin(np.pi * x)

        result = solve_wave_1d(
            L=1.0,
            c=1.0,
            Nx=100,
            T=0.0,
            C=0.9,
            I=I,
            save_history=True,
        )

        # At t=0, solution should match initial condition
        expected = I(result.x)
        np.testing.assert_allclose(result.u_history[0], expected, rtol=1e-10)

    def test_boundary_conditions(self):
        """Verify Dirichlet boundary conditions u(0,t) = u(L,t) = 0."""
        from src.wave import solve_wave_1d

        result = solve_wave_1d(
            L=1.0,
            c=1.0,
            Nx=50,
            T=0.5,
            C=0.9,
            save_history=True,
        )

        # Check boundaries at all time steps
        for n in range(len(result.t_history)):
            assert abs(result.u_history[n, 0]) < 1e-10, f"Left BC violated at t={result.t_history[n]}"
            assert abs(result.u_history[n, -1]) < 1e-10, f"Right BC violated at t={result.t_history[n]}"

    def test_standing_wave_accuracy(self):
        """Test accuracy against exact standing wave solution."""
        from src.wave import exact_standing_wave, solve_wave_1d

        L = 1.0
        c = 1.0
        T = 0.5

        result = solve_wave_1d(
            L=L,
            c=c,
            Nx=100,
            T=T,
            C=0.9,
        )

        u_exact = exact_standing_wave(result.x, T, L, c)
        error = np.sqrt(np.mean((result.u - u_exact)**2))

        # Should be reasonably accurate (allow some numerical error)
        assert error < 0.05, f"Error {error} too large"

    def test_convergence_second_order(self):
        """Verify at least second-order convergence in space.

        Note: For the standing wave solution with C close to 1, the leapfrog
        scheme can exhibit superconvergence (order > 2) because the discrete
        scheme is nearly exact for sinusoidal modes.
        """
        from src.wave import convergence_test_wave_1d

        grid_sizes, errors, observed_order = convergence_test_wave_1d(
            grid_sizes=[20, 40, 80],
            T=0.25,
            C=0.5,  # Use lower C to avoid superconvergence
        )

        # Should be at least second order
        assert observed_order > 1.5, f"Observed order {observed_order} < 1.5"

        # Verify errors decrease
        assert errors[1] < errors[0], "Errors should decrease with refinement"
        assert errors[2] < errors[1], "Errors should decrease with refinement"

    def test_courant_stability_violation_raises(self):
        """CFL > 1 should raise ValueError."""
        from src.wave import solve_wave_1d

        with pytest.raises(ValueError, match="CFL stability"):
            solve_wave_1d(
                L=1.0,
                c=1.0,
                Nx=50,
                T=0.1,
                C=1.5,  # Unstable!
            )

    def test_custom_initial_velocity(self):
        """Test with non-zero initial velocity."""
        from src.wave import solve_wave_1d

        def I(x):
            return np.zeros_like(x)

        def V(x):
            return np.sin(np.pi * x)

        result = solve_wave_1d(
            L=1.0,
            c=1.0,
            Nx=50,
            T=0.1,
            C=0.9,
            I=I,
            V=V,
            save_history=True,
        )

        # Solution should be non-zero due to initial velocity
        assert np.max(np.abs(result.u)) > 0.01

    def test_different_wave_speeds(self):
        """Test with different wave speeds."""
        from src.wave import solve_wave_1d

        for c in [0.5, 1.0, 2.0]:
            result = solve_wave_1d(
                L=1.0,
                c=c,
                Nx=50,
                T=0.1,
                C=0.9,
            )
            assert result.u is not None
            assert result.C <= 1.0  # CFL should be satisfied

    def test_result_dataclass(self):
        """Verify WaveResult contains all expected fields."""
        from src.wave import solve_wave_1d

        result = solve_wave_1d(
            L=1.0,
            c=1.0,
            Nx=50,
            T=0.1,
            C=0.9,
            save_history=True,
        )

        assert hasattr(result, 'u')
        assert hasattr(result, 'x')
        assert hasattr(result, 't')
        assert hasattr(result, 'dt')
        assert hasattr(result, 'u_history')
        assert hasattr(result, 't_history')
        assert hasattr(result, 'C')

        assert result.t == pytest.approx(0.1, rel=1e-3)
        assert result.u_history.shape[0] > 1
        assert result.u_history.shape[1] == 51


@pytest.mark.devito
class TestExactSolution:
    """Tests for the exact standing wave solution."""

    def test_exact_solution_at_t0(self):
        """Exact solution at t=0 should match initial condition."""
        from src.wave.wave1D_devito import exact_standing_wave

        x = np.linspace(0, 1, 101)
        L = 1.0
        c = 1.0

        u = exact_standing_wave(x, 0.0, L, c)
        expected = np.sin(np.pi * x / L)

        np.testing.assert_allclose(u, expected, rtol=1e-10)

    def test_exact_solution_periodicity(self):
        """Solution should be periodic with period 2*L/c."""
        from src.wave.wave1D_devito import exact_standing_wave

        x = np.linspace(0, 1, 101)
        L = 1.0
        c = 1.0
        period = 2 * L / c

        u_0 = exact_standing_wave(x, 0.0, L, c)
        u_T = exact_standing_wave(x, period, L, c)

        np.testing.assert_allclose(u_0, u_T, rtol=1e-10)

    def test_exact_solution_satisfies_wave_eq(self):
        """Verify exact solution satisfies u_tt = c^2 * u_xx analytically."""
        # This is a mathematical verification
        # u(x, t) = sin(pi*x/L) * cos(pi*c*t/L)
        #
        # u_tt = sin(pi*x/L) * (-(pi*c/L)^2 * cos(pi*c*t/L))
        #      = -(pi*c/L)^2 * u
        #
        # u_xx = -(pi/L)^2 * sin(pi*x/L) * cos(pi*c*t/L)
        #      = -(pi/L)^2 * u
        #
        # c^2 * u_xx = c^2 * (-(pi/L)^2 * u) = -(pi*c/L)^2 * u = u_tt
        #
        # QED - the solution satisfies the wave equation

        import sympy as sp

        x_sym = sp.Symbol('x', real=True)
        t_sym = sp.Symbol('t', real=True)
        L_sym = sp.Symbol('L', positive=True)
        c_sym = sp.Symbol('c', positive=True)

        u = sp.sin(sp.pi * x_sym / L_sym) * sp.cos(sp.pi * c_sym * t_sym / L_sym)

        u_tt = sp.diff(u, t_sym, 2)
        u_xx = sp.diff(u, x_sym, 2)

        residual = sp.simplify(u_tt - c_sym**2 * u_xx)
        assert residual == 0
