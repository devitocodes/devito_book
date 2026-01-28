"""Tests for Devito advection solvers."""

import numpy as np
import pytest

# Check if Devito is available
try:
    import devito  # noqa: F401

    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DEVITO_AVAILABLE, reason="Devito not installed"
)


class TestAdvectionUpwind:
    """Tests for the upwind advection solver."""

    def test_import(self):
        """Test that the module imports correctly."""
        from src.advec import solve_advection_upwind

        assert solve_advection_upwind is not None

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.advec import solve_advection_upwind

        result = solve_advection_upwind(L=1.0, c=1.0, Nx=50, T=0.1, C=0.8)

        assert result.u.shape == (51,)
        assert result.x.shape == (51,)
        assert result.t == pytest.approx(0.1, rel=0.1)
        assert result.C <= 1.0

    def test_initial_condition_preserved_at_t0(self):
        """Test that T=0 returns the initial condition."""
        from src.advec import solve_advection_upwind

        def I(x):
            return np.sin(2 * np.pi * x)

        result = solve_advection_upwind(L=1.0, Nx=50, T=0, I=I)

        expected = I(result.x)
        np.testing.assert_allclose(result.u, expected, rtol=1e-10)

    def test_courant_number_violation_raises(self):
        """Test that C > 1 raises ValueError."""
        from src.advec import solve_advection_upwind

        with pytest.raises(ValueError, match="Courant number"):
            solve_advection_upwind(L=1.0, Nx=50, T=0.1, C=1.1)

    def test_negative_velocity_raises(self):
        """Test that c <= 0 raises ValueError."""
        from src.advec import solve_advection_upwind

        with pytest.raises(ValueError, match="velocity"):
            solve_advection_upwind(L=1.0, c=-1.0, Nx=50, T=0.1, C=0.8)

    def test_exact_at_courant_one(self):
        """Test that C=1 gives very accurate solution."""
        from src.advec import exact_advection_periodic, solve_advection_upwind

        def I(x):
            return np.exp(-0.5 * ((x - 0.25) / 0.05) ** 2)

        result = solve_advection_upwind(
            L=1.0, c=1.0, Nx=100, T=0.5, C=1.0, I=I, periodic_bc=True
        )

        u_exact = exact_advection_periodic(result.x, result.t, 1.0, 1.0, I)
        error = np.max(np.abs(result.u - u_exact))

        # Should be very small (near exact for C=1)
        assert error < 1e-4

    def test_result_dataclass(self):
        """Test that result dataclass has expected attributes."""
        from src.advec import solve_advection_upwind

        result = solve_advection_upwind(
            L=1.0, Nx=50, T=0.1, C=0.8, save_history=True
        )

        assert hasattr(result, "u")
        assert hasattr(result, "x")
        assert hasattr(result, "t")
        assert hasattr(result, "dt")
        assert hasattr(result, "C")
        assert hasattr(result, "u_history")
        assert hasattr(result, "t_history")
        assert result.u_history is not None
        assert result.t_history is not None


class TestAdvectionLaxWendroff:
    """Tests for the Lax-Wendroff advection solver."""

    def test_import(self):
        """Test that the module imports correctly."""
        from src.advec import solve_advection_lax_wendroff

        assert solve_advection_lax_wendroff is not None

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.advec import solve_advection_lax_wendroff

        result = solve_advection_lax_wendroff(L=1.0, c=1.0, Nx=50, T=0.1, C=0.8)

        assert result.u.shape == (51,)
        assert result.x.shape == (51,)
        assert result.t == pytest.approx(0.1, rel=0.1)

    def test_courant_number_violation_raises(self):
        """Test that |C| > 1 raises ValueError."""
        from src.advec import solve_advection_lax_wendroff

        with pytest.raises(ValueError, match="Courant number"):
            solve_advection_lax_wendroff(L=1.0, Nx=50, T=0.1, C=1.1)

    def test_second_order_accuracy(self):
        """Test that Lax-Wendroff is second-order accurate."""
        from src.advec import exact_advection_periodic, solve_advection_lax_wendroff

        def I(x):
            return np.exp(-0.5 * ((x - 0.25) / 0.05) ** 2)

        errors = []
        grid_sizes = [50, 100, 200, 400]

        for Nx in grid_sizes:
            result = solve_advection_lax_wendroff(
                L=1.0, c=1.0, Nx=Nx, T=0.25, C=0.8, I=I, periodic_bc=True
            )
            u_exact = exact_advection_periodic(result.x, result.t, 1.0, 1.0, I)
            dx = 1.0 / Nx
            error = np.sqrt(dx * np.sum((result.u - u_exact) ** 2))
            errors.append(error)

        # Compute average convergence rate from finer grids
        rates = []
        for i in range(1, len(errors)):
            rate = np.log(errors[i - 1] / errors[i]) / np.log(
                grid_sizes[i] / grid_sizes[i - 1]
            )
            rates.append(rate)

        avg_rate = np.mean(rates)

        # Should be close to 2 for second-order (allow 1.5-2.3 for numerical effects)
        assert 1.5 < avg_rate < 2.3


class TestAdvectionLaxFriedrichs:
    """Tests for the Lax-Friedrichs advection solver."""

    def test_import(self):
        """Test that the module imports correctly."""
        from src.advec import solve_advection_lax_friedrichs

        assert solve_advection_lax_friedrichs is not None

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.advec import solve_advection_lax_friedrichs

        result = solve_advection_lax_friedrichs(L=1.0, c=1.0, Nx=50, T=0.1, C=0.8)

        assert result.u.shape == (51,)
        assert result.x.shape == (51,)

    def test_courant_number_violation_raises(self):
        """Test that |C| > 1 raises ValueError."""
        from src.advec import solve_advection_lax_friedrichs

        with pytest.raises(ValueError, match="Courant number"):
            solve_advection_lax_friedrichs(L=1.0, Nx=50, T=0.1, C=1.1)

    def test_solution_bounded(self):
        """Test that solution remains bounded."""
        from src.advec import solve_advection_lax_friedrichs

        def I(x):
            return np.exp(-0.5 * ((x - 0.25) / 0.05) ** 2)

        result = solve_advection_lax_friedrichs(
            L=1.0, c=1.0, Nx=100, T=1.0, C=0.8, I=I
        )

        # Solution should remain bounded (max of Gaussian is 1)
        assert result.u.max() <= 1.0
        assert result.u.min() >= 0.0


class TestExactAdvectionSolution:
    """Tests for the exact advection solution."""

    def test_exact_at_t0(self):
        """Test that exact solution at t=0 matches initial condition."""
        from src.advec import exact_advection

        def I(x):
            return np.sin(2 * np.pi * x)

        x = np.linspace(0, 1, 101)
        u = exact_advection(x, t=0, c=1.0, I=I)

        expected = I(x)
        np.testing.assert_allclose(u, expected, rtol=1e-10)

    def test_exact_translation(self):
        """Test that exact solution is translated initial condition."""
        from src.advec import exact_advection

        def I(x):
            return np.sin(2 * np.pi * x)

        x = np.linspace(0, 1, 101)
        c = 1.0
        t = 0.25

        u = exact_advection(x, t=t, c=c, I=I)
        expected = I(x - c * t)

        np.testing.assert_allclose(u, expected, rtol=1e-10)

    def test_periodic_wrapping(self):
        """Test that periodic exact solution wraps correctly."""
        from src.advec import exact_advection_periodic

        def I(x):
            return np.exp(-0.5 * ((x - 0.25) / 0.05) ** 2)

        L = 1.0
        c = 1.0

        # After one period, should return to initial condition
        # Exclude x=L since it's the same physical point as x=0 for periodic domains
        x = np.linspace(0, L, 101)[:-1]  # Exclude endpoint
        u = exact_advection_periodic(x, t=L / c, c=c, L=L, I=I)

        expected = I(x)
        np.testing.assert_allclose(u, expected, rtol=1e-10)


class TestInitialConditions:
    """Tests for initial condition utilities."""

    def test_gaussian_initial_condition(self):
        """Test Gaussian initial condition."""
        from src.advec import gaussian_initial_condition

        x = np.linspace(0, 1, 101)
        u = gaussian_initial_condition(x, L=1.0, sigma=0.05, x0=0.25)

        # Peak should be at x0
        peak_idx = np.argmax(u)
        assert x[peak_idx] == pytest.approx(0.25, abs=0.01)

        # Peak value should be 1
        assert u.max() == pytest.approx(1.0, rel=1e-10)

    def test_step_initial_condition(self):
        """Test step initial condition."""
        from src.advec import step_initial_condition

        x = np.linspace(0, 1, 101)
        u = step_initial_condition(x, L=1.0, x_step=0.25)

        # Left of step should be 1
        assert all(u[x < 0.24] == 1.0)

        # Right of step should be 0
        assert all(u[x > 0.26] == 0.0)


class TestConvergenceTest:
    """Tests for convergence testing utility."""

    def test_convergence_test_runs(self):
        """Test that convergence test executes."""
        from src.advec import convergence_test_advection, solve_advection_upwind

        sizes, errors, rate = convergence_test_advection(
            solve_advection_upwind,
            grid_sizes=[25, 50, 100],
            T=0.1,
            C=0.8,
        )

        assert len(sizes) == 3
        assert len(errors) == 3
        assert rate > 0  # Should have some positive rate

    def test_upwind_first_order(self):
        """Test that upwind converges at first order."""
        from src.advec import convergence_test_advection, solve_advection_upwind

        sizes, errors, rate = convergence_test_advection(
            solve_advection_upwind,
            grid_sizes=[25, 50, 100, 200],
            T=0.25,
            C=0.8,
        )

        # Should be close to 1 for first-order
        assert 0.8 < rate < 1.3

    def test_lax_wendroff_second_order(self):
        """Test that Lax-Wendroff converges at second order."""
        from src.advec import convergence_test_advection, solve_advection_lax_wendroff

        sizes, errors, rate = convergence_test_advection(
            solve_advection_lax_wendroff,
            grid_sizes=[25, 50, 100, 200],
            T=0.25,
            C=0.8,
        )

        # Should be close to 2 for second-order
        assert 1.7 < rate < 2.3


class TestSolutionProperties:
    """Tests for expected solution properties."""

    def test_upwind_amplitude_decay(self):
        """Test that upwind scheme has amplitude decay for C < 1."""
        from src.advec import solve_advection_upwind

        def I(x):
            return np.exp(-0.5 * ((x - 0.25) / 0.05) ** 2)

        result = solve_advection_upwind(
            L=1.0, c=1.0, Nx=100, T=1.0, C=0.8, I=I, periodic_bc=True
        )

        # Amplitude should have decayed (upwind is diffusive)
        assert result.u.max() < 1.0

    def test_periodic_integral_conservation(self):
        """Test that integral is approximately conserved with periodic BC."""
        from src.advec import solve_advection_lax_wendroff

        def I(x):
            return np.exp(-0.5 * ((x - 0.25) / 0.05) ** 2)

        result = solve_advection_lax_wendroff(
            L=1.0, c=1.0, Nx=100, T=0.5, C=0.8, I=I, periodic_bc=True, save_history=True
        )

        # Compute integral at start and end
        dx = 1.0 / 100
        integral_start = dx * np.sum(result.u_history[0])
        integral_end = dx * np.sum(result.u_history[-1])

        # Should be approximately conserved (within numerical error)
        np.testing.assert_allclose(integral_start, integral_end, rtol=0.1)
