"""Tests for 2D Burgers equation Devito solver."""

import numpy as np
import pytest

# Check if Devito is available
try:
    import devito  # noqa: F401

    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")


class TestBurgers2DBasic:
    """Basic tests for 2D Burgers equation solver."""

    def test_import(self):
        """Test that the module imports correctly."""
        from src.nonlin.burgers_devito import solve_burgers_2d

        assert solve_burgers_2d is not None

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.nonlin.burgers_devito import solve_burgers_2d

        result = solve_burgers_2d(Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0.01)

        assert result.u.shape == (21, 21)
        assert result.v.shape == (21, 21)
        assert result.x.shape == (21,)
        assert result.y.shape == (21,)
        assert result.t > 0
        assert result.dt > 0

    def test_t_equals_zero(self):
        """Test that T=0 returns initial condition."""
        from src.nonlin.burgers_devito import solve_burgers_2d

        result = solve_burgers_2d(Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0)

        # Default initial condition has hat function with value 2.0
        # in region [0.5, 1] x [0.5, 1]
        assert result.t == 0.0
        assert result.u.max() == pytest.approx(2.0, rel=1e-10)
        assert result.v.max() == pytest.approx(2.0, rel=1e-10)


class TestBurgers2DBoundaryConditions:
    """Tests for boundary conditions."""

    def test_dirichlet_bc_default(self):
        """Test that default Dirichlet BCs are applied (value=1.0)."""
        from src.nonlin.burgers_devito import solve_burgers_2d

        result = solve_burgers_2d(
            Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0.01, bc_value=1.0
        )

        # Check boundaries are at bc_value=1.0
        assert np.allclose(result.u[0, :], 1.0)
        assert np.allclose(result.u[-1, :], 1.0)
        assert np.allclose(result.u[:, 0], 1.0)
        assert np.allclose(result.u[:, -1], 1.0)

        assert np.allclose(result.v[0, :], 1.0)
        assert np.allclose(result.v[-1, :], 1.0)
        assert np.allclose(result.v[:, 0], 1.0)
        assert np.allclose(result.v[:, -1], 1.0)

    def test_dirichlet_bc_custom(self):
        """Test that custom Dirichlet BC value is applied."""
        from src.nonlin.burgers_devito import solve_burgers_2d

        result = solve_burgers_2d(
            Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0.01, bc_value=0.5
        )

        # Check boundaries are at bc_value=0.5
        assert np.allclose(result.u[0, :], 0.5)
        assert np.allclose(result.u[-1, :], 0.5)
        assert np.allclose(result.v[0, :], 0.5)
        assert np.allclose(result.v[-1, :], 0.5)


class TestBurgers2DPhysics:
    """Tests for physical behavior of the solution."""

    def test_solution_bounded(self):
        """Test that solution remains bounded (no blow-up)."""
        from src.nonlin.burgers_devito import solve_burgers_2d

        result = solve_burgers_2d(Lx=2.0, Ly=2.0, nu=0.01, Nx=31, Ny=31, T=0.1)

        # Solution should remain bounded by initial maximum
        # Burgers equation with viscosity should not blow up
        assert np.all(np.abs(result.u) < 10.0)
        assert np.all(np.abs(result.v) < 10.0)

    def test_viscosity_smoothing(self):
        """Test that higher viscosity leads to smoother solution."""
        from src.nonlin.burgers_devito import solve_burgers_2d

        # Low viscosity
        result_low = solve_burgers_2d(
            Lx=2.0, Ly=2.0, nu=0.001, Nx=31, Ny=31, T=0.01, sigma=0.00001
        )

        # High viscosity
        result_high = solve_burgers_2d(
            Lx=2.0, Ly=2.0, nu=0.1, Nx=31, Ny=31, T=0.01, sigma=0.001
        )

        # Higher viscosity should give smaller gradients
        grad_u_low = np.max(np.abs(np.diff(result_low.u, axis=0)))
        grad_u_high = np.max(np.abs(np.diff(result_high.u, axis=0)))

        assert grad_u_high < grad_u_low

    def test_advection_moves_solution(self):
        """Test that the solution evolves (not stationary)."""
        from src.nonlin.burgers_devito import solve_burgers_2d

        result_early = solve_burgers_2d(Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0.01)
        result_late = solve_burgers_2d(Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0.05)

        # Solutions at different times should be different
        assert not np.allclose(result_early.u, result_late.u)


class TestBurgers2DFirstDerivative:
    """Tests specifically for first_derivative usage with explicit order."""

    def test_first_derivative_imported(self):
        """Test that first_derivative is available."""
        from devito import first_derivative

        assert first_derivative is not None

    def test_upwind_differencing_used(self):
        """Test that the solver uses backward differences for advection.

        This is verified by checking that the solver runs without
        instability when using the explicit scheme.
        """
        from src.nonlin.burgers_devito import solve_burgers_2d

        # Run for many time steps - would become unstable with wrong differencing
        result = solve_burgers_2d(Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0.1)

        # Solution should remain bounded (stable)
        assert np.all(np.isfinite(result.u))
        assert np.all(np.isfinite(result.v))
        assert np.max(np.abs(result.u)) < 10.0


class TestBurgers2DVector:
    """Tests for VectorTimeFunction implementation."""

    def test_import_vector_solver(self):
        """Test that vector solver imports correctly."""
        from src.nonlin.burgers_devito import solve_burgers_2d_vector

        assert solve_burgers_2d_vector is not None

    def test_vector_solver_basic_run(self):
        """Test basic execution of vector solver."""
        from src.nonlin.burgers_devito import solve_burgers_2d_vector

        result = solve_burgers_2d_vector(Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0.01)

        assert result.u.shape == (21, 21)
        assert result.v.shape == (21, 21)
        assert result.t > 0

    def test_vector_solver_bounded(self):
        """Test that vector solver solution remains bounded."""
        from src.nonlin.burgers_devito import solve_burgers_2d_vector

        result = solve_burgers_2d_vector(Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0.1)

        assert np.all(np.abs(result.u) < 10.0)
        assert np.all(np.abs(result.v) < 10.0)

    def test_vector_solver_boundary_conditions(self):
        """Test boundary conditions in vector solver."""
        from src.nonlin.burgers_devito import solve_burgers_2d_vector

        result = solve_burgers_2d_vector(
            Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0.01, bc_value=1.0
        )

        # Check boundaries
        assert np.allclose(result.u[0, :], 1.0)
        assert np.allclose(result.u[-1, :], 1.0)
        assert np.allclose(result.v[0, :], 1.0)
        assert np.allclose(result.v[-1, :], 1.0)


class TestBurgers2DHistory:
    """Tests for solution history saving."""

    def test_save_history(self):
        """Test that history is saved correctly."""
        from src.nonlin.burgers_devito import solve_burgers_2d

        result = solve_burgers_2d(
            Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0.1, save_history=True, save_every=50
        )

        assert result.u_history is not None
        assert result.v_history is not None
        assert result.t_history is not None
        assert len(result.u_history) > 1
        assert len(result.u_history) == len(result.t_history)

    def test_history_none_when_not_saved(self):
        """Test that history is None when not requested."""
        from src.nonlin.burgers_devito import solve_burgers_2d

        result = solve_burgers_2d(
            Lx=2.0, Ly=2.0, nu=0.01, Nx=21, Ny=21, T=0.01, save_history=False
        )

        assert result.u_history is None
        assert result.v_history is None
        assert result.t_history is None


class TestBurgers2DInitialConditions:
    """Tests for initial condition functions."""

    def test_hat_initial_condition(self):
        """Test hat function initial condition."""
        import numpy as np

        from src.nonlin.burgers_devito import init_hat

        x = np.linspace(0, 2, 21)
        y = np.linspace(0, 2, 21)
        X, Y = np.meshgrid(x, y, indexing="ij")

        u0 = init_hat(X, Y, Lx=2.0, Ly=2.0, value=2.0)

        # Outside the hat region [0.5, 1] x [0.5, 1], value should be 1.0
        assert u0[0, 0] == pytest.approx(1.0)
        assert u0[-1, -1] == pytest.approx(1.0)

        # Inside the hat region, value should be 2.0
        # Find indices corresponding to center of hat region
        x_idx = np.argmin(np.abs(x - 0.75))
        y_idx = np.argmin(np.abs(y - 0.75))
        assert u0[x_idx, y_idx] == pytest.approx(2.0)

    def test_sinusoidal_initial_condition(self):
        """Test sinusoidal initial condition."""
        import numpy as np

        from src.nonlin.burgers_devito import sinusoidal_initial_condition

        x = np.linspace(0, 2, 21)
        y = np.linspace(0, 2, 21)
        X, Y = np.meshgrid(x, y, indexing="ij")

        u0 = sinusoidal_initial_condition(X, Y, Lx=2.0, Ly=2.0)

        # Should be zero at boundaries
        assert u0[0, :].max() == pytest.approx(0.0, abs=1e-10)
        assert u0[-1, :].max() == pytest.approx(0.0, abs=1e-10)
        assert u0[:, 0].max() == pytest.approx(0.0, abs=1e-10)
        assert u0[:, -1].max() == pytest.approx(0.0, abs=1e-10)

        # Maximum should be 1.0 at center
        center_idx = len(x) // 2
        assert u0[center_idx, center_idx] == pytest.approx(1.0, rel=0.1)

    def test_gaussian_initial_condition(self):
        """Test Gaussian initial condition."""
        import numpy as np

        from src.nonlin.burgers_devito import gaussian_initial_condition

        x = np.linspace(0, 2, 41)
        y = np.linspace(0, 2, 41)
        X, Y = np.meshgrid(x, y, indexing="ij")

        u0 = gaussian_initial_condition(X, Y, Lx=2.0, Ly=2.0, amplitude=2.0)

        # Background is 1.0, peak is at 1.0 + amplitude
        assert u0.min() >= 1.0
        assert u0.max() <= 3.0 + 1e-10

        # Peak should be near center
        center_idx = len(x) // 2
        peak_idx = np.unravel_index(np.argmax(u0), u0.shape)
        assert abs(peak_idx[0] - center_idx) <= 1
        assert abs(peak_idx[1] - center_idx) <= 1

    def test_custom_initial_condition(self):
        """Test using custom initial condition."""
        import numpy as np

        from src.nonlin.burgers_devito import solve_burgers_2d

        def custom_u(X, Y):
            return np.ones_like(X) * 1.5

        def custom_v(X, Y):
            return np.ones_like(X) * 0.5

        result = solve_burgers_2d(
            Lx=2.0, Ly=2.0, nu=0.1, Nx=21, Ny=21, T=0.0, I_u=custom_u, I_v=custom_v
        )

        # At T=0, should return initial condition
        assert np.allclose(result.u, 1.5)
        assert np.allclose(result.v, 0.5)


class TestBurgers2DResult:
    """Tests for Burgers2DResult dataclass."""

    def test_result_attributes(self):
        """Test that result has expected attributes."""
        from src.nonlin.burgers_devito import solve_burgers_2d

        result = solve_burgers_2d(
            Lx=2.0,
            Ly=2.0,
            nu=0.01,
            Nx=21,
            Ny=21,
            T=0.01,
            save_history=True,
            save_every=10,
        )

        assert hasattr(result, "u")
        assert hasattr(result, "v")
        assert hasattr(result, "x")
        assert hasattr(result, "y")
        assert hasattr(result, "t")
        assert hasattr(result, "dt")
        assert hasattr(result, "u_history")
        assert hasattr(result, "v_history")
        assert hasattr(result, "t_history")
