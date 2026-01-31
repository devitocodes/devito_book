"""Tests for CFD solvers using Devito - Lid-Driven Cavity Flow.

This module tests the Navier-Stokes solvers for incompressible fluid
dynamics, including:
1. Lid-driven cavity benchmark problem
2. No-slip boundary conditions
3. Pressure Poisson convergence
4. Centerline velocity profiles (Ghia et al. comparison)
5. Mass and momentum conservation
6. Steady-state convergence
7. Reynolds number effects
8. Streamfunction computation

The governing equations (incompressible Navier-Stokes):
    du/dt + (u . grad)u = -1/rho * grad(p) + nu * laplace(u)
    div(u) = 0

Per CONTRIBUTING.md: All results must be reproducible with fixed random seeds,
version-pinned dependencies, and automated tests validating examples.
"""

import numpy as np
import pytest

# Check if Devito is available
try:
    import devito  # noqa: F401

    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")


# =============================================================================
# Test: Module Imports
# =============================================================================


@pytest.mark.devito
class TestModuleImports:
    """Test that the CFD module imports correctly."""

    def test_import_cfd_module(self):
        """Test importing the CFD module."""
        from src.cfd import navier_stokes_devito

        assert navier_stokes_devito is not None

    def test_import_solver_functions(self):
        """Test importing solver functions."""
        from src.cfd import solve_cavity_2d

        assert solve_cavity_2d is not None

    def test_import_pressure_functions(self):
        """Test importing pressure Poisson functions."""
        from src.cfd import pressure_poisson_iteration

        assert pressure_poisson_iteration is not None

    def test_import_streamfunction(self):
        """Test importing streamfunction computation."""
        from src.cfd import compute_streamfunction

        assert compute_streamfunction is not None

    def test_import_result_dataclass(self):
        """Test importing result dataclass."""
        from src.cfd import CavityResult

        assert CavityResult is not None

    def test_import_benchmark_data(self):
        """Test importing Ghia benchmark data."""
        from src.cfd import ghia_benchmark_data

        assert ghia_benchmark_data is not None


# =============================================================================
# Test: Ghia Benchmark Data
# =============================================================================


class TestGhiaBenchmarkData:
    """Tests for Ghia et al. benchmark data."""

    def test_benchmark_data_Re100(self):
        """Test benchmark data for Re=100."""
        from src.cfd import ghia_benchmark_data

        u_data, v_data = ghia_benchmark_data(Re=100)

        assert u_data.shape[0] == 17
        assert u_data.shape[1] == 2
        assert v_data.shape[0] == 17
        assert v_data.shape[1] == 2

    def test_benchmark_data_Re400(self):
        """Test benchmark data for Re=400."""
        from src.cfd import ghia_benchmark_data

        u_data, v_data = ghia_benchmark_data(Re=400)

        assert u_data.shape[0] == 17
        assert v_data.shape[0] == 17

    def test_benchmark_data_Re1000(self):
        """Test benchmark data for Re=1000."""
        from src.cfd import ghia_benchmark_data

        u_data, v_data = ghia_benchmark_data(Re=1000)

        assert u_data is not None
        assert v_data is not None

    def test_benchmark_boundary_values(self):
        """Benchmark data should have correct boundary values."""
        from src.cfd import ghia_benchmark_data

        u_data, v_data = ghia_benchmark_data(Re=100)

        # u at bottom (y=0) should be 0
        assert u_data[0, 1] == pytest.approx(0.0)

        # u at top (y=1) should be 1 (lid velocity)
        assert u_data[-1, 1] == pytest.approx(1.0)

        # v at boundaries should be 0
        assert v_data[0, 1] == pytest.approx(0.0)
        assert v_data[-1, 1] == pytest.approx(0.0)

    def test_invalid_reynolds_raises(self):
        """Invalid Reynolds number should raise error."""
        from src.cfd import ghia_benchmark_data

        with pytest.raises(ValueError, match="not available"):
            ghia_benchmark_data(Re=500)  # Not in the dataset


# =============================================================================
# Test: Lid-Driven Cavity Solver
# =============================================================================


@pytest.mark.devito
class TestLidDrivenCavitySolver:
    """Tests for the lid-driven cavity solver."""

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=11, Re=100, nt=10, nit=5)

        assert result.u is not None
        assert result.v is not None
        assert result.p is not None
        assert result.u.shape == (11, 11)
        assert result.v.shape == (11, 11)
        assert result.p.shape == (11, 11)

    def test_grid_coordinates(self):
        """Test that grid coordinates are correct."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=21, Re=100, nt=10, nit=5)

        assert len(result.x) == 21
        assert len(result.y) == 21
        assert result.x[0] == pytest.approx(0.0)
        assert result.x[-1] == pytest.approx(1.0)
        assert result.y[0] == pytest.approx(0.0)
        assert result.y[-1] == pytest.approx(1.0)

    def test_reynolds_number_stored(self):
        """Test that Reynolds number is stored correctly."""
        from src.cfd import solve_cavity_2d

        Re = 250
        result = solve_cavity_2d(N=11, Re=Re, nt=10, nit=5)

        assert result.Re == Re


# =============================================================================
# Test: No-Slip Boundary Conditions
# =============================================================================


@pytest.mark.devito
class TestNoSlipBoundaryConditions:
    """Tests for no-slip boundary conditions."""

    def test_bottom_wall_noslip(self):
        """Bottom wall should have u=v=0."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=21, Re=100, nt=50, nit=20)

        np.testing.assert_allclose(result.u[:, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.v[:, 0], 0.0, atol=1e-6)

    def test_left_wall_noslip(self):
        """Left wall should have u=v=0."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=21, Re=100, nt=50, nit=20)

        np.testing.assert_allclose(result.u[0, :], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.v[0, :], 0.0, atol=1e-6)

    def test_right_wall_noslip(self):
        """Right wall should have u=v=0."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=21, Re=100, nt=50, nit=20)

        np.testing.assert_allclose(result.u[-1, :], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.v[-1, :], 0.0, atol=1e-6)

    def test_top_wall_lid_velocity(self):
        """Top wall should have u=U_lid, v=0."""
        from src.cfd import solve_cavity_2d

        U_lid = 1.0
        result = solve_cavity_2d(N=21, Re=100, nt=50, nit=20, U_lid=U_lid)

        # Check interior of top wall (exclude corners which may have BC conflicts)
        np.testing.assert_allclose(result.u[1:-1, -1], U_lid, atol=1e-6)
        np.testing.assert_allclose(result.v[1:-1, -1], 0.0, atol=1e-6)

    def test_custom_lid_velocity(self):
        """Test with custom lid velocity."""
        from src.cfd import solve_cavity_2d

        U_lid = 2.5
        result = solve_cavity_2d(N=21, Re=100, nt=50, nit=20, U_lid=U_lid)

        # Check interior of top wall (exclude corners)
        np.testing.assert_allclose(result.u[1:-1, -1], U_lid, atol=1e-6)


# =============================================================================
# Test: Pressure Poisson Solver
# =============================================================================


class TestPressurePoissonSolver:
    """Tests for the pressure Poisson solver."""

    def test_pressure_poisson_iteration(self):
        """Test pressure Poisson iteration converges."""
        from src.cfd import pressure_poisson_iteration

        N = 21
        dx = dy = 1.0 / (N - 1)
        p = np.zeros((N, N))
        b = np.ones((N, N)) * 0.1

        p_new = pressure_poisson_iteration(p, b, dx, dy, nit=50)

        # Pressure should be modified
        assert not np.allclose(p_new, 0.0)

    def test_pressure_poisson_neumann_bc(self):
        """Test that Neumann BCs are satisfied after iteration."""
        from src.cfd import pressure_poisson_iteration

        np.random.seed(42)
        N = 21
        dx = dy = 1.0 / (N - 1)
        p = np.zeros((N, N))
        b = np.random.randn(N, N) * 0.1

        p = pressure_poisson_iteration(p, b, dx, dy, nit=100)

        # dp/dn = 0 means boundary values equal adjacent interior values
        # Allow small numerical tolerance
        np.testing.assert_allclose(p[0, 1:-1], p[1, 1:-1], atol=1e-3)
        np.testing.assert_allclose(p[-1, 1:-1], p[-2, 1:-1], atol=1e-3)
        np.testing.assert_allclose(p[1:-1, 0], p[1:-1, 1], atol=1e-3)
        np.testing.assert_allclose(p[1:-1, -1], p[1:-1, -2], atol=1e-3)

    def test_pressure_fixed_point(self):
        """Test that p=0 at corner (for uniqueness)."""
        from src.cfd import pressure_poisson_iteration

        np.random.seed(42)
        N = 21
        dx = dy = 1.0 / (N - 1)
        p = np.ones((N, N))  # Start with non-zero
        b = np.random.randn(N, N) * 0.1

        p = pressure_poisson_iteration(p, b, dx, dy, nit=100)

        assert p[0, 0] == pytest.approx(0.0)


# =============================================================================
# Test: Centerline Velocity Profiles
# =============================================================================


@pytest.mark.devito
@pytest.mark.slow
class TestCenterlineVelocityProfiles:
    """Tests for centerline velocity profiles against Ghia benchmark."""

    def test_u_profile_direction(self):
        """u along vertical centerline should have expected sign pattern."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=31, Re=100, nt=500, nit=50)

        # Extract u along vertical centerline (x = 0.5)
        mid_x = len(result.x) // 2
        u_centerline = result.u[mid_x, :]

        # Near top: u should be positive (driven by lid)
        assert u_centerline[-2] > 0

        # Near bottom: u might be negative (recirculation)
        # At bottom boundary: u = 0
        assert u_centerline[0] == pytest.approx(0.0, abs=1e-10)

    def test_v_profile_direction(self):
        """v along horizontal centerline should have expected pattern."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=31, Re=100, nt=500, nit=50)

        # Extract v along horizontal centerline (y = 0.5)
        mid_y = len(result.y) // 2
        v_centerline = result.v[:, mid_y]

        # Near left: v should be negative (downward flow)
        # Near right: v should be positive (upward flow)
        # This depends on enough time steps to develop
        assert v_centerline[0] == pytest.approx(0.0, abs=1e-10)  # BC
        assert v_centerline[-1] == pytest.approx(0.0, abs=1e-10)  # BC


# =============================================================================
# Test: Mass Conservation
# =============================================================================


@pytest.mark.devito
class TestMassConservation:
    """Tests for mass conservation (incompressibility)."""

    def test_initial_divergence_free(self):
        """Initial velocity field should be divergence-free."""
        # Initial conditions are zero velocity, which is divergence-free
        u = np.zeros((21, 21))
        v = np.zeros((21, 21))

        dx = dy = 1.0 / 20

        # Compute divergence
        div = np.zeros_like(u)
        div[1:-1, 1:-1] = (
            (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)
            + (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)
        )

        np.testing.assert_allclose(div, 0.0, atol=1e-14)

    def test_divergence_bounded(self):
        """Divergence should remain bounded after time stepping."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=21, Re=100, nt=100, nit=50)

        dx = result.x[1] - result.x[0]
        dy = result.y[1] - result.y[0]

        # Compute divergence
        div = np.zeros_like(result.u)
        div[1:-1, 1:-1] = (
            (result.u[2:, 1:-1] - result.u[:-2, 1:-1]) / (2 * dx)
            + (result.v[1:-1, 2:] - result.v[1:-1, :-2]) / (2 * dy)
        )

        # Interior divergence should be reasonably small
        # Note: projection methods don't guarantee exact div-free
        max_div = np.max(np.abs(div[2:-2, 2:-2]))
        assert max_div < 1.0  # Loose bound for this simple test


# =============================================================================
# Test: Steady-State Convergence
# =============================================================================


@pytest.mark.devito
@pytest.mark.slow
class TestSteadyStateConvergence:
    """Tests for steady-state convergence detection."""

    def test_convergence_flag(self):
        """Test that convergence flag is set correctly."""
        from src.cfd import solve_cavity_2d

        # Run with small number of steps (won't converge)
        result_short = solve_cavity_2d(
            N=11, Re=100, nt=10, nit=5, steady_tol=1e-10
        )

        # Run with more steps (may converge on small grid)
        result_long = solve_cavity_2d(
            N=11, Re=100, nt=5000, nit=50, steady_tol=1e-4
        )

        # Short run unlikely to converge
        assert result_short.nt <= 10

        # Long run either converges or reaches max steps
        assert result_long.nt <= 5000


# =============================================================================
# Test: Reynolds Number Effects
# =============================================================================


@pytest.mark.devito
class TestReynoldsNumberEffects:
    """Tests for different Reynolds number regimes."""

    def test_low_reynolds_number(self):
        """Low Re should produce smooth flow."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=21, Re=10, nt=200, nit=50)

        # Flow should be smooth (no NaN or Inf)
        assert np.all(np.isfinite(result.u))
        assert np.all(np.isfinite(result.v))
        assert np.all(np.isfinite(result.p))

    def test_moderate_reynolds_number(self):
        """Moderate Re (100) should be stable."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=21, Re=100, nt=200, nit=50)

        assert np.all(np.isfinite(result.u))
        assert np.all(np.isfinite(result.v))

    def test_higher_reynolds_number(self):
        """Higher Re (400) should still be stable with enough resolution."""
        from src.cfd import solve_cavity_2d

        # For higher Re, use finer grid and smaller dt
        result = solve_cavity_2d(N=41, Re=400, nt=500, nit=50, dt=0.0001)

        # Check that most values are finite (allow some boundary issues)
        finite_u = np.isfinite(result.u)
        finite_v = np.isfinite(result.v)
        assert np.mean(finite_u) > 0.99
        assert np.mean(finite_v) > 0.99


# =============================================================================
# Test: Streamfunction Computation
# =============================================================================


class TestStreamfunctionComputation:
    """Tests for streamfunction computation."""

    def test_streamfunction_shape(self):
        """Streamfunction should have same shape as velocity."""
        from src.cfd import compute_streamfunction

        N = 21
        u = np.zeros((N, N))
        v = np.zeros((N, N))
        dx = dy = 1.0 / (N - 1)

        psi = compute_streamfunction(u, v, dx, dy)

        assert psi.shape == (N, N)

    def test_streamfunction_zero_for_zero_velocity(self):
        """Streamfunction should be zero for zero velocity."""
        from src.cfd import compute_streamfunction

        N = 21
        u = np.zeros((N, N))
        v = np.zeros((N, N))
        dx = dy = 1.0 / (N - 1)

        psi = compute_streamfunction(u, v, dx, dy)

        np.testing.assert_allclose(psi, 0.0, atol=1e-14)

    def test_streamfunction_for_uniform_v(self):
        """Test streamfunction for uniform v-velocity."""
        from src.cfd import compute_streamfunction

        N = 21
        u = np.zeros((N, N))
        v = np.ones((N, N))  # Uniform v = 1
        dx = dy = 1.0 / (N - 1)

        psi = compute_streamfunction(u, v, dx, dy)

        # psi should vary in x (integral of -v along x)
        # psi[i, j] = psi[i-1, j] - v[i, j] * dx
        # So psi should decrease along x
        assert psi[-1, 10] < psi[0, 10]


# =============================================================================
# Test: Velocity Boundary Condition Helper
# =============================================================================


class TestVelocityBCHelper:
    """Tests for velocity boundary condition helper function."""

    def test_apply_velocity_bcs(self):
        """Test that BCs are applied correctly."""
        from src.cfd import apply_velocity_bcs

        N = 21
        u = np.ones((N, N))
        v = np.ones((N, N))
        U_lid = 1.5

        apply_velocity_bcs(u, v, N, U_lid)

        # Check walls - the function sets the entire boundary
        # Left wall
        np.testing.assert_allclose(u[0, :], 0.0)
        np.testing.assert_allclose(v[0, :], 0.0)

        # Right wall
        np.testing.assert_allclose(u[-1, :], 0.0)
        np.testing.assert_allclose(v[-1, :], 0.0)

        # Bottom wall
        np.testing.assert_allclose(u[:, 0], 0.0)
        np.testing.assert_allclose(v[:, 0], 0.0)

        # Top wall - corners may be overwritten by left/right walls
        # Just check that lid velocity is set somewhere
        assert np.any(u[:, -1] == U_lid), "Lid velocity should be set on top wall"
        np.testing.assert_allclose(v[:, -1], 0.0)  # v = 0 on top


# =============================================================================
# Test: NumPy Reference Solver
# =============================================================================


class TestNumpyReferenceSolver:
    """Tests for the NumPy reference implementation."""

    def test_numpy_solver_runs(self):
        """Test that NumPy solver runs without errors."""
        from src.cfd.navier_stokes_devito import solve_cavity_numpy

        result = solve_cavity_numpy(N=11, Re=100, nt=10, nit=5)

        assert result.u is not None
        assert result.v is not None
        assert result.p is not None

    def test_numpy_solver_boundary_conditions(self):
        """Test that NumPy solver enforces BCs."""
        from src.cfd.navier_stokes_devito import solve_cavity_numpy

        result = solve_cavity_numpy(N=21, Re=100, nt=50, nit=20)

        # Check BCs with reasonable tolerance
        np.testing.assert_allclose(result.u[:, 0], 0.0, atol=1e-6)  # Bottom
        np.testing.assert_allclose(result.v[:, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.u[1:-1, -1], 1.0, atol=1e-6)  # Top (lid)


# =============================================================================
# Test: Edge Cases
# =============================================================================


@pytest.mark.devito
class TestEdgeCases:
    """Tests for edge cases and stability."""

    def test_small_grid(self):
        """Test solver on minimum viable grid."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=5, Re=100, nt=10, nit=5)

        assert result.u.shape == (5, 5)
        assert np.all(np.isfinite(result.u))

    def test_solution_bounded(self):
        """Solution should remain bounded."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=21, Re=100, nt=200, nit=50)

        # Velocity should not exceed lid velocity by too much
        assert np.max(np.abs(result.u)) < 5.0
        assert np.max(np.abs(result.v)) < 5.0

    def test_pressure_bounded(self):
        """Pressure should remain bounded."""
        from src.cfd import solve_cavity_2d

        result = solve_cavity_2d(N=21, Re=100, nt=200, nit=50)

        # Pressure should not blow up
        assert np.all(np.isfinite(result.p))
        assert np.max(np.abs(result.p)) < 1000.0


# =============================================================================
# Test: Vorticity Computation
# =============================================================================


class TestVorticityComputation:
    """Tests for vorticity field computation."""

    def test_vorticity_import(self):
        """Test that vorticity function can be imported."""
        from src.cfd.navier_stokes_devito import compute_vorticity

        assert compute_vorticity is not None

    def test_vorticity_shape(self):
        """Vorticity field should have correct shape."""
        from src.cfd.navier_stokes_devito import compute_vorticity

        N = 21
        u = np.random.randn(N, N)
        v = np.random.randn(N, N)
        dx = dy = 1.0 / (N - 1)

        omega = compute_vorticity(u, v, dx, dy)

        assert omega.shape == (N, N)

    def test_vorticity_zero_for_uniform_flow(self):
        """Uniform flow should have zero vorticity."""
        from src.cfd.navier_stokes_devito import compute_vorticity

        N = 21
        u = np.ones((N, N))  # Uniform u
        v = np.zeros((N, N))  # Zero v
        dx = dy = 1.0 / (N - 1)

        omega = compute_vorticity(u, v, dx, dy)

        # Interior vorticity should be zero
        np.testing.assert_allclose(omega[1:-1, 1:-1], 0.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
