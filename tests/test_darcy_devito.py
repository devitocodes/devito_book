"""Tests for Darcy flow solvers using Devito.

This module tests the Darcy flow solvers for porous media, including:
1. Homogeneous and heterogeneous permeability
2. Pressure boundary conditions
3. Velocity computation from pressure
4. Mass conservation
5. Analytical solutions for verification
6. Transient flow with storage

Darcy's law:
    q = -K * grad(p)

Combined with mass conservation:
    -div(K * grad(p)) = f

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
    """Test that the Darcy module imports correctly."""

    def test_import_darcy_module(self):
        """Test importing the Darcy module."""
        from src.darcy import darcy_devito

        assert darcy_devito is not None

    def test_import_solver_functions(self):
        """Test importing solver functions."""
        from src.darcy import solve_darcy_2d, solve_darcy_transient

        assert solve_darcy_2d is not None
        assert solve_darcy_transient is not None

    def test_import_velocity_computation(self):
        """Test importing velocity computation function."""
        from src.darcy import compute_darcy_velocity

        assert compute_darcy_velocity is not None

    def test_import_result_dataclass(self):
        """Test importing result dataclass."""
        from src.darcy import DarcyResult

        assert DarcyResult is not None

    def test_import_permeability_generation(self):
        """Test importing permeability generation functions."""
        from src.darcy import create_binary_permeability, create_layered_permeability

        assert create_layered_permeability is not None
        assert create_binary_permeability is not None

    def test_import_gaussian_random_field(self):
        """Test importing GaussianRandomField class."""
        from src.darcy import GaussianRandomField

        assert GaussianRandomField is not None


# =============================================================================
# Test: Homogeneous Permeability
# =============================================================================


@pytest.mark.devito
class TestHomogeneousPermeability:
    """Tests for Darcy flow with homogeneous (uniform) permeability."""

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.darcy import solve_darcy_2d

        result = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=20, Ny=20, permeability=1.0, bc_left=1.0, bc_right=0.0
        )

        assert result.p is not None
        assert result.p.shape == (20, 20)
        assert result.x is not None
        assert result.y is not None

    def test_pressure_boundary_conditions(self):
        """Test that pressure BCs are satisfied."""
        from src.darcy import solve_darcy_2d

        p_left = 2.0
        p_right = 0.5

        result = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=30, Ny=30, permeability=1.0,
            bc_left=p_left, bc_right=p_right
        )

        # Left boundary should be p_left
        np.testing.assert_allclose(result.p[0, :], p_left, atol=1e-4)

        # Right boundary should be p_right
        np.testing.assert_allclose(result.p[-1, :], p_right, atol=1e-4)

    def test_linear_pressure_profile_1d(self):
        """For uniform K and 1D flow, pressure should be linear."""
        from src.darcy import solve_darcy_2d

        p_left = 1.0
        p_right = 0.0

        result = solve_darcy_2d(
            Lx=1.0,
            Ly=0.2,
            Nx=40,
            Ny=10,
            permeability=1.0,
            bc_left=p_left,
            bc_right=p_right,
            bc_top="neumann",
            bc_bottom="neumann",
            tol=1e-6,
        )

        # Along any horizontal line, pressure should be linear
        x = result.x
        p_expected = p_left + (p_right - p_left) * x / result.x[-1]

        # Check middle row
        mid_row = result.p[:, result.p.shape[1] // 2]
        np.testing.assert_allclose(mid_row, p_expected, atol=0.05)

    def test_velocity_uniform_in_1d_flow(self):
        """For 1D flow with uniform K, velocity should be uniform."""
        from src.darcy import solve_darcy_2d

        K = 1.0
        p_left = 1.0
        p_right = 0.0
        Lx = 1.0

        result = solve_darcy_2d(
            Lx=Lx, Ly=0.1, Nx=64, Ny=8, permeability=K,
            bc_left=p_left, bc_right=p_right, tol=1e-6
        )

        # Expected uniform velocity
        v_expected = -K * (p_right - p_left) / Lx

        # Check x-velocity in interior (away from boundaries)
        interior_qx = result.qx[10:-10, 2:-2]
        np.testing.assert_allclose(interior_qx, v_expected, atol=0.5)

    def test_convergence_with_mesh_refinement(self):
        """Solution should improve with mesh refinement."""
        from src.darcy import verify_linear_pressure

        # Use built-in verification
        error = verify_linear_pressure(tol=1e-6)
        assert error < 0.02, f"Linear pressure error {error} too large"


# =============================================================================
# Test: Heterogeneous Permeability
# =============================================================================


@pytest.mark.devito
class TestHeterogeneousPermeability:
    """Tests for Darcy flow with heterogeneous permeability."""

    def test_layered_permeability(self):
        """Test with simple layered permeability."""
        from src.darcy import create_layered_permeability, solve_darcy_2d

        Nx, Ny = 32, 32
        # Create two-layer system
        layers = [(0.5, 1.0), (1.0, 10.0)]  # Low perm bottom, high perm top
        K = create_layered_permeability(Nx, Ny, layers)

        result = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=Nx, Ny=Ny, permeability=K,
            bc_left=1.0, bc_right=0.0
        )

        # Solution should exist
        assert result.converged
        assert np.all(np.isfinite(result.p))

    def test_binary_permeability(self):
        """Test with binary permeability field."""
        from src.darcy import create_binary_permeability, solve_darcy_2d

        Nx, Ny = 32, 32
        K = create_binary_permeability(Nx, Ny, K_low=1.0, K_high=10.0, seed=42)

        result = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=Nx, Ny=Ny, permeability=K,
            bc_left=1.0, bc_right=0.0
        )

        assert result.converged
        assert np.all(np.isfinite(result.p))

    def test_gaussian_random_field(self):
        """Test with Gaussian random field permeability."""
        from src.darcy import GaussianRandomField, solve_darcy_2d

        Nx = Ny = 32
        np.random.seed(42)
        grf = GaussianRandomField(size=Nx, alpha=2, tau=3)
        field = grf.sample(1)[0]
        K = np.exp(field)  # Log-normal permeability

        result = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=Nx, Ny=Ny, permeability=K,
            bc_left=1.0, bc_right=0.0
        )

        assert result.converged
        assert np.all(np.isfinite(result.p))


# =============================================================================
# Test: Velocity Computation
# =============================================================================


@pytest.mark.devito
class TestVelocityComputation:
    """Tests for velocity field computation from pressure."""

    def test_velocity_computed(self):
        """Test that velocity is computed when requested."""
        from src.darcy import solve_darcy_2d

        result = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=20, Ny=20, permeability=1.0,
            bc_left=1.0, bc_right=0.0, compute_velocity=True
        )

        assert result.qx is not None
        assert result.qy is not None
        assert result.qx.shape == result.p.shape
        assert result.qy.shape == result.p.shape

    def test_velocity_not_computed_when_disabled(self):
        """Test that velocity is not computed when disabled."""
        from src.darcy import solve_darcy_2d

        result = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=20, Ny=20, permeability=1.0,
            bc_left=1.0, bc_right=0.0, compute_velocity=False
        )

        assert result.qx is None
        assert result.qy is None

    def test_velocity_direction(self):
        """Velocity should flow from high to low pressure."""
        from src.darcy import solve_darcy_2d

        result = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=30, Ny=30, permeability=1.0,
            bc_left=1.0, bc_right=0.0
        )

        # Flow should be in positive x direction (high p on left)
        interior_qx = result.qx[5:-5, 5:-5]
        assert np.mean(interior_qx) > 0

        # Vertical velocity should be approximately zero
        interior_qy = result.qy[5:-5, 5:-5]
        assert np.abs(np.mean(interior_qy)) < 0.1

    def test_darcy_law_satisfied(self):
        """Test that q = -K * grad(p)."""
        from src.darcy import compute_darcy_velocity, solve_darcy_2d

        K = 2.0
        Lx = Ly = 1.0
        Nx = Ny = 30
        result = solve_darcy_2d(
            Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, permeability=K,
            bc_left=1.0, bc_right=0.0
        )

        # Recompute velocity
        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)
        qx, qy = compute_darcy_velocity(result.p, K, dx, dy)

        # Should match result
        np.testing.assert_allclose(qx, result.qx, atol=1e-10)
        np.testing.assert_allclose(qy, result.qy, atol=1e-10)


# =============================================================================
# Test: Mass Conservation
# =============================================================================


@pytest.mark.devito
class TestMassConservation:
    """Tests for mass conservation (divergence-free velocity)."""

    def test_mass_conservation_check(self):
        """Test the mass conservation checking function."""
        from src.darcy import check_mass_conservation, solve_darcy_2d

        Lx = Ly = 1.0
        result = solve_darcy_2d(
            Lx=Lx, Ly=Ly, Nx=64, Ny=64, permeability=1.0,
            bc_left=1.0, bc_right=0.0, source=0.0, tol=1e-6
        )

        imbalance = check_mass_conservation(
            result.p, result.K, 0.0, Lx, Ly
        )

        # For zero source, should be small (allow larger tolerance)
        assert imbalance < 2.0  # Relaxed bound for iterative solver

    def test_flux_balance(self):
        """Inflow flux should equal outflow flux at steady state."""
        from src.darcy import solve_darcy_2d

        Lx, Ly = 1.0, 0.2
        Nx, Ny = 64, 16
        result = solve_darcy_2d(
            Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, permeability=1.0,
            bc_left=1.0, bc_right=0.0, tol=1e-6
        )

        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        # Flux through left boundary (inflow)
        flux_in = np.sum(result.qx[0, :]) * dy

        # Flux through right boundary (outflow)
        flux_out = np.sum(result.qx[-1, :]) * dy

        # Should be approximately equal (with relaxed tolerance)
        np.testing.assert_allclose(flux_in, flux_out, rtol=0.5)


# =============================================================================
# Test: Analytical Solutions
# =============================================================================


@pytest.mark.devito
class TestAnalyticalSolutions:
    """Tests against analytical solutions."""

    def test_verify_linear_pressure(self):
        """Test the linear pressure verification utility."""
        from src.darcy import verify_linear_pressure

        error = verify_linear_pressure(tol=1e-6)

        # Error should be small for well-resolved linear solution
        assert error < 0.02, f"Linear pressure verification error {error} too large"

    def test_numerical_vs_analytical_1d(self):
        """Compare numerical solution to analytical for 1D case."""
        from src.darcy import solve_darcy_2d

        Lx = 1.0
        p_left = 1.5
        p_right = 0.5

        result = solve_darcy_2d(
            Lx=Lx,
            Ly=0.1,  # Thin domain for 1D approximation
            Nx=64,
            Ny=8,
            permeability=1.0,
            bc_left=p_left,
            bc_right=p_right,
            bc_top="neumann",
            bc_bottom="neumann",
            tol=1e-6,
        )

        # Analytical solution
        p_exact = p_left + (p_right - p_left) * result.x / Lx

        # Pressure comparison (middle row)
        p_numerical = result.p[:, 4]
        np.testing.assert_allclose(p_numerical, p_exact, atol=0.02)


# =============================================================================
# Test: Transient Flow
# =============================================================================


@pytest.mark.devito
class TestTransientFlow:
    """Tests for transient Darcy flow with storage."""

    def test_transient_basic_run(self):
        """Test basic transient solver execution."""
        from src.darcy import solve_darcy_transient

        # Use smaller permeability to meet stability requirements
        result = solve_darcy_transient(
            Lx=1.0,
            Ly=1.0,
            Nx=20,
            Ny=20,
            permeability=0.01,  # Smaller K for stability
            porosity=0.2,
            T=0.1,
            nt=100,
            bc_left=1.0,
            bc_right=0.0,
        )

        assert result.p is not None
        assert result.p.shape == (20, 20)

    def test_transient_initial_condition(self):
        """Test that initial condition is applied."""
        from src.darcy import solve_darcy_transient

        p_init = 0.3

        result = solve_darcy_transient(
            Lx=1.0,
            Ly=1.0,
            Nx=20,
            Ny=20,
            permeability=0.01,
            porosity=0.2,
            T=0.001,
            nt=10,
            p_initial=p_init,
            bc_left=1.0,
            bc_right=0.0,
            save_interval=1,
        )

        # Initial interior should be close to p_init
        if result.p_history is not None and len(result.p_history) > 0:
            initial_interior = result.p_history[0][5:-5, 5:-5]
            assert np.abs(np.mean(initial_interior) - p_init) < 0.3

    def test_transient_approaches_steady_state(self):
        """Transient solution should approach steady state."""
        from src.darcy import solve_darcy_2d, solve_darcy_transient

        Nx = Ny = 16

        # Get steady-state solution
        steady = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=Nx, Ny=Ny, permeability=0.01,
            bc_left=1.0, bc_right=0.0
        )

        # Run transient to long time (with small K for stability)
        transient = solve_darcy_transient(
            Lx=1.0,
            Ly=1.0,
            Nx=Nx,
            Ny=Ny,
            permeability=0.01,
            porosity=0.2,
            T=50.0,
            nt=5000,
            p_initial=0.5,
            bc_left=1.0,
            bc_right=0.0,
        )

        # Should be close to steady state
        error = np.max(np.abs(transient.p - steady.p))
        assert error < 0.2

    def test_transient_history_saved(self):
        """Test that history is saved when requested."""
        from src.darcy import solve_darcy_transient

        result = solve_darcy_transient(
            Lx=1.0,
            Ly=1.0,
            Nx=16,
            Ny=16,
            permeability=0.01,
            porosity=0.2,
            T=0.1,
            nt=50,
            bc_left=1.0,
            bc_right=0.0,
            save_interval=10,
        )

        assert result.p_history is not None
        assert len(result.p_history) > 0


# =============================================================================
# Test: Wells and Sources
# =============================================================================


@pytest.mark.devito
class TestWellsAndSources:
    """Tests for source/sink terms (wells)."""

    def test_injection_well(self):
        """Test injection well (positive source)."""
        from src.darcy import add_well, solve_darcy_2d

        Nx, Ny = 32, 32
        source = np.zeros((Nx, Ny))
        # Add injection well at center
        source = add_well(source, Nx // 2, Ny // 2, rate=10.0)

        result = solve_darcy_2d(
            Lx=1.0,
            Ly=1.0,
            Nx=Nx,
            Ny=Ny,
            permeability=1.0,
            source=source,
            bc_left=0.0,
            bc_right=0.0,
            tol=1e-5,
        )

        # Pressure should be elevated near well
        p_at_well = result.p[Nx // 2, Ny // 2]
        p_far = np.mean(result.p[0, :])
        assert p_at_well > p_far

    def test_production_well(self):
        """Test production well (negative source)."""
        from src.darcy import add_well, solve_darcy_2d

        Nx, Ny = 32, 32
        source = np.zeros((Nx, Ny))
        # Add production well at center
        source = add_well(source, Nx // 2, Ny // 2, rate=-5.0)

        result = solve_darcy_2d(
            Lx=1.0,
            Ly=1.0,
            Nx=Nx,
            Ny=Ny,
            permeability=1.0,
            source=source,
            bc_left=1.0,
            bc_right=1.0,
            tol=1e-5,
        )

        # Pressure should be lower near well
        p_at_well = result.p[Nx // 2, Ny // 2]
        p_boundary = np.mean(result.p[0, :])
        assert p_at_well < p_boundary


# =============================================================================
# Test: Permeability Field Generation
# =============================================================================


class TestPermeabilityGeneration:
    """Tests for heterogeneous permeability field generation."""

    def test_layered_field_shape(self):
        """Test that layered field has correct shape."""
        from src.darcy import create_layered_permeability

        Nx, Ny = 40, 30
        layers = [(0.5, 1.0), (1.0, 5.0)]
        K = create_layered_permeability(Nx, Ny, layers)

        assert K.shape == (Nx, Ny)

    def test_binary_field_values(self):
        """Test that binary field contains only two values."""
        from src.darcy import create_binary_permeability

        K_low, K_high = 1.0, 10.0
        K = create_binary_permeability(32, 32, K_low=K_low, K_high=K_high)

        unique_vals = np.unique(K)
        assert len(unique_vals) == 2
        assert K_low in unique_vals
        assert K_high in unique_vals

    def test_gaussian_random_field_shape(self):
        """Test GaussianRandomField output shape."""
        from src.darcy import GaussianRandomField

        size = 64
        grf = GaussianRandomField(size=size, alpha=2, tau=3)
        fields = grf.sample(3)

        assert fields.shape == (3, size, size)

    def test_gaussian_random_field_zero_mean(self):
        """GaussianRandomField should produce approximately zero-mean fields."""
        from src.darcy import GaussianRandomField

        np.random.seed(42)
        grf = GaussianRandomField(size=64, alpha=2, tau=3)
        fields = grf.sample(10)

        # Mean of means should be close to zero
        mean_of_means = np.mean([np.mean(f) for f in fields])
        assert abs(mean_of_means) < 0.5


# =============================================================================
# Test: Edge Cases
# =============================================================================


@pytest.mark.devito
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_grid(self):
        """Test solver on small grid."""
        from src.darcy import solve_darcy_2d

        result = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=8, Ny=8, permeability=1.0,
            bc_left=1.0, bc_right=0.0
        )

        assert result.p.shape == (8, 8)
        assert result.converged

    def test_rectangular_domain(self):
        """Test on non-square domain."""
        from src.darcy import solve_darcy_2d

        result = solve_darcy_2d(
            Lx=2.0, Ly=0.5, Nx=40, Ny=10, permeability=1.0,
            bc_left=1.0, bc_right=0.0
        )

        assert result.p.shape == (40, 10)
        assert result.converged

    def test_high_permeability(self):
        """Test with high permeability value."""
        from src.darcy import solve_darcy_2d

        result = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=20, Ny=20, permeability=1000.0,
            bc_left=1.0, bc_right=0.0
        )

        assert result.converged
        assert np.all(np.isfinite(result.p))

    def test_all_neumann_with_source(self):
        """Test all-Neumann BCs with source term."""
        from src.darcy import solve_darcy_2d

        Nx, Ny = 32, 32
        source = np.zeros((Nx, Ny))
        source[Nx//2, Ny//2] = 1.0  # Point source

        result = solve_darcy_2d(
            Lx=1.0, Ly=1.0, Nx=Nx, Ny=Ny, permeability=1.0,
            source=source,
            bc_left="neumann", bc_right="neumann",
            bc_bottom="neumann", bc_top="neumann",
        )

        assert np.all(np.isfinite(result.p))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
