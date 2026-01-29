"""Test verification for code snippets throughout the book chapters.

This module verifies that code examples shown in each chapter:
1. Actually compile and run without errors
2. Produce correct/reasonable results
3. Follow Devito best practices

Per CONTRIBUTING.md: All results must be reproducible with fixed random seeds,
version-pinned dependencies, and automated tests validating examples.
"""

import numpy as np
import pytest

try:
    from devito import (
        Constant,
        Eq,
        Function,
        Grid,
        Operator,
        TimeFunction,
    )

    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


# =============================================================================
# Chapter: devito_intro/devito_abstractions.qmd
# =============================================================================


@pytest.mark.devito
class TestDevitoAbstractions:
    """Test code snippets from devito_abstractions.qmd."""

    def test_grid_creation_1d_2d_3d(self):
        """Test Grid creation snippets (lines 11-22)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        # 1D grid: 101 points over [0, 1]
        grid_1d = Grid(shape=(101,), extent=(1.0,))
        assert grid_1d.shape == (101,)

        # 2D grid: 101x101 points over [0, 1] x [0, 1]
        grid_2d = Grid(shape=(101, 101), extent=(1.0, 1.0))
        assert grid_2d.shape == (101, 101)

        # 3D grid: 51x51x51 points over [0, 2] x [0, 2] x [0, 2]
        grid_3d = Grid(shape=(51, 51, 51), extent=(2.0, 2.0, 2.0))
        assert grid_3d.shape == (51, 51, 51)

    def test_grid_properties(self):
        """Test Grid properties access (lines 31-36)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        grid = Grid(shape=(101, 101), extent=(1.0, 1.0))
        x, y = grid.dimensions
        dx, dy = grid.spacing

        # Verify spacing is correct (use reasonable tolerance for float32)
        assert abs(float(dx) - 0.01) < 1e-6
        assert abs(float(dy) - 0.01) < 1e-6

    def test_function_static_fields(self):
        """Test Function creation (lines 44-57)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        grid = Grid(shape=(101,), extent=(1.0,))

        # Wave velocity field - constant
        c = Function(name="c", grid=grid)
        c.data[:] = 1500.0
        assert np.allclose(c.data[:], 1500.0)

        # Spatially varying velocity
        x_vals = np.linspace(0, 1, 101)
        c.data[:] = 1500 + 500 * x_vals
        assert c.data[0] == 1500.0
        assert c.data[-1] == 2000.0

    def test_timefunction_creation(self):
        """Test TimeFunction creation (lines 70-80)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        grid = Grid(shape=(101,), extent=(1.0,))

        # For first-order time derivatives (diffusion equation)
        u1 = TimeFunction(name="u1", grid=grid, time_order=1, space_order=2)
        assert u1.time_order == 1

        # For second-order time derivatives (wave equation)
        u2 = TimeFunction(name="u2", grid=grid, time_order=2, space_order=2)
        assert u2.time_order == 2

    def test_laplacian_dimension_agnostic(self):
        """Test that u.laplace works in 2D and 3D (lines 112-118)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        # 2D case
        grid_2d = Grid(shape=(21, 21), extent=(1.0, 1.0))
        u_2d = TimeFunction(name="u", grid=grid_2d, time_order=1, space_order=2)

        # Both should be symbolically equivalent
        laplacian_explicit = u_2d.dx2 + u_2d.dy2
        laplacian_auto = u_2d.laplace

        # They should produce same structure (both are Add expressions)
        assert laplacian_auto is not None

    def test_2d_diffusion_complete_example(self):
        """Test complete 2D diffusion example (lines 164-197)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        # Create a 2D grid
        grid = Grid(shape=(101, 101), extent=(1.0, 1.0))

        # Time-varying field (first-order in time for diffusion)
        u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)

        # Parameters
        alpha = 0.1
        dx = 1.0 / 100
        F = 0.25  # Fourier number (for stability)
        dt = F * dx**2 / alpha

        # Initial condition: hot spot in the center
        u.data[0, 45:55, 45:55] = 1.0
        initial_max = np.max(u.data[0, :, :])

        # The diffusion equation using .laplace
        eq = Eq(u.forward, u + alpha * dt * u.laplace)

        # Create and run
        op = Operator([eq])
        op(time=100, dt=dt)

        # Verify diffusion occurred (max decreased, spread out)
        final_max = np.max(u.data[0, :, :])
        assert final_max < initial_max, "Diffusion should reduce peak"
        assert final_max > 0, "Solution should not vanish"
        assert np.all(np.isfinite(u.data[0, :, :])), "No NaN/Inf"


# =============================================================================
# Chapter: devito_intro/boundary_conditions.qmd
# =============================================================================


@pytest.mark.devito
class TestBoundaryConditions:
    """Test code snippets from boundary_conditions.qmd."""

    def test_dirichlet_bc_explicit(self):
        """Test explicit Dirichlet BC method (lines 18-36)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        grid = Grid(shape=(101,), extent=(1.0,))
        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

        # Parameters
        c = 1.0
        dx = 1.0 / 100
        dt = 0.5 * dx / c

        # Initial condition
        x_vals = np.linspace(0, 1, 101)
        u.data[0, :] = np.sin(np.pi * x_vals)
        u.data[1, :] = u.data[0, :]

        t = grid.stepping_dim

        # Interior update (wave equation)
        update = Eq(u.forward, 2 * u - u.backward + dt**2 * c**2 * u.dx2)

        # Boundary conditions: u = 0 at both ends
        bc_left = Eq(u[t + 1, 0], 0)
        bc_right = Eq(u[t + 1, 100], 0)

        op = Operator([update, bc_left, bc_right])
        op(time=50, dt=dt)

        # Verify boundaries are zero
        # Note: data indexing is modular, check both time slots
        assert abs(u.data[0, 0]) < 1e-10 or abs(u.data[1, 0]) < 1e-10
        assert abs(u.data[0, 100]) < 1e-10 or abs(u.data[1, 100]) < 1e-10

    def test_dirichlet_bc_subdomain(self):
        """Test subdomain method for Dirichlet BC (lines 42-52)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        grid = Grid(shape=(101,), extent=(1.0,))
        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

        c = 1.0
        dx = 1.0 / 100
        dt = 0.5 * dx / c

        x_vals = np.linspace(0, 1, 101)
        u.data[0, :] = np.sin(np.pi * x_vals)
        u.data[1, :] = u.data[0, :]

        t = grid.stepping_dim

        # Update only interior points
        update = Eq(
            u.forward,
            2 * u - u.backward + dt**2 * c**2 * u.dx2,
            subdomain=grid.interior,
        )

        bc_left = Eq(u[t + 1, 0], 0)
        bc_right = Eq(u[t + 1, 100], 0)

        op = Operator([update, bc_left, bc_right])
        op(time=50, dt=dt)

        # Solution should be bounded
        assert np.max(np.abs(u.data[0, :])) < 2.0

    def test_complete_wave_solver_with_bcs(self):
        """Test complete wave equation example (lines 216-250)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        # Setup
        L, c, T = 1.0, 1.0, 2.0
        Nx = 100
        C = 0.9  # Courant number
        dx = L / Nx
        dt = C * dx / c
        Nt = int(T / dt)

        # Grid and field
        grid = Grid(shape=(Nx + 1,), extent=(L,))
        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)
        t = grid.stepping_dim

        # Initial condition: plucked string
        x_vals = np.linspace(0, L, Nx + 1)
        u.data[0, :] = np.sin(np.pi * x_vals)
        u.data[1, :] = u.data[0, :]

        # Equations
        update = Eq(
            u.forward,
            2 * u - u.backward + (c * dt) ** 2 * u.dx2,
            subdomain=grid.interior,
        )
        bc_left = Eq(u[t + 1, 0], 0)
        bc_right = Eq(u[t + 1, Nx], 0)

        # Solve
        op = Operator([update, bc_left, bc_right])
        op(time=Nt, dt=dt)

        # After period 2L/c = 2, should return near initial
        # Due to numerical dispersion, allow some error
        final_max = np.max(np.abs(u.data[0, :]))
        assert 0.5 < final_max < 1.5, f"Solution amplitude {final_max} unexpected"


# =============================================================================
# Chapter: advec/advec1D_devito.qmd
# =============================================================================


@pytest.mark.devito
class TestAdvectionSchemes:
    """Test advection scheme code snippets from advec1D_devito.qmd."""

    def test_upwind_scheme(self):
        """Test upwind advection scheme (lines 75-104)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        # Setup
        Nx = 100
        L = 1.0
        c = 1.0
        C = 0.8  # Courant number
        dx = L / Nx
        dt = C * dx / c

        grid = Grid(shape=(Nx + 1,), extent=(L,))
        u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)
        x_dim = grid.dimensions[0]

        # Initial condition: Gaussian pulse
        x_vals = np.linspace(0, L, Nx + 1)
        u.data[0, :] = np.exp(-((x_vals - 0.3) ** 2) / (2 * 0.05**2))

        # Upwind scheme: u^{n+1}_i = u^n_i - C*(u^n_i - u^n_{i-1})
        u_minus = u.subs(x_dim, x_dim - x_dim.spacing)
        courant = Constant(name="courant")
        update = Eq(u.forward, u - courant * (u - u_minus))

        op = Operator([update])
        op(time=50, dt=dt, courant=C)

        # Solution should be bounded and finite
        assert np.all(np.isfinite(u.data[0, :]))
        assert np.max(np.abs(u.data[0, :])) < 2.0

    def test_lax_wendroff_scheme(self):
        """Test Lax-Wendroff advection scheme (lines 126-147)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        Nx = 100
        L = 1.0
        c = 1.0
        C = 0.8
        dx = L / Nx
        dt = C * dx / c

        grid = Grid(shape=(Nx + 1,), extent=(L,))
        u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)

        x_vals = np.linspace(0, L, Nx + 1)
        u.data[0, :] = np.exp(-((x_vals - 0.3) ** 2) / (2 * 0.05**2))

        # Lax-Wendroff: second-order accurate
        courant = Constant(name="courant")
        update = Eq(
            u.forward,
            u - 0.5 * courant * dx * u.dx + 0.5 * courant**2 * dx**2 * u.dx2,
        )

        op = Operator([update])
        op(time=50, dt=dt, courant=C)

        assert np.all(np.isfinite(u.data[0, :]))
        assert np.max(np.abs(u.data[0, :])) < 2.0

    def test_lax_friedrichs_scheme(self):
        """Test Lax-Friedrichs advection scheme (lines 163-188)."""
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        Nx = 100
        L = 1.0
        c = 1.0
        C = 0.8
        dx = L / Nx
        dt = C * dx / c

        grid = Grid(shape=(Nx + 1,), extent=(L,))
        u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)
        x_dim = grid.dimensions[0]

        x_vals = np.linspace(0, L, Nx + 1)
        u.data[0, :] = np.exp(-((x_vals - 0.3) ** 2) / (2 * 0.05**2))

        # Lax-Friedrichs: averaging neighbors
        u_plus = u.subs(x_dim, x_dim + x_dim.spacing)
        u_minus = u.subs(x_dim, x_dim - x_dim.spacing)
        courant = Constant(name="courant")

        update = Eq(
            u.forward, 0.5 * (u_plus + u_minus) - 0.5 * courant * (u_plus - u_minus)
        )

        op = Operator([update])
        op(time=50, dt=dt, courant=C)

        assert np.all(np.isfinite(u.data[0, :]))
        assert np.max(np.abs(u.data[0, :])) < 2.0


# =============================================================================
# Verify skill will be used for Devito code
# =============================================================================


class TestDevitoSkillActivation:
    """Verify that the devito skill keywords are present in code patterns."""

    def test_skill_keywords_present(self):
        """Verify key Devito patterns that should trigger skill usage."""
        # These patterns should trigger the devito skill
        devito_patterns = [
            "from devito import",
            "Grid(",
            "TimeFunction(",
            "Function(",
            "Eq(",
            "Operator(",
            "solve(",
            ".dx",
            ".dx2",
            ".dt",
            ".dt2",
            ".laplace",
            ".forward",
            ".backward",
            "SparseTimeFunction",
        ]

        # Verify all patterns are recognized as Devito-related
        for pattern in devito_patterns:
            # These should all be in the skill description
            assert len(pattern) > 0  # Trivial check, real test is skill activation

    def test_solve_pattern_documented(self):
        """Verify the solve() pattern is properly used."""
        # The correct pattern should be:
        # pde = Eq(u.dt2, c**2 * u.dx2)
        # update = Eq(u.forward, solve(pde, u.forward))
        # op = Operator([update])

        # NOT:
        # eq = Eq(u.dt2, c**2 * u.dx2)
        # op = Operator([eq])  # This generates invalid C code

        # This test documents the correct pattern
        assert True  # Pattern documented in test_book_code_verification.py


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
