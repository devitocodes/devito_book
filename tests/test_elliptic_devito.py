"""Tests for Devito elliptic PDE solvers (Laplace and Poisson equations).

This module tests elliptic PDE solvers implemented using Devito, including:
1. Laplace equation: nabla^2 u = 0 (steady-state, no time derivative)
2. Poisson equation: nabla^2 u = f (with source term)

Elliptic PDEs require iterative methods since there is no time evolution.
Common approaches:
- Jacobi iteration with dual buffers
- Pseudo-timestepping (diffusion to steady state)
- Direct solvers (not typically done in Devito)

Per CONTRIBUTING.md: All results must be reproducible with fixed random seeds,
version-pinned dependencies, and automated tests validating examples.
"""

import numpy as np
import pytest

# Check if Devito is available
try:
    from devito import Constant, Eq, Function, Grid, Operator, TimeFunction

    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DEVITO_AVAILABLE, reason="Devito not installed"
)


# =============================================================================
# Test: Grid and Function Creation for Elliptic Problems
# =============================================================================


@pytest.mark.devito
class TestEllipticGridCreation:
    """Test grid and Function creation patterns for elliptic problems."""

    def test_function_vs_timefunction_for_elliptic(self):
        """Test that Function (not TimeFunction) is appropriate for elliptic PDEs.

        For elliptic equations with no time derivative, we use Function
        for static fields. TimeFunction is used only for pseudo-timestepping.
        """
        grid = Grid(shape=(21, 21), extent=(1.0, 1.0))

        # Static field for elliptic problem
        p = Function(name="p", grid=grid, space_order=2)

        # Verify it's a static field (no time dimension)
        assert p.shape == (21, 21)
        assert "time" not in [str(d) for d in p.dimensions]

        # TimeFunction for pseudo-timestepping approach
        u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)
        assert u.time_order == 1
        # Has time buffer slots
        assert u.data.shape[0] > 1

    def test_dual_buffer_pattern_with_functions(self):
        """Test the dual-buffer pattern using two Function objects.

        For iterative Jacobi-style methods, we need two buffers:
        - p: current iteration values
        - p_new: next iteration values
        """
        grid = Grid(shape=(21, 21), extent=(1.0, 1.0))

        # Two separate buffers for Jacobi iteration
        p = Function(name="p", grid=grid, space_order=2)
        p_new = Function(name="p_new", grid=grid, space_order=2)

        # Initialize p with some values
        p.data[:, :] = 0.0
        p_new.data[:, :] = 0.0

        # Verify independent buffers
        p.data[10, 10] = 1.0
        assert p_new.data[10, 10] == 0.0  # p_new unaffected

    def test_grid_dimensions_access(self):
        """Test accessing grid dimensions for boundary condition indexing."""
        grid = Grid(shape=(21, 21), extent=(1.0, 1.0))
        x, y = grid.dimensions

        # Verify dimension properties
        assert str(x) == "x"
        assert str(y) == "y"

        # Access spacing
        hx, hy = grid.spacing
        expected_h = 1.0 / 20  # extent / (shape - 1)
        # Use reasonable tolerance for float32 (Devito default dtype)
        assert abs(float(hx) - expected_h) < 1e-6
        assert abs(float(hy) - expected_h) < 1e-6


# =============================================================================
# Test: Laplace Equation Solver
# =============================================================================


@pytest.mark.devito
class TestLaplaceEquationSolver:
    """Tests for the Laplace equation: nabla^2 p = 0."""

    def test_laplace_jacobi_single_iteration(self):
        """Test a single Jacobi iteration for Laplace equation.

        Jacobi update: p_new[i,j] = (p[i+1,j] + p[i-1,j] + p[i,j+1] + p[i,j-1]) / 4
        """
        Nx, Ny = 21, 21
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))

        p = Function(name="p", grid=grid, space_order=2)
        p_new = Function(name="p_new", grid=grid, space_order=2)

        # Initialize with boundary conditions
        p.data[:, :] = 0.0
        p.data[0, :] = 0.0  # Bottom
        p.data[-1, :] = 1.0  # Top = 1 (Dirichlet)
        p.data[:, 0] = 0.0  # Left
        p.data[:, -1] = 0.0  # Right

        # Initial guess for interior
        p.data[1:-1, 1:-1] = 0.5

        # Jacobi update equation using Laplacian
        # For uniform grid: p_new = (p[i+1,j] + p[i-1,j] + p[i,j+1] + p[i,j-1]) / 4
        # This is equivalent to: p_new = p + (1/4) * h^2 * laplace(p)
        # where laplace uses second-order stencil
        hx, hy = grid.spacing
        h2 = hx * hy  # For uniform grid hx = hy

        # Direct Jacobi formula
        x, y = grid.dimensions
        eq = Eq(
            p_new,
            0.25 * (p.subs(x, x + x.spacing) + p.subs(x, x - x.spacing) +
                    p.subs(y, y + y.spacing) + p.subs(y, y - y.spacing)),
            subdomain=grid.interior,
        )

        op = Operator([eq])
        op.apply()

        # Verify interior was updated (not boundary)
        assert p_new.data[0, 10] == 0.0  # Bottom boundary unchanged
        assert p_new.data[-1, 10] == 0.0  # p_new not set at boundary
        # Interior should have been updated
        assert p_new.data[10, 10] != 0.0

    def test_laplace_dirichlet_bc_enforcement(self):
        """Test Dirichlet boundary condition enforcement in elliptic solve."""
        Nx, Ny = 21, 21
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))
        x, y = grid.dimensions  # Get dimensions before using them
        t = grid.stepping_dim

        # Use TimeFunction for pseudo-timestepping
        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)

        # Set Dirichlet BCs
        p.data[0, :, :] = 0.0
        p.data[1, :, :] = 0.0

        # Specific boundary values
        top_val = 1.0
        p.data[:, -1, :] = top_val  # Top boundary
        p.data[:, 0, :] = 0.0  # Bottom boundary
        p.data[:, :, 0] = 0.0  # Left boundary
        p.data[:, :, -1] = 0.0  # Right boundary

        # Pseudo-timestepping update
        alpha = 0.25  # Diffusion coefficient for stability
        eq = Eq(p.forward, p + alpha * p.laplace, subdomain=grid.interior)

        # Boundary equations to enforce Dirichlet BCs at t+1
        bc_top = Eq(p[t + 1, Ny - 1, y], top_val)
        bc_bottom = Eq(p[t + 1, 0, y], 0)
        bc_left = Eq(p[t + 1, x, 0], 0)
        bc_right = Eq(p[t + 1, x, Ny - 1], 0)

        op = Operator([eq, bc_top, bc_bottom, bc_left, bc_right])

        # Run several iterations
        for _ in range(100):
            op.apply(time_m=0, time_M=0)

        # Verify boundary conditions are maintained
        # Note: corners may have different values due to BC ordering
        # Check interior boundary points (excluding corners)
        assert np.allclose(p.data[0, -1, 1:-1], top_val, atol=1e-6)
        assert np.allclose(p.data[0, 0, 1:-1], 0.0, atol=1e-6)
        assert np.allclose(p.data[0, 1:-1, 0], 0.0, atol=1e-6)
        assert np.allclose(p.data[0, 1:-1, -1], 0.0, atol=1e-6)

    def test_laplace_neumann_bc_copy_trick(self):
        """Test Neumann BC using the copy trick: dp/dy = 0 at boundary.

        For zero-gradient (Neumann) BC at y=0: p[i,0] = p[i,1]
        This implements dp/dy = 0 using first-order approximation.
        """
        Nx, Ny = 21, 21
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))
        x, y = grid.dimensions

        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)
        t = grid.stepping_dim

        # Initialize
        p.data[:, :, :] = 0.5

        # Apply Dirichlet on top, Neumann on bottom
        p.data[:, -1, :] = 1.0  # Top: p = 1

        # Interior update
        alpha = 0.25
        eq = Eq(p.forward, p + alpha * p.laplace, subdomain=grid.interior)

        # Neumann BC at bottom: copy interior value to boundary
        # p[t+1, 0, j] = p[t+1, 1, j] implements dp/dy = 0
        bc_neumann_bottom = Eq(p[t + 1, 0, y], p[t + 1, 1, y])

        # Dirichlet at top
        bc_top = Eq(p[t + 1, Ny - 1, y], 1.0)

        # Periodic-like or Neumann on sides
        bc_left = Eq(p[t + 1, x, 0], p[t + 1, x, 1])
        bc_right = Eq(p[t + 1, x, Ny - 1], p[t + 1, x, Ny - 2])

        op = Operator([eq, bc_neumann_bottom, bc_top, bc_left, bc_right])

        # Run to approach steady state
        for _ in range(200):
            op.apply(time_m=0, time_M=0)

        # Verify Neumann condition: gradient at bottom should be ~0
        # p[1,:] should be approximately equal to p[0,:]
        grad_bottom = np.abs(p.data[0, 1, 1:-1] - p.data[0, 0, 1:-1])
        assert np.max(grad_bottom) < 0.1  # Gradient approaches zero

    def test_laplace_convergence_to_steady_state(self):
        """Test that pseudo-timestepping converges to steady state."""
        Nx, Ny = 21, 21
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))
        x, y = grid.dimensions
        t = grid.stepping_dim

        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)

        # Set initial guess and boundary conditions
        # Initialize with linear interpolation as good initial guess
        y_coords = np.linspace(0, 1, Ny)
        for i in range(Nx):
            p.data[0, i, :] = y_coords
            p.data[1, i, :] = y_coords

        # Enforce BCs
        p.data[:, 0, :] = 0.0  # Bottom = 0
        p.data[:, -1, :] = 1.0  # Top = 1

        # Pseudo-timestepping
        alpha = 0.2
        eq = Eq(p.forward, p + alpha * p.laplace, subdomain=grid.interior)

        # Boundary equations - with Dirichlet on all sides for simpler test
        bc_top = Eq(p[t + 1, Ny - 1, y], 1.0)
        bc_bottom = Eq(p[t + 1, 0, y], 0.0)
        # Linear interpolation on left and right
        bc_left = Eq(p[t + 1, x, 0], x / (Nx - 1))
        bc_right = Eq(p[t + 1, x, Ny - 1], x / (Nx - 1))

        op = Operator([eq, bc_top, bc_bottom, bc_left, bc_right])

        # Track convergence
        prev_norm = np.inf
        tolerances = []

        for iteration in range(500):
            op.apply(time_m=0, time_M=0)

            # Measure change from previous iteration
            current_norm = np.sum(p.data[0, 1:-1, 1:-1] ** 2)
            change = abs(current_norm - prev_norm)
            tolerances.append(change)
            prev_norm = current_norm

            if change < 1e-8:
                break

        # Should have converged
        assert tolerances[-1] < 1e-4, f"Did not converge: final change = {tolerances[-1]}"

        # Verify solution is physically reasonable
        # For this setup with linear BCs, solution should be approximately linear
        center_col = p.data[0, :, Nx // 2]
        x_coords = np.linspace(0, 1, Nx)
        # Check that values are monotonically increasing (roughly)
        assert center_col[0] < center_col[-1], "Solution should increase from bottom to top"
        # Check boundaries
        assert abs(p.data[0, 0, Nx // 2]) < 0.1, "Bottom should be near 0"
        assert abs(p.data[0, -1, Nx // 2] - 1.0) < 0.1, "Top should be near 1"

    def test_buffer_swapping_via_argument_substitution(self):
        """Test the buffer swapping pattern using argument substitution.

        In Devito, when using two Functions for Jacobi iteration,
        we can swap buffers by passing them as arguments.
        """
        Nx, Ny = 11, 11
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))
        x, y = grid.dimensions

        # Create symbolic functions
        p = Function(name="p", grid=grid, space_order=2)
        p_new = Function(name="p_new", grid=grid, space_order=2)

        # Initialize
        p.data[:, :] = 0.0
        p.data[-1, :] = 1.0  # Top = 1
        p_new.data[:, :] = 0.0

        # Jacobi update
        eq = Eq(
            p_new,
            0.25 * (p.subs(x, x + x.spacing) + p.subs(x, x - x.spacing) +
                    p.subs(y, y + y.spacing) + p.subs(y, y - y.spacing)),
            subdomain=grid.interior,
        )

        # Boundary update for p_new
        bc_top = Eq(p_new.indexed[Nx - 1, y], 1.0)
        bc_bottom = Eq(p_new.indexed[0, y], 0.0)
        bc_left = Eq(p_new.indexed[x, 0], 0.0)
        bc_right = Eq(p_new.indexed[x, Ny - 1], 0.0)

        op = Operator([eq, bc_top, bc_bottom, bc_left, bc_right])

        # Run iterations with manual buffer swap
        for _ in range(50):
            op.apply()
            # Swap: copy p_new to p
            p.data[:, :] = p_new.data[:, :]

        # Solution should be developing
        assert not np.allclose(p.data[5, 5], 0.0)
        assert p.data[-1, 5] == 1.0  # Top boundary maintained


# =============================================================================
# Test: Poisson Equation Solver
# =============================================================================


@pytest.mark.devito
class TestPoissonEquationSolver:
    """Tests for the Poisson equation: nabla^2 p = f."""

    def test_poisson_with_point_source(self):
        """Test Poisson equation with a point source.

        nabla^2 p = f where f is nonzero at a single point (source).
        We use the formulation: p_{t} = laplace(p) + f
        which converges to laplace(p) = -f at steady state.
        """
        Nx, Ny = 31, 31
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))
        x, y = grid.dimensions
        t = grid.stepping_dim

        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)
        f = Function(name="f", grid=grid)  # Source term

        # Initialize with small positive values
        p.data[:, :, :] = 0.01

        # Point source at center (positive source will create a peak)
        f.data[:, :] = 0.0
        center = Nx // 2
        f.data[center, center] = 5.0  # Positive source

        # Pseudo-timestepping for Poisson: p_t = laplace(p) + f
        # At steady state: laplace(p) = -f
        alpha = 0.15
        eq = Eq(
            p.forward,
            p + alpha * (p.laplace + f),
            subdomain=grid.interior,
        )

        # Homogeneous Dirichlet BCs
        bc_top = Eq(p[t + 1, Nx - 1, y], 0.0)
        bc_bottom = Eq(p[t + 1, 0, y], 0.0)
        bc_left = Eq(p[t + 1, x, 0], 0.0)
        bc_right = Eq(p[t + 1, x, Ny - 1], 0.0)

        op = Operator([eq, bc_top, bc_bottom, bc_left, bc_right])

        # Run to steady state with many iterations
        for _ in range(2000):
            op.apply(time_m=0, time_M=0)

        # Solution should have elevated values near the source
        solution = p.data[0, :, :]

        # The interior should have positive values due to the source
        interior = solution[5:-5, 5:-5]
        assert np.mean(interior) > 0, "Interior mean should be positive with positive source"

        # Check that value at center region is higher than near boundaries
        center_val = solution[center, center]
        edge_avg = (np.mean(solution[2, :]) + np.mean(solution[-3, :]) +
                    np.mean(solution[:, 2]) + np.mean(solution[:, -3])) / 4
        assert center_val > edge_avg, "Center should have higher value than near boundaries"

    def test_poisson_timefunction_pseudo_timestepping(self):
        """Test TimeFunction approach for pseudo-timestepping Poisson solver.

        Uses u_t = a * laplace(u) + f to iterate to steady state.
        At steady state: laplace(u) = -f/a (approximately)
        """
        Nx, Ny = 21, 21
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))
        x, y = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)
        source = Function(name="source", grid=grid)

        # Uniform positive source term
        source.data[:, :] = 0.5

        # Initialize with small positive values to help convergence
        u.data[:, :, :] = 0.05

        # Pseudo-time diffusion with source
        a = Constant(name="a")
        eq = Eq(u.forward, u + a * (u.laplace + source), subdomain=grid.interior)

        # Dirichlet BCs
        bc_top = Eq(u[t + 1, Nx - 1, y], 0.0)
        bc_bottom = Eq(u[t + 1, 0, y], 0.0)
        bc_left = Eq(u[t + 1, x, 0], 0.0)
        bc_right = Eq(u[t + 1, x, Ny - 1], 0.0)

        op = Operator([eq, bc_top, bc_bottom, bc_left, bc_right])

        # Run with small pseudo-timestep for many iterations
        for _ in range(1000):
            op.apply(time_m=0, time_M=0, a=0.1)

        # Solution should be positive in interior with positive source
        interior = u.data[0, 2:-2, 2:-2]  # Away from boundaries
        assert np.mean(interior) > 0, "Interior mean should be positive with positive source"

        # Boundaries should remain close to zero
        assert np.allclose(u.data[0, 0, 1:-1], 0.0, atol=0.05)
        assert np.allclose(u.data[0, -1, 1:-1], 0.0, atol=0.05)

    def test_poisson_boundary_conditions_at_t_plus_1(self):
        """Test that boundary conditions are properly applied at t+1.

        Critical for pseudo-timestepping: BCs must be applied to the
        new time level, not the current one.
        """
        Nx, Ny = 11, 11
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))
        t = grid.stepping_dim
        x, y = grid.dimensions

        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)

        # Initialize
        p.data[:, :, :] = 0.5  # Arbitrary initial value

        # Non-zero Dirichlet BC
        bc_value = 2.0

        # Interior update
        eq = Eq(p.forward, p + 0.25 * p.laplace, subdomain=grid.interior)

        # BC at t+1
        bc = Eq(p[t + 1, Nx - 1, y], bc_value)

        op = Operator([eq, bc])
        op.apply(time_m=0, time_M=0)

        # Check that boundary was set correctly at new time level
        # After one step, data[1] contains the new values
        assert np.allclose(p.data[1, Nx - 1, :], bc_value)


# =============================================================================
# Test: Verification Against Analytical Solutions
# =============================================================================


@pytest.mark.devito
class TestEllipticVerification:
    """Verification tests against analytical solutions."""

    def test_laplace_1d_linear_solution(self):
        """Test 1D Laplace: d^2p/dx^2 = 0 with p(0)=0, p(1)=1.

        Analytical solution: p(x) = x
        """
        Nx = 51
        grid = Grid(shape=(Nx,), extent=(1.0,))
        x_dim = grid.dimensions[0]
        t = grid.stepping_dim

        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)

        # Initialize with linear interpolation (good initial guess)
        x_coords = np.linspace(0, 1, Nx)
        p.data[0, :] = x_coords
        p.data[1, :] = x_coords

        # BCs
        p.data[:, 0] = 0.0
        p.data[:, -1] = 1.0

        # Pseudo-timestepping with smaller alpha for stability
        eq = Eq(p.forward, p + 0.3 * p.dx2, subdomain=grid.interior)
        bc_left = Eq(p[t + 1, 0], 0.0)
        bc_right = Eq(p[t + 1, Nx - 1], 1.0)

        op = Operator([eq, bc_left, bc_right])

        for _ in range(200):
            op.apply(time_m=0, time_M=0)

        # Compare to analytical solution
        analytical = x_coords
        numerical = p.data[0, :]

        error = np.max(np.abs(numerical - analytical))
        assert error < 0.05, f"Error {error} exceeds tolerance"

    def test_laplace_2d_known_solution(self):
        """Test 2D Laplace with known harmonic solution.

        If p(x,y) = x + y, then laplace(p) = 0.
        Test with boundary conditions consistent with this solution.
        """
        Nx, Ny = 21, 21
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))
        x_dim, y_dim = grid.dimensions
        t = grid.stepping_dim

        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)

        # Create coordinate arrays for BCs
        x_coords = np.linspace(0, 1, Nx)
        y_coords = np.linspace(0, 1, Ny)

        # Initialize with analytical solution (this should be preserved)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        p.data[0, :, :] = X + Y
        p.data[1, :, :] = X + Y

        # Set boundary conditions from analytical solution
        # Bottom (x, 0): p = x
        # Top (x, 1): p = x + 1
        # Left (0, y): p = y
        # Right (1, y): p = 1 + y

        # Update interior only
        eq = Eq(p.forward, p + 0.25 * p.laplace, subdomain=grid.interior)

        op = Operator([eq])

        # Run a few iterations
        for _ in range(10):
            op.apply(time_m=0, time_M=0)
            # Re-apply boundary conditions
            p.data[0, 0, :] = y_coords  # Left
            p.data[0, -1, :] = 1.0 + y_coords  # Right
            p.data[0, :, 0] = x_coords  # Bottom
            p.data[0, :, -1] = x_coords + 1.0  # Top

        # Solution should remain close to x + y
        analytical = X + Y
        error = np.max(np.abs(p.data[0, :, :] - analytical))
        assert error < 0.05, f"Solution deviates from analytical: error = {error}"

    def test_solution_boundedness(self):
        """Test that elliptic solution remains bounded by boundary values.

        Maximum principle: solution of Laplace equation achieves its
        max and min on the boundary, not in the interior.
        """
        Nx, Ny = 21, 21
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))
        x, y = grid.dimensions

        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)

        # Set boundary values
        bc_min = 0.0
        bc_max = 1.0
        p.data[:, :, :] = 0.5  # Interior guess

        # Bottom = 0, Top = 1, Left/Right = linear interpolation
        p.data[:, 0, :] = bc_min
        p.data[:, -1, :] = bc_max
        y_vals = np.linspace(bc_min, bc_max, Ny)
        p.data[:, :, 0] = y_vals
        p.data[:, :, -1] = y_vals

        # Pseudo-timestepping
        t = grid.stepping_dim
        eq = Eq(p.forward, p + 0.2 * p.laplace, subdomain=grid.interior)
        bc_bottom = Eq(p[t + 1, 0, y], bc_min)
        bc_top = Eq(p[t + 1, Nx - 1, y], bc_max)
        bc_left = Eq(p[t + 1, x, 0], p[t, x, 0])  # Keep interpolated values
        bc_right = Eq(p[t + 1, x, Ny - 1], p[t, x, Ny - 1])

        op = Operator([eq, bc_bottom, bc_top, bc_left, bc_right])

        for _ in range(200):
            op.apply(time_m=0, time_M=0)

        # Interior solution should be bounded by boundary values
        interior = p.data[0, 1:-1, 1:-1]
        assert np.min(interior) >= bc_min - 0.01
        assert np.max(interior) <= bc_max + 0.01

    def test_conservation_with_zero_source(self):
        """Test that Laplace equation conserves the mean value property.

        For Laplace equation, the value at any interior point equals
        the average of values in a neighborhood (discrete version).
        """
        Nx, Ny = 21, 21
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))
        x, y = grid.dimensions

        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)
        t = grid.stepping_dim

        # Simple boundary conditions
        p.data[:, :, :] = 0.0
        p.data[:, -1, :] = 1.0  # Top = 1

        # Run to steady state
        eq = Eq(p.forward, p + 0.2 * p.laplace, subdomain=grid.interior)
        bc_top = Eq(p[t + 1, Nx - 1, y], 1.0)
        bc_bottom = Eq(p[t + 1, 0, y], 0.0)
        bc_left = Eq(p[t + 1, x, 0], p[t + 1, x, 1])  # Neumann
        bc_right = Eq(p[t + 1, x, Ny - 1], p[t + 1, x, Ny - 2])  # Neumann

        op = Operator([eq, bc_top, bc_bottom, bc_left, bc_right])

        for _ in range(500):
            op.apply(time_m=0, time_M=0)

        # Test mean value property at interior point
        i, j = 10, 10
        val = p.data[0, i, j]
        avg_neighbors = 0.25 * (
            p.data[0, i + 1, j]
            + p.data[0, i - 1, j]
            + p.data[0, i, j + 1]
            + p.data[0, i, j - 1]
        )

        # At steady state, value should equal average of neighbors
        assert abs(val - avg_neighbors) < 0.05


# =============================================================================
# Test: Edge Cases and Error Handling
# =============================================================================


@pytest.mark.devito
class TestEllipticEdgeCases:
    """Test edge cases for elliptic solvers."""

    def test_uniform_dirichlet_gives_uniform_solution(self):
        """Test that uniform Dirichlet BCs give uniform solution."""
        Nx, Ny = 11, 11
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))
        x, y = grid.dimensions
        t = grid.stepping_dim

        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)

        # All boundaries = 0.5, initialize interior to same
        bc_val = 0.5
        p.data[:, :, :] = bc_val

        eq = Eq(p.forward, p + 0.2 * p.laplace, subdomain=grid.interior)

        # Include boundary equations in operator
        bc_top = Eq(p[t + 1, Nx - 1, y], bc_val)
        bc_bottom = Eq(p[t + 1, 0, y], bc_val)
        bc_left = Eq(p[t + 1, x, 0], bc_val)
        bc_right = Eq(p[t + 1, x, Ny - 1], bc_val)

        op = Operator([eq, bc_top, bc_bottom, bc_left, bc_right])

        # Run iterations
        for _ in range(50):
            op.apply(time_m=0, time_M=0)

        # Solution should remain uniformly 0.5 (it's already at equilibrium)
        interior = p.data[0, 1:-1, 1:-1]
        assert np.allclose(interior, bc_val, atol=0.01)

    def test_small_grid(self):
        """Test solver works on minimum viable grid size."""
        Nx, Ny = 5, 5
        grid = Grid(shape=(Nx, Ny), extent=(1.0, 1.0))

        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)

        # Initialize
        p.data[:, :, :] = 0.0
        p.data[:, -1, :] = 1.0

        eq = Eq(p.forward, p + 0.2 * p.laplace, subdomain=grid.interior)

        op = Operator([eq])

        # Should run without error
        for _ in range(10):
            op.apply(time_m=0, time_M=0)
            p.data[0, -1, :] = 1.0  # Maintain BC
            p.data[0, 0, :] = 0.0

        # Verify something happened
        assert not np.allclose(p.data[0, :, :], 0.0)

    def test_asymmetric_domain(self):
        """Test solver on non-square domain."""
        Nx, Ny = 31, 11  # Rectangular domain
        grid = Grid(shape=(Nx, Ny), extent=(3.0, 1.0))
        x, y = grid.dimensions
        t = grid.stepping_dim

        p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)

        # Initialize
        p.data[:, :, :] = 0.0
        p.data[:, -1, :] = 1.0  # Top = 1

        eq = Eq(p.forward, p + 0.15 * p.laplace, subdomain=grid.interior)
        bc_top = Eq(p[t + 1, Nx - 1, y], 1.0)
        bc_bottom = Eq(p[t + 1, 0, y], 0.0)

        op = Operator([eq, bc_top, bc_bottom])

        for _ in range(200):
            op.apply(time_m=0, time_M=0)

        # Solution should vary primarily in x direction (short axis)
        # Check boundaries maintained
        assert np.allclose(p.data[0, 0, :], 0.0, atol=1e-10)
        assert np.allclose(p.data[0, -1, :], 1.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
