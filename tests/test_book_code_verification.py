"""Test verification for code examples in the book.

This module verifies that code examples shown in the book:
1. Actually run without errors
2. Produce correct results
3. Follow Devito best practices

Per CONTRIBUTING.md: All results must be reproducible with fixed random seeds,
version-pinned dependencies, and automated tests validating examples.
"""

import numpy as np
import pytest

try:
    from devito import Constant, Eq, Grid, Operator, TimeFunction, solve
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@pytest.mark.devito
class TestIndexQmdWaveEquation:
    """Test the wave equation code shown in index.qmd (the landing page)."""

    def test_symbolic_pde_form_needs_solve(self):
        """Test that Eq(u.dt2, c**2 * u.dx2) needs solve() for correct behavior.

        The code in index.qmd line 39 uses:
            eq = Eq(u.dt2, c**2 * u.dx2)

        This is the symbolic PDE form. Devito needs an explicit update
        equation, so we should use solve() to derive the stencil:
            pde = Eq(u.dt2, c**2 * u.dx2)
            update = Eq(u.forward, solve(pde, u.forward))
        """
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        grid = Grid(shape=(101,), extent=(1.0,))
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)
        c = 1.0

        # Set initial condition (Gaussian pulse)
        x_coords = np.linspace(0, 1.0, 101)
        u.data[0, :] = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        u.data[1, :] = u.data[0, :]

        # Method 1: Original code from index.qmd (symbolic PDE form)
        # This may not work correctly as it's not an explicit update
        eq_symbolic = Eq(u.dt2, c**2 * u.dx2)

        # Method 2: Correct approach using solve()
        dt_const = Constant(name='dt')
        pde = Eq(u.dt2, c**2 * u.dx2)
        update = Eq(u.forward, solve(pde, u.forward))

        # Both should create operators, but behavior differs
        try:
            op_symbolic = Operator([eq_symbolic])
            # This test documents that the symbolic form may not work as expected
            # The operator might be created but not produce a useful update
        except Exception as e:
            pytest.fail(f"Symbolic PDE form failed to create operator: {e}")

        op_correct = Operator([update])

        # Run the correct version and verify it works
        u.data[0, :] = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        u.data[1, :] = u.data[0, :]
        op_correct.apply(time_M=10, dt=0.001)

        # Solution should still be bounded (not exploding)
        assert np.max(np.abs(u.data[0, :])) < 10.0, "Solution exploded"

    def test_correct_wave_equation_pattern(self):
        """Test the recommended pattern for wave equation in Devito.

        This is the pattern that should be used in index.qmd:
            pde = Eq(u.dt2, c**2 * u.dx2)
            update = Eq(u.forward, solve(pde, u.forward))
            op = Operator([update])
        """
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        # Problem setup
        L = 1.0
        Nx = 100
        c = 1.0
        C = 0.5  # Courant number

        dx = L / Nx
        dt = C * dx / c
        Nt = 100

        grid = Grid(shape=(Nx + 1,), extent=(L,))
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

        # Initial condition: Gaussian pulse
        x_coords = np.linspace(0, L, Nx + 1)
        u.data[0, :] = np.exp(-((x_coords - 0.5 * L) ** 2) / (2 * 0.1**2))
        u.data[1, :] = u.data[0, :]

        # Correct pattern: Define PDE symbolically, then solve for u.forward
        c_const = Constant(name='c')
        pde = Eq(u.dt2, c_const**2 * u.dx2)
        update = Eq(u.forward, solve(pde, u.forward))

        op = Operator([update])
        op.apply(time_M=Nt, dt=dt, c=c)

        # Verify solution is reasonable
        max_val = np.max(np.abs(u.data[0, :]))
        assert max_val < 2.0, f"Solution amplitude {max_val} too large"
        assert max_val > 0.01, f"Solution amplitude {max_val} too small"


@pytest.mark.devito
class TestWhatIsDevitoQmdDiffusion:
    """Test the diffusion code shown in what_is_devito.qmd."""

    def test_diffusion_missing_variables(self):
        """Test that the diffusion code in what_is_devito.qmd has issues.

        The code at lines 51-69 uses `alpha` and `dt` without defining them:
            eq = Eq(u.forward, u + alpha * dt * u.dx2)

        This test verifies the corrected version works.
        """
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        # Create computational grid
        grid = Grid(shape=(101,), extent=(1.0,))

        # Define the unknown field
        u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)

        # Set initial condition
        u.data[0, 50] = 1.0

        # CORRECTED: Define alpha and dt properly
        alpha = 1.0  # diffusion coefficient
        Nx = 100
        L = 1.0
        dx = L / Nx
        F = 0.5  # Fourier number for stability
        dt = F * dx**2 / alpha  # Time step from stability

        # Define the PDE update equation (CORRECTED)
        eq = Eq(u.forward, u + alpha * dt * u.dx2)

        # Create and run the operator
        op = Operator([eq])
        op(time=100, dt=dt)

        # Verify solution is bounded and diffusing
        max_val = np.max(u.data[0, :])
        assert max_val < 1.0, "Diffusion should reduce peak"
        assert max_val > 0.0, "Solution should not go to zero immediately"

    def test_diffusion_with_solve_pattern(self):
        """Test the recommended pattern using solve() for diffusion.

        Better approach:
            a_const = Constant(name='a')
            pde = u.dt - a_const * u.dx2
            stencil = Eq(u.forward, solve(pde, u.forward))
        """
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        # Parameters
        L = 1.0
        Nx = 100
        alpha = 1.0
        F = 0.5

        dx = L / Nx
        dt = F * dx**2 / alpha

        grid = Grid(shape=(Nx + 1,), extent=(L,))
        u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)

        # Initial condition: sinusoidal
        x_coords = np.linspace(0, L, Nx + 1)
        u.data[0, :] = np.sin(np.pi * x_coords / L)

        # Use solve() pattern
        a_const = Constant(name='a')
        pde = u.dt - a_const * u.dx2
        stencil = Eq(u.forward, solve(pde, u.forward))

        # Boundary conditions
        bc_left = Eq(u[grid.stepping_dim + 1, 0], 0)
        bc_right = Eq(u[grid.stepping_dim + 1, Nx], 0)

        op = Operator([stencil, bc_left, bc_right])

        # Run for a short time
        T = 0.1
        Nt = int(T / dt)
        for _ in range(Nt):
            op.apply(time_m=0, time_M=0, dt=dt, a=alpha)
            u.data[0, :] = u.data[1, :]

        # Verify against exact solution
        exact = np.exp(-alpha * (np.pi / L) ** 2 * T) * np.sin(np.pi * x_coords / L)
        error = np.max(np.abs(u.data[0, :] - exact))

        assert error < 0.01, f"Error {error:.4f} too large"


@pytest.mark.devito
class TestFirstPdeQmdWaveEquation:
    """Test the wave equation code shown in first_pde.qmd."""

    def test_first_pde_manual_stencil_correct(self):
        """Verify the manual stencil form in first_pde.qmd works.

        The code uses the manually derived stencil:
            eq = Eq(u.forward, 2*u - u.backward + (c*dt)**2 * u.dx2)

        This is correct because it explicitly defines u.forward.
        """
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        # Problem parameters
        L = 1.0
        c = 1.0
        T = 1.0
        Nx = 100
        C = 0.5  # Courant number

        # Derived parameters
        dx = L / Nx
        dt = C * dx / c
        Nt = int(T / dt)

        # Create the computational grid
        grid = Grid(shape=(Nx + 1,), extent=(L,))

        # Create a time-varying field
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

        # Set initial condition: a Gaussian pulse
        x_coord = 0.5 * L
        sigma = 0.1
        x_coords = np.linspace(0, L, Nx + 1)
        u.data[0, :] = np.exp(-((x_coords - x_coord) ** 2) / (2 * sigma**2))
        u.data[1, :] = u.data[0, :]  # Zero initial velocity

        # Define the update equation (manual stencil form)
        eq = Eq(u.forward, 2 * u - u.backward + (c * dt) ** 2 * u.dx2)

        # Create the operator
        op = Operator([eq])

        # Run the simulation
        op(time=Nt, dt=dt)

        # Verify solution is reasonable
        max_val = np.max(np.abs(u.data[0, :]))
        assert max_val < 2.0, f"Solution amplitude {max_val} too large"


@pytest.mark.devito
class TestDiffu1DDevitoQmd:
    """Test the diffusion code shown in diffu1D_devito.qmd."""

    def test_diffu1d_devito_dt_defined(self):
        """Verify dt is properly defined before use in the diffusion example.

        The code should define dt from the Fourier number:
            dt = F * dx**2 / a
        """
        if not DEVITO_AVAILABLE:
            pytest.skip("Devito not available")

        # Domain and discretization
        L = 1.0
        Nx = 100
        a = 1.0  # Diffusion coefficient
        F = 0.5  # Fourier number

        dx = L / Nx
        dt = F * dx**2 / a  # Time step from stability condition

        # Verify dt is defined and reasonable
        assert dt > 0, "dt must be positive"
        assert dt < 1.0, "dt seems too large"

        # Create Devito grid
        grid = Grid(shape=(Nx + 1,), extent=(L,))
        u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)

        # Set initial condition
        x_coords = np.linspace(0, L, Nx + 1)
        u.data[0, :] = np.sin(np.pi * x_coords)

        # Using solve() pattern
        a_const = Constant(name='a_const')
        pde = u.dt - a_const * u.dx2
        stencil = Eq(u.forward, solve(pde, u.forward))

        op = Operator([stencil])

        # Run one time step to verify operator works
        op.apply(time_m=0, time_M=0, dt=dt, a_const=a)

        # Solution should be bounded
        assert np.all(np.isfinite(u.data[1, :])), "Solution contains NaN/Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
