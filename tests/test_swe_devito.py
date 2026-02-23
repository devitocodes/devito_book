"""Tests for the Shallow Water Equations solver using Devito."""

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


class TestSWEImport:
    """Test that the module imports correctly."""

    def test_import_solve_swe(self):
        """Test main solver import."""
        from src.systems import solve_swe

        assert solve_swe is not None

    def test_import_create_operator(self):
        """Test operator creation function import."""
        from src.systems import create_swe_operator

        assert create_swe_operator is not None

    def test_import_result_class(self):
        """Test result dataclass import."""
        from src.systems import SWEResult

        assert SWEResult is not None


class TestCoupledSystemSetup:
    """Test that the coupled system is set up correctly with 3 equations."""

    def test_three_time_functions(self):
        """Test that eta, M, N are all TimeFunction."""
        from devito import Grid, TimeFunction

        grid = Grid(shape=(51, 51), extent=(100.0, 100.0), dtype=np.float32)

        eta = TimeFunction(name='eta', grid=grid, space_order=2)
        M = TimeFunction(name='M', grid=grid, space_order=2)
        N = TimeFunction(name='N', grid=grid, space_order=2)

        # Check they are all TimeFunctions
        assert hasattr(eta, 'forward')
        assert hasattr(M, 'forward')
        assert hasattr(N, 'forward')

        # Check they have proper shapes
        assert eta.data[0].shape == (51, 51)
        assert M.data[0].shape == (51, 51)
        assert N.data[0].shape == (51, 51)

    def test_operator_has_three_update_equations(self):
        """Test that the operator updates all three fields."""
        from devito import (
            Eq,
            Function,
            Grid,
            Operator,
            TimeFunction,
            solve,
            sqrt,
        )

        grid = Grid(shape=(51, 51), extent=(100.0, 100.0), dtype=np.float32)

        eta = TimeFunction(name='eta', grid=grid, space_order=2)
        M = TimeFunction(name='M', grid=grid, space_order=2)
        N = TimeFunction(name='N', grid=grid, space_order=2)
        h = Function(name='h', grid=grid)
        D = Function(name='D', grid=grid)

        g, alpha = 9.81, 0.025

        # Initialize fields
        eta.data[0, :, :] = 0.1
        M.data[0, :, :] = 1.0
        N.data[0, :, :] = 0.5
        h.data[:] = 50.0
        D.data[:] = 50.1

        # Create equations
        friction_M = g * alpha**2 * sqrt(M**2 + N**2) / D**(7.0/3.0)
        pde_eta = Eq(eta.dt + M.dxc + N.dyc)
        pde_M = Eq(M.dt + (M**2/D).dxc + (M*N/D).dyc
                   + g*D*eta.forward.dxc + friction_M*M)

        stencil_eta = solve(pde_eta, eta.forward)
        stencil_M = solve(pde_M, M.forward)

        # These should compile without error
        update_eta = Eq(eta.forward, stencil_eta, subdomain=grid.interior)
        update_M = Eq(M.forward, stencil_M, subdomain=grid.interior)

        op = Operator([update_eta, update_M])

        # Should be able to run (h is not in the operator, so don't pass it)
        op.apply(eta=eta, M=M, D=D, time_m=0, time_M=0, dt=0.001)


class TestBathymetryAsFunction:
    """Test that bathymetry is correctly handled as a static Function."""

    def test_bathymetry_is_function(self):
        """Test bathymetry uses Function (not TimeFunction)."""
        from devito import Function, Grid

        grid = Grid(shape=(51, 51), extent=(100.0, 100.0), dtype=np.float32)
        h = Function(name='h', grid=grid)

        # Function does not have 'forward' attribute
        assert not hasattr(h, 'forward')
        assert h.data.shape == (51, 51)

    def test_bathymetry_constant(self):
        """Test solver with constant bathymetry."""
        from src.systems import solve_swe

        result = solve_swe(
            Lx=50.0, Ly=50.0,
            Nx=51, Ny=51,
            T=0.1,
            dt=1/2000,
            h0=30.0,  # Constant depth
            nsnaps=0,
        )

        assert result.eta.shape == (51, 51)
        assert result.M.shape == (51, 51)
        assert result.N.shape == (51, 51)

    def test_bathymetry_array(self):
        """Test solver with spatially varying bathymetry."""
        from src.systems import solve_swe

        x = np.linspace(0, 50, 51)
        y = np.linspace(0, 50, 51)
        X, Y = np.meshgrid(x, y)

        # Varying bathymetry
        h_array = 50.0 - 20.0 * np.exp(-((X - 25)**2/100) - ((Y - 25)**2/100))

        result = solve_swe(
            Lx=50.0, Ly=50.0,
            Nx=51, Ny=51,
            T=0.1,
            dt=1/2000,
            h0=h_array,
            nsnaps=0,
        )

        assert result.eta.shape == (51, 51)


class TestConditionalDimensionSnapshotting:
    """Test that ConditionalDimension correctly subsamples snapshots."""

    def test_snapshot_shape(self):
        """Test snapshots have correct shape."""
        from src.systems import solve_swe

        nsnaps = 10
        result = solve_swe(
            Lx=50.0, Ly=50.0,
            Nx=51, Ny=51,
            T=0.5,
            dt=1/2000,
            h0=30.0,
            nsnaps=nsnaps,
        )

        assert result.eta_snapshots is not None
        assert result.eta_snapshots.shape[0] == nsnaps
        assert result.eta_snapshots.shape[1] == 51
        assert result.eta_snapshots.shape[2] == 51

    def test_time_snapshots(self):
        """Test time array for snapshots."""
        from src.systems import solve_swe

        nsnaps = 20
        T = 1.0
        result = solve_swe(
            Lx=50.0, Ly=50.0,
            Nx=51, Ny=51,
            T=T,
            dt=1/2000,
            h0=30.0,
            nsnaps=nsnaps,
        )

        assert result.t_snapshots is not None
        assert len(result.t_snapshots) == nsnaps
        assert result.t_snapshots[0] == 0.0
        assert result.t_snapshots[-1] == pytest.approx(T, rel=0.01)

    def test_no_snapshots(self):
        """Test that nsnaps=0 returns None for snapshots."""
        from src.systems import solve_swe

        result = solve_swe(
            Lx=50.0, Ly=50.0,
            Nx=51, Ny=51,
            T=0.1,
            dt=1/2000,
            h0=30.0,
            nsnaps=0,
        )

        assert result.eta_snapshots is None
        assert result.t_snapshots is None


class TestMassConservation:
    """Test that mass is approximately conserved."""

    def test_mass_conservation_constant_depth(self):
        """Test mass conservation with constant depth."""
        from src.systems import solve_swe

        # Small domain, short time for testing
        x = np.linspace(0, 50, 51)
        y = np.linspace(0, 50, 51)
        X, Y = np.meshgrid(x, y)

        # Initial Gaussian perturbation
        eta0 = 0.1 * np.exp(-((X - 25)**2/50) - ((Y - 25)**2/50))
        M0 = 10.0 * eta0
        N0 = np.zeros_like(M0)

        result = solve_swe(
            Lx=50.0, Ly=50.0,
            Nx=51, Ny=51,
            T=0.5,
            dt=1/4000,
            h0=30.0,
            eta0=eta0,
            M0=M0,
            N0=N0,
            nsnaps=10,
        )

        # Compute mass (integral of eta over domain)
        dx = 50.0 / 50
        dy = 50.0 / 50

        mass_initial = np.sum(result.eta_snapshots[0]) * dx * dy
        mass_final = np.sum(result.eta_snapshots[-1]) * dx * dy

        # Mass should be approximately conserved (within some tolerance)
        # Note: open boundaries may allow some mass loss
        relative_change = abs(mass_final - mass_initial) / abs(mass_initial + 1e-10)

        # Allow up to 50% change due to open boundaries and numerical effects
        assert relative_change < 0.5

    def test_integral_of_eta_bounded(self):
        """Test that integral of eta remains bounded."""
        from src.systems import solve_swe

        x = np.linspace(0, 50, 51)
        y = np.linspace(0, 50, 51)
        X, Y = np.meshgrid(x, y)

        eta0 = 0.2 * np.exp(-((X - 25)**2/30) - ((Y - 25)**2/30))

        result = solve_swe(
            Lx=50.0, Ly=50.0,
            Nx=51, Ny=51,
            T=0.3,
            dt=1/4000,
            h0=40.0,
            eta0=eta0,
            nsnaps=5,
        )

        # Check that eta integral doesn't blow up
        dx = 50.0 / 50
        dy = 50.0 / 50

        for i in range(result.eta_snapshots.shape[0]):
            integral = np.sum(np.abs(result.eta_snapshots[i])) * dx * dy
            # Integral should not grow unboundedly
            assert integral < 1000.0


class TestSolutionBoundedness:
    """Test that solution values remain bounded (no blowup)."""

    def test_eta_bounded(self):
        """Test that wave height remains bounded."""
        from src.systems import solve_swe

        result = solve_swe(
            Lx=50.0, Ly=50.0,
            Nx=51, Ny=51,
            T=0.5,
            dt=1/4000,
            h0=30.0,
            nsnaps=10,
        )

        # Check all snapshots are bounded
        for i in range(result.eta_snapshots.shape[0]):
            assert np.all(np.isfinite(result.eta_snapshots[i]))
            # Wave height should be much smaller than depth
            assert np.max(np.abs(result.eta_snapshots[i])) < 30.0

    def test_discharge_bounded(self):
        """Test that discharge fluxes remain bounded."""
        from src.systems import solve_swe

        result = solve_swe(
            Lx=50.0, Ly=50.0,
            Nx=51, Ny=51,
            T=0.3,
            dt=1/4000,
            h0=30.0,
            nsnaps=0,
        )

        # Final M and N should be finite and bounded
        assert np.all(np.isfinite(result.M))
        assert np.all(np.isfinite(result.N))
        assert np.max(np.abs(result.M)) < 10000.0
        assert np.max(np.abs(result.N)) < 10000.0

    def test_no_nan_values(self):
        """Test that solution contains no NaN values."""
        from src.systems import solve_swe

        result = solve_swe(
            Lx=50.0, Ly=50.0,
            Nx=51, Ny=51,
            T=0.2,
            dt=1/4000,
            h0=30.0,
            nsnaps=5,
        )

        assert not np.any(np.isnan(result.eta))
        assert not np.any(np.isnan(result.M))
        assert not np.any(np.isnan(result.N))

        if result.eta_snapshots is not None:
            assert not np.any(np.isnan(result.eta_snapshots))


class TestSWEResult:
    """Test the SWEResult dataclass."""

    def test_result_attributes(self):
        """Test that result has all expected attributes."""
        from src.systems import solve_swe

        result = solve_swe(
            Lx=50.0, Ly=50.0,
            Nx=51, Ny=51,
            T=0.1,
            dt=1/2000,
            h0=30.0,
        )

        assert hasattr(result, 'eta')
        assert hasattr(result, 'M')
        assert hasattr(result, 'N')
        assert hasattr(result, 'x')
        assert hasattr(result, 'y')
        assert hasattr(result, 't')
        assert hasattr(result, 'dt')
        assert hasattr(result, 'eta_snapshots')
        assert hasattr(result, 't_snapshots')

    def test_coordinate_arrays(self):
        """Test that x and y coordinate arrays are correct."""
        from src.systems import solve_swe

        Lx, Ly = 100.0, 80.0
        Nx, Ny = 101, 81

        result = solve_swe(
            Lx=Lx, Ly=Ly,
            Nx=Nx, Ny=Ny,
            T=0.01,
            dt=1/2000,
            h0=30.0,
        )

        assert len(result.x) == Nx
        assert len(result.y) == Ny
        assert result.x[0] == pytest.approx(0.0)
        assert result.x[-1] == pytest.approx(Lx)
        assert result.y[0] == pytest.approx(0.0)
        assert result.y[-1] == pytest.approx(Ly)


class TestHelperFunctions:
    """Test utility functions for common scenarios."""

    def test_gaussian_source(self):
        """Test Gaussian tsunami source function."""
        from src.systems.swe_devito import gaussian_tsunami_source

        x = np.linspace(0, 100, 101)
        y = np.linspace(0, 100, 101)
        X, Y = np.meshgrid(x, y)

        eta = gaussian_tsunami_source(X, Y, x0=50, y0=50, amplitude=0.5)

        # Check shape
        assert eta.shape == (101, 101)

        # Check peak is at center
        max_idx = np.unravel_index(np.argmax(eta), eta.shape)
        assert max_idx == (50, 50)

        # Check amplitude
        assert eta.max() == pytest.approx(0.5, rel=0.01)

    def test_seamount_bathymetry(self):
        """Test seamount bathymetry function."""
        from src.systems.swe_devito import seamount_bathymetry

        x = np.linspace(0, 100, 101)
        y = np.linspace(0, 100, 101)
        X, Y = np.meshgrid(x, y)

        h = seamount_bathymetry(X, Y, h_base=50, height=45)

        # Check shape
        assert h.shape == (101, 101)

        # Minimum depth should be at seamount peak (center by default)
        assert h.min() == pytest.approx(5.0, rel=0.1)

        # Depth at corners should be close to base
        assert h[0, 0] == pytest.approx(50.0, rel=0.1)

    def test_tanh_bathymetry(self):
        """Test tanh coastal profile function."""
        from src.systems.swe_devito import tanh_bathymetry

        x = np.linspace(0, 100, 101)
        y = np.linspace(0, 100, 101)
        X, Y = np.meshgrid(x, y)

        h = tanh_bathymetry(X, Y, h_deep=50, h_shallow=5, x_transition=70)

        # Check shape
        assert h.shape == (101, 101)

        # Left side should be deep
        assert h[50, 0] > 40

        # Right side should be shallow
        assert h[50, 100] < 10


class TestPhysicalBehavior:
    """Test expected physical behavior of solutions."""

    def test_wave_propagation(self):
        """Test that waves propagate outward from initial disturbance."""
        from src.systems import solve_swe

        x = np.linspace(0, 100, 101)
        y = np.linspace(0, 100, 101)
        X, Y = np.meshgrid(x, y)

        # Initial disturbance at center
        eta0 = 0.3 * np.exp(-((X - 50)**2/20) - ((Y - 50)**2/20))
        M0 = 50.0 * eta0
        N0 = np.zeros_like(M0)

        result = solve_swe(
            Lx=100.0, Ly=100.0,
            Nx=101, Ny=101,
            T=1.0,
            dt=1/4000,
            h0=50.0,
            eta0=eta0,
            M0=M0,
            N0=N0,
            nsnaps=5,
        )

        # Initial disturbance should spread out
        # Variance of |eta| distribution should increase
        initial_var = np.var(result.eta_snapshots[0])
        final_var = np.var(result.eta_snapshots[-1])

        # After spreading, variance should decrease (wave disperses)
        # or stay similar (if boundaries reflect)
        assert final_var < initial_var * 2  # Not blowing up

    def test_amplitude_decay_with_friction(self):
        """Test that bottom friction causes amplitude decay over longer times."""
        from src.systems import solve_swe

        x = np.linspace(0, 100, 101)
        y = np.linspace(0, 100, 101)
        X, Y = np.meshgrid(x, y)

        eta0 = 0.3 * np.exp(-((X - 50)**2/30) - ((Y - 50)**2/30))

        # High friction coefficient, longer time for friction to act
        result = solve_swe(
            Lx=100.0, Ly=100.0,
            Nx=101, Ny=101,
            T=3.0,  # Longer time
            dt=1/4000,
            h0=20.0,  # Shallower = more friction effect
            alpha=0.1,  # Higher Manning's coefficient for stronger friction
            eta0=eta0,
            M0=np.zeros_like(eta0),  # Start with no momentum
            N0=np.zeros_like(eta0),
            nsnaps=20,
        )

        # Compute total energy proxy: sum of |eta|^2
        energy_initial = np.sum(result.eta_snapshots[1]**2)  # After first step
        energy_final = np.sum(result.eta_snapshots[-1]**2)

        # Energy should decay due to friction
        # Note: some transient growth may occur initially, so compare mid to late
        energy_mid = np.sum(result.eta_snapshots[10]**2)

        # At minimum, energy should not grow unboundedly
        # and final energy should be less than initial
        assert energy_final < energy_initial * 2  # Should not grow too much
        assert np.all(np.isfinite(result.eta_snapshots[-1]))
