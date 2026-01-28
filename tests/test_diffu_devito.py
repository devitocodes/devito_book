"""Tests for Devito diffusion solvers."""

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


class TestDiffusion1DSolver:
    """Tests for the 1D diffusion solver."""

    def test_import(self):
        """Test that the module imports correctly."""
        from src.diffu import solve_diffusion_1d
        assert solve_diffusion_1d is not None

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.diffu import solve_diffusion_1d

        result = solve_diffusion_1d(L=1.0, a=1.0, Nx=50, T=0.1, F=0.5)

        assert result.u.shape == (51,)
        assert result.x.shape == (51,)
        assert result.t == pytest.approx(0.1, rel=0.01)
        assert result.F <= 0.5

    def test_initial_condition_preserved_at_t0(self):
        """Test that T=0 returns the initial condition."""
        from src.diffu import solve_diffusion_1d

        def I(x):
            return np.sin(2 * np.pi * x)

        result = solve_diffusion_1d(L=1.0, Nx=50, T=0, I=I)

        expected = I(result.x)
        np.testing.assert_allclose(result.u, expected, rtol=1e-10)

    def test_boundary_conditions(self):
        """Test that Dirichlet BCs are enforced."""
        from src.diffu import solve_diffusion_1d

        result = solve_diffusion_1d(L=1.0, Nx=50, T=0.1, F=0.5)

        assert result.u[0] == pytest.approx(0.0, abs=1e-10)
        assert result.u[-1] == pytest.approx(0.0, abs=1e-10)

    def test_exact_solution_accuracy(self):
        """Test accuracy against exact sinusoidal solution."""
        from src.diffu import exact_diffusion_sine, solve_diffusion_1d

        result = solve_diffusion_1d(
            L=1.0, a=1.0, Nx=100, T=0.1, F=0.5,
            I=lambda x: np.sin(np.pi * x),
        )

        u_exact = exact_diffusion_sine(result.x, result.t, L=1.0, a=1.0)
        error = np.max(np.abs(result.u - u_exact))

        # With Nx=100 and second-order spatial discretization
        assert error < 0.01

    def test_convergence_second_order(self):
        """Test that spatial convergence is second order."""
        from src.diffu import convergence_test_diffusion_1d

        grid_sizes, errors, rate = convergence_test_diffusion_1d(
            grid_sizes=[10, 20, 40, 80],
            T=0.1,
            F=0.5,
        )

        # Should be close to 2.0 for second-order spatial convergence
        assert 1.8 < rate < 2.2

    def test_fourier_stability_violation_raises(self):
        """Test that F > 0.5 raises ValueError."""
        from src.diffu import solve_diffusion_1d

        with pytest.raises(ValueError, match="Fourier number"):
            solve_diffusion_1d(L=1.0, Nx=50, T=0.1, F=0.51)

    def test_solution_decay(self):
        """Test that the solution decays over time."""
        from src.diffu import solve_diffusion_1d

        result = solve_diffusion_1d(
            L=1.0, a=1.0, Nx=50, T=0.5, F=0.5,
            I=lambda x: np.sin(np.pi * x),
            save_history=True,
        )

        # Energy should decrease monotonically
        energies = [np.sum(u**2) for u in result.u_history]
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i - 1]

    def test_result_dataclass(self):
        """Test that result dataclass has expected attributes."""
        from src.diffu import solve_diffusion_1d

        result = solve_diffusion_1d(L=1.0, Nx=50, T=0.1, F=0.5, save_history=True)

        assert hasattr(result, 'u')
        assert hasattr(result, 'x')
        assert hasattr(result, 't')
        assert hasattr(result, 'dt')
        assert hasattr(result, 'F')
        assert hasattr(result, 'u_history')
        assert hasattr(result, 't_history')
        assert result.u_history is not None
        assert result.t_history is not None


class TestExactDiffusionSolution:
    """Tests for the exact diffusion solution."""

    def test_exact_solution_at_t0(self):
        """Test that exact solution at t=0 matches initial condition."""
        from src.diffu import exact_diffusion_sine

        x = np.linspace(0, 1, 101)
        u = exact_diffusion_sine(x, t=0, L=1.0, a=1.0)

        expected = np.sin(np.pi * x)
        np.testing.assert_allclose(u, expected, rtol=1e-10)

    def test_exact_solution_decay_rate(self):
        """Test that exact solution decays with correct rate."""
        from src.diffu import exact_diffusion_sine

        x = np.array([0.5])  # Single point at center
        L = 1.0
        a = 1.0

        t1 = 0.0
        t2 = 0.1

        u1 = exact_diffusion_sine(x, t1, L, a)
        u2 = exact_diffusion_sine(x, t2, L, a)

        # Decay rate should be exp(-a * (pi/L)^2 * dt)
        expected_ratio = np.exp(-a * (np.pi / L)**2 * (t2 - t1))
        actual_ratio = u2 / u1

        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-10)

    def test_higher_mode(self):
        """Test exact solution for higher modes."""
        from src.diffu import exact_diffusion_sine

        x = np.linspace(0, 1, 101)
        m = 3  # Third mode

        u = exact_diffusion_sine(x, t=0, L=1.0, a=1.0, m=m)
        expected = np.sin(m * np.pi * x)

        np.testing.assert_allclose(u, expected, rtol=1e-10)


class TestDiffusion2DSolver:
    """Tests for the 2D diffusion solver."""

    def test_import(self):
        """Test that the module imports correctly."""
        from src.diffu import solve_diffusion_2d
        assert solve_diffusion_2d is not None

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.diffu import solve_diffusion_2d

        result = solve_diffusion_2d(
            Lx=1.0, Ly=1.0, a=1.0, Nx=20, Ny=20, T=0.05, F=0.25
        )

        assert result.u.shape == (21, 21)
        assert result.x.shape == (21,)
        assert result.y.shape == (21,)
        assert result.t == pytest.approx(0.05, rel=0.01)

    def test_initial_condition_preserved_at_t0(self):
        """Test that T=0 returns the initial condition."""
        from src.diffu import solve_diffusion_2d

        def I(X, Y):
            return np.sin(np.pi * X) * np.sin(np.pi * Y)

        result = solve_diffusion_2d(Lx=1.0, Ly=1.0, Nx=20, Ny=20, T=0, I=I)

        X, Y = np.meshgrid(result.x, result.y, indexing='ij')
        expected = I(X, Y)
        np.testing.assert_allclose(result.u, expected, rtol=1e-10)

    def test_boundary_conditions(self):
        """Test that Dirichlet BCs are enforced on all sides."""
        from src.diffu import solve_diffusion_2d

        result = solve_diffusion_2d(
            Lx=1.0, Ly=1.0, Nx=20, Ny=20, T=0.05, F=0.25
        )

        # All boundaries should be zero
        np.testing.assert_allclose(result.u[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.u[-1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.u[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.u[:, -1], 0.0, atol=1e-10)

    def test_exact_solution_accuracy(self):
        """Test accuracy against exact 2D sinusoidal solution."""
        from src.diffu import exact_diffusion_2d, solve_diffusion_2d

        result = solve_diffusion_2d(
            Lx=1.0, Ly=1.0, a=1.0, Nx=40, Ny=40, T=0.05, F=0.25,
            I=lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y),
        )

        X, Y = np.meshgrid(result.x, result.y, indexing='ij')
        u_exact = exact_diffusion_2d(X, Y, result.t, 1.0, 1.0, 1.0)
        error = np.max(np.abs(result.u - u_exact))

        # With Nx=Ny=40 and second-order spatial discretization
        assert error < 0.01

    def test_convergence_second_order(self):
        """Test that spatial convergence is second order."""
        from src.diffu import convergence_test_diffusion_2d

        grid_sizes, errors, rate = convergence_test_diffusion_2d(
            grid_sizes=[10, 20, 40],
            T=0.05,
            F=0.25,
        )

        # Should be close to 2.0 for second-order spatial convergence
        assert 1.7 < rate < 2.3

    def test_fourier_stability_violation_raises(self):
        """Test that F > 0.25 raises ValueError for equal spacing."""
        from src.diffu import solve_diffusion_2d

        with pytest.raises(ValueError, match="stability condition"):
            solve_diffusion_2d(
                Lx=1.0, Ly=1.0, Nx=20, Ny=20, T=0.05, F=0.26
            )

    def test_result_dataclass(self):
        """Test that result dataclass has expected attributes."""
        from src.diffu import solve_diffusion_2d

        result = solve_diffusion_2d(
            Lx=1.0, Ly=1.0, Nx=20, Ny=20, T=0.05, F=0.25,
            save_history=True,
        )

        assert hasattr(result, 'u')
        assert hasattr(result, 'x')
        assert hasattr(result, 'y')
        assert hasattr(result, 't')
        assert hasattr(result, 'dt')
        assert hasattr(result, 'F')
        assert hasattr(result, 'u_history')
        assert hasattr(result, 't_history')
        assert result.u_history is not None
        assert result.t_history is not None


class TestInitialConditions:
    """Tests for initial condition utilities."""

    def test_gaussian_initial_condition(self):
        """Test Gaussian initial condition."""
        from src.diffu import gaussian_initial_condition

        x = np.linspace(0, 1, 101)
        u = gaussian_initial_condition(x, L=1.0, sigma=0.1)

        # Peak should be at center
        center_idx = len(x) // 2
        assert u[center_idx] == pytest.approx(1.0, rel=1e-10)

        # Should be symmetric
        np.testing.assert_allclose(u, u[::-1], rtol=1e-10)

    def test_plug_initial_condition(self):
        """Test plug (discontinuous) initial condition."""
        from src.diffu import plug_initial_condition

        x = np.linspace(0, 1, 101)
        u = plug_initial_condition(x, L=1.0, width=0.2)

        # Center should be 1
        center_idx = len(x) // 2
        assert u[center_idx] == pytest.approx(1.0)

        # Edges should be 0
        assert u[0] == pytest.approx(0.0)
        assert u[-1] == pytest.approx(0.0)

    def test_gaussian_2d_initial_condition(self):
        """Test 2D Gaussian initial condition."""
        from src.diffu import gaussian_2d_initial_condition

        x = np.linspace(0, 1, 51)
        y = np.linspace(0, 1, 51)
        X, Y = np.meshgrid(x, y, indexing='ij')

        u = gaussian_2d_initial_condition(X, Y, Lx=1.0, Ly=1.0, sigma=0.1)

        # Peak should be at center
        center_idx = len(x) // 2
        assert u[center_idx, center_idx] == pytest.approx(1.0, rel=1e-10)

        # Should be radially symmetric for equal domain
        assert u[center_idx + 5, center_idx] == pytest.approx(
            u[center_idx, center_idx + 5], rel=1e-10
        )
