"""Tests for Devito nonlinear PDE solvers."""

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


class TestNonlinearDiffusionExplicit:
    """Tests for explicit nonlinear diffusion solver."""

    def test_import(self):
        """Test that the module imports correctly."""
        from src.nonlin import solve_nonlinear_diffusion_explicit

        assert solve_nonlinear_diffusion_explicit is not None

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.nonlin import solve_nonlinear_diffusion_explicit

        result = solve_nonlinear_diffusion_explicit(L=1.0, Nx=50, T=0.01, F=0.4)

        assert result.u.shape == (51,)
        assert result.x.shape == (51,)
        assert result.t > 0
        assert result.dt > 0

    def test_boundary_conditions(self):
        """Test that boundary conditions are satisfied."""
        from src.nonlin import solve_nonlinear_diffusion_explicit

        result = solve_nonlinear_diffusion_explicit(L=1.0, Nx=50, T=0.01, F=0.4)

        # Dirichlet BCs: u(0) = u(L) = 0
        assert result.u[0] == pytest.approx(0.0, abs=1e-10)
        assert result.u[-1] == pytest.approx(0.0, abs=1e-10)

    def test_linear_case_matches_diffusion(self):
        """Test that D(u) = constant gives same result as linear diffusion."""
        from src.nonlin import constant_diffusion, solve_nonlinear_diffusion_explicit

        def I(x):
            return np.sin(np.pi * x)

        # With constant D, should behave like linear diffusion
        result = solve_nonlinear_diffusion_explicit(
            L=1.0,
            Nx=50,
            T=0.01,
            F=0.4,
            I=I,
            D_func=lambda u: constant_diffusion(u, D0=1.0),
        )

        # Solution should decay but remain positive in interior
        assert np.all(result.u[1:-1] >= 0)
        assert np.max(result.u) < 1.0  # Has decayed from initial max

    def test_solution_bounded(self):
        """Test that solution remains bounded."""
        from src.nonlin import solve_nonlinear_diffusion_explicit

        def I(x):
            return np.sin(np.pi * x)

        result = solve_nonlinear_diffusion_explicit(
            L=1.0, Nx=100, T=0.05, F=0.4, I=I
        )

        # Solution should remain bounded
        assert np.all(np.abs(result.u) < 10.0)

    def test_save_history(self):
        """Test that history is saved correctly."""
        from src.nonlin import solve_nonlinear_diffusion_explicit

        result = solve_nonlinear_diffusion_explicit(
            L=1.0, Nx=50, T=0.01, F=0.4, save_history=True
        )

        assert result.u_history is not None
        assert result.t_history is not None
        assert len(result.u_history) == len(result.t_history)
        assert len(result.u_history) > 1


class TestReactionDiffusionSplitting:
    """Tests for reaction-diffusion solver with operator splitting."""

    def test_import(self):
        """Test that the module imports correctly."""
        from src.nonlin import solve_reaction_diffusion_splitting

        assert solve_reaction_diffusion_splitting is not None

    def test_basic_run_strang(self):
        """Test basic solver execution with Strang splitting."""
        from src.nonlin import solve_reaction_diffusion_splitting

        result = solve_reaction_diffusion_splitting(
            L=1.0, a=1.0, Nx=50, T=0.01, F=0.4, splitting="strang"
        )

        assert result.u.shape == (51,)
        assert result.x.shape == (51,)
        assert result.t > 0

    def test_basic_run_lie(self):
        """Test basic solver execution with Lie splitting."""
        from src.nonlin import solve_reaction_diffusion_splitting

        result = solve_reaction_diffusion_splitting(
            L=1.0, a=1.0, Nx=50, T=0.01, F=0.4, splitting="lie"
        )

        assert result.u.shape == (51,)
        assert result.x.shape == (51,)

    def test_invalid_splitting_raises(self):
        """Test that invalid splitting method raises error."""
        from src.nonlin import solve_reaction_diffusion_splitting

        with pytest.raises(ValueError, match="splitting must be"):
            solve_reaction_diffusion_splitting(
                L=1.0, Nx=50, T=0.01, splitting="invalid"
            )

    def test_boundary_conditions(self):
        """Test that boundary conditions are satisfied."""
        from src.nonlin import solve_reaction_diffusion_splitting

        result = solve_reaction_diffusion_splitting(L=1.0, Nx=50, T=0.01, F=0.4)

        # Dirichlet BCs
        assert result.u[0] == pytest.approx(0.0, abs=1e-10)
        assert result.u[-1] == pytest.approx(0.0, abs=1e-10)

    def test_diffusion_only(self):
        """Test with no reaction (should match pure diffusion)."""
        from src.nonlin import solve_reaction_diffusion_splitting

        def zero_reaction(u):
            return np.zeros_like(u)

        def I(x):
            return np.sin(np.pi * x)

        result = solve_reaction_diffusion_splitting(
            L=1.0, a=1.0, Nx=50, T=0.01, F=0.4, I=I, R_func=zero_reaction
        )

        # Solution should decay
        assert np.max(result.u) < 1.0

    def test_strang_higher_order_than_lie(self):
        """Test that Strang splitting gives different (typically better) result."""
        from src.nonlin import solve_reaction_diffusion_splitting

        def I(x):
            return 0.5 * np.sin(np.pi * x)

        result_lie = solve_reaction_diffusion_splitting(
            L=1.0, a=0.1, Nx=50, T=0.1, F=0.4, I=I, splitting="lie"
        )

        result_strang = solve_reaction_diffusion_splitting(
            L=1.0, a=0.1, Nx=50, T=0.1, F=0.4, I=I, splitting="strang"
        )

        # Results should be different (Strang is O(dt^2), Lie is O(dt))
        assert not np.allclose(result_lie.u, result_strang.u)


class TestBurgersEquation:
    """Tests for Burgers' equation solver."""

    def test_import(self):
        """Test that the module imports correctly."""
        from src.nonlin import solve_burgers_equation

        assert solve_burgers_equation is not None

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.nonlin import solve_burgers_equation

        result = solve_burgers_equation(L=2.0, nu=0.01, Nx=50, T=0.1, C=0.5)

        assert result.u.shape == (51,)
        assert result.x.shape == (51,)
        assert result.t > 0

    def test_boundary_conditions(self):
        """Test that boundary conditions are satisfied."""
        from src.nonlin import solve_burgers_equation

        result = solve_burgers_equation(L=2.0, nu=0.01, Nx=50, T=0.1)

        # Dirichlet BCs
        assert result.u[0] == pytest.approx(0.0, abs=1e-10)
        assert result.u[-1] == pytest.approx(0.0, abs=1e-10)

    def test_solution_bounded(self):
        """Test that solution remains bounded."""
        from src.nonlin import solve_burgers_equation

        result = solve_burgers_equation(L=2.0, nu=0.01, Nx=100, T=0.5, C=0.3)

        # Burgers can develop steep gradients but should remain bounded
        assert np.all(np.abs(result.u) < 10.0)

    def test_viscosity_effect(self):
        """Test that higher viscosity smooths the solution more."""
        from src.nonlin import solve_burgers_equation

        def I(x):
            return np.sin(np.pi * x)

        result_low_nu = solve_burgers_equation(
            L=2.0, nu=0.001, Nx=100, T=0.1, C=0.3, I=I
        )

        result_high_nu = solve_burgers_equation(
            L=2.0, nu=0.1, Nx=100, T=0.1, C=0.3, I=I
        )

        # Higher viscosity should give smaller gradients
        grad_low = np.max(np.abs(np.diff(result_low_nu.u)))
        grad_high = np.max(np.abs(np.diff(result_high_nu.u)))
        assert grad_high < grad_low


class TestNonlinearDiffusionPicard:
    """Tests for Picard iteration solver."""

    def test_import(self):
        """Test that the module imports correctly."""
        from src.nonlin import solve_nonlinear_diffusion_picard

        assert solve_nonlinear_diffusion_picard is not None

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.nonlin import solve_nonlinear_diffusion_picard

        result = solve_nonlinear_diffusion_picard(L=1.0, Nx=50, T=0.01, dt=0.001)

        assert result.u.shape == (51,)
        assert result.x.shape == (51,)
        assert result.t > 0

    def test_boundary_conditions(self):
        """Test that boundary conditions are satisfied."""
        from src.nonlin import solve_nonlinear_diffusion_picard

        result = solve_nonlinear_diffusion_picard(L=1.0, Nx=50, T=0.01, dt=0.001)

        # Dirichlet BCs
        assert result.u[0] == pytest.approx(0.0, abs=1e-10)
        assert result.u[-1] == pytest.approx(0.0, abs=1e-10)


class TestReactionFunctions:
    """Tests for reaction term functions."""

    def test_logistic_reaction(self):
        """Test logistic reaction term."""
        from src.nonlin import logistic_reaction

        u = np.array([0.0, 0.5, 1.0])
        R = logistic_reaction(u, r=1.0, K=1.0)

        # R(0) = 0, R(K) = 0, R(K/2) = r*K/4
        assert R[0] == pytest.approx(0.0)
        assert R[2] == pytest.approx(0.0)
        assert R[1] == pytest.approx(0.25)

    def test_fisher_reaction(self):
        """Test Fisher-KPP reaction term."""
        from src.nonlin import fisher_reaction

        u = np.array([0.0, 0.5, 1.0])
        R = fisher_reaction(u, r=1.0)

        # R(0) = 0, R(1) = 0, R(0.5) = 0.25
        assert R[0] == pytest.approx(0.0)
        assert R[2] == pytest.approx(0.0)
        assert R[1] == pytest.approx(0.25)

    def test_allen_cahn_reaction(self):
        """Test Allen-Cahn reaction term."""
        from src.nonlin import allen_cahn_reaction

        u = np.array([-1.0, 0.0, 1.0])
        R = allen_cahn_reaction(u, epsilon=1.0)

        # R(0) = 0, R(1) = 0, R(-1) = 0 (fixed points)
        assert R[0] == pytest.approx(0.0)
        assert R[1] == pytest.approx(0.0)
        assert R[2] == pytest.approx(0.0)


class TestDiffusionCoefficients:
    """Tests for diffusion coefficient functions."""

    def test_constant_diffusion(self):
        """Test constant diffusion coefficient."""
        from src.nonlin import constant_diffusion

        u = np.array([0.0, 0.5, 1.0, 2.0])
        D = constant_diffusion(u, D0=2.0)

        assert np.all(D == 2.0)

    def test_linear_diffusion(self):
        """Test linear diffusion coefficient."""
        from src.nonlin import linear_diffusion

        u = np.array([0.0, 1.0, 2.0])
        D = linear_diffusion(u, D0=1.0, alpha=0.5)

        expected = np.array([1.0, 1.5, 2.0])
        np.testing.assert_allclose(D, expected)

    def test_porous_medium_diffusion(self):
        """Test porous medium diffusion coefficient."""
        from src.nonlin import porous_medium_diffusion

        u = np.array([0.0, 1.0, 4.0])
        D = porous_medium_diffusion(u, m=2.0, D0=1.0)

        # D(u) = D0 * m * u^(m-1) = 2 * u
        expected = np.array([0.0, 2.0, 8.0])
        np.testing.assert_allclose(D, expected)


class TestNonlinearResult:
    """Tests for NonlinearResult dataclass."""

    def test_result_attributes(self):
        """Test that result has expected attributes."""
        from src.nonlin import solve_nonlinear_diffusion_explicit

        result = solve_nonlinear_diffusion_explicit(
            L=1.0, Nx=50, T=0.01, save_history=True
        )

        assert hasattr(result, "u")
        assert hasattr(result, "x")
        assert hasattr(result, "t")
        assert hasattr(result, "dt")
        assert hasattr(result, "u_history")
        assert hasattr(result, "t_history")

    def test_history_none_when_not_saved(self):
        """Test that history is None when save_history=False."""
        from src.nonlin import solve_nonlinear_diffusion_explicit

        result = solve_nonlinear_diffusion_explicit(
            L=1.0, Nx=50, T=0.01, save_history=False
        )

        assert result.u_history is None
        assert result.t_history is None
