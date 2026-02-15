"""Tests for src.elliptic Laplace and Poisson solvers."""

import numpy as np
import pytest


def _devito_importable() -> bool:
    try:
        import devito  # noqa: F401
    except Exception:
        return False
    return True


_skip_no_devito = pytest.mark.skipif(
    not _devito_importable(), reason="Devito not importable in this environment"
)


# ---------------------------------------------------------------------------
# Pure-NumPy utilities (no Devito required)
# ---------------------------------------------------------------------------


class TestCreatePointSource:
    """Tests for create_point_source (pure NumPy)."""

    def test_single_source_location(self):
        from src.elliptic import create_point_source

        b = create_point_source(Nx=21, Ny=21, Lx=2.0, Ly=1.0,
                                x_src=1.0, y_src=0.5, value=100.0)
        assert b.shape == (21, 21)
        assert b.sum() == pytest.approx(100.0)
        # Exactly one non-zero entry
        assert np.count_nonzero(b) == 1

    def test_source_at_origin(self):
        from src.elliptic import create_point_source

        b = create_point_source(Nx=11, Ny=11, Lx=1.0, Ly=1.0,
                                x_src=0.0, y_src=0.0, value=50.0)
        assert b[0, 0] == pytest.approx(50.0)

    def test_source_at_corner(self):
        from src.elliptic import create_point_source

        b = create_point_source(Nx=11, Ny=11, Lx=1.0, Ly=1.0,
                                x_src=1.0, y_src=1.0, value=-30.0)
        assert b[10, 10] == pytest.approx(-30.0)

    def test_source_clamped_to_grid(self):
        """Source outside domain is clamped to boundary."""
        from src.elliptic import create_point_source

        b = create_point_source(Nx=11, Ny=11, Lx=1.0, Ly=1.0,
                                x_src=5.0, y_src=5.0, value=1.0)
        assert np.count_nonzero(b) == 1
        assert b[10, 10] == pytest.approx(1.0)


class TestCreateGaussianSource:
    """Tests for create_gaussian_source (pure NumPy)."""

    def test_shape_and_peak(self):
        from src.elliptic import create_gaussian_source

        x = np.linspace(0, 1, 51)
        y = np.linspace(0, 1, 51)
        X, Y = np.meshgrid(x, y, indexing="ij")

        b = create_gaussian_source(X, Y, x0=0.5, y0=0.5,
                                   sigma=0.1, amplitude=10.0)
        assert b.shape == (51, 51)
        assert b.max() == pytest.approx(10.0, rel=1e-3)

    def test_peak_location(self):
        from src.elliptic import create_gaussian_source

        x = np.linspace(0, 2, 101)
        y = np.linspace(0, 1, 51)
        X, Y = np.meshgrid(x, y, indexing="ij")

        b = create_gaussian_source(X, Y, x0=1.0, y0=0.5, sigma=0.05)
        imax, jmax = np.unravel_index(b.argmax(), b.shape)
        assert x[imax] == pytest.approx(1.0, abs=0.03)
        assert y[jmax] == pytest.approx(0.5, abs=0.03)

    def test_symmetry(self):
        from src.elliptic import create_gaussian_source

        x = np.linspace(0, 1, 101)
        X, Y = np.meshgrid(x, x, indexing="ij")
        b = create_gaussian_source(X, Y, x0=0.5, y0=0.5, sigma=0.1)
        # Symmetric about center
        assert np.allclose(b, b[::-1, :], atol=1e-12)
        assert np.allclose(b, b[:, ::-1], atol=1e-12)


class TestExactPoissonPointSource:
    """Tests for exact_poisson_point_source (pure NumPy)."""

    def test_boundary_values_zero(self):
        from src.elliptic import exact_poisson_point_source

        x = np.linspace(0, 1, 51)
        y = np.linspace(0, 1, 51)
        X, Y = np.meshgrid(x, y, indexing="ij")

        p = exact_poisson_point_source(X, Y, Lx=1.0, Ly=1.0,
                                       x_src=0.5, y_src=0.5,
                                       strength=100.0, n_terms=10)
        # Dirichlet BCs: p=0 on all boundaries
        assert np.allclose(p[0, :], 0.0, atol=1e-10)
        assert np.allclose(p[-1, :], 0.0, atol=1e-10)
        assert np.allclose(p[:, 0], 0.0, atol=1e-10)
        assert np.allclose(p[:, -1], 0.0, atol=1e-10)

    def test_symmetry_for_centered_source(self):
        from src.elliptic import exact_poisson_point_source

        x = np.linspace(0, 1, 51)
        X, Y = np.meshgrid(x, x, indexing="ij")

        p = exact_poisson_point_source(X, Y, Lx=1.0, Ly=1.0,
                                       x_src=0.5, y_src=0.5,
                                       strength=100.0, n_terms=20)
        assert np.allclose(p, p[::-1, :], atol=1e-10)
        assert np.allclose(p, p[:, ::-1], atol=1e-10)


class TestExactLaplaceLinear:
    """Tests for exact_laplace_linear (pure NumPy)."""

    def test_dp_dy_zero(self):
        from src.elliptic import exact_laplace_linear

        Lx, Ly = 2.0, 1.0
        x = np.linspace(0, Lx, 50)
        y = np.linspace(0, Ly, 50)
        X, Y = np.meshgrid(x, y, indexing="ij")
        p = exact_laplace_linear(X, Y, Lx, Ly)

        dp_dy = np.diff(p, axis=1)
        assert np.allclose(dp_dy, 0.0, atol=1e-14)

    def test_boundary_conditions(self):
        from src.elliptic import exact_laplace_linear

        Lx, Ly = 2.0, 1.0
        x = np.linspace(0, Lx, 50)
        y = np.linspace(0, Ly, 50)
        X, Y = np.meshgrid(x, y, indexing="ij")
        p = exact_laplace_linear(X, Y, Lx, Ly)

        assert np.allclose(p[0, :], 0.0, atol=1e-14)
        assert np.allclose(p[-1, :], 1.0, atol=1e-14)

    def test_harmonic(self):
        from src.elliptic import exact_laplace_linear

        Lx, Ly = 2.0, 1.0
        x = np.linspace(0, Lx, 50)
        y = np.linspace(0, Ly, 50)
        X, Y = np.meshgrid(x, y, indexing="ij")
        p = exact_laplace_linear(X, Y, Lx, Ly)

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        laplacian = (
            (p[2:, 1:-1] - 2 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / dx**2
            + (p[1:-1, 2:] - 2 * p[1:-1, 1:-1] + p[1:-1, :-2]) / dy**2
        )
        assert np.allclose(laplacian, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Laplace solvers (require Devito)
# ---------------------------------------------------------------------------


@pytest.mark.devito
@_skip_no_devito
class TestSolveLaplace2d:
    """Tests for the dual-buffer Laplace solver."""

    def test_smoke_neumann(self):
        """Converges with Neumann + constant Dirichlet BCs."""
        from src.elliptic import solve_laplace_2d

        result = solve_laplace_2d(
            Lx=2.0, Ly=1.0, Nx=21, Ny=21,
            bc_left=0.0, bc_right=1.0,
            bc_bottom="neumann", bc_top="neumann",
            tol=1e-4,
        )
        assert result.converged
        assert result.iterations > 0

    def test_result_fields(self):
        """All LaplaceResult fields are populated correctly."""
        from src.elliptic import solve_laplace_2d

        result = solve_laplace_2d(
            Lx=2.0, Ly=1.0, Nx=21, Ny=21,
            bc_left=0.0, bc_right=1.0,
            bc_bottom="neumann", bc_top="neumann",
            tol=1e-4,
        )
        assert result.p.shape == (21, 21)
        assert result.x.shape == (21,)
        assert result.y.shape == (21,)
        assert result.final_l1norm <= 1e-4
        assert result.converged
        assert result.p_history is None

    def test_callable_bc(self):
        """Callable boundary condition (Dirichlet profile)."""
        from src.elliptic import solve_laplace_2d

        result = solve_laplace_2d(
            Lx=1.0, Ly=1.0, Nx=21, Ny=21,
            bc_left=0.0,
            bc_right=lambda y: np.sin(np.pi * y),
            bc_bottom=0.0,
            bc_top=0.0,
            tol=1e-5,
        )
        assert result.converged
        # Right boundary should match sin(pi*y)
        y = result.y
        expected_right = np.sin(np.pi * y)
        np.testing.assert_allclose(result.p[-1, :], expected_right, atol=1e-2)

    def test_save_interval(self):
        """save_interval populates p_history."""
        from src.elliptic import solve_laplace_2d

        result = solve_laplace_2d(
            Lx=2.0, Ly=1.0, Nx=11, Ny=11,
            bc_left=0.0, bc_right=1.0,
            bc_bottom="neumann", bc_top="neumann",
            tol=1e-4, save_interval=50,
        )
        assert result.converged
        assert result.p_history is not None
        assert len(result.p_history) > 0

    def test_max_iterations_no_convergence(self):
        """Returns converged=False when max_iterations is hit."""
        from src.elliptic import solve_laplace_2d

        result = solve_laplace_2d(
            Lx=2.0, Ly=1.0, Nx=21, Ny=21,
            bc_left=0.0, bc_right=1.0,
            bc_bottom="neumann", bc_top="neumann",
            tol=1e-15, max_iterations=5,
        )
        assert not result.converged
        assert result.iterations == 5

    def test_linear_exact_solution(self):
        """Solver recovers the linear exact solution x/Lx."""
        from src.elliptic import exact_laplace_linear, solve_laplace_2d

        Lx, Ly = 2.0, 1.0
        result = solve_laplace_2d(
            Lx=Lx, Ly=Ly, Nx=21, Ny=21,
            bc_left=0.0, bc_right=1.0,
            bc_bottom="neumann", bc_top="neumann",
            tol=1e-6,
        )
        X, Y = np.meshgrid(result.x, result.y, indexing="ij")
        p_exact = exact_laplace_linear(X, Y, Lx, Ly)
        np.testing.assert_allclose(result.p, p_exact, atol=1e-3)


@pytest.mark.devito
@_skip_no_devito
class TestSolveLaplace2dWithCopy:
    """Tests for the copy-based Laplace solver."""

    def test_smoke(self):
        from src.elliptic import solve_laplace_2d_with_copy

        result = solve_laplace_2d_with_copy(
            Lx=2.0, Ly=1.0, Nx=21, Ny=21,
            bc_left=0.0, bc_right=1.0,
            bc_bottom="neumann", bc_top="neumann",
            tol=1e-4,
        )
        assert result.converged
        assert result.iterations > 0

    def test_result_fields(self):
        from src.elliptic import solve_laplace_2d_with_copy

        result = solve_laplace_2d_with_copy(
            Lx=2.0, Ly=1.0, Nx=21, Ny=21,
            bc_left=0.0, bc_right=1.0,
            bc_bottom="neumann", bc_top="neumann",
            tol=1e-4,
        )
        assert result.p.shape == (21, 21)
        assert result.x.shape == (21,)
        assert result.y.shape == (21,)
        assert result.final_l1norm <= 1e-4

    def test_agrees_with_dual_buffer(self):
        """Copy-based and dual-buffer solvers give the same answer."""
        from src.elliptic import solve_laplace_2d, solve_laplace_2d_with_copy

        kwargs = dict(
            Lx=2.0, Ly=1.0, Nx=21, Ny=21,
            bc_left=0.0, bc_right=1.0,
            bc_bottom="neumann", bc_top="neumann",
            tol=1e-6,
        )
        r1 = solve_laplace_2d(**kwargs)
        r2 = solve_laplace_2d_with_copy(**kwargs)
        np.testing.assert_allclose(r1.p, r2.p, atol=1e-4)

    def test_callable_bc(self):
        from src.elliptic import solve_laplace_2d_with_copy

        result = solve_laplace_2d_with_copy(
            Lx=1.0, Ly=1.0, Nx=21, Ny=21,
            bc_left=0.0,
            bc_right=lambda y: np.sin(np.pi * y),
            bc_bottom=0.0, bc_top=0.0,
            tol=1e-5,
        )
        assert result.converged
        expected_right = np.sin(np.pi * result.y)
        np.testing.assert_allclose(result.p[-1, :], expected_right, atol=1e-2)


@pytest.mark.devito
@_skip_no_devito
class TestConvergenceTestLaplace:
    """Tests for convergence_test_laplace_2d."""

    def test_returns_valid_structure(self):
        from src.elliptic import convergence_test_laplace_2d

        grid_sizes, errors, observed_order = convergence_test_laplace_2d(
            grid_sizes=[11, 21], tol=1e-8,
        )
        assert len(grid_sizes) == 2
        assert len(errors) == 2
        assert isinstance(observed_order, float)

    def test_errors_small_for_linear_solution(self):
        """Linear exact solution gives near-zero discretization error."""
        from src.elliptic import convergence_test_laplace_2d

        grid_sizes, errors, _ = convergence_test_laplace_2d(
            grid_sizes=[11, 21, 41], tol=1e-8,
        )
        for N, err in zip(grid_sizes, errors):
            assert err < 1e-3, f"Error for N={N} is {err}"


# ---------------------------------------------------------------------------
# Poisson solvers (require Devito)
# ---------------------------------------------------------------------------

# Manufactured solution: p = sin(pi*x)*sin(pi*y), b = -2*pi^2*sin(pi*x)*sin(pi*y)
def _poisson_mms_source(X, Y):
    return -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)


def _poisson_mms_exact(X, Y):
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


@pytest.mark.devito
@_skip_no_devito
class TestSolvePoisson2d:
    """Tests for the dual-buffer Poisson solver."""

    def test_smoke_point_source(self):
        from src.elliptic import solve_poisson_2d

        result = solve_poisson_2d(
            Lx=2.0, Ly=1.0, Nx=31, Ny=31,
            source_points=[(1.0, 0.5, 100.0)],
            n_iterations=200,
        )
        assert result.iterations == 200
        assert result.p.shape == (31, 31)

    def test_result_fields(self):
        from src.elliptic import solve_poisson_2d

        result = solve_poisson_2d(
            Lx=1.0, Ly=1.0, Nx=21, Ny=21,
            source_points=[(0.5, 0.5, 50.0)],
            n_iterations=100,
        )
        assert result.p.shape == (21, 21)
        assert result.x.shape == (21,)
        assert result.y.shape == (21,)
        assert result.b.shape == (21, 21)
        assert result.iterations == 100
        assert result.p_history is None

    def test_callable_source(self):
        """Accepts a callable b(X,Y) source term."""
        from src.elliptic import solve_poisson_2d

        result = solve_poisson_2d(
            Lx=1.0, Ly=1.0, Nx=31, Ny=31,
            b=_poisson_mms_source,
            n_iterations=500,
        )
        assert result.p.shape == (31, 31)
        # Source term should be non-zero in interior
        assert np.any(result.b != 0)

    def test_array_source(self):
        """Accepts a numpy array source term."""
        from src.elliptic import solve_poisson_2d

        Nx, Ny = 21, 21
        b_arr = np.zeros((Nx, Ny))
        b_arr[10, 10] = 50.0
        result = solve_poisson_2d(
            Lx=1.0, Ly=1.0, Nx=Nx, Ny=Ny,
            b=b_arr, n_iterations=100,
        )
        assert result.p.shape == (Nx, Ny)

    def test_save_interval(self):
        from src.elliptic import solve_poisson_2d

        result = solve_poisson_2d(
            Lx=1.0, Ly=1.0, Nx=21, Ny=21,
            source_points=[(0.5, 0.5, 50.0)],
            n_iterations=100, save_interval=25,
        )
        assert result.p_history is not None
        # Initial snapshot + snapshots at 25, 50, 75, 100
        assert len(result.p_history) >= 4

    def test_boundary_values_zero(self):
        """Dirichlet BC: solution is zero on boundaries."""
        from src.elliptic import solve_poisson_2d

        result = solve_poisson_2d(
            Lx=1.0, Ly=1.0, Nx=31, Ny=31,
            source_points=[(0.5, 0.5, 100.0)],
            n_iterations=500, bc_value=0.0,
        )
        np.testing.assert_allclose(result.p[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.p[-1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.p[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.p[:, -1], 0.0, atol=1e-10)


@pytest.mark.devito
@_skip_no_devito
class TestSolvePoisson2dTimefunction:
    """Tests for the TimeFunction-based Poisson solver."""

    def test_smoke(self):
        from src.elliptic import solve_poisson_2d_timefunction

        result = solve_poisson_2d_timefunction(
            Lx=1.0, Ly=1.0, Nx=31, Ny=31,
            source_points=[(0.5, 0.5, 100.0)],
            n_iterations=200,
        )
        assert result.iterations == 200
        assert result.p.shape == (31, 31)

    def test_callable_source(self):
        from src.elliptic import solve_poisson_2d_timefunction

        result = solve_poisson_2d_timefunction(
            Lx=1.0, Ly=1.0, Nx=31, Ny=31,
            b=_poisson_mms_source,
            n_iterations=500,
        )
        assert result.p.shape == (31, 31)

    def test_boundary_values_zero(self):
        from src.elliptic import solve_poisson_2d_timefunction

        result = solve_poisson_2d_timefunction(
            Lx=1.0, Ly=1.0, Nx=31, Ny=31,
            source_points=[(0.5, 0.5, 100.0)],
            n_iterations=500, bc_value=0.0,
        )
        np.testing.assert_allclose(result.p[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.p[-1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.p[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.p[:, -1], 0.0, atol=1e-10)

    def test_agrees_with_dual_buffer(self):
        """TimeFunction and dual-buffer solvers give similar answers."""
        from src.elliptic import solve_poisson_2d, solve_poisson_2d_timefunction

        kwargs = dict(
            Lx=1.0, Ly=1.0, Nx=31, Ny=31,
            source_points=[(0.5, 0.5, 100.0)],
            n_iterations=500, bc_value=0.0,
        )
        r1 = solve_poisson_2d(**kwargs)
        r2 = solve_poisson_2d_timefunction(**kwargs)
        np.testing.assert_allclose(r1.p, r2.p, atol=1e-2)


@pytest.mark.devito
@_skip_no_devito
class TestSolvePoisson2dWithCopy:
    """Tests for the copy-based Poisson solver."""

    def test_smoke(self):
        from src.elliptic import solve_poisson_2d_with_copy

        result = solve_poisson_2d_with_copy(
            Lx=1.0, Ly=1.0, Nx=31, Ny=31,
            source_points=[(0.5, 0.5, 100.0)],
            n_iterations=200,
        )
        assert result.iterations == 200
        assert result.p.shape == (31, 31)

    def test_callable_source(self):
        from src.elliptic import solve_poisson_2d_with_copy

        result = solve_poisson_2d_with_copy(
            Lx=1.0, Ly=1.0, Nx=31, Ny=31,
            b=_poisson_mms_source,
            n_iterations=500,
        )
        assert result.p.shape == (31, 31)

    def test_agrees_with_dual_buffer(self):
        """Copy-based and dual-buffer Poisson solvers give the same answer."""
        from src.elliptic import solve_poisson_2d, solve_poisson_2d_with_copy

        kwargs = dict(
            Lx=1.0, Ly=1.0, Nx=31, Ny=31,
            source_points=[(0.5, 0.5, 100.0)],
            n_iterations=500, bc_value=0.0,
        )
        r1 = solve_poisson_2d(**kwargs)
        r2 = solve_poisson_2d_with_copy(**kwargs)
        np.testing.assert_allclose(r1.p, r2.p, atol=1e-2)

    def test_boundary_values_zero(self):
        from src.elliptic import solve_poisson_2d_with_copy

        result = solve_poisson_2d_with_copy(
            Lx=1.0, Ly=1.0, Nx=31, Ny=31,
            source_points=[(0.5, 0.5, 100.0)],
            n_iterations=500, bc_value=0.0,
        )
        np.testing.assert_allclose(result.p[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.p[-1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.p[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.p[:, -1], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Convergence order verification (Poisson manufactured solution)
# ---------------------------------------------------------------------------


@pytest.mark.devito
@_skip_no_devito
class TestConvergenceTestPoisson:
    """Tests for convergence_test_poisson_2d with manufactured solution."""

    def test_returns_valid_structure(self):
        from src.elliptic import convergence_test_poisson_2d

        grid_sizes, errors = convergence_test_poisson_2d(
            grid_sizes=[11, 21], n_iterations=500,
        )
        assert len(grid_sizes) == 2
        assert len(errors) == 2

    def test_errors_decrease(self):
        """Errors decrease with grid refinement."""
        from src.elliptic import convergence_test_poisson_2d

        grid_sizes, errors = convergence_test_poisson_2d(
            grid_sizes=[11, 21, 41], n_iterations=5000,
        )
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1], (
                f"Error should decrease: errors[{i}]={errors[i]} "
                f">= errors[{i-1}]={errors[i-1]}"
            )

    def test_second_order_convergence(self):
        """Observed convergence order is approximately 2."""
        from src.elliptic import convergence_test_poisson_2d

        # Need enough iterations so iterative error << discretization error
        grid_sizes, errors = convergence_test_poisson_2d(
            grid_sizes=[11, 21, 41], n_iterations=5000,
        )
        # Compute observed order from finest pair
        log_h = np.log(1.0 / grid_sizes)
        log_err = np.log(errors + 1e-15)
        observed_order = np.polyfit(log_h, log_err, 1)[0]
        assert observed_order >= 1.5, (
            f"Observed convergence order {observed_order:.2f} < 1.5"
        )
