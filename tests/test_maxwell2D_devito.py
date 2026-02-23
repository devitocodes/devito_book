"""Tests for src.em.maxwell2D_devito — 2D Maxwell FDTD solver."""

import numpy as np
import pytest

from src.em.maxwell2D_devito import (
    MaxwellResult2D,
    create_pml_profile,
    gaussian_source_2d,
    line_source_2d,
    solve_maxwell_2d,
)

# 2D CFL must be <= 1/sqrt(2) ≈ 0.707
CFL_2D = 0.5


# ---------------------------------------------------------------------------
# PML profile (pure NumPy)
# ---------------------------------------------------------------------------


class TestCreatePmlProfile:

    def test_shape(self):
        sigma = create_pml_profile(N=100, pml_width=10)
        assert sigma.shape == (100,)

    def test_zero_in_interior(self):
        sigma = create_pml_profile(N=100, pml_width=10)
        assert np.all(sigma[15:85] == 0.0)

    def test_nonzero_in_pml_region(self):
        sigma = create_pml_profile(N=100, pml_width=10, sigma_max=1.0)
        assert sigma[0] > 0
        assert sigma[-1] > 0

    def test_symmetric(self):
        sigma = create_pml_profile(N=100, pml_width=10, sigma_max=1.0)
        np.testing.assert_allclose(sigma[:10], sigma[-10:][::-1])

    def test_max_at_boundary(self):
        """Maximum conductivity should be at domain edges."""
        sigma = create_pml_profile(N=100, pml_width=10, sigma_max=2.0)
        assert sigma[0] == pytest.approx(2.0, rel=1e-10)
        assert sigma[-1] == pytest.approx(2.0, rel=1e-10)

    def test_order_affects_grading(self):
        """Higher order should give steeper profile."""
        s1 = create_pml_profile(N=100, pml_width=10, sigma_max=1.0, order=2)
        s3 = create_pml_profile(N=100, pml_width=10, sigma_max=1.0, order=4)
        # At midpoint of PML, higher order should have lower value
        assert s3[5] < s1[5]


# ---------------------------------------------------------------------------
# Source functions (pure NumPy)
# ---------------------------------------------------------------------------


class TestGaussianSource2d:

    def test_peak_at_center(self):
        x = np.linspace(0, 1, 101)
        y = np.linspace(0, 1, 101)
        X, Y = np.meshgrid(x, y, indexing='ij')
        E = gaussian_source_2d(X, Y, x0=0.5, y0=0.5, sigma=0.05)
        i, j = np.unravel_index(np.argmax(E), E.shape)
        assert x[i] == pytest.approx(0.5, abs=0.02)
        assert y[j] == pytest.approx(0.5, abs=0.02)

    def test_amplitude(self):
        x = np.linspace(0, 1, 101)
        y = np.linspace(0, 1, 101)
        X, Y = np.meshgrid(x, y, indexing='ij')
        E = gaussian_source_2d(X, Y, x0=0.5, y0=0.5, sigma=0.05, amplitude=3.0)
        assert np.max(E) == pytest.approx(3.0, rel=1e-6)

    def test_shape(self):
        x = np.linspace(0, 1, 51)
        y = np.linspace(0, 1, 61)
        X, Y = np.meshgrid(x, y, indexing='ij')
        E = gaussian_source_2d(X, Y, x0=0.5, y0=0.5, sigma=0.05)
        assert E.shape == (51, 61)


class TestLineSource2d:

    def test_uniform_in_y(self):
        """Line source should be constant along y."""
        x = np.linspace(0, 1, 101)
        y = np.linspace(0, 1, 101)
        X, Y = np.meshgrid(x, y, indexing='ij')
        E = line_source_2d(X, Y, x0=0.5, sigma=0.05)
        # Each row (constant x) should have same value
        for i in range(E.shape[0]):
            assert np.allclose(E[i, :], E[i, 0])

    def test_peak_at_x0(self):
        x = np.linspace(0, 1, 101)
        y = np.linspace(0, 1, 101)
        X, Y = np.meshgrid(x, y, indexing='ij')
        E = line_source_2d(X, Y, x0=0.3, sigma=0.05)
        peak_row = np.argmax(E[:, 50])
        assert x[peak_row] == pytest.approx(0.3, abs=0.02)


# ---------------------------------------------------------------------------
# 2D Maxwell solver
# ---------------------------------------------------------------------------


class TestSolveMaxwell2dBasic:

    def test_returns_result_dataclass(self):
        result = solve_maxwell_2d(Lx=1.0, Ly=1.0, Nx=30, Ny=30, T=1e-9, CFL=CFL_2D)
        assert isinstance(result, MaxwellResult2D)

    def test_array_shapes(self):
        Nx, Ny = 40, 50
        result = solve_maxwell_2d(Lx=1.0, Ly=1.0, Nx=Nx, Ny=Ny, T=1e-9, CFL=CFL_2D)
        assert result.E_z.shape == (Nx + 1, Ny + 1)
        assert result.H_x.shape == (Nx + 1, Ny)
        assert result.H_y.shape == (Nx, Ny + 1)
        assert result.x.shape == (Nx + 1,)
        assert result.y.shape == (Ny + 1,)

    def test_cfl_violation_raises(self):
        """CFL > 1/sqrt(2) should raise."""
        with pytest.raises(ValueError, match="CFL"):
            solve_maxwell_2d(Lx=1.0, Ly=1.0, Nx=30, Ny=30, T=1e-9, CFL=0.8)

    def test_coordinate_arrays(self):
        result = solve_maxwell_2d(Lx=2.0, Ly=3.0, Nx=30, Ny=30, T=1e-9, CFL=CFL_2D)
        assert result.x[0] == 0.0
        assert result.x[-1] == 2.0
        assert result.y[0] == 0.0
        assert result.y[-1] == 3.0


class TestSolveMaxwell2dPEC:

    def test_pec_boundaries(self):
        """E_z should be zero at PEC boundaries."""
        def E_init(X, Y):
            return gaussian_source_2d(X, Y, 0.5, 0.5, 0.05)

        result = solve_maxwell_2d(
            Lx=1.0, Ly=1.0, Nx=50, Ny=50, T=2e-9,
            CFL=CFL_2D, E_init=E_init, pml_width=0,
        )
        np.testing.assert_allclose(result.E_z[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.E_z[-1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.E_z[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.E_z[:, -1], 0.0, atol=1e-10)


class TestSolveMaxwell2dPML:

    def test_pml_reduces_reflection(self):
        """PML should reduce reflected energy compared to PEC."""
        def E_init(X, Y):
            return gaussian_source_2d(X, Y, 0.5, 0.5, 0.05)

        pec = solve_maxwell_2d(
            Lx=1.0, Ly=1.0, Nx=50, Ny=50, T=3e-9,
            CFL=CFL_2D, E_init=E_init, pml_width=0,
        )

        pml = solve_maxwell_2d(
            Lx=1.0, Ly=1.0, Nx=50, Ny=50, T=3e-9,
            CFL=CFL_2D, E_init=E_init, pml_width=10,
        )

        # Interior energy should be lower with PML (wave absorbed)
        inner = slice(15, -15)
        energy_pec = np.sum(pec.E_z[inner, inner]**2)
        energy_pml = np.sum(pml.E_z[inner, inner]**2)
        assert energy_pml < energy_pec


class TestSolveMaxwell2dLossy:

    def test_lossy_attenuates(self):
        """Lossy medium should attenuate the field."""
        def E_init(X, Y):
            return gaussian_source_2d(X, Y, 0.5, 0.5, 0.05)

        lossless = solve_maxwell_2d(
            Lx=1.0, Ly=1.0, Nx=50, Ny=50, T=2e-9,
            CFL=CFL_2D, E_init=E_init, sigma=0.0,
        )

        lossy = solve_maxwell_2d(
            Lx=1.0, Ly=1.0, Nx=50, Ny=50, T=2e-9,
            CFL=CFL_2D, E_init=E_init, sigma=0.5,
        )

        energy_lossless = np.sum(lossless.E_z**2)
        energy_lossy = np.sum(lossy.E_z**2)
        assert energy_lossy < energy_lossless


class TestSolveMaxwell2dDispersive:

    def test_variable_eps_r(self):
        """Solver should accept spatially varying eps_r."""
        Nx, Ny = 50, 50
        eps_r = np.ones((Nx + 1, Ny + 1))
        eps_r[Nx // 2:, :] = 4.0

        def E_init(X, Y):
            return gaussian_source_2d(X, Y, 0.25, 0.5, 0.05)

        result = solve_maxwell_2d(
            Lx=1.0, Ly=1.0, Nx=Nx, Ny=Ny, T=2e-9,
            CFL=CFL_2D, E_init=E_init, eps_r=eps_r,
        )
        assert result.E_z.shape == (Nx + 1, Ny + 1)


class TestSolveMaxwell2dSource:

    def test_source_injection(self):
        """Source should inject energy into domain."""
        f0 = 1e9

        def src(t):
            tau = np.pi * f0 * (t - 1.0 / f0)
            return (1 - 2 * tau**2) * np.exp(-tau**2)

        result = solve_maxwell_2d(
            Lx=1.0, Ly=1.0, Nx=50, Ny=50, T=3e-9,
            CFL=CFL_2D, source_func=src, source_position=(0.5, 0.5),
            pml_width=10,
        )
        assert np.max(np.abs(result.E_z)) > 0

    def test_source_requires_position(self):
        """source_func without source_position should raise."""
        with pytest.raises(ValueError, match="source_position"):
            solve_maxwell_2d(
                Lx=1.0, Ly=1.0, Nx=30, Ny=30, T=1e-9,
                CFL=CFL_2D, source_func=lambda t: 1.0,
            )


class TestSolveMaxwell2dHistory:

    def test_history_saved(self):
        result = solve_maxwell_2d(
            Lx=1.0, Ly=1.0, Nx=30, Ny=30, T=2e-9,
            CFL=CFL_2D,
            E_init=lambda X, Y: gaussian_source_2d(X, Y, 0.5, 0.5, 0.05),
            save_history=True, save_every=5,
        )
        assert result.E_history is not None
        assert result.t_history is not None
        assert len(result.E_history) > 1

    def test_history_not_saved_by_default(self):
        result = solve_maxwell_2d(
            Lx=1.0, Ly=1.0, Nx=30, Ny=30, T=1e-9, CFL=CFL_2D,
        )
        assert result.E_history is None
        assert result.t_history is None
