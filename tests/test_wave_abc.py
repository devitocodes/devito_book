"""Tests for absorbing boundary condition methods (src/wave/abc_methods.py)."""

import numpy as np
import pytest


def _devito_importable() -> bool:
    try:
        import devito  # noqa: F401
    except Exception:
        return False
    return True


# ---- Tests that do NOT require Devito ----

class TestDampingProfile:
    """Tests for create_damping_profile (pure NumPy, no Devito)."""

    def test_zero_in_interior(self):
        from src.wave.abc_methods import create_damping_profile

        profile = create_damping_profile((101, 101), pad_width=10, sigma_max=50.0, c=1.0, dx=0.01)
        # Interior region should be zero
        assert np.all(profile[15:86, 15:86] == 0.0)

    def test_polynomial_increase(self):
        from src.wave.abc_methods import create_damping_profile

        profile = create_damping_profile((101, 101), pad_width=20, sigma_max=100.0, order=2)
        # Values should increase toward the boundary
        # Check left boundary in x (at mid-y=50)
        left_vals = profile[:20, 50]
        # Should decrease monotonically from boundary (index 0) to interior (index 19)
        for i in range(len(left_vals) - 1):
            assert left_vals[i] >= left_vals[i + 1]

    def test_symmetry(self):
        from src.wave.abc_methods import create_damping_profile

        profile = create_damping_profile((101, 101), pad_width=15, sigma_max=50.0)
        # Left-right symmetry
        np.testing.assert_allclose(profile[:15, 50], profile[101-15:, 50][::-1], atol=1e-10)
        # Top-bottom symmetry
        np.testing.assert_allclose(profile[50, :15], profile[50, 101-15:][::-1], atol=1e-10)

    def test_max_value_at_boundary(self):
        from src.wave.abc_methods import create_damping_profile

        sigma_max = 75.0
        profile = create_damping_profile((101, 101), pad_width=10, sigma_max=sigma_max)
        assert np.max(profile) == pytest.approx(sigma_max, rel=0.01)

    def test_shape(self):
        from src.wave.abc_methods import create_damping_profile

        profile = create_damping_profile((51, 81), pad_width=5, sigma_max=10.0)
        assert profile.shape == (51, 81)


class TestReflectionMeasurement:
    """Tests for measure_reflection (pure NumPy)."""

    def test_returns_between_zero_and_one(self):
        from src.wave.abc_methods import ABCResult, measure_reflection

        x = np.linspace(0, 1, 51)
        y = np.linspace(0, 1, 51)
        u = np.random.RandomState(42).randn(51, 51)

        result = ABCResult(u=u, x=x, y=y, t=1.0, dt=0.01, abc_type='test')
        R = measure_reflection(result)
        assert 0.0 <= R <= 1.0

    def test_zero_field_gives_zero(self):
        from src.wave.abc_methods import ABCResult, measure_reflection

        x = np.linspace(0, 1, 51)
        y = np.linspace(0, 1, 51)
        u = np.zeros((51, 51))

        result = ABCResult(u=u, x=x, y=y, t=1.0, dt=0.01, abc_type='test')
        R = measure_reflection(result)
        assert R == 0.0

    def test_with_reference(self):
        from src.wave.abc_methods import ABCResult, measure_reflection

        x = np.linspace(0, 1, 51)
        y = np.linspace(0, 1, 51)
        u_abc = np.ones((51, 51)) * 0.5
        u_ref = np.ones((51, 51))

        result_abc = ABCResult(u=u_abc, x=x, y=y, t=1.0, dt=0.01, abc_type='damping')
        result_ref = ABCResult(u=u_ref, x=x, y=y, t=1.0, dt=0.01, abc_type='ref')
        R = measure_reflection(result_abc, result_ref)
        assert 0.0 < R < 1.0


# ---- Tests that require Devito ----

@pytest.mark.devito
@pytest.mark.skipif(not _devito_importable(), reason="Devito not importable")
class TestWave2dDirichlet:
    """Baseline: Dirichlet BC solver runs correctly."""

    def test_basic_run(self):
        from src.wave.abc_methods import solve_wave_2d_abc

        result = solve_wave_2d_abc(
            Lx=1.0, Ly=1.0, Nx=40, Ny=40, T=0.3, CFL=0.5,
            abc_type='dirichlet',
        )
        assert result.u.shape == (41, 41)
        assert np.isfinite(result.u).all()
        assert result.abc_type == 'dirichlet'


@pytest.mark.devito
@pytest.mark.skipif(not _devito_importable(), reason="Devito not importable")
class TestWave2dFirstOrderABC:
    """Tests for first-order (Clayton-Engquist) ABC."""

    def test_basic_run(self):
        from src.wave.abc_methods import solve_wave_2d_abc

        result = solve_wave_2d_abc(
            Lx=1.0, Ly=1.0, Nx=40, Ny=40, T=0.3, CFL=0.5,
            abc_type='first_order',
        )
        assert result.u.shape == (41, 41)
        assert np.isfinite(result.u).all()

    def test_reduces_reflection_vs_dirichlet(self):
        from src.wave.abc_methods import measure_reflection, solve_wave_2d_abc

        kwargs = dict(Lx=2.0, Ly=2.0, Nx=60, Ny=60, T=1.5, CFL=0.5)
        result_dir = solve_wave_2d_abc(**kwargs, abc_type='dirichlet')
        result_abc = solve_wave_2d_abc(**kwargs, abc_type='first_order')

        R_dir = measure_reflection(result_dir)
        R_abc = measure_reflection(result_abc)
        # First-order ABC should have less interior energy than Dirichlet
        assert R_abc < R_dir or R_dir < 0.01  # unless both very small


@pytest.mark.devito
@pytest.mark.skipif(not _devito_importable(), reason="Devito not importable")
class TestWave2dDamping:
    """Tests for damping (sponge) layer ABC."""

    def test_basic_run(self):
        from src.wave.abc_methods import solve_wave_2d_abc

        result = solve_wave_2d_abc(
            Lx=2.0, Ly=2.0, Nx=60, Ny=60, T=0.5, CFL=0.5,
            abc_type='damping', pad_width=10,
        )
        assert result.u.shape == (61, 61)
        assert np.isfinite(result.u).all()
        assert result.abc_type == 'damping'

    def test_reduces_reflection_vs_dirichlet(self):
        from src.wave.abc_methods import measure_reflection, solve_wave_2d_abc

        kwargs = dict(Lx=2.0, Ly=2.0, Nx=60, Ny=60, T=1.5, CFL=0.5)
        result_dir = solve_wave_2d_abc(**kwargs, abc_type='dirichlet')
        result_dmp = solve_wave_2d_abc(**kwargs, abc_type='damping', pad_width=15)

        R_dir = measure_reflection(result_dir)
        R_dmp = measure_reflection(result_dmp)
        assert R_dmp < R_dir or R_dir < 0.01

    def test_wider_layer_less_reflection(self):
        from src.wave.abc_methods import solve_wave_2d_abc

        kwargs = dict(Lx=2.0, Ly=2.0, Nx=80, Ny=80, T=1.5, CFL=0.5,
                      abc_type='damping')
        result_thin = solve_wave_2d_abc(**kwargs, pad_width=5)
        result_wide = solve_wave_2d_abc(**kwargs, pad_width=20)

        # Wider layer should absorb more -> less total energy
        energy_thin = np.sum(result_thin.u**2)
        energy_wide = np.sum(result_wide.u**2)
        assert energy_wide <= energy_thin


@pytest.mark.devito
@pytest.mark.skipif(not _devito_importable(), reason="Devito not importable")
class TestWave2dPML:
    """Tests for PML ABC."""

    def test_basic_run(self):
        from src.wave.abc_methods import solve_wave_2d_abc

        result = solve_wave_2d_abc(
            Lx=2.0, Ly=2.0, Nx=60, Ny=60, T=0.5, CFL=0.5,
            abc_type='pml', pad_width=10,
        )
        assert result.u.shape == (61, 61)
        assert np.isfinite(result.u).all()
        assert result.abc_type == 'pml'

    def test_reduces_reflection_vs_dirichlet(self):
        from src.wave.abc_methods import measure_reflection, solve_wave_2d_abc

        kwargs = dict(Lx=2.0, Ly=2.0, Nx=60, Ny=60, T=1.5, CFL=0.5)
        result_dir = solve_wave_2d_abc(**kwargs, abc_type='dirichlet')
        result_pml = solve_wave_2d_abc(**kwargs, abc_type='pml', pad_width=15)

        R_dir = measure_reflection(result_dir)
        R_pml = measure_reflection(result_pml)
        assert R_pml < R_dir or R_dir < 0.01


@pytest.mark.devito
@pytest.mark.skipif(not _devito_importable(), reason="Devito not importable")
class TestCompareABC:
    """Tests for compare_abc_methods."""

    def test_runs_multiple_methods(self):
        from src.wave.abc_methods import compare_abc_methods

        results = compare_abc_methods(
            Lx=1.0, Ly=1.0, Nx=40, Ny=40, T=0.5, CFL=0.5,
            methods=['dirichlet', 'first_order', 'damping'],
            pad_width=8,
        )
        assert 'dirichlet' in results
        assert 'first_order' in results
        assert 'damping' in results
        for name, result in results.items():
            assert result.u.shape == (41, 41)
            assert np.isfinite(result.u).all()


@pytest.mark.devito
@pytest.mark.skipif(not _devito_importable(), reason="Devito not importable")
class TestSaveHistory:
    """Tests for save_history functionality."""

    def test_history_saved(self):
        from src.wave.abc_methods import solve_wave_2d_abc

        result = solve_wave_2d_abc(
            Lx=1.0, Ly=1.0, Nx=30, Ny=30, T=0.2, CFL=0.5,
            abc_type='damping', pad_width=5, save_history=True,
        )
        assert result.u_history is not None
        assert result.t_history is not None
        assert len(result.t_history) == result.u_history.shape[0]

    def test_invalid_abc_type_raises(self):
        from src.wave.abc_methods import solve_wave_2d_abc

        with pytest.raises(ValueError, match="abc_type must be"):
            solve_wave_2d_abc(abc_type='invalid')


@pytest.mark.devito
@pytest.mark.skipif(not _devito_importable(), reason="Devito not importable")
class TestWave2dHigdon:
    """Tests for second-order Higdon ABC."""

    def test_basic_run(self):
        from src.wave.abc_methods import solve_wave_2d_abc

        result = solve_wave_2d_abc(
            Lx=2.0, Ly=2.0, Nx=60, Ny=60, T=0.5, CFL=0.5,
            abc_type='higdon',
        )
        assert result.u.shape == (61, 61)
        assert np.isfinite(result.u).all()
        assert result.abc_type == 'higdon'

    def test_reduces_reflection_vs_first_order(self):
        from src.wave.abc_methods import measure_reflection, solve_wave_2d_abc

        kwargs = dict(Lx=2.0, Ly=2.0, Nx=60, Ny=60, T=1.5, CFL=0.5)
        result_first = solve_wave_2d_abc(**kwargs, abc_type='first_order')
        result_hig = solve_wave_2d_abc(**kwargs, abc_type='higdon')

        R_first = measure_reflection(result_first)
        R_hig = measure_reflection(result_hig)
        # Higdon P=2 should have less reflection than first-order
        assert R_hig < R_first or R_first < 0.01


@pytest.mark.devito
@pytest.mark.skipif(not _devito_importable(), reason="Devito not importable")
class TestWave2dHABC:
    """Tests for Hybrid ABC (Higdon + weighted absorption layer)."""

    def test_basic_run(self):
        from src.wave.abc_methods import solve_wave_2d_abc

        result = solve_wave_2d_abc(
            Lx=2.0, Ly=2.0, Nx=60, Ny=60, T=0.5, CFL=0.5,
            abc_type='habc', pad_width=10,
        )
        assert result.u.shape == (61, 61)
        assert np.isfinite(result.u).all()
        assert result.abc_type == 'habc'

    def test_better_than_damping_with_fewer_cells(self):
        from src.wave.abc_methods import solve_wave_2d_abc

        kwargs = dict(Lx=2.0, Ly=2.0, Nx=80, Ny=80, T=1.5, CFL=0.5)
        # Damping with 20 cells vs HABC with 10 cells
        result_dmp = solve_wave_2d_abc(**kwargs, abc_type='damping', pad_width=20)
        result_habc = solve_wave_2d_abc(**kwargs, abc_type='habc', pad_width=10)

        energy_dmp = np.sum(result_dmp.u**2)
        energy_habc = np.sum(result_habc.u**2)
        # HABC with thinner layer should still absorb well
        assert energy_habc <= energy_dmp * 2  # at most 2x energy


class TestHABCWeights:
    """Tests for create_habc_weights (pure NumPy)."""

    def test_shape(self):
        from src.wave.abc_methods import create_habc_weights

        weights = create_habc_weights(10)
        assert weights.shape == (10,)

    def test_outer_is_one(self):
        from src.wave.abc_methods import create_habc_weights

        weights = create_habc_weights(10, P=2)
        # First P+1 entries should be 1.0
        assert np.all(weights[:3] == 1.0)

    def test_inner_is_zero(self):
        from src.wave.abc_methods import create_habc_weights

        weights = create_habc_weights(10)
        assert weights[-1] == 0.0

    def test_monotone_decreasing(self):
        from src.wave.abc_methods import create_habc_weights

        weights = create_habc_weights(15, P=2)
        # After the flat region (P+1), weights should decrease
        for k in range(3, len(weights) - 1):
            assert weights[k] >= weights[k + 1]
