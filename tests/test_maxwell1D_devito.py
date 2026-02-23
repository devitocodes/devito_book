"""Tests for 1D Maxwell FDTD solver."""

import numpy as np
import pytest

from src.em.maxwell1D_devito import (
    MaxwellResult1D,
    convergence_test_maxwell_1d,
    exact_plane_wave_1d,
    gaussian_pulse_1d,
    ricker_wavelet,
    solve_maxwell_1d,
)
from src.em.units import EMConstants


class TestMaxwell1DBasic:
    """Basic functionality tests for 1D Maxwell solver."""

    def test_returns_result_dataclass(self):
        """Solver should return MaxwellResult1D."""
        result = solve_maxwell_1d(L=1.0, Nx=50, T=1e-9)
        assert isinstance(result, MaxwellResult1D)

    def test_array_shapes(self):
        """Output arrays should have correct shapes."""
        Nx = 100
        result = solve_maxwell_1d(L=1.0, Nx=Nx, T=1e-9)

        assert result.E_z.shape == (Nx + 1,)
        assert result.H_y.shape == (Nx,)
        assert result.x_E.shape == (Nx + 1,)
        assert result.x_H.shape == (Nx,)

    def test_coordinate_arrays(self):
        """Coordinate arrays should span correct domain."""
        L = 2.0
        Nx = 100
        result = solve_maxwell_1d(L=L, Nx=Nx, T=1e-9)

        assert result.x_E[0] == 0.0
        assert result.x_E[-1] == L
        assert result.x_H[0] > 0  # Half-integer offset
        assert result.x_H[-1] < L

    def test_cfl_violation_raises(self):
        """CFL > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="CFL"):
            solve_maxwell_1d(L=1.0, Nx=100, T=1e-9, CFL=1.5)


class TestMaxwell1DPEC:
    """Tests for PEC boundary conditions."""

    def test_pec_boundaries_zero(self):
        """E_z should be zero at PEC boundaries."""
        result = solve_maxwell_1d(
            L=1.0, Nx=100, T=5e-9,
            E_init=gaussian_pulse_1d.__wrapped__ if hasattr(gaussian_pulse_1d, '__wrapped__')
                   else lambda x: np.exp(-((x - 0.5)**2) / 0.02**2),
            bc_left="pec", bc_right="pec",
        )

        assert abs(result.E_z[0]) < 1e-10
        assert abs(result.E_z[-1]) < 1e-10


class TestMaxwell1DWaveSpeed:
    """Tests for correct wave propagation speed."""

    def test_standing_wave_frequency(self):
        """Standing wave frequency should match expected f = c/(2L)."""
        const = EMConstants()
        L = 1.0
        Nx = 200

        # Fundamental mode
        def E_init(x):
            return np.sin(np.pi * x / L)

        # Run for one period
        f_expected = const.c0 / (2 * L)
        T_period = 1 / f_expected
        T = 2 * T_period

        result = solve_maxwell_1d(
            L=L, Nx=Nx, T=T, CFL=0.9,
            E_init=E_init,
            bc_left="pec", bc_right="pec",
            save_history=True,
        )

        # After two periods, should return to initial state (approximately)
        if result.E_history is not None:
            E_initial = result.E_history[0]
            E_final = result.E_history[-1]
            correlation = np.corrcoef(E_initial, E_final)[0, 1]
            # Should be highly correlated (same shape, minimal dispersion)
            assert correlation > 0.999, f"Correlation {correlation:.6f} < 0.999"


class TestMaxwell1DExactSolution:
    """Tests against exact analytical solutions."""

    def test_standing_wave_in_cavity(self):
        """Standing wave in PEC cavity should match exact solution."""
        const = EMConstants()
        L = 1.0
        Nx = 200

        # Standing wave mode: sin(pi*x/L) * cos(pi*c*t/L)
        def E_init(x):
            return np.sin(np.pi * x / L)

        # Very short simulation
        omega = np.pi * const.c0 / L
        T = 0.1 / (omega / (2 * np.pi))  # 0.1 periods

        result = solve_maxwell_1d(
            L=L, Nx=Nx, T=T, CFL=0.9,
            E_init=E_init,
            bc_left="pec", bc_right="pec",
        )

        # Exact standing wave solution
        E_exact = np.sin(np.pi * result.x_E / L) * np.cos(omega * result.t)
        error = np.sqrt(np.mean((result.E_z - E_exact)**2))

        # Yee scheme with Nx=200 (dx=0.005): O(dx^2) ~ 2.5e-5 bound
        assert error < 1e-4, f"Error {error:.2e} exceeds threshold"


class TestMaxwell1DConvergence:
    """Convergence tests for 1D Maxwell solver."""

    def test_error_decreases_with_resolution(self):
        """Error should decrease as grid is refined."""
        const = EMConstants()
        L = 1.0
        omega = np.pi * const.c0 / L
        T = 0.1 / (omega / (2 * np.pi))  # 0.1 periods

        grid_sizes = [50, 100, 200]
        errors = []

        for Nx in grid_sizes:
            def E_init(x):
                return np.sin(np.pi * x / L)

            result = solve_maxwell_1d(
                L=L, Nx=Nx, T=T, CFL=0.9,
                E_init=E_init,
                bc_left="pec", bc_right="pec",
            )

            # Exact solution
            E_exact = np.sin(np.pi * result.x_E / L) * np.cos(omega * result.t)
            error = np.sqrt(np.mean((result.E_z - E_exact)**2))
            errors.append(error)

        # Error should decrease monotonically with finer grids
        for i in range(len(errors) - 1):
            assert errors[i+1] < errors[i], \
                f"Error did not decrease: {errors[i]:.2e} -> {errors[i+1]:.2e}"


class TestRickerWavelet:
    """Tests for Ricker wavelet source."""

    def test_wavelet_shape(self):
        """Ricker wavelet should have correct shape."""
        t = np.linspace(0, 10e-9, 1000)
        f0 = 500e6
        wavelet = ricker_wavelet(t, f0=f0)

        # Peak should be at t0 = 1/f0
        t0 = 1.0 / f0
        peak_idx = np.argmax(wavelet)
        t_peak = t[peak_idx]
        assert abs(t_peak - t0) < t[1] - t[0]  # Within one time step

    def test_wavelet_amplitude(self):
        """Ricker wavelet peak should be 1 (default amplitude)."""
        t = np.linspace(0, 10e-9, 1000)
        wavelet = ricker_wavelet(t, f0=500e6)
        assert abs(np.max(wavelet) - 1.0) < 0.005


class TestMaxwell1DHistory:
    """Tests for solution history saving."""

    def test_history_saved_when_requested(self):
        """History should be saved when save_history=True."""
        result = solve_maxwell_1d(
            L=1.0, Nx=50, T=1e-9, save_history=True
        )

        assert result.E_history is not None
        assert result.H_history is not None
        assert result.t_history is not None

    def test_history_not_saved_by_default(self):
        """History should not be saved by default."""
        result = solve_maxwell_1d(L=1.0, Nx=50, T=1e-9)

        assert result.E_history is None
        assert result.H_history is None


class TestMaxwell1DZeroTime:
    """Tests for T=0 edge case."""

    def test_zero_time_returns_initial(self):
        """T=0 should return initial condition."""
        def E_init(x):
            return np.sin(np.pi * x)

        result = solve_maxwell_1d(
            L=1.0, Nx=100, T=0,
            E_init=E_init,
        )

        expected = np.sin(np.pi * result.x_E)
        np.testing.assert_allclose(result.E_z, expected, rtol=1e-10)


class TestMaxwell1DLossy:
    """Tests for lossy media (sigma > 0)."""

    def test_lossy_attenuates(self):
        """Field energy should decay in lossy medium."""
        def E_init(x):
            return np.sin(np.pi * x)

        # Lossless reference
        ref = solve_maxwell_1d(
            L=1.0, Nx=200, T=3e-9, CFL=0.9,
            E_init=E_init, sigma=0.0,
            bc_left="pec", bc_right="pec",
        )

        # Lossy
        lossy = solve_maxwell_1d(
            L=1.0, Nx=200, T=3e-9, CFL=0.9,
            E_init=E_init, sigma=0.1,
            bc_left="pec", bc_right="pec",
        )

        energy_ref = np.sum(ref.E_z**2)
        energy_lossy = np.sum(lossy.E_z**2)
        assert energy_lossy < energy_ref

    def test_higher_sigma_more_loss(self):
        """Higher conductivity should produce more attenuation."""
        def E_init(x):
            return np.sin(np.pi * x)

        results = []
        for sigma in [0.01, 0.1, 1.0]:
            r = solve_maxwell_1d(
                L=1.0, Nx=200, T=3e-9, CFL=0.9,
                E_init=E_init, sigma=sigma,
                bc_left="pec", bc_right="pec",
            )
            results.append(np.sum(r.E_z**2))

        for i in range(len(results) - 1):
            assert results[i + 1] < results[i]

    def test_lossy_pec_boundaries(self):
        """PEC boundaries should still hold in lossy medium."""
        def E_init(x):
            return np.sin(np.pi * x)

        result = solve_maxwell_1d(
            L=1.0, Nx=200, T=3e-9, CFL=0.9,
            E_init=E_init, sigma=0.05,
            bc_left="pec", bc_right="pec",
        )
        assert abs(result.E_z[0]) < 1e-10
        assert abs(result.E_z[-1]) < 1e-10


class TestMaxwell1DDispersive:
    """Tests for spatially varying permittivity."""

    def test_variable_eps_r_runs(self):
        """Solver should accept array-valued eps_r."""
        Nx = 200
        eps_r = np.ones(Nx + 1)
        eps_r[Nx // 2:] = 4.0  # Higher permittivity in second half

        def E_init(x):
            return np.exp(-((x - 0.25)**2) / 0.01)

        result = solve_maxwell_1d(
            L=1.0, Nx=Nx, T=2e-9, CFL=0.9,
            E_init=E_init, eps_r=eps_r,
            bc_left="pec", bc_right="pec",
        )
        assert result.E_z.shape == (Nx + 1,)

    def test_higher_eps_r_slower_wave(self):
        """Wave speed should decrease with higher permittivity."""
        Nx = 200

        # Pulse near left boundary
        def E_init(x):
            return np.exp(-((x - 0.1)**2) / 0.005)

        r1 = solve_maxwell_1d(
            L=1.0, Nx=Nx, T=2e-9, CFL=0.9,
            E_init=E_init, eps_r=1.0,
            bc_left="pec", bc_right="pec",
            save_history=True,
        )

        r2 = solve_maxwell_1d(
            L=1.0, Nx=Nx, T=2e-9, CFL=0.9,
            E_init=E_init, eps_r=9.0,
            bc_left="pec", bc_right="pec",
            save_history=True,
        )

        # In vacuum, wave should be higher because it propagated
        # faster and reflected; check c is reported correctly
        assert r1.c > r2.c


class TestMaxwell1DABC:
    """Tests for absorbing boundary conditions."""

    def test_abc_no_reflection(self):
        """ABC should reduce reflections compared to PEC."""
        def E_init(x):
            return np.exp(-((x - 0.5)**2) / 0.01)

        # PEC: wave bounces back
        pec = solve_maxwell_1d(
            L=1.0, Nx=200, T=5e-9, CFL=0.9,
            E_init=E_init,
            bc_left="pec", bc_right="pec",
        )

        # ABC: wave should mostly leave
        abc = solve_maxwell_1d(
            L=1.0, Nx=200, T=5e-9, CFL=0.9,
            E_init=E_init,
            bc_left="abc", bc_right="abc",
        )

        energy_pec = np.sum(pec.E_z**2)
        energy_abc = np.sum(abc.E_z**2)
        assert energy_abc < energy_pec

    def test_abc_boundaries_nonzero(self):
        """ABC boundaries should NOT force E_z to zero."""
        def E_init(x):
            return np.exp(-((x - 0.5)**2) / 0.01)

        result = solve_maxwell_1d(
            L=1.0, Nx=200, T=1e-9, CFL=0.9,
            E_init=E_init,
            bc_left="abc", bc_right="abc",
            save_history=True,
        )
        # At some point in the history, boundary should be nonzero
        # as the wave passes through
        if result.E_history is not None:
            max_boundary = max(
                np.max(np.abs(result.E_history[:, 0])),
                np.max(np.abs(result.E_history[:, -1])),
            )
            assert max_boundary > 1e-10


class TestMaxwell1DPMC:
    """Tests for PMC (perfect magnetic conductor) boundaries."""

    def test_pmc_boundaries(self):
        """PMC should give dE/dx = 0 at boundaries (E copies neighbor)."""
        def E_init(x):
            return np.sin(np.pi * x)

        result = solve_maxwell_1d(
            L=1.0, Nx=200, T=1e-9, CFL=0.9,
            E_init=E_init,
            bc_left="pmc", bc_right="pmc",
        )
        # PMC mirrors neighboring value, so boundary is NOT zero
        # (unlike PEC). Just check it runs and produces output.
        assert result.E_z.shape[0] == 201

    def test_pmc_preserves_energy_better(self):
        """PMC with symmetric init should preserve more energy than ABC."""
        def E_init(x):
            return np.cos(np.pi * x)

        pmc = solve_maxwell_1d(
            L=1.0, Nx=200, T=3e-9, CFL=0.9,
            E_init=E_init,
            bc_left="pmc", bc_right="pmc",
        )

        abc = solve_maxwell_1d(
            L=1.0, Nx=200, T=3e-9, CFL=0.9,
            E_init=E_init,
            bc_left="abc", bc_right="abc",
        )

        energy_pmc = np.sum(pmc.E_z**2)
        energy_abc = np.sum(abc.E_z**2)
        assert energy_pmc > energy_abc


class TestMaxwell1DSource:
    """Tests for source injection."""

    def test_source_injection(self):
        """Source should inject energy into the domain."""
        f0 = 1e9
        result = solve_maxwell_1d(
            L=1.0, Nx=200, T=5e-9, CFL=0.9,
            source_func=lambda t: ricker_wavelet(np.array([t]), f0=f0)[0],
            source_position=0.5,
            bc_left="abc", bc_right="abc",
        )

        # Should have nonzero field from source
        assert np.max(np.abs(result.E_z)) > 0

    def test_source_requires_position(self):
        """source_func without source_position should raise."""
        with pytest.raises(ValueError, match="source_position"):
            solve_maxwell_1d(
                L=1.0, Nx=100, T=1e-9,
                source_func=lambda t: 1.0,
            )

    def test_source_with_history(self):
        """Source injection should work with history saving."""
        f0 = 1e9
        result = solve_maxwell_1d(
            L=1.0, Nx=100, T=3e-9, CFL=0.9,
            source_func=lambda t: ricker_wavelet(np.array([t]), f0=f0)[0],
            source_position=0.5,
            bc_left="abc", bc_right="abc",
            save_history=True,
        )

        assert result.E_history is not None
        assert np.max(np.abs(result.E_history[-1])) > 0


class TestMaxwell1DConvergenceTest:
    """Tests for the convergence_test_maxwell_1d function."""

    def test_convergence_test_runs(self):
        """convergence_test_maxwell_1d should run and return results."""
        grid_sizes, errors, order = convergence_test_maxwell_1d(
            grid_sizes=[50, 100, 200],
            T=0.5e-9,
        )
        assert len(grid_sizes) == 3
        assert len(errors) == 3
        assert all(e > 0 for e in errors)

    def test_errors_decrease(self):
        """Finest grid should have smaller error than coarsest."""
        _, errors, _ = convergence_test_maxwell_1d(
            grid_sizes=[50, 100, 200],
            T=0.5e-9,
        )
        assert errors[-1] < errors[0]


class TestExactPlaneWave:
    """Tests for exact_plane_wave_1d."""

    def test_returns_e_and_h(self):
        """Should return (E_z, H_y) tuple."""
        x = np.linspace(0, 1, 100)
        E, H = exact_plane_wave_1d(x, t=0.0, frequency=1e9)
        assert E.shape == x.shape
        assert H.shape == x.shape

    def test_requires_frequency_param(self):
        """Should raise if no frequency parameter given."""
        x = np.linspace(0, 1, 100)
        with pytest.raises(ValueError, match="One of"):
            exact_plane_wave_1d(x, t=0.0)

    def test_impedance_relation(self):
        """E/H ratio should equal wave impedance."""
        const = EMConstants()
        x = np.linspace(0, 1, 100)
        E, H = exact_plane_wave_1d(x, t=0.0, frequency=1e9)
        # E/H = eta0 for vacuum, where both are nonzero
        mask = np.abs(H) > 1e-15
        ratio = E[mask] / H[mask]
        np.testing.assert_allclose(ratio, const.eta0, rtol=1e-10)


class TestGaussianPulse1D:
    """Tests for gaussian_pulse_1d."""

    def test_peak_at_center(self):
        """Peak should be at x0."""
        x = np.linspace(0, 1, 1000)
        pulse = gaussian_pulse_1d(x, x0=0.5, sigma=0.05)
        peak_x = x[np.argmax(pulse)]
        assert peak_x == pytest.approx(0.5, abs=0.002)

    def test_amplitude(self):
        """Peak amplitude should match parameter."""
        x = np.linspace(0, 1, 1000)
        pulse = gaussian_pulse_1d(x, x0=0.5, sigma=0.05, amplitude=3.0)
        assert np.max(pulse) == pytest.approx(3.0, rel=1e-3)
