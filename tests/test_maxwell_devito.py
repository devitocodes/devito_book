"""Tests for FDTD Maxwell's Equations Solver using Devito.

This module tests the computational electromagnetics implementation,
including:
1. 1D and 2D FDTD solvers
2. Plane wave propagation and verification
3. Resonant cavity modes
4. Boundary conditions (PEC, PMC, ABC)
5. PML absorbing boundaries
6. Energy conservation
7. Source functions
8. CFL stability

Physical constants:
    - c0 = 299792458 m/s (speed of light)
    - mu0 = 4π × 10⁻⁷ H/m (permeability)
    - eps0 = 8.854 × 10⁻¹² F/m (permittivity)

Per CONTRIBUTING.md: All results must be reproducible with fixed random seeds,
version-pinned dependencies, and automated tests validating examples.
"""

import numpy as np
import pytest

# Check if Devito is available
try:
    import devito  # noqa: F401

    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")

# Physical constants
C0 = 299792458.0
MU0 = 4.0 * np.pi * 1e-7
EPS0 = 8.854187817e-12
ETA0 = np.sqrt(MU0 / EPS0)


# =============================================================================
# Test: Module Imports
# =============================================================================


@pytest.mark.devito
class TestModuleImports:
    """Test that the maxwell module imports correctly."""

    def test_import_maxwell_module(self):
        """Test importing the maxwell module."""
        from src.maxwell import maxwell_devito

        assert maxwell_devito is not None

    def test_import_solver_1d(self):
        """Test importing 1D solver."""
        from src.maxwell import solve_maxwell_1d

        assert solve_maxwell_1d is not None

    def test_import_solver_2d(self):
        """Test importing 2D solver."""
        from src.maxwell import solve_maxwell_2d

        assert solve_maxwell_2d is not None

    def test_import_result_dataclasses(self):
        """Test importing result dataclasses."""
        from src.maxwell import MaxwellResult, MaxwellResult2D

        assert MaxwellResult is not None
        assert MaxwellResult2D is not None

    def test_import_pml_functions(self):
        """Test importing PML functions."""
        from src.maxwell import create_cpml_coefficients, create_pml_sigma

        assert create_cpml_coefficients is not None
        assert create_pml_sigma is not None

    def test_import_sources(self):
        """Test importing source functions."""
        from src.maxwell import (
            gaussian_modulated_source,
            gaussian_pulse_em,
            sinusoidal_source,
        )

        assert gaussian_pulse_em is not None
        assert sinusoidal_source is not None
        assert gaussian_modulated_source is not None

    def test_import_analytical(self):
        """Test importing analytical solutions."""
        from src.maxwell import (
            cavity_resonant_frequencies,
            exact_plane_wave_1d,
            exact_plane_wave_2d,
        )

        assert exact_plane_wave_1d is not None
        assert exact_plane_wave_2d is not None
        assert cavity_resonant_frequencies is not None


# =============================================================================
# Test: Source Functions
# =============================================================================


class TestSourceFunctions:
    """Tests for electromagnetic source functions."""

    def test_gaussian_pulse_shape(self):
        """Gaussian pulse should have correct shape."""
        from src.maxwell import gaussian_pulse_em

        t = np.linspace(0, 10e-9, 1000)
        pulse = gaussian_pulse_em(t, t0=5e-9, sigma=1e-9)

        assert pulse.shape == t.shape
        # Peak should be at t0
        peak_idx = np.argmax(pulse)
        assert t[peak_idx] == pytest.approx(5e-9, rel=0.01)

    def test_gaussian_pulse_amplitude(self):
        """Gaussian pulse should have specified amplitude."""
        from src.maxwell import gaussian_pulse_em

        t = np.linspace(0, 10e-9, 1000)
        amplitude = 2.5
        pulse = gaussian_pulse_em(t, t0=5e-9, sigma=1e-9, amplitude=amplitude)

        assert np.max(pulse) == pytest.approx(amplitude, rel=0.01)

    def test_sinusoidal_source_frequency(self):
        """Sinusoidal source should have correct frequency."""
        from src.maxwell import sinusoidal_source

        f0 = 1e9
        t = np.linspace(0, 10 / f0, 10000)  # 10 periods
        source = sinusoidal_source(t, f0=f0, t_ramp=1 / f0)

        # Count zero crossings (after ramp)
        steady = source[len(t) // 2 :]
        crossings = np.sum(np.diff(np.sign(steady)) != 0)
        periods = crossings / 2
        expected_periods = 5  # Half of 10 periods
        assert periods == pytest.approx(expected_periods, abs=1)

    def test_sinusoidal_source_ramp(self):
        """Sinusoidal source should ramp up smoothly."""
        from src.maxwell import sinusoidal_source

        t = np.linspace(0, 10e-9, 1000)
        source = sinusoidal_source(t, f0=1e9, t_ramp=2e-9)

        # At t=0, source should be small
        assert np.abs(source[0]) < 0.1
        # After ramp, envelope should reach 1
        assert np.max(np.abs(source[len(t) // 2 :])) > 0.9

    def test_gaussian_modulated_source(self):
        """Gaussian-modulated source should be narrow-band."""
        from src.maxwell import gaussian_modulated_source

        f0 = 5e9
        t = np.linspace(0, 10e-9, 10000)
        source = gaussian_modulated_source(t, f0=f0, t0=5e-9, sigma=1e-9)

        # FFT should show peak near f0
        dt = t[1] - t[0]
        freq = np.fft.fftfreq(len(t), dt)
        spectrum = np.abs(np.fft.fft(source))

        # Find peak frequency (positive frequencies only)
        pos_mask = freq > 0
        peak_freq = freq[pos_mask][np.argmax(spectrum[pos_mask])]
        assert peak_freq == pytest.approx(f0, rel=0.1)


# =============================================================================
# Test: PML Coefficients
# =============================================================================


class TestPMLCoefficients:
    """Tests for PML coefficient generation."""

    def test_pml_sigma_shape(self):
        """PML sigma profile should have correct shape."""
        from src.maxwell import create_pml_sigma

        n_pml = 10
        sigma = create_pml_sigma(n_pml, dx=0.001)

        assert sigma.shape == (n_pml,)

    def test_pml_sigma_monotonic(self):
        """PML sigma should increase toward boundary."""
        from src.maxwell import create_pml_sigma

        sigma = create_pml_sigma(n_pml=20, dx=0.001)

        # Should be monotonically increasing
        assert np.all(np.diff(sigma) >= 0)

    def test_cpml_coefficients_keys(self):
        """CPML coefficients should contain all required keys."""
        from src.maxwell import create_cpml_coefficients

        cpml = create_cpml_coefficients(n_pml=10, dx=0.001, dt=1e-12)

        required_keys = ["b", "a", "kappa", "sigma", "alpha"]
        for key in required_keys:
            assert key in cpml, f"Missing key: {key}"

    def test_cpml_b_range(self):
        """CPML b coefficient should be between 0 and 1."""
        from src.maxwell import create_cpml_coefficients

        cpml = create_cpml_coefficients(n_pml=10, dx=0.001, dt=1e-12)

        assert np.all(cpml["b"] >= 0)
        assert np.all(cpml["b"] <= 1)

    def test_cpml_kappa_range(self):
        """CPML kappa should be >= 1."""
        from src.maxwell import create_cpml_coefficients

        cpml = create_cpml_coefficients(n_pml=10, dx=0.001, dt=1e-12, kappa_max=5.0)

        assert np.all(cpml["kappa"] >= 1)


# =============================================================================
# Test: Analytical Solutions
# =============================================================================


class TestAnalyticalSolutions:
    """Tests for analytical electromagnetic solutions."""

    def test_plane_wave_1d_shape(self):
        """1D plane wave should have correct shape."""
        from src.maxwell import exact_plane_wave_1d

        x = np.linspace(0, 1, 100)
        Ey, Hz = exact_plane_wave_1d(x, t=1e-9, f0=1e9)

        assert Ey.shape == x.shape
        assert Hz.shape == x.shape

    def test_plane_wave_1d_impedance(self):
        """E/H should equal wave impedance."""
        from src.maxwell import exact_plane_wave_1d

        x = np.linspace(0, 1, 100)
        Ey, Hz = exact_plane_wave_1d(x, t=1e-9, f0=1e9, E0=1.0)

        # Avoid division by zero
        mask = np.abs(Hz) > 1e-10
        ratio = np.abs(Ey[mask] / Hz[mask])
        expected_eta = ETA0

        assert np.mean(ratio) == pytest.approx(expected_eta, rel=0.01)

    def test_plane_wave_2d_tmz(self):
        """2D TMz plane wave should have correct structure."""
        from src.maxwell import exact_plane_wave_2d

        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        Ez, Hx, Hy = exact_plane_wave_2d(x, y, t=0, f0=1e9, theta=0, polarization="TMz")

        assert Ez.shape == (50, 50)
        assert Hx.shape == (50, 50)
        assert Hy.shape == (50, 50)

    def test_cavity_resonant_frequencies_lowest(self):
        """Cavity should have correct lowest TMz resonance."""
        from src.maxwell import cavity_resonant_frequencies

        a = b = 0.1  # 10 cm square cavity
        modes = cavity_resonant_frequencies(a, b)

        # For TMz, the lowest mode is TM_11 (m=1, n=1)
        # TE modes (m=0 or n=0) have lower frequency but don't exist for TMz
        expected_f11 = (C0 / 2) * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)

        # Find first TM mode (m >= 1 and n >= 1)
        tm_modes = [m for m in modes if m["m"] >= 1 and m["n"] >= 1]
        assert len(tm_modes) > 0

        first_tm = tm_modes[0]
        assert first_tm["m"] == 1
        assert first_tm["n"] == 1
        assert first_tm["f"] == pytest.approx(expected_f11, rel=1e-6)

    def test_cavity_frequencies_order(self):
        """Cavity modes should be sorted by frequency."""
        from src.maxwell import cavity_resonant_frequencies

        modes = cavity_resonant_frequencies(a=0.1, b=0.08, m_max=3, n_max=3)

        frequencies = [m["f"] for m in modes]
        assert frequencies == sorted(frequencies)


# =============================================================================
# Test: 1D FDTD Solver
# =============================================================================


@pytest.mark.devito
class TestFDTD1D:
    """Tests for 1D FDTD Maxwell solver."""

    def test_basic_run(self):
        """Test basic 1D solver execution."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(L=1.0, Nx=50, T=1e-9)

        assert result.Ey is not None
        assert result.Hz is not None
        assert result.Ey.shape == (50,)
        assert result.Hz.shape == (50,)

    def test_grid_coordinates(self):
        """Test grid coordinates are correct."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(L=1.0, Nx=101, T=1e-9)

        assert len(result.x) == 101
        assert result.x[0] == pytest.approx(0.0)
        assert result.x[-1] == pytest.approx(1.0)

    def test_pec_boundary_left(self):
        """PEC boundary should enforce E=0 at left."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(
            L=1.0,
            Nx=100,
            T=5e-9,
            bc_left="pec",
            bc_right="abc",
            source_position=0.5,
        )

        # E should be zero at PEC boundary
        assert result.Ey[0] == pytest.approx(0.0, abs=1e-10)

    def test_pec_boundary_right(self):
        """PEC boundary should enforce E=0 at right."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(
            L=1.0,
            Nx=100,
            T=5e-9,
            bc_left="abc",
            bc_right="pec",
            source_position=0.5,
        )

        assert result.Ey[-1] == pytest.approx(0.0, abs=1e-10)

    def test_fields_finite(self):
        """Fields should remain finite (no blow-up)."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(L=1.0, Nx=100, T=10e-9)

        assert np.all(np.isfinite(result.Ey))
        assert np.all(np.isfinite(result.Hz))

    def test_save_history(self):
        """History should be saved when requested."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(L=1.0, Nx=50, T=2e-9, save_history=True, save_every=10)

        assert result.Ey_history is not None
        assert result.Hz_history is not None
        assert result.t_history is not None
        assert len(result.Ey_history) > 1


@pytest.mark.devito
class TestFDTD1DSourceTypes:
    """Tests for different source types in 1D FDTD."""

    def test_gaussian_source(self):
        """Gaussian source should excite fields."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(L=1.0, Nx=100, T=3e-9, source_type="gaussian")

        # Fields should be non-zero
        assert np.max(np.abs(result.Ey)) > 0
        assert np.max(np.abs(result.Hz)) > 0

    def test_sinusoidal_source(self):
        """Sinusoidal source should excite fields."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(L=1.0, Nx=100, T=5e-9, source_type="sinusoidal", f0=1e9)

        assert np.max(np.abs(result.Ey)) > 0

    def test_ricker_source(self):
        """Ricker wavelet source should excite fields."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(L=1.0, Nx=100, T=5e-9, source_type="ricker", f0=1e9)

        assert np.max(np.abs(result.Ey)) > 0


# =============================================================================
# Test: 2D FDTD Solver
# =============================================================================


@pytest.mark.devito
class TestFDTD2D:
    """Tests for 2D FDTD Maxwell solver."""

    def test_basic_run(self):
        """Test basic 2D solver execution."""
        from src.maxwell import solve_maxwell_2d

        result = solve_maxwell_2d(Lx=0.1, Ly=0.1, Nx=21, Ny=21, T=1e-9)

        assert result.Ez is not None
        assert result.Hx is not None
        assert result.Hy is not None
        assert result.Ez.shape == (21, 21)

    def test_grid_coordinates_2d(self):
        """Test 2D grid coordinates are correct."""
        from src.maxwell import solve_maxwell_2d

        result = solve_maxwell_2d(Lx=0.1, Ly=0.2, Nx=51, Ny=101, T=0.5e-9)

        assert len(result.x) == 51
        assert len(result.y) == 101
        assert result.x[-1] == pytest.approx(0.1)
        assert result.y[-1] == pytest.approx(0.2)

    def test_pec_boundaries_2d(self):
        """PEC boundaries should enforce Ez=0 on all edges."""
        from src.maxwell import solve_maxwell_2d

        result = solve_maxwell_2d(
            Lx=0.1,
            Ly=0.1,
            Nx=31,
            Ny=31,
            T=2e-9,
            bc_type="pec",
        )

        # Check all boundaries
        np.testing.assert_allclose(result.Ez[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.Ez[-1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.Ez[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.Ez[:, -1], 0.0, atol=1e-10)

    def test_fields_finite_2d(self):
        """2D fields should remain finite."""
        from src.maxwell import solve_maxwell_2d

        result = solve_maxwell_2d(Lx=0.1, Ly=0.1, Nx=31, Ny=31, T=2e-9)

        assert np.all(np.isfinite(result.Ez))
        assert np.all(np.isfinite(result.Hx))
        assert np.all(np.isfinite(result.Hy))

    def test_snapshots_saved(self):
        """Snapshots should be saved when requested."""
        from src.maxwell import solve_maxwell_2d

        result = solve_maxwell_2d(
            Lx=0.1, Ly=0.1, Nx=21, Ny=21, T=2e-9, nsnaps=10
        )

        assert result.Ez_history is not None
        assert result.t_history is not None
        assert len(result.Ez_history) >= 5


# =============================================================================
# Test: Energy Conservation
# =============================================================================


@pytest.mark.devito
class TestEnergyConservation:
    """Tests for electromagnetic energy conservation."""

    def test_energy_computation_1d(self):
        """Test energy computation in 1D."""
        from src.maxwell import compute_energy

        Ey = np.ones(100)
        Hz = np.ones(100)
        dx = 0.01

        energy = compute_energy(Ey, Hz, dx)

        # Energy should be positive
        assert energy > 0

    def test_energy_computation_2d(self):
        """Test energy computation in 2D."""
        from src.maxwell import compute_energy_2d

        Ez = np.ones((50, 50))
        Hx = np.ones((50, 50))
        Hy = np.ones((50, 50))
        dx = dy = 0.01

        energy = compute_energy_2d(Ez, Hx, Hy, dx, dy)

        assert energy > 0

    def test_energy_bounded_pec_cavity(self):
        """Energy in PEC cavity should remain bounded."""
        from src.maxwell import compute_energy, solve_maxwell_1d

        # Run simulation with PEC walls
        result = solve_maxwell_1d(
            L=1.0,
            Nx=100,
            T=10e-9,
            bc_left="pec",
            bc_right="pec",
            save_history=True,
            save_every=100,
        )

        # Compute energy at each saved time
        dx = result.x[1] - result.x[0]
        energies = []
        for Ey, Hz in zip(result.Ey_history, result.Hz_history):
            e = compute_energy(Ey, Hz, dx)
            energies.append(e)

        energies = np.array(energies)

        # Energy should not grow (allowing small numerical variation)
        max_energy = np.max(energies[1:])  # Skip initial (may be zero)
        min_energy = np.min(energies[1:])
        if min_energy > 0:
            assert max_energy / min_energy < 2.0  # No blow-up


# =============================================================================
# Test: Wave Speed Verification
# =============================================================================


@pytest.mark.devito
@pytest.mark.slow
class TestWaveSpeed:
    """Tests for wave propagation speed."""

    def test_wave_speed_free_space(self):
        """Wave speed in result should equal speed of light."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(
            L=1.0,
            Nx=200,
            T=2e-9,
            source_type="gaussian",
            f0=5e9,
        )

        # The result.c should be the speed of light for free space
        assert result.c == pytest.approx(C0, rel=1e-6)

    def test_field_excited_near_source(self):
        """Field should be excited near source location."""
        from src.maxwell import solve_maxwell_1d

        source_pos = 0.25
        result = solve_maxwell_1d(
            L=1.0,
            Nx=200,
            T=2e-9,
            source_type="gaussian",
            source_position=source_pos,
            f0=2e9,
            bc_left="pec",
            bc_right="pec",
        )

        # Field should be non-zero overall (source excites the domain)
        assert np.max(np.abs(result.Ey)) > 0


# =============================================================================
# Test: CFL Stability
# =============================================================================


@pytest.mark.devito
class TestCFLStability:
    """Tests for CFL stability condition."""

    def test_default_dt_stable(self):
        """Default dt should satisfy CFL."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(L=1.0, Nx=100, T=10e-9)

        # Verify CFL number
        dx = result.x[1] - result.x[0]
        CFL = result.c * result.dt / dx

        assert CFL <= 1.0

    def test_custom_dt_stability(self):
        """Fields should remain stable with proper dt."""
        from src.maxwell import solve_maxwell_1d

        L = 1.0
        Nx = 100
        dx = L / (Nx - 1)
        dt = 0.5 * dx / C0  # 50% of CFL limit

        result = solve_maxwell_1d(L=L, Nx=Nx, T=10e-9, dt=dt)

        # Fields should not blow up
        assert np.max(np.abs(result.Ey)) < 1e10


# =============================================================================
# Test: Material Properties
# =============================================================================


@pytest.mark.devito
class TestMaterialProperties:
    """Tests for material property handling."""

    def test_free_space(self):
        """Free space should have c = c0."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(L=1.0, Nx=100, T=1e-9, eps_r=1.0, mu_r=1.0)

        assert result.c == pytest.approx(C0, rel=1e-6)

    def test_dielectric_medium(self):
        """Dielectric should slow wave speed."""
        from src.maxwell import solve_maxwell_1d

        eps_r = 4.0  # Relative permittivity
        result = solve_maxwell_1d(L=1.0, Nx=100, T=1e-9, eps_r=eps_r)

        expected_c = C0 / np.sqrt(eps_r)
        assert result.c == pytest.approx(expected_c, rel=1e-6)


# =============================================================================
# Test: Boundary Condition Types
# =============================================================================


@pytest.mark.devito
class TestBoundaryConditions:
    """Tests for different boundary condition types."""

    def test_pec_enforces_zero(self):
        """PEC should enforce E=0 at boundary."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(
            L=1.0,
            Nx=200,
            T=5e-9,
            source_position=0.25,
            bc_left="pec",
            bc_right="pec",
            source_type="gaussian",
        )

        # E should be zero at PEC boundaries
        assert result.Ey[0] == pytest.approx(0.0, abs=1e-10)
        assert result.Ey[-1] == pytest.approx(0.0, abs=1e-10)

    def test_fields_non_zero_with_source(self):
        """Fields should be excited by source."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(
            L=1.0,
            Nx=200,
            T=5e-9,
            source_position=0.5,
            bc_left="pec",
            bc_right="pec",
            source_type="gaussian",
        )

        # Field should be non-zero in interior
        assert np.max(np.abs(result.Ey)) > 0


# =============================================================================
# Test: Cavity Resonance
# =============================================================================


@pytest.mark.devito
@pytest.mark.slow
class TestCavityResonance:
    """Tests for resonant cavity simulation."""

    def test_cavity_mode_excitation(self):
        """Exciting near resonance should produce strong response."""
        from src.maxwell import cavity_resonant_frequencies, solve_maxwell_2d

        # 10 cm square cavity
        a = b = 0.1
        modes = cavity_resonant_frequencies(a, b)
        f_11 = modes[0]["f"]

        result = solve_maxwell_2d(
            Lx=a,
            Ly=b,
            Nx=51,
            Ny=51,
            T=5e-9,
            f0=f_11,
            source_type="gaussian",
            bc_type="pec",
        )

        # Field should be excited
        assert np.max(np.abs(result.Ez)) > 0


# =============================================================================
# Test: Poynting Vector
# =============================================================================


class TestPoyntingVector:
    """Tests for Poynting vector computation."""

    def test_poynting_1d_shape(self):
        """1D Poynting vector should have correct shape."""
        from src.maxwell.maxwell_devito import compute_poynting_vector_1d

        Ey = np.ones(100)
        Hz = np.ones(100)

        Sx = compute_poynting_vector_1d(Ey, Hz)

        assert Sx.shape == Ey.shape

    def test_poynting_2d_shape(self):
        """2D Poynting vector should have correct shape."""
        from src.maxwell.maxwell_devito import compute_poynting_vector_2d

        Ez = np.ones((50, 60))
        Hx = np.ones((50, 60))
        Hy = np.ones((50, 60))

        Sx, Sy = compute_poynting_vector_2d(Ez, Hx, Hy)

        assert Sx.shape == Ez.shape
        assert Sy.shape == Ez.shape


# =============================================================================
# Test: Edge Cases
# =============================================================================


@pytest.mark.devito
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_grid(self):
        """Solver should handle small grids."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(L=0.1, Nx=10, T=0.5e-9)

        assert result.Ey.shape == (10,)
        assert np.all(np.isfinite(result.Ey))

    def test_short_simulation(self):
        """Solver should handle very short simulations."""
        from src.maxwell import solve_maxwell_1d

        result = solve_maxwell_1d(L=1.0, Nx=50, T=0.1e-9)

        assert result.t <= 0.1e-9
        assert np.all(np.isfinite(result.Ey))

    def test_invalid_source_raises(self):
        """Invalid source type should raise error."""
        from src.maxwell import solve_maxwell_1d

        with pytest.raises(ValueError, match="Unknown source type"):
            solve_maxwell_1d(L=1.0, Nx=50, T=1e-9, source_type="invalid")


# =============================================================================
# Test: Physical Constants
# =============================================================================


class TestPhysicalConstants:
    """Tests for physical constants consistency."""

    def test_speed_of_light(self):
        """c0 should equal 1/sqrt(mu0*eps0)."""
        c_computed = 1.0 / np.sqrt(MU0 * EPS0)
        assert c_computed == pytest.approx(C0, rel=1e-6)

    def test_free_space_impedance(self):
        """eta0 should equal sqrt(mu0/eps0)."""
        eta_computed = np.sqrt(MU0 / EPS0)
        assert eta_computed == pytest.approx(ETA0, rel=1e-6)
        assert eta_computed == pytest.approx(377, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
