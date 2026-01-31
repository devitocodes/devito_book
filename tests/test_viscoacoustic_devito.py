"""Tests for the Viscoacoustic Wave Equations solver using Devito."""

import importlib.util

import numpy as np
import pytest

# Check if Devito is available
DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None

pytestmark = pytest.mark.skipif(
    not DEVITO_AVAILABLE, reason="Devito not installed"
)


class TestViscoacousticImport:
    """Test that the module imports correctly."""

    def test_import_solve_viscoacoustic_sls(self):
        """Test SLS solver import."""
        from src.systems import solve_viscoacoustic_sls

        assert solve_viscoacoustic_sls is not None

    def test_import_solve_viscoacoustic_kv(self):
        """Test Kelvin-Voigt solver import."""
        from src.systems import solve_viscoacoustic_kv

        assert solve_viscoacoustic_kv is not None

    def test_import_solve_viscoacoustic_maxwell(self):
        """Test Maxwell solver import."""
        from src.systems import solve_viscoacoustic_maxwell

        assert solve_viscoacoustic_maxwell is not None

    def test_import_result_class(self):
        """Test result dataclass import."""
        from src.systems import ViscoacousticResult

        assert ViscoacousticResult is not None

    def test_import_helper_functions(self):
        """Test helper function imports."""
        from src.systems import (
            compute_sls_relaxation_parameters,
            create_damping_field,
        )

        assert compute_sls_relaxation_parameters is not None
        assert create_damping_field is not None


class TestRelaxationParameters:
    """Test relaxation parameter computation."""

    def test_compute_sls_parameters(self):
        """Test SLS relaxation parameter computation."""
        from src.systems import compute_sls_relaxation_parameters

        Q = 50.0
        f0 = 0.01

        t_s, t_ep, tau = compute_sls_relaxation_parameters(Q, f0)

        # All parameters should be positive
        assert t_s > 0
        assert t_ep > 0
        assert tau > 0

    def test_sls_parameters_high_q(self):
        """Test that high Q gives small tau."""
        from src.systems import compute_sls_relaxation_parameters

        f0 = 0.01

        _, _, tau_low_q = compute_sls_relaxation_parameters(20.0, f0)
        _, _, tau_high_q = compute_sls_relaxation_parameters(500.0, f0)

        # Higher Q means less attenuation, smaller tau
        assert tau_high_q < tau_low_q

    def test_sls_parameters_array(self):
        """Test that array Q values work."""
        from src.systems import compute_sls_relaxation_parameters

        Q = np.array([[50.0, 100.0], [200.0, 300.0]])
        f0 = 0.01

        t_s, t_ep, tau = compute_sls_relaxation_parameters(Q, f0)

        assert t_s.shape == Q.shape
        assert t_ep.shape == Q.shape
        assert tau.shape == Q.shape


class TestViscoacousticSLS:
    """Test the SLS (Standard Linear Solid) viscoacoustic solver."""

    def test_sls_basic_run(self):
        """Test that SLS solver runs without errors."""
        from src.systems import solve_viscoacoustic_sls

        result = solve_viscoacoustic_sls(
            Lx=1000.0, Lz=1000.0,
            Nx=51, Nz=51,
            T=100.0,
            vp=2.0,
            rho=1.0,
            Q=50.0,
            f0=0.01,
            space_order=4,
        )

        assert result.p is not None
        assert result.vx is not None
        assert result.vz is not None

    def test_sls_result_shapes(self):
        """Test that SLS results have correct shapes."""
        from src.systems import solve_viscoacoustic_sls

        Nx, Nz = 41, 51

        result = solve_viscoacoustic_sls(
            Lx=800.0, Lz=1000.0,
            Nx=Nx, Nz=Nz,
            T=50.0,
            space_order=4,
        )

        assert result.p.shape == (Nx, Nz)
        assert result.vx.shape == (Nx, Nz)
        assert result.vz.shape == (Nx, Nz)
        assert len(result.x) == Nx
        assert len(result.z) == Nz

    def test_sls_wavefield_finite(self):
        """Test that SLS wavefield values are finite."""
        from src.systems import solve_viscoacoustic_sls

        result = solve_viscoacoustic_sls(
            Lx=1000.0, Lz=1000.0,
            Nx=51, Nz=51,
            T=100.0,
            space_order=4,
        )

        assert np.all(np.isfinite(result.p))
        assert np.all(np.isfinite(result.vx))
        assert np.all(np.isfinite(result.vz))

    def test_sls_no_nan(self):
        """Test that SLS solution contains no NaN."""
        from src.systems import solve_viscoacoustic_sls

        result = solve_viscoacoustic_sls(
            Lx=1000.0, Lz=1000.0,
            Nx=51, Nz=51,
            T=100.0,
            space_order=4,
        )

        assert not np.any(np.isnan(result.p))
        assert not np.any(np.isnan(result.vx))
        assert not np.any(np.isnan(result.vz))


class TestViscoacousticKelvinVoigt:
    """Test the Kelvin-Voigt viscoacoustic solver."""

    def test_kv_basic_run(self):
        """Test that Kelvin-Voigt solver runs without errors."""
        from src.systems import solve_viscoacoustic_kv

        result = solve_viscoacoustic_kv(
            Lx=1000.0, Lz=1000.0,
            Nx=51, Nz=51,
            T=100.0,
            vp=2.0,
            Q=50.0,
            space_order=4,
        )

        assert result.p is not None
        assert result.vx is not None
        assert result.vz is not None

    def test_kv_wavefield_finite(self):
        """Test that Kelvin-Voigt wavefield values are finite."""
        from src.systems import solve_viscoacoustic_kv

        result = solve_viscoacoustic_kv(
            Lx=1000.0, Lz=1000.0,
            Nx=51, Nz=51,
            T=100.0,
            space_order=4,
        )

        assert np.all(np.isfinite(result.p))
        assert np.all(np.isfinite(result.vx))
        assert np.all(np.isfinite(result.vz))


class TestViscoacousticMaxwell:
    """Test the Maxwell viscoacoustic solver."""

    def test_maxwell_basic_run(self):
        """Test that Maxwell solver runs without errors."""
        from src.systems import solve_viscoacoustic_maxwell

        result = solve_viscoacoustic_maxwell(
            Lx=1000.0, Lz=1000.0,
            Nx=51, Nz=51,
            T=100.0,
            vp=2.0,
            Q=50.0,
            space_order=4,
        )

        assert result.p is not None
        assert result.vx is not None
        assert result.vz is not None

    def test_maxwell_wavefield_finite(self):
        """Test that Maxwell wavefield values are finite."""
        from src.systems import solve_viscoacoustic_maxwell

        result = solve_viscoacoustic_maxwell(
            Lx=1000.0, Lz=1000.0,
            Nx=51, Nz=51,
            T=100.0,
            space_order=4,
        )

        assert np.all(np.isfinite(result.p))
        assert np.all(np.isfinite(result.vx))
        assert np.all(np.isfinite(result.vz))

    def test_maxwell_no_nan(self):
        """Test that Maxwell solution contains no NaN."""
        from src.systems import solve_viscoacoustic_maxwell

        result = solve_viscoacoustic_maxwell(
            Lx=1000.0, Lz=1000.0,
            Nx=51, Nz=51,
            T=100.0,
            space_order=4,
        )

        assert not np.any(np.isnan(result.p))


class TestAttenuationBehavior:
    """Test physical attenuation behavior."""

    def test_low_q_causes_attenuation(self):
        """Test that low Q causes wave amplitude decay.

        Note: This test uses SLS model which has more robust Q implementation.
        The attenuation effect depends on simulation time, frequency, and Q value.
        """
        from src.systems import solve_viscoacoustic_sls

        # Use larger grid and longer simulation to see attenuation effects
        # Run with very high Q (essentially no attenuation)
        result_high_q = solve_viscoacoustic_sls(
            Lx=4000.0, Lz=4000.0,
            Nx=101, Nz=101,
            T=1000.0,
            Q=1000.0,  # High Q = low attenuation
            f0=0.01,   # Higher reference frequency
            space_order=4,
            use_damp=False,  # No boundary damping to isolate Q effect
        )

        # Run with low Q (high attenuation)
        result_low_q = solve_viscoacoustic_sls(
            Lx=4000.0, Lz=4000.0,
            Nx=101, Nz=101,
            T=1000.0,
            Q=5.0,  # Very low Q = high attenuation
            f0=0.01,
            space_order=4,
            use_damp=False,
        )

        # Compare total energy (L2 norm) - more robust than max amplitude
        energy_high_q = np.linalg.norm(result_high_q.p)
        energy_low_q = np.linalg.norm(result_low_q.p)

        # Low Q should have attenuated the wave - allow 1% tolerance for numerical effects
        assert energy_low_q <= energy_high_q * 1.01, \
            f"Low Q energy ({energy_low_q}) should not exceed high Q energy ({energy_high_q})"


class TestViscoacousticResult:
    """Test the ViscoacousticResult dataclass."""

    def test_result_attributes(self):
        """Test that result has all expected attributes."""
        from src.systems import solve_viscoacoustic_sls

        result = solve_viscoacoustic_sls(
            Lx=500.0, Lz=500.0,
            Nx=31, Nz=31,
            T=50.0,
            space_order=4,
        )

        assert hasattr(result, 'p')
        assert hasattr(result, 'vx')
        assert hasattr(result, 'vz')
        assert hasattr(result, 'x')
        assert hasattr(result, 'z')
        assert hasattr(result, 't')
        assert hasattr(result, 'dt')

    def test_time_attributes(self):
        """Test time-related attributes."""
        from src.systems import solve_viscoacoustic_sls

        T = 100.0
        result = solve_viscoacoustic_sls(
            Lx=500.0, Lz=500.0,
            Nx=31, Nz=31,
            T=T,
            space_order=4,
        )

        assert result.t == T
        assert result.dt > 0
        assert result.dt < T


class TestSourceInjection:
    """Test source injection produces waves."""

    def test_source_generates_wavefield(self):
        """Test that source injection generates non-zero wavefield."""
        from src.systems import solve_viscoacoustic_sls

        result = solve_viscoacoustic_sls(
            Lx=1000.0, Lz=1000.0,
            Nx=51, Nz=51,
            T=200.0,
            src_coords=(500.0, 500.0),
            space_order=4,
        )

        # After simulation, pressure field should be non-zero
        max_p = np.max(np.abs(result.p))
        assert max_p > 0, "Pressure field is zero - source injection may have failed"


class TestCoordinateArrays:
    """Test coordinate arrays."""

    def test_coordinate_range(self):
        """Test that coordinate arrays cover the domain."""
        from src.systems import solve_viscoacoustic_sls

        Lx, Lz = 1500.0, 1000.0
        Nx, Nz = 31, 21

        result = solve_viscoacoustic_sls(
            Lx=Lx, Lz=Lz,
            Nx=Nx, Nz=Nz,
            T=10.0,
            space_order=4,
        )

        assert result.x[0] == pytest.approx(0.0)
        assert result.x[-1] == pytest.approx(Lx)
        assert result.z[0] == pytest.approx(0.0)
        assert result.z[-1] == pytest.approx(Lz)


class TestVaryingParameters:
    """Test with spatially varying parameters."""

    def test_varying_velocity(self):
        """Test with spatially varying velocity."""
        from src.systems import solve_viscoacoustic_sls

        Nx, Nz = 51, 51

        # Create linearly varying velocity
        x = np.linspace(0, 1, Nx)
        z = np.linspace(0, 1, Nz)
        X, Z = np.meshgrid(x, z, indexing='ij')
        vp = 1.5 + 1.0 * Z  # Velocity increases with depth

        result = solve_viscoacoustic_sls(
            Lx=1000.0, Lz=1000.0,
            Nx=Nx, Nz=Nz,
            T=100.0,
            vp=vp.astype(np.float32),
            space_order=4,
        )

        assert np.all(np.isfinite(result.p))

    def test_varying_q(self):
        """Test with spatially varying Q."""
        from src.systems import solve_viscoacoustic_sls

        Nx, Nz = 51, 51

        # Create varying Q (higher Q at depth)
        x = np.linspace(0, 1, Nx)
        z = np.linspace(0, 1, Nz)
        X, Z = np.meshgrid(x, z, indexing='ij')
        Q = 30.0 + 70.0 * Z

        result = solve_viscoacoustic_sls(
            Lx=1000.0, Lz=1000.0,
            Nx=Nx, Nz=Nz,
            T=100.0,
            Q=Q.astype(np.float32),
            space_order=4,
        )

        assert np.all(np.isfinite(result.p))
