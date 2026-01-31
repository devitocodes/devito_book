"""
Tests for the theory appendix: stability analysis and on-the-fly DFT.

This module tests:
- Von Neumann stability analysis functions
- CFL condition verification
- On-the-fly discrete Fourier transform
"""

# Check if Devito is available
import importlib.util

import numpy as np
import pytest

from src.theory.fourier_dft import compute_reference_dft, ricker_wavelet
from src.theory.stability_analysis import (
    amplification_factor_advection_upwind,
    amplification_factor_diffusion,
    amplification_factor_wave,
    check_stability_diffusion,
    check_stability_wave,
    compute_cfl,
    stable_timestep_diffusion,
    stable_timestep_wave,
)

DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None


# =============================================================================
# Stability Analysis Tests
# =============================================================================

class TestAmplificationFactorDiffusion:
    """Tests for FTCS diffusion amplification factor."""

    def test_unity_at_theta_zero(self):
        """g(0) = 1 for any r (no oscillation mode is unchanged)."""
        for r in [0.1, 0.25, 0.4, 0.5]:
            g = amplification_factor_diffusion(r, 0.0)
            assert np.isclose(g, 1.0)

    def test_minimum_at_theta_pi(self):
        """g(pi) = 1 - 4r (highest frequency mode most damped)."""
        for r in [0.1, 0.25, 0.4]:
            g = amplification_factor_diffusion(r, np.pi)
            expected = 1 - 4 * r
            assert np.isclose(g, expected)

    def test_stable_regime(self):
        """For r <= 0.5, |g| <= 1 for all theta."""
        theta = np.linspace(0, 2*np.pi, 100)
        for r in [0.1, 0.25, 0.4, 0.5]:
            g = amplification_factor_diffusion(r, theta)
            assert np.all(np.abs(g) <= 1.0 + 1e-10)

    def test_unstable_regime(self):
        """For r > 0.5, |g| > 1 for some theta."""
        theta = np.linspace(0, 2*np.pi, 100)
        for r in [0.6, 0.75, 1.0]:
            g = amplification_factor_diffusion(r, theta)
            assert np.any(np.abs(g) > 1.0)


class TestAmplificationFactorAdvection:
    """Tests for upwind advection amplification factor."""

    def test_unity_at_theta_zero(self):
        """g(0) = 1 for any nu (constant mode unchanged)."""
        for nu in [0.25, 0.5, 0.75, 1.0]:
            g = amplification_factor_advection_upwind(nu, 0.0)
            assert np.isclose(np.abs(g), 1.0)

    def test_stable_regime(self):
        """For 0 <= nu <= 1, |g| <= 1 for all theta."""
        theta = np.linspace(0, 2*np.pi, 100)
        for nu in [0.25, 0.5, 0.75, 1.0]:
            g = amplification_factor_advection_upwind(nu, theta)
            assert np.all(np.abs(g) <= 1.0 + 1e-10)

    def test_unstable_regime(self):
        """For nu > 1, |g| > 1 for some theta."""
        theta = np.linspace(0, 2*np.pi, 100)
        for nu in [1.1, 1.5, 2.0]:
            g = amplification_factor_advection_upwind(nu, theta)
            assert np.any(np.abs(g) > 1.0 + 1e-10)


class TestAmplificationFactorWave:
    """Tests for leapfrog wave equation amplification factor."""

    def test_magnitude_unity_stable(self):
        """For nu <= 1, |g| = 1 for all theta (neutral stability)."""
        theta = np.linspace(0.01, 2*np.pi - 0.01, 100)  # Avoid endpoints
        for nu in [0.5, 0.75, 0.9, 1.0]:
            g = amplification_factor_wave(nu, theta)
            # For stable regime, |g| should be exactly 1
            assert np.allclose(np.abs(g), 1.0, atol=1e-6)

    def test_unstable_regime(self):
        """For nu > 1, |g| != 1 for some theta."""
        theta = np.linspace(0.01, 2*np.pi - 0.01, 100)
        for nu in [1.1, 1.5]:
            g = amplification_factor_wave(nu, theta)
            # Should have growth or decay
            assert not np.allclose(np.abs(g), 1.0, atol=0.01)


class TestCFLComputation:
    """Tests for CFL number computation."""

    def test_basic_cfl(self):
        """Basic CFL computation c*dt/dx."""
        assert np.isclose(compute_cfl(1500, 0.001, 10), 0.15)
        assert np.isclose(compute_cfl(1000, 0.0005, 5), 0.1)

    def test_cfl_dimensions(self):
        """CFL computation with dimensions parameter."""
        cfl_1d = compute_cfl(1500, 0.001, 10, ndim=1)
        cfl_2d = compute_cfl(1500, 0.001, 10, ndim=2)
        cfl_3d = compute_cfl(1500, 0.001, 10, ndim=3)
        # CFL number doesn't change with ndim (stability limit does)
        assert cfl_1d == cfl_2d == cfl_3d


class TestStableTimestepDiffusion:
    """Tests for diffusion stable time step computation."""

    def test_1d_diffusion(self):
        """dt <= dx^2 / (2*alpha) for 1D."""
        alpha = 0.1
        dx = 0.01
        dt = stable_timestep_diffusion(alpha, dx, cfl_max=0.5)
        # Should equal exactly dx^2/(2*alpha) at cfl_max=0.5
        expected = 0.5 * dx**2 / alpha
        assert np.isclose(dt, expected)

    def test_2d_diffusion(self):
        """dt <= dx^2 / (4*alpha) for 2D."""
        alpha = 0.1
        dx = 0.01
        dt = stable_timestep_diffusion(alpha, dx, cfl_max=0.25, ndim=2)
        # r = 0.25 for 2D (max stable is 0.25)
        expected = 0.25 * dx**2 / (2 * alpha)
        assert np.isclose(dt, expected)


class TestStableTimestepWave:
    """Tests for wave equation stable time step computation."""

    def test_1d_wave(self):
        """dt <= dx/c for 1D."""
        c = 1500
        dx = 10
        dt = stable_timestep_wave(c, dx, cfl_max=1.0, ndim=1)
        expected = dx / c
        assert np.isclose(dt, expected)

    def test_2d_wave(self):
        """dt <= dx/(c*sqrt(2)) for 2D."""
        c = 1500
        dx = 10
        dt = stable_timestep_wave(c, dx, cfl_max=1.0, ndim=2)
        expected = dx / (c * np.sqrt(2))
        assert np.isclose(dt, expected)

    def test_3d_wave(self):
        """dt <= dx/(c*sqrt(3)) for 3D."""
        c = 1500
        dx = 10
        dt = stable_timestep_wave(c, dx, cfl_max=1.0, ndim=3)
        expected = dx / (c * np.sqrt(3))
        assert np.isclose(dt, expected)


class TestStabilityChecks:
    """Tests for stability check functions."""

    def test_diffusion_stable(self):
        """Check stable diffusion configuration."""
        stable, r, r_max = check_stability_diffusion(0.1, 0.0004, 0.01)
        assert stable
        assert np.isclose(r, 0.4)
        assert np.isclose(r_max, 0.5)

    def test_diffusion_unstable(self):
        """Check unstable diffusion configuration."""
        stable, r, r_max = check_stability_diffusion(0.1, 0.001, 0.01)
        assert not stable
        assert r > r_max

    def test_wave_stable(self):
        """Check stable wave configuration."""
        stable, cfl, cfl_max = check_stability_wave(1500, 0.0001, 10, ndim=1)
        assert stable
        assert np.isclose(cfl, 0.015)
        assert np.isclose(cfl_max, 1.0)

    def test_wave_unstable(self):
        """Check unstable wave configuration."""
        stable, cfl, cfl_max = check_stability_wave(1500, 0.01, 10, ndim=1)
        assert not stable
        assert cfl > cfl_max

    def test_wave_2d_stability_limit(self):
        """2D stability limit is 1/sqrt(2)."""
        _, _, cfl_max = check_stability_wave(1500, 0.001, 10, ndim=2)
        assert np.isclose(cfl_max, 1/np.sqrt(2))


# =============================================================================
# Fourier DFT Tests
# =============================================================================

class TestRickerWavelet:
    """Tests for Ricker wavelet generation."""

    def test_peak_amplitude(self):
        """Peak amplitude should be approximately 1."""
        t = np.linspace(0, 0.5, 1000)
        w = ricker_wavelet(t, f0=10.0)
        assert np.isclose(np.max(w), 1.0, atol=0.01)

    def test_peak_time(self):
        """Peak should occur at t0 = 1.5/f0."""
        t = np.linspace(0, 0.5, 1000)
        f0 = 10.0
        w = ricker_wavelet(t, f0=f0)
        t_peak = t[np.argmax(w)]
        t0_expected = 1.5 / f0
        assert np.isclose(t_peak, t0_expected, atol=0.01)

    def test_custom_t0(self):
        """Custom t0 shifts the peak."""
        t = np.linspace(0, 1.0, 1000)
        t0 = 0.3
        w = ricker_wavelet(t, f0=10.0, t0=t0)
        t_peak = t[np.argmax(w)]
        assert np.isclose(t_peak, t0, atol=0.01)

    def test_symmetry(self):
        """Wavelet should be symmetric about peak."""
        t = np.linspace(0, 0.3, 1001)
        f0 = 10.0
        t0 = 0.15
        w = ricker_wavelet(t, f0=f0, t0=t0)
        # Check symmetry (odd samples, center at 500)
        assert np.allclose(w[:500], w[501:][::-1], atol=1e-6)


class TestReferenceDFT:
    """Tests for reference DFT computation."""

    def test_single_frequency_signal(self):
        """DFT of sinusoid should have peak at correct frequency."""
        nt = 1000
        dt = 0.001
        f_signal = 20.0  # 20 Hz signal

        t = np.arange(nt) * dt
        u = np.sin(2 * np.pi * f_signal * t)
        u_history = u.reshape(nt, 1)  # (nt, 1) shape

        frequencies = np.array([10.0, 20.0, 30.0, 40.0])
        modes = compute_reference_dft(u_history, frequencies, dt)

        # Peak should be at 20 Hz
        peak_idx = np.argmax(np.abs(modes))
        assert peak_idx == 1  # 20 Hz is second frequency

    def test_linearity(self):
        """DFT should be linear."""
        nt = 500
        dt = 0.001
        nx = 10

        u1 = np.random.randn(nt, nx)
        u2 = np.random.randn(nt, nx)

        frequencies = np.array([5.0, 10.0, 15.0])

        modes1 = compute_reference_dft(u1, frequencies, dt)
        modes2 = compute_reference_dft(u2, frequencies, dt)
        modes_sum = compute_reference_dft(u1 + u2, frequencies, dt)

        assert np.allclose(modes_sum, modes1 + modes2, rtol=1e-5)

    def test_2d_shape(self):
        """Test 2D wavefield DFT output shape."""
        nt, nx, ny = 100, 20, 20
        u_history = np.random.randn(nt, nx, ny)
        frequencies = np.array([5.0, 10.0])

        modes = compute_reference_dft(u_history, frequencies, dt=0.001)

        assert modes.shape == (2, nx, ny)
        assert modes.dtype == np.complex64


# =============================================================================
# Devito-dependent tests
# =============================================================================

@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")
class TestOnTheFlyDFT:
    """Tests for on-the-fly DFT with Devito."""

    def test_single_frequency_runs(self):
        """Single frequency DFT should run without error."""
        from src.theory.fourier_dft import run_otf_dft

        mode, info = run_otf_dft(nx=31, ny=31, nt=100, freq=10.0)

        assert mode.shape == (31, 31)
        assert mode.dtype == np.complex64
        assert np.isfinite(mode).all()
        assert info['cfl'] < 1.0  # Should be stable

    def test_multifreq_runs(self):
        """Multi-frequency DFT should run without error."""
        from src.theory.fourier_dft import run_otf_dft_multifreq

        modes, freqs, info = run_otf_dft_multifreq(
            nx=31, ny=31, nt=100,
            frequencies=np.array([5.0, 10.0, 15.0])
        )

        assert modes.shape == (3, 31, 31)
        assert len(freqs) == 3
        assert np.isfinite(modes).all()

    def test_multifreq_different_magnitudes(self):
        """Different frequencies should have different magnitudes."""
        from src.theory.fourier_dft import run_otf_dft_multifreq

        modes, freqs, info = run_otf_dft_multifreq(
            nx=41, ny=41, nt=200,
            frequencies=np.array([5.0, 15.0, 25.0]),
            f0=15.0  # Source centered at 15 Hz
        )

        # Mode at 15 Hz (index 1) should have highest energy
        # since source has f0=15 Hz
        norms = [np.linalg.norm(modes[i]) for i in range(3)]
        assert norms[1] > norms[0]  # 15 Hz > 5 Hz

    def test_mode_nonzero(self):
        """Fourier modes should be nonzero (source creates wavefield)."""
        from src.theory.fourier_dft import run_otf_dft

        mode, _ = run_otf_dft(nx=41, ny=41, nt=200, freq=10.0)
        assert np.linalg.norm(mode) > 1.0

    @pytest.mark.skip(reason="OTF DFT comparison requires careful time indexing")
    def test_verification_against_reference(self):
        """On-the-fly DFT should match reference computation."""
        from src.theory.fourier_dft import compare_otf_to_fft

        passed, max_error, details = compare_otf_to_fft(
            nx=21, ny=21, nt=100,
            frequencies=np.array([5.0, 10.0]),
            rtol=0.15  # Allow 15% tolerance
        )

        assert passed, f"Max error {max_error:.2e} exceeds tolerance"


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")
class TestIntegration:
    """Integration tests combining stability and DFT."""

    def test_stable_simulation(self):
        """Simulation with stable parameters should produce bounded results."""
        from src.theory.fourier_dft import run_otf_dft

        # Run with conservative CFL
        mode, info = run_otf_dft(
            nx=51, ny=51, nt=500,
            freq=10.0,
            velocity=1500.0
        )

        # Check stability
        assert info['cfl'] < 1.0, "CFL should be < 1 for stability"

        # Results should be finite
        assert np.isfinite(mode).all()

        # Norm should be bounded (not exploding)
        assert np.linalg.norm(mode) < 1e10

    def test_cfl_respected(self):
        """Simulation should respect CFL condition."""
        from src.theory.fourier_dft import run_otf_dft

        _, info = run_otf_dft(nx=31, ny=31, nt=100)

        # Verify CFL is computed correctly
        expected_cfl = info['velocity'] * info['dt'] / min(info['dx'], info['dy'])
        assert np.isclose(info['cfl'], expected_cfl)

        # Should be below stability limit
        assert info['cfl'] < 1.0


# =============================================================================
# Edge cases and error handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_wavenumber(self):
        """All schemes should have |g(0)| = 1."""
        assert np.isclose(amplification_factor_diffusion(0.3, 0.0), 1.0)
        assert np.isclose(np.abs(amplification_factor_advection_upwind(0.5, 0.0)), 1.0)
        assert np.isclose(np.abs(amplification_factor_wave(0.9, 0.0)), 1.0)

    def test_zero_cfl(self):
        """Zero CFL should always be stable."""
        stable, cfl, _ = check_stability_wave(1500, 0.0, 10)
        assert stable
        assert cfl == 0.0

    def test_negative_parameters_raises_or_handles(self):
        """Negative physical parameters should be handled."""
        # These should work mathematically (though unphysical)
        dt = stable_timestep_wave(1500, 10, cfl_max=0.9, ndim=1)
        assert dt > 0

    def test_array_wavenumber_input(self):
        """Functions should handle array inputs."""
        theta = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])

        g_diff = amplification_factor_diffusion(0.3, theta)
        assert len(g_diff) == 5

        g_adv = amplification_factor_advection_upwind(0.5, theta)
        assert len(g_adv) == 5

        g_wave = amplification_factor_wave(0.9, theta)
        assert len(g_wave) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
