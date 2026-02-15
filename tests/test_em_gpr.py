"""Tests for src.em.gpr — GPR wavelets, time/depth conversion, and simulations."""

import numpy as np
import pytest

from src.em.gpr import (
    blackman_harris_wavelet,
    depth_from_travel_time,
    fit_hyperbola,
    gaussian_derivative_wavelet,
    hyperbola_travel_time,
    ricker_wavelet,
    two_way_travel_time,
    wavelet_spectrum,
)
from src.em.units import EMConstants

# ---------------------------------------------------------------------------
# Wavelets (pure NumPy)
# ---------------------------------------------------------------------------


class TestRickerWavelet:

    def test_peak_at_t0(self):
        t = np.linspace(0, 10e-9, 10000)
        f0 = 500e6
        w = ricker_wavelet(t, f0)
        t0 = 1.0 / f0
        peak_t = t[np.argmax(w)]
        assert peak_t == pytest.approx(t0, abs=t[1] - t[0])

    def test_peak_amplitude(self):
        t = np.linspace(0, 10e-9, 10000)
        w = ricker_wavelet(t, f0=500e6, amplitude=2.5)
        assert np.max(w) == pytest.approx(2.5, rel=0.01)

    def test_custom_t0(self):
        t = np.linspace(0, 20e-9, 10000)
        w = ricker_wavelet(t, f0=500e6, t0=5e-9)
        peak_t = t[np.argmax(w)]
        assert peak_t == pytest.approx(5e-9, abs=t[1] - t[0])

    def test_shape_length(self):
        t = np.linspace(0, 10e-9, 500)
        w = ricker_wavelet(t, f0=500e6)
        assert w.shape == t.shape


class TestGaussianDerivativeWavelet:

    def test_shape(self):
        t = np.linspace(0, 10e-9, 500)
        w = gaussian_derivative_wavelet(t, f0=500e6)
        assert w.shape == t.shape

    def test_zero_crossing_near_t0(self):
        """Gaussian derivative should cross zero near t0."""
        t = np.linspace(0, 10e-9, 10000)
        w = gaussian_derivative_wavelet(t, f0=500e6)
        t0 = 1.0 / 500e6
        # Find zero crossing nearest to t0
        sign_changes = np.where(np.diff(np.sign(w)))[0]
        if len(sign_changes) > 0:
            zero_t = t[sign_changes[np.argmin(np.abs(t[sign_changes] - t0))]]
            assert abs(zero_t - t0) < 2e-9  # Within 2 ns


class TestBlackmanHarrisWavelet:

    def test_shape(self):
        t = np.linspace(0, 20e-9, 1000)
        w = blackman_harris_wavelet(t, f0=500e6)
        assert w.shape == t.shape

    def test_finite_duration(self):
        """Wavelet should be zero outside its window."""
        t = np.linspace(-20e-9, 40e-9, 10000)
        w = blackman_harris_wavelet(t, f0=500e6, n_cycles=4)
        # Far from the pulse, should be zero
        assert np.all(np.abs(w[:100]) < 1e-10)
        assert np.all(np.abs(w[-100:]) < 1e-10)

    def test_amplitude(self):
        t = np.linspace(0, 20e-9, 10000)
        w = blackman_harris_wavelet(t, f0=500e6, amplitude=3.0)
        assert np.max(np.abs(w)) <= 3.0 + 0.01


class TestWaveletSpectrum:

    def test_returns_freq_and_spectrum(self):
        t = np.linspace(0, 20e-9, 1024)
        dt = t[1] - t[0]
        w = ricker_wavelet(t, f0=500e6)
        freq, spec = wavelet_spectrum(w, dt)
        assert len(freq) == len(spec)
        assert freq[0] == 0.0
        assert np.all(spec >= 0)

    def test_ricker_peak_near_f0(self):
        """Ricker spectrum should peak near f0."""
        f0 = 500e6
        t = np.linspace(0, 40e-9, 4096)
        dt = t[1] - t[0]
        w = ricker_wavelet(t, f0)
        freq, spec = wavelet_spectrum(w, dt)
        peak_freq = freq[np.argmax(spec)]
        assert peak_freq == pytest.approx(f0, rel=0.15)


# ---------------------------------------------------------------------------
# Time/depth conversion (pure NumPy)
# ---------------------------------------------------------------------------


class TestTravelTime:

    def test_round_trip(self):
        """depth → twtt → depth should be identity."""
        depth = 1.5
        eps_r = 9.0
        twtt = two_way_travel_time(depth, eps_r)
        recovered = depth_from_travel_time(twtt, eps_r)
        assert recovered == pytest.approx(depth, rel=1e-10)

    def test_twtt_positive(self):
        assert two_way_travel_time(1.0, 4.0) > 0

    def test_higher_eps_slower(self):
        """Higher permittivity → longer travel time."""
        t1 = two_way_travel_time(1.0, 4.0)
        t2 = two_way_travel_time(1.0, 16.0)
        assert t2 > t1

    def test_free_space_speed(self):
        """In vacuum (eps_r=1), v=c0."""
        const = EMConstants()
        twtt = two_way_travel_time(1.0, 1.0)
        expected = 2 * 1.0 / const.c0
        assert twtt == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Hyperbola functions (pure NumPy)
# ---------------------------------------------------------------------------


class TestHyperbolaTravelTime:

    def test_minimum_at_target_position(self):
        """Travel time is minimized when antenna is directly above target."""
        x_positions = np.linspace(-1, 1, 100)
        times = [hyperbola_travel_time(x, 0.0, 0.5, 1e8) for x in x_positions]
        min_idx = np.argmin(times)
        assert x_positions[min_idx] == pytest.approx(0.0, abs=0.03)

    def test_symmetric_about_target(self):
        t_left = hyperbola_travel_time(-0.5, 0.0, 1.0, 1e8)
        t_right = hyperbola_travel_time(0.5, 0.0, 1.0, 1e8)
        assert t_left == pytest.approx(t_right, rel=1e-10)

    def test_directly_above(self):
        """Directly above: t = 2*depth/v."""
        v = 1e8
        depth = 0.5
        t = hyperbola_travel_time(0.0, 0.0, depth, v)
        assert t == pytest.approx(2 * depth / v, rel=1e-10)


class TestFitHyperbola:

    def test_recovers_known_parameters(self):
        """Fit should recover known target position, depth, and velocity."""
        # Use soil-like velocity (~0.1*c0) matching function's initial guess
        from src.em.units import EMConstants
        v_true = 0.1 * EMConstants().c0  # ~3e7 m/s
        x0_true, z0_true = 0.5, 0.5
        x = np.linspace(-1, 2, 200)
        t = np.array([hyperbola_travel_time(xi, x0_true, z0_true, v_true) for xi in x])

        x0_fit, z0_fit, v_fit = fit_hyperbola(x, t)
        assert x0_fit == pytest.approx(x0_true, abs=0.05)
        assert z0_fit == pytest.approx(z0_true, abs=0.1)
        assert v_fit == pytest.approx(v_true, rel=0.1)

    def test_noisy_data(self):
        """Fit should handle moderate noise."""
        from src.em.units import EMConstants
        rng = np.random.default_rng(42)
        v_true = 0.1 * EMConstants().c0
        x0_true, z0_true = 1.0, 0.5
        x = np.linspace(-0.5, 2.5, 200)
        t = np.array([hyperbola_travel_time(xi, x0_true, z0_true, v_true) for xi in x])
        t_noisy = t + rng.normal(0, 0.1e-9, len(t))

        x0_fit, z0_fit, v_fit = fit_hyperbola(x, t_noisy)
        assert x0_fit == pytest.approx(x0_true, abs=0.2)
        assert z0_fit == pytest.approx(z0_true, abs=0.3)
        assert v_fit == pytest.approx(v_true, rel=0.3)


# ---------------------------------------------------------------------------
# GPR simulation (requires Devito)
# ---------------------------------------------------------------------------


def _devito_importable() -> bool:
    try:
        import devito  # noqa: F401
    except Exception:
        return False
    return True


_skip_no_devito = pytest.mark.skipif(
    not _devito_importable(), reason="Devito not importable"
)


@pytest.mark.devito
@_skip_no_devito
class TestRunGpr1d:

    def test_smoke(self):
        from src.em.gpr import run_gpr_1d

        result = run_gpr_1d(
            depth=1.0, eps_r_soil=9.0, sigma_soil=0.001,
            frequency=500e6, Nx=200,
        )
        assert result.ascan is not None
        assert len(result.t) > 0
        assert len(result.x) > 0

    def test_with_target(self):
        from src.em.gpr import run_gpr_1d

        result = run_gpr_1d(
            depth=1.0, eps_r_soil=9.0, sigma_soil=0.001,
            frequency=500e6, target_depth=0.5, target_eps_r=1.0,
            Nx=200,
        )
        assert result.ascan is not None
        assert result.depth_axis is not None
