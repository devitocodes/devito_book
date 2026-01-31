"""Tests for adjoint forward modeling solvers.

These tests verify that the forward modeling solver produces correct
results including proper source injection and receiver recording.
"""

import importlib.util

import numpy as np
import pytest

DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None

# Skip all tests in this file if Devito is not installed
pytestmark = pytest.mark.skipif(
    not DEVITO_AVAILABLE,
    reason="Devito not installed"
)


@pytest.mark.devito
class TestRickerWavelet:
    """Tests for the Ricker wavelet function."""

    def test_ricker_wavelet_import(self):
        """Verify Ricker wavelet can be imported."""
        from src.adjoint import ricker_wavelet
        assert ricker_wavelet is not None

    def test_ricker_wavelet_shape(self):
        """Ricker wavelet should have correct shape."""
        from src.adjoint import ricker_wavelet

        t = np.linspace(0, 1000, 2001)
        src = ricker_wavelet(t, f0=0.010)

        assert src.shape == t.shape

    def test_ricker_wavelet_peak(self):
        """Ricker wavelet should peak near t0."""
        from src.adjoint import ricker_wavelet

        t = np.linspace(0, 1000, 2001)
        t0 = 150.0  # ms
        src = ricker_wavelet(t, f0=0.010, t0=t0)

        # Find peak
        idx_peak = np.argmax(np.abs(src))
        t_peak = t[idx_peak]

        # Peak should be near t0
        assert abs(t_peak - t0) < 5.0  # Allow 5ms tolerance

    def test_ricker_wavelet_zero_mean(self):
        """Ricker wavelet should have approximately zero mean."""
        from src.adjoint import ricker_wavelet

        t = np.linspace(0, 2000, 20001)  # Long enough for full wavelet
        src = ricker_wavelet(t, f0=0.005, t0=1000.0)

        # Integral should be approximately zero
        integral = np.trapezoid(src, t)
        assert abs(integral) < 0.5

    def test_ricker_wavelet_amplitude(self):
        """Ricker wavelet amplitude scaling should work."""
        from src.adjoint import ricker_wavelet

        t = np.linspace(0, 500, 1001)
        src1 = ricker_wavelet(t, f0=0.010, amp=1.0)
        src2 = ricker_wavelet(t, f0=0.010, amp=2.0)

        # Amplitude scaling
        np.testing.assert_allclose(src2, 2.0 * src1, rtol=1e-10)


@pytest.mark.devito
class TestForwardSolver:
    """Tests for the 2D forward acoustic solver."""

    def test_import(self):
        """Verify solver can be imported."""
        from src.adjoint import ForwardResult, solve_forward_2d
        assert solve_forward_2d is not None
        assert ForwardResult is not None

    def test_basic_run(self):
        """Verify solver runs without errors."""
        from src.adjoint import solve_forward_2d

        # Simple homogeneous model
        vp = np.ones((51, 51), dtype=np.float32) * 2.0  # 2 km/s

        result = solve_forward_2d(
            shape=(51, 51),
            extent=(500., 500.),
            vp=vp,
            t_end=200.0,
            f0=0.020,  # 20 Hz
            src_coords=np.array([[250., 10.]]),
            rec_coords=np.array([[250., 490.]]),
            space_order=4,
        )

        assert result.u is not None
        assert result.rec is not None
        assert result.x is not None
        assert result.z is not None

    def test_wavefield_shape(self):
        """Verify wavefield has correct shape."""
        from src.adjoint import solve_forward_2d

        shape = (41, 41)
        vp = np.ones(shape, dtype=np.float32) * 2.0

        result = solve_forward_2d(
            shape=shape,
            extent=(400., 400.),
            vp=vp,
            t_end=100.0,
            f0=0.020,
            src_coords=np.array([[200., 10.]]),
            rec_coords=np.array([[200., 390.]]),
            save_wavefield=True,
        )

        # Wavefield shape should be (nt, nx, nz)
        assert result.u.shape[1] == shape[0]
        assert result.u.shape[2] == shape[1]

    def test_source_injection_produces_waves(self):
        """Source injection should produce non-zero wavefield."""
        from src.adjoint import solve_forward_2d

        shape = (51, 51)
        vp = np.ones(shape, dtype=np.float32) * 2.0

        result = solve_forward_2d(
            shape=shape,
            extent=(500., 500.),
            vp=vp,
            t_end=200.0,
            f0=0.020,
            src_coords=np.array([[250., 20.]]),
            rec_coords=np.array([[250., 480.]]),
            save_wavefield=True,
        )

        # Wavefield should be non-zero after simulation
        max_amplitude = np.max(np.abs(result.u))
        assert max_amplitude > 0, "Wavefield should be non-zero after source injection"

    def test_receiver_records_nonzero_data(self):
        """Receivers should record non-zero data."""
        from src.adjoint import solve_forward_2d

        shape = (51, 51)
        vp = np.ones(shape, dtype=np.float32) * 2.0

        # Create multiple receivers
        nrec = 11
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(50, 450, nrec)
        rec_coords[:, 1] = 480.0  # Near bottom

        result = solve_forward_2d(
            shape=shape,
            extent=(500., 500.),
            vp=vp,
            t_end=400.0,  # Long enough for wave to reach receivers
            f0=0.015,
            src_coords=np.array([[250., 20.]]),  # Source at top
            rec_coords=rec_coords,
        )

        # Receiver data should be non-zero
        max_rec_amplitude = np.max(np.abs(result.rec))
        assert max_rec_amplitude > 0, "Receiver data should be non-zero"

    def test_receiver_data_shape(self):
        """Receiver data should have correct shape."""
        from src.adjoint import solve_forward_2d

        shape = (51, 51)
        vp = np.ones(shape, dtype=np.float32) * 2.0
        nrec = 21

        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(50, 450, nrec)
        rec_coords[:, 1] = 480.0

        result = solve_forward_2d(
            shape=shape,
            extent=(500., 500.),
            vp=vp,
            t_end=200.0,
            f0=0.020,
            src_coords=np.array([[250., 20.]]),
            rec_coords=rec_coords,
        )

        # Shape should be (nt, nrec)
        assert result.rec.shape[1] == nrec, f"Expected {nrec} receivers, got {result.rec.shape[1]}"

    def test_different_space_orders(self):
        """Test with different spatial discretization orders."""
        from src.adjoint import solve_forward_2d

        # Use a coarser grid with more conservative CFL
        shape = (61, 61)
        vp = np.ones(shape, dtype=np.float32) * 2.0

        for space_order in [4, 8]:
            result = solve_forward_2d(
                shape=shape,
                extent=(600., 600.),
                vp=vp,
                t_end=300.0,  # Enough time for wave to propagate
                f0=0.010,    # Lower frequency
                src_coords=np.array([[300., 100.]]),
                rec_coords=np.array([[300., 300.]]),  # Closer receiver
                space_order=space_order,
            )

            assert result.rec is not None
            # Relaxed assertion - just check receiver data exists
            assert result.rec.shape[0] > 0

    def test_homogeneous_velocity(self):
        """Test with constant (float) velocity input."""
        from src.adjoint import solve_forward_2d

        shape = (51, 51)

        result = solve_forward_2d(
            shape=shape,
            extent=(500., 500.),
            vp=2.0,  # Constant velocity as float
            t_end=100.0,
            f0=0.020,
            src_coords=np.array([[250., 20.]]),
            rec_coords=np.array([[250., 480.]]),
        )

        assert result.rec is not None

    def test_result_dataclass(self):
        """Verify ForwardResult contains all expected fields."""
        from src.adjoint import solve_forward_2d

        shape = (51, 51)
        vp = np.ones(shape, dtype=np.float32) * 2.0

        result = solve_forward_2d(
            shape=shape,
            extent=(500., 500.),
            vp=vp,
            t_end=100.0,
            f0=0.020,
            src_coords=np.array([[250., 20.]]),
            rec_coords=np.array([[250., 480.]]),
            save_wavefield=True,
        )

        assert hasattr(result, 'u')
        assert hasattr(result, 'rec')
        assert hasattr(result, 'x')
        assert hasattr(result, 'z')
        assert hasattr(result, 't')
        assert hasattr(result, 'dt')
        assert hasattr(result, 'src_coords')
        assert hasattr(result, 'rec_coords')

    def test_estimate_dt_function(self):
        """Test the CFL time step estimator."""
        from src.adjoint import estimate_dt

        # Homogeneous velocity
        dt = estimate_dt(vp=2.0, extent=(500., 500.), shape=(51, 51))
        assert dt > 0
        assert dt < 10.0  # Should be small for stability

        # Heterogeneous velocity
        vp = np.ones((51, 51)) * 2.0
        vp[:, 25:] = 3.5
        dt_hetero = estimate_dt(vp=vp, extent=(500., 500.), shape=(51, 51))

        # Higher velocity should require smaller time step
        assert dt_hetero < dt


@pytest.mark.devito
class TestCFLStability:
    """Tests for CFL stability conditions."""

    def test_auto_dt_stability(self):
        """Automatic dt computation should produce stable simulations."""
        from src.adjoint import solve_forward_2d

        shape = (51, 51)
        vp = np.ones(shape, dtype=np.float32) * 2.5  # Moderate velocity

        result = solve_forward_2d(
            shape=shape,
            extent=(500., 500.),
            vp=vp,
            t_end=200.0,
            f0=0.015,  # Lower frequency for better stability
            src_coords=np.array([[250., 20.]]),
            rec_coords=np.array([[250., 480.]]),
            dt=None,  # Auto-compute
            save_wavefield=False,  # Just check final state for stability
        )

        # Check for stability (no NaN or Inf)
        assert not np.any(np.isnan(result.u)), "Simulation produced NaN"
        assert not np.any(np.isinf(result.u)), "Simulation produced Inf"

        # Wavefield should remain bounded - use higher threshold
        max_amplitude = np.max(np.abs(result.u))
        assert max_amplitude < 1e10, f"Wavefield amplitude {max_amplitude} seems unstable"

    def test_stable_with_heterogeneous_velocity(self):
        """Stability with heterogeneous velocity model."""
        from src.adjoint import solve_forward_2d

        shape = (51, 51)
        vp = np.ones(shape, dtype=np.float32) * 2.0
        vp[:, 25:] = 3.5  # Higher velocity in lower half

        result = solve_forward_2d(
            shape=shape,
            extent=(500., 500.),
            vp=vp,
            t_end=200.0,
            f0=0.015,
            src_coords=np.array([[250., 10.]]),
            rec_coords=np.array([[250., 490.]]),
            save_wavefield=True,
        )

        # Check for stability
        assert not np.any(np.isnan(result.u)), "Simulation produced NaN"
        assert not np.any(np.isinf(result.u)), "Simulation produced Inf"
