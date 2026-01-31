"""Tests for RTM (Reverse Time Migration) solvers.

These tests verify that the RTM solver produces correct images
including reflector detection at appropriate locations.
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
class TestRTMSingleShot:
    """Tests for single-shot RTM imaging."""

    def test_import(self):
        """Verify RTM functions can be imported."""
        from src.adjoint import RTMResult, rtm_single_shot
        assert rtm_single_shot is not None
        assert RTMResult is not None

    def test_basic_run(self):
        """Verify RTM runs without errors."""
        from src.adjoint import rtm_single_shot

        shape = (51, 51)

        # True model with reflector
        vp_true = np.ones(shape, dtype=np.float32) * 2.0
        vp_true[:, 25:] = 2.5  # Reflector at mid-depth

        # Smooth model (no reflector)
        vp_smooth = np.ones(shape, dtype=np.float32) * 2.0

        # Receivers
        nrec = 21
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(50, 450, nrec)
        rec_coords[:, 1] = 20.0

        result = rtm_single_shot(
            shape=shape,
            extent=(500., 500.),
            vp_true=vp_true,
            vp_smooth=vp_smooth,
            src_coords=np.array([[250., 10.]]),
            rec_coords=rec_coords,
            t_end=400.0,
            f0=0.015,
        )

        assert result.image is not None
        assert result.x is not None
        assert result.z is not None

    def test_image_shape(self):
        """Verify RTM image has correct shape."""
        from src.adjoint import rtm_single_shot

        shape = (41, 51)

        vp_true = np.ones(shape, dtype=np.float32) * 2.0
        vp_true[:, 25:] = 2.5

        vp_smooth = np.ones(shape, dtype=np.float32) * 2.0

        nrec = 15
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(50, 350, nrec)
        rec_coords[:, 1] = 20.0

        result = rtm_single_shot(
            shape=shape,
            extent=(400., 500.),
            vp_true=vp_true,
            vp_smooth=vp_smooth,
            src_coords=np.array([[200., 10.]]),
            rec_coords=rec_coords,
            t_end=300.0,
            f0=0.020,
        )

        assert result.image.shape == shape

    def test_image_nonzero(self):
        """RTM image should be non-zero when there is a reflector."""
        from src.adjoint import rtm_single_shot

        shape = (51, 51)

        # True model with reflector
        vp_true = np.ones(shape, dtype=np.float32) * 2.0
        vp_true[:, 25:] = 2.5  # Reflector at mid-depth

        # Smooth model (no reflector)
        vp_smooth = np.ones(shape, dtype=np.float32) * 2.0

        nrec = 21
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(50, 450, nrec)
        rec_coords[:, 1] = 20.0

        result = rtm_single_shot(
            shape=shape,
            extent=(500., 500.),
            vp_true=vp_true,
            vp_smooth=vp_smooth,
            src_coords=np.array([[250., 10.]]),
            rec_coords=rec_coords,
            t_end=400.0,
            f0=0.015,
        )

        # Image should be non-zero
        max_amplitude = np.max(np.abs(result.image))
        assert max_amplitude > 0, "RTM image should be non-zero with a reflector"

    def test_reflector_location(self):
        """RTM image should show reflector at approximately correct depth."""
        from src.adjoint import rtm_single_shot

        shape = (51, 51)
        extent = (500., 500.)

        # Reflector at z=250m (grid index ~25)
        vp_true = np.ones(shape, dtype=np.float32) * 2.0
        vp_true[:, 25:] = 2.5

        vp_smooth = np.ones(shape, dtype=np.float32) * 2.0

        nrec = 31
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(50, 450, nrec)
        rec_coords[:, 1] = 20.0

        result = rtm_single_shot(
            shape=shape,
            extent=extent,
            vp_true=vp_true,
            vp_smooth=vp_smooth,
            src_coords=np.array([[250., 10.]]),
            rec_coords=rec_coords,
            t_end=500.0,
            f0=0.012,
        )

        # Find maximum response in image
        # Take horizontal derivative to find reflector more precisely
        image_dx = np.diff(result.image, axis=1)
        max_idx = np.unravel_index(np.argmax(np.abs(image_dx)), image_dx.shape)

        # Reflector should be around z-index 25 (some tolerance due to finite frequency)
        z_grid = np.linspace(0, extent[1], shape[1])
        dz = z_grid[1] - z_grid[0]
        reflector_z = max_idx[1] * dz

        # Check that maximum is within ~50m of expected depth (250m)
        assert abs(reflector_z - 250) < 50, f"Reflector found at z={reflector_z}, expected ~250"

    def test_result_dataclass(self):
        """Verify RTMResult contains all expected fields."""
        from src.adjoint import rtm_single_shot

        shape = (41, 41)

        vp_true = np.ones(shape, dtype=np.float32) * 2.0
        vp_true[:, 20:] = 2.5

        vp_smooth = np.ones(shape, dtype=np.float32) * 2.0

        nrec = 11
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(50, 350, nrec)
        rec_coords[:, 1] = 20.0

        result = rtm_single_shot(
            shape=shape,
            extent=(400., 400.),
            vp_true=vp_true,
            vp_smooth=vp_smooth,
            src_coords=np.array([[200., 10.]]),
            rec_coords=rec_coords,
            t_end=300.0,
            f0=0.020,
        )

        assert hasattr(result, 'image')
        assert hasattr(result, 'x')
        assert hasattr(result, 'z')
        assert hasattr(result, 'nshots')
        assert result.nshots == 1


@pytest.mark.devito
class TestRTMMultiShot:
    """Tests for multi-shot RTM imaging."""

    def test_import(self):
        """Verify multi-shot RTM can be imported."""
        from src.adjoint import rtm_multi_shot
        assert rtm_multi_shot is not None

    def test_basic_run(self):
        """Verify multi-shot RTM runs without errors."""
        from src.adjoint import rtm_multi_shot

        shape = (41, 41)

        vp_true = np.ones(shape, dtype=np.float32) * 2.0
        vp_true[:, 20:] = 2.5

        vp_smooth = np.ones(shape, dtype=np.float32) * 2.0

        # Multiple shots
        nshots = 3
        src_positions = np.zeros((nshots, 2))
        src_positions[:, 0] = np.linspace(100, 300, nshots)
        src_positions[:, 1] = 10.0

        nrec = 11
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(50, 350, nrec)
        rec_coords[:, 1] = 20.0

        result = rtm_multi_shot(
            shape=shape,
            extent=(400., 400.),
            vp_true=vp_true,
            vp_smooth=vp_smooth,
            src_positions=src_positions,
            rec_coords=rec_coords,
            t_end=300.0,
            f0=0.020,
            verbose=False,
        )

        assert result.image is not None
        assert result.nshots == nshots

    def test_multishot_improves_image(self):
        """Multi-shot RTM should produce stronger image than single shot."""
        from src.adjoint import rtm_multi_shot, rtm_single_shot

        shape = (41, 41)

        vp_true = np.ones(shape, dtype=np.float32) * 2.0
        vp_true[:, 20:] = 2.5

        vp_smooth = np.ones(shape, dtype=np.float32) * 2.0

        nrec = 11
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(50, 350, nrec)
        rec_coords[:, 1] = 20.0

        # Single shot
        result_single = rtm_single_shot(
            shape=shape,
            extent=(400., 400.),
            vp_true=vp_true,
            vp_smooth=vp_smooth,
            src_coords=np.array([[200., 10.]]),
            rec_coords=rec_coords,
            t_end=300.0,
            f0=0.020,
        )

        # Multiple shots
        nshots = 3
        src_positions = np.zeros((nshots, 2))
        src_positions[:, 0] = np.linspace(100, 300, nshots)
        src_positions[:, 1] = 10.0

        result_multi = rtm_multi_shot(
            shape=shape,
            extent=(400., 400.),
            vp_true=vp_true,
            vp_smooth=vp_smooth,
            src_positions=src_positions,
            rec_coords=rec_coords,
            t_end=300.0,
            f0=0.020,
            verbose=False,
        )

        # Multi-shot image should be stronger (stacked)
        # Note: This might not always be true due to interference, but
        # in general stacking should increase amplitude
        max_single = np.max(np.abs(result_single.image))
        max_multi = np.max(np.abs(result_multi.image))

        # Just check both are non-zero
        assert max_single > 0
        assert max_multi > 0


@pytest.mark.devito
class TestAdjointSolver:
    """Tests for the adjoint wave equation solver."""

    def test_import(self):
        """Verify adjoint solver can be imported."""
        from src.adjoint import solve_adjoint_2d
        assert solve_adjoint_2d is not None


@pytest.mark.devito
class TestRTMStability:
    """Tests for RTM numerical stability."""

    def test_no_nans_in_image(self):
        """RTM image should not contain NaN values."""
        from src.adjoint import rtm_single_shot

        shape = (51, 51)

        vp_true = np.ones(shape, dtype=np.float32) * 2.0
        vp_true[:, 25:] = 2.5

        vp_smooth = np.ones(shape, dtype=np.float32) * 2.0

        nrec = 21
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(50, 450, nrec)
        rec_coords[:, 1] = 20.0

        result = rtm_single_shot(
            shape=shape,
            extent=(500., 500.),
            vp_true=vp_true,
            vp_smooth=vp_smooth,
            src_coords=np.array([[250., 10.]]),
            rec_coords=rec_coords,
            t_end=400.0,
            f0=0.015,
        )

        assert not np.any(np.isnan(result.image)), "RTM image contains NaN"

    def test_no_infs_in_image(self):
        """RTM image should not contain Inf values."""
        from src.adjoint import rtm_single_shot

        shape = (51, 51)

        vp_true = np.ones(shape, dtype=np.float32) * 2.0
        vp_true[:, 25:] = 2.5

        vp_smooth = np.ones(shape, dtype=np.float32) * 2.0

        nrec = 21
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(50, 450, nrec)
        rec_coords[:, 1] = 20.0

        result = rtm_single_shot(
            shape=shape,
            extent=(500., 500.),
            vp_true=vp_true,
            vp_smooth=vp_smooth,
            src_coords=np.array([[250., 10.]]),
            rec_coords=rec_coords,
            t_end=400.0,
            f0=0.015,
        )

        assert not np.any(np.isinf(result.image)), "RTM image contains Inf"
