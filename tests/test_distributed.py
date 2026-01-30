"""Tests for distributed computing utilities.

These tests verify the Dask-based parallel execution of Devito
computations. Tests use LocalCluster for testing without requiring
a full distributed setup.

Tests are organized as:
- TestImports: Verify module imports correctly
- TestRickerWavelet: Test source wavelet generation
- TestFGPair: Test functional-gradient pair operations
- TestForwardShot: Test single-shot forward modeling
- TestFWIGradientSingleShot: Test single-shot gradient computation
- TestParallelForwardModeling: Test parallel forward modeling
- TestParallelFWIGradient: Test parallel gradient computation
"""

import importlib.util

import numpy as np
import pytest

# Check if dependencies are available
DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None
DASK_AVAILABLE = importlib.util.find_spec("dask") is not None

# Skip all tests if Dask is not available
pytestmark = [
    pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed"),
    pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not installed"),
]


class TestImports:
    """Test that distributed module imports correctly."""

    def test_import_fg_pair(self):
        """Test FGPair import."""
        from src.distributed import FGPair

        assert FGPair is not None

    def test_import_create_local_cluster(self):
        """Test create_local_cluster import."""
        from src.distributed import create_local_cluster

        assert create_local_cluster is not None

    def test_import_forward_shot(self):
        """Test forward_shot import."""
        from src.distributed import forward_shot

        assert forward_shot is not None

    def test_import_fwi_gradient_single_shot(self):
        """Test fwi_gradient_single_shot import."""
        from src.distributed import fwi_gradient_single_shot

        assert fwi_gradient_single_shot is not None

    def test_import_parallel_forward_modeling(self):
        """Test parallel_forward_modeling import."""
        from src.distributed import parallel_forward_modeling

        assert parallel_forward_modeling is not None

    def test_import_parallel_fwi_gradient(self):
        """Test parallel_fwi_gradient import."""
        from src.distributed import parallel_fwi_gradient

        assert parallel_fwi_gradient is not None

    def test_import_ricker_wavelet(self):
        """Test ricker_wavelet import."""
        from src.distributed import ricker_wavelet

        assert ricker_wavelet is not None


class TestRickerWavelet:
    """Test Ricker wavelet generation."""

    def test_ricker_shape(self):
        """Test wavelet has correct shape."""
        from src.distributed import ricker_wavelet

        t = np.linspace(0, 1000, 1001)
        src = ricker_wavelet(t, f0=0.01)

        assert src.shape == t.shape

    def test_ricker_peak_at_t0(self):
        """Test wavelet peaks near t0."""
        from src.distributed import ricker_wavelet

        t = np.linspace(0, 500, 5001)
        t0 = 100.0
        src = ricker_wavelet(t, f0=0.01, t0=t0)

        idx_peak = np.argmax(src)
        t_peak = t[idx_peak]

        assert abs(t_peak - t0) < 1.0

    def test_ricker_default_t0(self):
        """Test default t0 = 1.5/f0."""
        from src.distributed import ricker_wavelet

        t = np.linspace(0, 500, 5001)
        f0 = 0.01
        expected_t0 = 1.5 / f0
        src = ricker_wavelet(t, f0=f0)

        idx_peak = np.argmax(src)
        t_peak = t[idx_peak]

        assert abs(t_peak - expected_t0) < 2.0


class TestFGPair:
    """Test FGPair functional-gradient pair operations."""

    def test_fg_pair_creation(self):
        """Test FGPair can be created."""
        from src.distributed import FGPair

        fg = FGPair(f=10.0, g=np.array([1.0, 2.0, 3.0]))

        assert fg.f == 10.0
        assert np.array_equal(fg.g, [1.0, 2.0, 3.0])

    def test_fg_pair_addition(self):
        """Test FGPair addition."""
        from src.distributed import FGPair

        fg1 = FGPair(f=10.0, g=np.array([1.0, 2.0]))
        fg2 = FGPair(f=5.0, g=np.array([3.0, 4.0]))

        fg_sum = fg1 + fg2

        assert fg_sum.f == 15.0
        np.testing.assert_array_equal(fg_sum.g, [4.0, 6.0])

    def test_fg_pair_radd_with_zero(self):
        """Test FGPair right addition with zero (for sum())."""
        from src.distributed import FGPair

        fg = FGPair(f=10.0, g=np.array([1.0, 2.0]))

        result = 0 + fg

        assert result.f == 10.0
        np.testing.assert_array_equal(result.g, [1.0, 2.0])

    def test_fg_pair_sum(self):
        """Test summing multiple FGPairs."""
        from src.distributed import FGPair

        fg_list = [
            FGPair(f=10.0, g=np.array([1.0, 2.0])),
            FGPair(f=20.0, g=np.array([3.0, 4.0])),
            FGPair(f=30.0, g=np.array([5.0, 6.0])),
        ]

        total = sum(fg_list)

        assert total.f == 60.0
        np.testing.assert_array_equal(total.g, [9.0, 12.0])


class TestSumFGPairs:
    """Test sum_fg_pairs utility function."""

    def test_sum_fg_pairs(self):
        """Test sum_fg_pairs function."""
        from src.distributed import FGPair, sum_fg_pairs

        fg_list = [
            FGPair(f=10.0, g=np.array([1.0, 2.0])),
            FGPair(f=20.0, g=np.array([3.0, 4.0])),
        ]

        total = sum_fg_pairs(fg_list)

        assert total.f == 30.0
        np.testing.assert_array_equal(total.g, [4.0, 6.0])


class TestCreateLocalCluster:
    """Test LocalCluster creation."""

    def test_create_cluster(self):
        """Test cluster creation and cleanup."""
        from src.distributed import create_local_cluster

        cluster, client = create_local_cluster(n_workers=2)

        try:
            # Verify client is connected
            assert client.status == "running"

            # Verify number of workers
            info = client.scheduler_info()
            assert len(info["workers"]) == 2
        finally:
            client.close()
            cluster.close()


@pytest.mark.slow
class TestForwardShot:
    """Test single-shot forward modeling.

    These tests are marked slow as they run Devito simulations.
    """

    def test_forward_shot_runs(self):
        """Test forward_shot completes without error."""
        from src.distributed import forward_shot

        shape = (31, 31)
        extent = (300.0, 300.0)
        velocity = np.full(shape, 2.5, dtype=np.float32)

        src_coord = np.array([150.0, 20.0])
        nrec = 11
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(20.0, 280.0, nrec)
        rec_coords[:, 1] = 280.0

        nt = 201
        dt = 0.5
        f0 = 0.025

        result = forward_shot(
            shot_id=0,
            velocity=velocity,
            src_coord=src_coord,
            rec_coords=rec_coords,
            nt=nt,
            dt=dt,
            f0=f0,
            extent=extent,
        )

        assert result.shape == (nt, nrec)
        assert np.all(np.isfinite(result))

    def test_forward_shot_nonzero_output(self):
        """Test forward_shot produces non-zero data."""
        from src.distributed import forward_shot

        shape = (31, 31)
        extent = (300.0, 300.0)
        velocity = np.full(shape, 2.5, dtype=np.float32)

        src_coord = np.array([150.0, 20.0])
        nrec = 11
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(20.0, 280.0, nrec)
        rec_coords[:, 1] = 280.0

        nt = 401
        dt = 0.5
        f0 = 0.025

        result = forward_shot(
            shot_id=0,
            velocity=velocity,
            src_coord=src_coord,
            rec_coords=rec_coords,
            nt=nt,
            dt=dt,
            f0=f0,
            extent=extent,
        )

        # Should have non-zero values after wavefield reaches receivers
        assert np.max(np.abs(result)) > 0


@pytest.mark.slow
class TestFWIGradientSingleShot:
    """Test single-shot FWI gradient computation.

    These tests are marked slow as they run forward and adjoint simulations.
    """

    def test_fwi_gradient_runs(self):
        """Test fwi_gradient_single_shot completes without error."""
        from src.distributed import forward_shot, fwi_gradient_single_shot

        shape = (31, 31)
        extent = (300.0, 300.0)
        spacing = (10.0, 10.0)

        # True velocity (with anomaly)
        vp_true = np.full(shape, 2.5, dtype=np.float32)
        center = (shape[0] // 2, shape[1] // 2)
        for i in range(shape[0]):
            for j in range(shape[1]):
                dist = np.sqrt(
                    (i * spacing[0] - center[0] * spacing[0]) ** 2
                    + (j * spacing[1] - center[1] * spacing[1]) ** 2
                )
                if dist < 50:
                    vp_true[i, j] = 3.0

        # Current velocity (smooth)
        vp_current = np.full(shape, 2.5, dtype=np.float32)

        src_coord = np.array([150.0, 20.0])
        nrec = 11
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(20.0, 280.0, nrec)
        rec_coords[:, 1] = 280.0

        nt = 201
        dt = 0.5
        f0 = 0.025

        # Generate observed data
        d_obs = forward_shot(
            shot_id=0,
            velocity=vp_true,
            src_coord=src_coord,
            rec_coords=rec_coords,
            nt=nt,
            dt=dt,
            f0=f0,
            extent=extent,
        )

        # Compute gradient
        objective, gradient = fwi_gradient_single_shot(
            velocity=vp_current,
            src_coord=src_coord,
            rec_coords=rec_coords,
            d_obs=d_obs,
            shape=shape,
            extent=extent,
            nt=nt,
            dt=dt,
            f0=f0,
        )

        assert np.isfinite(objective)
        assert gradient.shape == shape
        assert np.all(np.isfinite(gradient))

    def test_zero_objective_for_matching_data(self):
        """Test objective is zero when observed matches synthetic."""
        from src.distributed import forward_shot, fwi_gradient_single_shot

        shape = (31, 31)
        extent = (300.0, 300.0)
        velocity = np.full(shape, 2.5, dtype=np.float32)

        src_coord = np.array([150.0, 20.0])
        nrec = 11
        rec_coords = np.zeros((nrec, 2))
        rec_coords[:, 0] = np.linspace(20.0, 280.0, nrec)
        rec_coords[:, 1] = 280.0

        nt = 201
        dt = 0.5
        f0 = 0.025

        # Observed = synthetic (same velocity)
        d_obs = forward_shot(
            shot_id=0,
            velocity=velocity,
            src_coord=src_coord,
            rec_coords=rec_coords,
            nt=nt,
            dt=dt,
            f0=f0,
            extent=extent,
        )

        objective, gradient = fwi_gradient_single_shot(
            velocity=velocity,
            src_coord=src_coord,
            rec_coords=rec_coords,
            d_obs=d_obs,
            shape=shape,
            extent=extent,
            nt=nt,
            dt=dt,
            f0=f0,
        )

        # Objective should be very small (numerical precision)
        assert objective < 1e-6


@pytest.mark.slow
class TestParallelForwardModeling:
    """Test parallel forward modeling.

    These tests run multiple shots in parallel using LocalCluster.
    """

    def test_parallel_forward_modeling(self):
        """Test parallel forward modeling completes correctly."""
        from src.distributed import create_local_cluster, parallel_forward_modeling

        cluster, client = create_local_cluster(n_workers=2)

        try:
            shape = (31, 31)
            extent = (300.0, 300.0)
            velocity = np.full(shape, 2.5, dtype=np.float32)

            # 4 shots
            src_positions = np.array(
                [[100.0, 20.0], [150.0, 20.0], [200.0, 20.0], [250.0, 20.0]]
            )
            nrec = 11
            rec_coords = np.zeros((nrec, 2))
            rec_coords[:, 0] = np.linspace(20.0, 280.0, nrec)
            rec_coords[:, 1] = 280.0

            nt = 201
            dt = 0.5
            f0 = 0.025

            results = parallel_forward_modeling(
                client=client,
                velocity=velocity,
                src_positions=src_positions,
                rec_coords=rec_coords,
                nt=nt,
                dt=dt,
                f0=f0,
                extent=extent,
            )

            # Should have 4 shot records
            assert len(results) == 4

            # Each should have correct shape
            for i, result in enumerate(results):
                assert result.shape == (nt, nrec), f"Shot {i} has wrong shape"
                assert np.all(np.isfinite(result)), f"Shot {i} has non-finite values"

        finally:
            client.close()
            cluster.close()


@pytest.mark.slow
class TestParallelFWIGradient:
    """Test parallel FWI gradient computation.

    These tests run multiple shots in parallel and sum gradients.
    """

    def test_parallel_fwi_gradient(self):
        """Test parallel gradient computation."""
        from src.distributed import (
            create_local_cluster,
            parallel_forward_modeling,
            parallel_fwi_gradient,
        )

        cluster, client = create_local_cluster(n_workers=2)

        try:
            shape = (31, 31)
            extent = (300.0, 300.0)
            spacing = (10.0, 10.0)

            # True velocity (with anomaly)
            vp_true = np.full(shape, 2.5, dtype=np.float32)
            center = (shape[0] // 2, shape[1] // 2)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    dist = np.sqrt(
                        (i * spacing[0] - center[0] * spacing[0]) ** 2
                        + (j * spacing[1] - center[1] * spacing[1]) ** 2
                    )
                    if dist < 50:
                        vp_true[i, j] = 3.0

            # Current velocity (smooth)
            vp_current = np.full(shape, 2.5, dtype=np.float32)

            # 2 shots
            src_positions = np.array([[100.0, 20.0], [200.0, 20.0]])
            nrec = 11
            rec_coords = np.zeros((nrec, 2))
            rec_coords[:, 0] = np.linspace(20.0, 280.0, nrec)
            rec_coords[:, 1] = 280.0

            nt = 201
            dt = 0.5
            f0 = 0.025

            # Generate observed data
            observed_data = parallel_forward_modeling(
                client=client,
                velocity=vp_true,
                src_positions=src_positions,
                rec_coords=rec_coords,
                nt=nt,
                dt=dt,
                f0=f0,
                extent=extent,
            )

            # Compute gradient
            objective, gradient = parallel_fwi_gradient(
                client=client,
                velocity=vp_current,
                src_positions=src_positions,
                rec_coords=rec_coords,
                observed_data=observed_data,
                shape=shape,
                extent=extent,
                nt=nt,
                dt=dt,
                f0=f0,
            )

            assert np.isfinite(objective)
            assert objective > 0  # Should have misfit
            assert gradient.shape == shape
            assert np.all(np.isfinite(gradient))

        finally:
            client.close()
            cluster.close()

    def test_gradient_additivity(self):
        """Test that parallel gradient equals sum of individual gradients."""
        from src.distributed import (
            create_local_cluster,
            forward_shot,
            fwi_gradient_single_shot,
            parallel_fwi_gradient,
        )

        cluster, client = create_local_cluster(n_workers=2)

        try:
            shape = (31, 31)
            extent = (300.0, 300.0)
            spacing = (10.0, 10.0)

            # Create simple anomaly
            vp_true = np.full(shape, 2.5, dtype=np.float32)
            vp_true[12:18, 12:18] = 3.0
            vp_current = np.full(shape, 2.5, dtype=np.float32)

            # 2 shots
            src_positions = np.array([[100.0, 20.0], [200.0, 20.0]])
            nrec = 11
            rec_coords = np.zeros((nrec, 2))
            rec_coords[:, 0] = np.linspace(20.0, 280.0, nrec)
            rec_coords[:, 1] = 280.0

            nt = 201
            dt = 0.5
            f0 = 0.025

            # Generate observed data for each shot individually
            d_obs_0 = forward_shot(
                0, vp_true, src_positions[0], rec_coords, nt, dt, f0, extent
            )
            d_obs_1 = forward_shot(
                1, vp_true, src_positions[1], rec_coords, nt, dt, f0, extent
            )
            observed_data = [d_obs_0, d_obs_1]

            # Compute individual gradients
            obj_0, grad_0 = fwi_gradient_single_shot(
                vp_current,
                src_positions[0],
                rec_coords,
                d_obs_0,
                shape,
                extent,
                nt,
                dt,
                f0,
            )
            obj_1, grad_1 = fwi_gradient_single_shot(
                vp_current,
                src_positions[1],
                rec_coords,
                d_obs_1,
                shape,
                extent,
                nt,
                dt,
                f0,
            )

            # Compute parallel gradient
            obj_parallel, grad_parallel = parallel_fwi_gradient(
                client=client,
                velocity=vp_current,
                src_positions=src_positions,
                rec_coords=rec_coords,
                observed_data=observed_data,
                shape=shape,
                extent=extent,
                nt=nt,
                dt=dt,
                f0=f0,
            )

            # Should match (within numerical precision)
            expected_obj = obj_0 + obj_1
            expected_grad = grad_0 + grad_1

            assert np.isclose(
                obj_parallel, expected_obj, rtol=1e-5
            ), f"Objectives differ: {obj_parallel} vs {expected_obj}"
            np.testing.assert_allclose(
                grad_parallel, expected_grad, rtol=1e-5, atol=1e-10
            )

        finally:
            client.close()
            cluster.close()


@pytest.mark.slow
class TestScipyIntegration:
    """Test integration with scipy.optimize."""

    def test_create_scipy_loss_function(self):
        """Test scipy-compatible loss function creation."""
        from src.distributed import create_local_cluster, parallel_forward_modeling
        from src.distributed.dask_utils import create_scipy_loss_function

        cluster, client = create_local_cluster(n_workers=2)

        try:
            shape = (21, 21)
            extent = (200.0, 200.0)

            vp_true = np.full(shape, 2.5, dtype=np.float32)
            vp_true[8:12, 8:12] = 3.0

            src_positions = np.array([[100.0, 20.0]])
            nrec = 11
            rec_coords = np.zeros((nrec, 2))
            rec_coords[:, 0] = np.linspace(10.0, 190.0, nrec)
            rec_coords[:, 1] = 180.0

            nt = 201
            dt = 0.5
            f0 = 0.025

            # Generate observed data
            observed_data = parallel_forward_modeling(
                client, vp_true, src_positions, rec_coords, nt, dt, f0, extent
            )

            # Create loss function
            loss_fn = create_scipy_loss_function(
                client, shape, extent, src_positions, rec_coords, observed_data, nt, dt, f0
            )

            # Test with initial model
            vp_init = np.full(shape, 2.5, dtype=np.float32)
            m0 = (1.0 / vp_init**2).flatten()

            objective, gradient = loss_fn(m0)

            assert np.isfinite(objective)
            assert objective > 0  # Should have misfit
            assert gradient.shape == (np.prod(shape),)
            assert gradient.dtype == np.float64  # scipy requires float64
            assert np.all(np.isfinite(gradient))

        finally:
            client.close()
            cluster.close()
