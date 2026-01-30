"""Tests for memory management and snapshotting utilities.

These tests verify that the memory estimation and snapshotting
utilities work correctly for wave propagation simulations.
"""

import importlib.util
import os
import tempfile

import numpy as np
import pytest

# Check if Devito is available
DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None

# Check if h5py is available
H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None


class TestMemoryEstimation:
    """Tests for memory estimation functions (no Devito required)."""

    def test_import(self):
        """Verify module can be imported."""
        from src.memory import estimate_wavefield_memory
        assert estimate_wavefield_memory is not None

    def test_2d_memory_estimate(self):
        """Test memory estimation for 2D grid."""
        from src.memory import estimate_wavefield_memory

        shape = (101, 101)
        nt = 500
        mem = estimate_wavefield_memory(shape, nt)

        assert mem['grid_points'] == 101 * 101
        assert mem['dimensions'] == 2
        assert mem['time_steps'] == 500

        # Per snapshot should be 101*101*4 bytes
        expected_per_snap = 101 * 101 * 4
        assert mem['per_snapshot_bytes'] == expected_per_snap

        # Full storage
        expected_full = nt * expected_per_snap
        assert mem['full_storage_bytes'] == expected_full

    def test_3d_memory_estimate(self):
        """Test memory estimation for 3D grid."""
        from src.memory import estimate_wavefield_memory

        shape = (101, 101, 101)
        nt = 1000
        mem = estimate_wavefield_memory(shape, nt)

        assert mem['grid_points'] == 101**3
        assert mem['dimensions'] == 3

        # Full storage should be ~4 GB
        assert mem['full_storage_GB'] > 3.5
        assert mem['full_storage_GB'] < 4.5

    def test_snapshot_estimates(self):
        """Test that snapshot estimates are computed correctly."""
        from src.memory import estimate_wavefield_memory

        shape = (100, 100)
        nt = 1000
        mem = estimate_wavefield_memory(shape, nt)

        # Factor 10 should give 100 snapshots
        assert mem['snapshot_factor_10_nsnaps'] == 100

        # Factor 50 should give 20 snapshots
        assert mem['snapshot_factor_50_nsnaps'] == 20

        # Snapshot memory should be proportionally less
        full_gb = mem['full_storage_GB']
        snap10_gb = mem['snapshot_factor_10_GB']
        assert abs(snap10_gb - full_gb / 10) < 0.01

    def test_rolling_buffer_size(self):
        """Test rolling buffer estimation with different time orders."""
        from src.memory import estimate_wavefield_memory

        shape = (100, 100)
        nt = 1000

        # Time order 2 -> 3 time levels
        mem_order2 = estimate_wavefield_memory(shape, nt, time_order=2)
        expected_buffer = 3 * 100 * 100 * 4
        assert mem_order2['rolling_buffer_bytes'] == expected_buffer

        # Time order 4 -> 5 time levels
        mem_order4 = estimate_wavefield_memory(shape, nt, time_order=4)
        expected_buffer = 5 * 100 * 100 * 4
        assert mem_order4['rolling_buffer_bytes'] == expected_buffer

    def test_dtype_affects_memory(self):
        """Test that dtype bytes affects memory estimates."""
        from src.memory import estimate_wavefield_memory

        shape = (100, 100)
        nt = 100

        mem_float32 = estimate_wavefield_memory(shape, nt, dtype_bytes=4)
        mem_float64 = estimate_wavefield_memory(shape, nt, dtype_bytes=8)

        # Float64 should be exactly 2x float32
        assert mem_float64['full_storage_bytes'] == 2 * mem_float32['full_storage_bytes']


class TestWavefieldIO:
    """Tests for wavefield I/O functions (no Devito required)."""

    def test_save_load_binary(self):
        """Test saving and loading raw binary files."""
        from src.memory import load_wavefield, save_wavefield

        shape = (50, 50, 50)
        data = np.random.randn(*shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            filename = f.name

        try:
            stats = save_wavefield(data, filename)
            assert stats['shape'] == shape
            assert os.path.exists(filename)

            loaded = load_wavefield(filename, shape=shape)
            np.testing.assert_allclose(data, loaded)
        finally:
            os.remove(filename)

    def test_save_load_compressed(self):
        """Test saving and loading compressed files."""
        from src.memory import load_wavefield, save_wavefield

        shape = (50, 50, 50)
        data = np.random.randn(*shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filename = f.name

        try:
            stats = save_wavefield(data, filename, compressed=True)
            assert stats['compression_ratio'] >= 1.0  # Should have some compression
            assert os.path.exists(filename)

            loaded = load_wavefield(filename)
            np.testing.assert_allclose(data, loaded)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_compression_ratio(self):
        """Test that compression achieves meaningful ratio."""
        from src.memory import save_wavefield

        # Highly compressible data (sparse)
        shape = (100, 100, 100)
        data = np.zeros(shape, dtype=np.float32)
        data[40:60, 40:60, 40:60] = 1.0  # Small non-zero region

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filename = f.name

        try:
            stats = save_wavefield(data, filename, compressed=True)
            # Sparse data should compress well
            assert stats['compression_ratio'] > 2.0
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_save_load_hdf5(self):
        """Test HDF5 I/O with compression."""
        from src.memory import load_wavefield_hdf5, save_wavefield_hdf5

        shape = (50, 50, 50)
        data = np.random.randn(*shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filename = f.name

        try:
            stats = save_wavefield_hdf5(data, filename)
            assert stats['compression'] == 'gzip'
            assert os.path.exists(filename)

            loaded = load_wavefield_hdf5(filename)
            np.testing.assert_allclose(data, loaded)
        finally:
            os.remove(filename)

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_hdf5_partial_load(self):
        """Test partial loading from HDF5 with slicing."""
        from src.memory import load_wavefield_hdf5, save_wavefield_hdf5

        shape = (100, 50, 50)
        data = np.random.randn(*shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filename = f.name

        try:
            save_wavefield_hdf5(data, filename)

            # Load first 10 time steps
            partial = load_wavefield_hdf5(
                filename,
                slices=(slice(0, 10), slice(None), slice(None))
            )
            assert partial.shape == (10, 50, 50)
            np.testing.assert_allclose(data[:10], partial)
        finally:
            os.remove(filename)


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")
@pytest.mark.devito
class TestSnapshotTimeFunction:
    """Tests for snapshotting with Devito."""

    def test_create_snapshot_timefunction(self):
        """Test creation of snapshotted TimeFunction."""
        from src.memory import create_snapshot_timefunction

        shape = (51, 51)
        extent = (500., 500.)
        nt = 100
        snapshot_factor = 10

        grid, usave = create_snapshot_timefunction(
            shape=shape,
            extent=extent,
            nt=nt,
            snapshot_factor=snapshot_factor
        )

        # Grid should have correct shape
        assert grid.shape == shape

        # usave should have correct number of snapshots
        nsnaps = nt // snapshot_factor
        assert usave.data.shape[0] == nsnaps
        assert usave.data.shape[1:] == shape

    def test_snapshot_dimensions(self):
        """Test snapshot array dimensions for different factors."""
        from src.memory import create_snapshot_timefunction

        shape = (31, 31)
        nt = 500

        for factor in [5, 10, 25, 50]:
            _, usave = create_snapshot_timefunction(
                shape=shape,
                extent=(100., 100.),
                nt=nt,
                snapshot_factor=factor
            )

            expected_nsnaps = nt // factor
            assert usave.data.shape[0] == expected_nsnaps


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")
@pytest.mark.devito
class TestWavePropagationWithSnapshotting:
    """Tests for wave propagation with snapshotting."""

    def test_basic_propagation(self):
        """Test basic wave propagation with snapshotting."""
        from src.memory import wave_propagation_with_snapshotting

        result = wave_propagation_with_snapshotting(
            shape=(51, 51),
            extent=(500., 500.),
            nt=100,
            snapshot_factor=10
        )

        assert result.snapshots is not None
        assert len(result.time_indices) == 10
        assert result.memory_savings > 1.0  # Should save memory

    def test_snapshot_count(self):
        """Test that correct number of snapshots is saved."""
        from src.memory import wave_propagation_with_snapshotting

        nt = 200
        for factor in [5, 10, 20]:
            result = wave_propagation_with_snapshotting(
                shape=(31, 31),
                extent=(100., 100.),
                nt=nt,
                snapshot_factor=factor
            )

            expected_nsnaps = nt // factor
            assert len(result.time_indices) == expected_nsnaps
            assert result.snapshots.shape[0] == expected_nsnaps

    def test_memory_savings_factor(self):
        """Test that memory savings are computed correctly."""
        from src.memory import wave_propagation_with_snapshotting

        # Higher snapshot factor should give more savings
        result_10 = wave_propagation_with_snapshotting(
            shape=(51, 51),
            extent=(500., 500.),
            nt=500,
            snapshot_factor=10
        )

        result_50 = wave_propagation_with_snapshotting(
            shape=(51, 51),
            extent=(500., 500.),
            nt=500,
            snapshot_factor=50
        )

        # Factor 50 should give more savings than factor 10
        assert result_50.memory_savings > result_10.memory_savings

    def test_gaussian_initial_condition(self):
        """Test Gaussian initial condition."""
        from src.memory import wave_propagation_with_snapshotting

        result = wave_propagation_with_snapshotting(
            shape=(51, 51),
            extent=(500., 500.),
            nt=100,
            snapshot_factor=10,
            initial_condition='gaussian'
        )

        # Later snapshots should have non-zero values from wave propagation
        # (check middle snapshot after wave has propagated)
        assert np.max(np.abs(result.snapshots[len(result.snapshots) // 2])) > 0

    def test_plane_initial_condition(self):
        """Test plane wave initial condition."""
        from src.memory import wave_propagation_with_snapshotting

        result = wave_propagation_with_snapshotting(
            shape=(51, 51),
            extent=(500., 500.),
            nt=100,
            snapshot_factor=10,
            initial_condition='plane'
        )

        # Later snapshots should have non-zero values from wave propagation
        assert np.max(np.abs(result.snapshots[len(result.snapshots) // 2])) > 0

    def test_time_indices_correct(self):
        """Test that time indices are correctly computed."""
        from src.memory import wave_propagation_with_snapshotting

        nt = 100
        factor = 20

        result = wave_propagation_with_snapshotting(
            shape=(31, 31),
            extent=(100., 100.),
            nt=nt,
            snapshot_factor=factor
        )

        expected_indices = np.arange(0, nt, factor)
        np.testing.assert_array_equal(result.time_indices, expected_indices)

    def test_wavefield_evolves(self):
        """Test that wavefield evolves over time."""
        from src.memory import wave_propagation_with_snapshotting

        result = wave_propagation_with_snapshotting(
            shape=(51, 51),
            extent=(500., 500.),
            vel=2.0,
            nt=200,
            dt=1.0,
            snapshot_factor=20,
            initial_condition='gaussian'
        )

        # Snapshots should differ as wave propagates
        # (initial Gaussian spreads out)
        first_snap = result.snapshots[0]
        last_snap = result.snapshots[-1]

        # Calculate difference
        diff = np.mean(np.abs(first_snap - last_snap))
        assert diff > 0.01  # Should be noticeably different


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")
@pytest.mark.devito
class TestSnapshotResult:
    """Tests for SnapshotResult dataclass."""

    def test_result_fields(self):
        """Test that SnapshotResult has all expected fields."""
        from src.memory import wave_propagation_with_snapshotting

        result = wave_propagation_with_snapshotting(
            shape=(31, 31),
            extent=(100., 100.),
            nt=100,
            snapshot_factor=10
        )

        assert hasattr(result, 'snapshots')
        assert hasattr(result, 'time_indices')
        assert hasattr(result, 'memory_savings')
        assert hasattr(result, 'snapshot_factor')
        assert hasattr(result, 'grid_shape')

    def test_result_consistency(self):
        """Test that result fields are consistent."""
        from src.memory import wave_propagation_with_snapshotting

        shape = (41, 41)
        factor = 10

        result = wave_propagation_with_snapshotting(
            shape=shape,
            extent=(200., 200.),
            nt=100,
            snapshot_factor=factor
        )

        assert result.snapshot_factor == factor
        assert result.grid_shape == shape
        assert result.snapshots.shape[1:] == shape
