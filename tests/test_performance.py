"""Tests for performance benchmarking utilities."""

import numpy as np
import pytest

# Check if Devito is available
try:
    import devito  # noqa: F401
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


class TestBenchmarkUtilities:
    """Tests for benchmark utility functions (no Devito required)."""

    def test_import(self):
        """Test that the module imports correctly."""
        from src.performance import (
            BenchmarkResult,
            estimate_stencil_flops,
            roofline_analysis,
        )
        assert BenchmarkResult is not None
        assert estimate_stencil_flops is not None
        assert roofline_analysis is not None

    def test_estimate_stencil_flops_order2(self):
        """Test FLOPS estimation for 2nd order stencil."""
        from src.performance.benchmark import estimate_stencil_flops

        # 2nd order stencil in 3D
        flops = estimate_stencil_flops(space_order=2, ndim=3)

        # Each 2nd derivative: (2+1) muls + 2 adds = 5 ops
        # Laplacian: 3 dims * 5 + 2 additions = 17 ops
        # Time update: ~4 ops
        # Total: ~21 ops
        assert flops > 15
        assert flops < 30

    def test_estimate_stencil_flops_order4(self):
        """Test FLOPS estimation for 4th order stencil."""
        from src.performance.benchmark import estimate_stencil_flops

        flops_o2 = estimate_stencil_flops(space_order=2, ndim=3)
        flops_o4 = estimate_stencil_flops(space_order=4, ndim=3)

        # Higher order should have more FLOPS
        assert flops_o4 > flops_o2

    def test_estimate_stencil_flops_ndim(self):
        """Test FLOPS scales with dimensions."""
        from src.performance.benchmark import estimate_stencil_flops

        flops_2d = estimate_stencil_flops(space_order=4, ndim=2)
        flops_3d = estimate_stencil_flops(space_order=4, ndim=3)

        # 3D should have more FLOPS than 2D
        assert flops_3d > flops_2d

    def test_estimate_memory_traffic(self):
        """Test memory traffic estimation."""
        from src.performance.benchmark import estimate_memory_traffic

        grid_shape = (100, 100, 100)
        bytes_traffic = estimate_memory_traffic(grid_shape, dtype_size=4)

        # Should access 3 arrays of grid_size * 4 bytes
        expected = 3 * 100**3 * 4
        assert bytes_traffic == expected

    def test_benchmark_result_dataclass(self):
        """Test BenchmarkResult dataclass."""
        from src.performance import BenchmarkResult

        result = BenchmarkResult(
            grid_shape=(100, 100, 100),
            time_steps=50,
            space_order=4,
            elapsed_time=1.5,
            gflops=100.0,
            bandwidth_gb_s=50.0,
            arithmetic_intensity=2.0,
            points_per_second=1e8,
        )

        assert result.grid_shape == (100, 100, 100)
        assert result.gflops == 100.0
        assert "100.00 GFLOPS" in result.summary()

    def test_roofline_analysis_memory_bound(self):
        """Test roofline analysis for memory-bound case."""
        from src.performance import roofline_analysis

        result = roofline_analysis(
            gflops=50.0,
            bandwidth=40.0,
            arithmetic_intensity=1.0,  # Low AI = memory bound
            peak_gflops=500.0,
            peak_bandwidth=100.0,
        )

        assert result['is_memory_bound'] is True
        assert result['roofline_limit'] == pytest.approx(100.0)  # 100 * 1.0
        assert result['efficiency_percent'] == pytest.approx(50.0)

    def test_roofline_analysis_compute_bound(self):
        """Test roofline analysis for compute-bound case."""
        from src.performance import roofline_analysis

        result = roofline_analysis(
            gflops=400.0,
            bandwidth=40.0,
            arithmetic_intensity=10.0,  # High AI = compute bound
            peak_gflops=500.0,
            peak_bandwidth=100.0,
        )

        assert result['is_memory_bound'] is False
        assert result['roofline_limit'] == pytest.approx(500.0)  # Peak FLOPS
        assert result['efficiency_percent'] == pytest.approx(80.0)

    def test_roofline_ridge_point(self):
        """Test roofline ridge point calculation."""
        from src.performance import roofline_analysis

        result = roofline_analysis(
            gflops=100.0,
            bandwidth=50.0,
            arithmetic_intensity=2.0,
            peak_gflops=500.0,
            peak_bandwidth=100.0,
        )

        # Ridge point = peak_gflops / peak_bandwidth
        assert result['ridge_point'] == pytest.approx(5.0)


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")
class TestBenchmarkWithDevito:
    """Tests that require Devito."""

    def test_benchmark_operator_runs(self):
        """Test that benchmark_operator executes successfully."""
        from src.performance import benchmark_operator

        # Small grid for fast test
        result = benchmark_operator(
            grid_shape=(20, 20, 20),
            time_steps=5,
            space_order=2,
            warmup_steps=1,
        )

        assert result.grid_shape == (20, 20, 20)
        assert result.time_steps == 5
        assert result.elapsed_time > 0
        assert result.gflops > 0
        assert result.bandwidth_gb_s > 0

    def test_benchmark_operator_space_order(self):
        """Test benchmark with different space orders."""
        from src.performance import benchmark_operator

        result_o2 = benchmark_operator(
            grid_shape=(20, 20, 20),
            time_steps=5,
            space_order=2,
            warmup_steps=0,
        )

        result_o4 = benchmark_operator(
            grid_shape=(20, 20, 20),
            time_steps=5,
            space_order=4,
            warmup_steps=0,
        )

        # Both should complete
        assert result_o2.elapsed_time > 0
        assert result_o4.elapsed_time > 0

        # Higher order should report more FLOPS (but may be slower)
        # Note: timing can vary, so we just check it runs

    def test_measure_performance(self):
        """Test simplified measure_performance interface."""
        from src.performance import measure_performance

        result = measure_performance(nx=20, nt=5, space_order=2)

        assert 'grid_size' in result
        assert 'elapsed' in result
        assert 'gflops' in result
        assert 'bandwidth_gb_s' in result
        assert result['grid_size'] == 20

    def test_compare_platforms_cpu(self):
        """Test platform comparison (CPU only)."""
        from src.performance import compare_platforms

        results = compare_platforms(
            grid_shape=(20, 20, 20),
            time_steps=5,
            space_order=2,
            platforms=['cpu'],
        )

        assert 'cpu' in results
        assert results['cpu'].elapsed_time > 0

    def test_benchmark_result_summary(self):
        """Test that summary is formatted correctly."""
        from src.performance import benchmark_operator

        result = benchmark_operator(
            grid_shape=(20, 20, 20),
            time_steps=5,
            space_order=2,
        )

        summary = result.summary()
        assert "Grid shape" in summary
        assert "GFLOPS" in summary
        assert "GB/s" in summary


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")
class TestOperatorOptimizations:
    """Tests for verifying optimization options work."""

    def test_operator_with_openmp(self):
        """Test operator creation with OpenMP enabled."""
        from devito import Eq, Grid, Operator, TimeFunction

        grid = Grid(shape=(20, 20, 20))
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

        # Should not raise
        op = Operator(
            [Eq(u.forward, 2*u - u.backward + u.laplace)],
            opt=('advanced', {'openmp': True})
        )

        assert op is not None

    def test_operator_noop(self):
        """Test operator with noop optimization."""
        from devito import Eq, Grid, Operator, TimeFunction

        grid = Grid(shape=(20, 20, 20))
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

        op = Operator(
            [Eq(u.forward, 2*u - u.backward + u.laplace)],
            opt='noop'
        )

        # Should still work, just unoptimized
        u.data[:] = np.random.rand(*u.data.shape).astype(np.float32)
        op.apply(time_M=2, dt=0.001)

    def test_operator_print_code(self):
        """Test that generated code can be printed."""
        from devito import Eq, Grid, Operator, TimeFunction

        grid = Grid(shape=(20, 20, 20))
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

        op = Operator([Eq(u.forward, 2*u - u.backward + u.laplace)])

        # print(op) returns the generated C code
        code = str(op)
        assert 'for' in code.lower()  # Should have loops

    def test_autotuning_basic(self):
        """Test that autotuning runs without errors."""
        from devito import Eq, Grid, Operator, TimeFunction

        grid = Grid(shape=(30, 30, 30))
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

        # Use 'advanced' without OpenMP for portability
        op = Operator(
            [Eq(u.forward, 2*u - u.backward + u.laplace)],
            opt='advanced'
        )

        u.data[:] = np.random.rand(*u.data.shape).astype(np.float32)

        # Basic autotuning should work
        summary = op.apply(time_M=5, dt=0.001, autotune='basic')
        assert summary is not None


@pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")
class TestGPUSupport:
    """Tests for GPU-related functionality.

    These tests verify that GPU-related code paths exist and are syntactically
    correct, but skip actual GPU execution if no GPU is available.
    """

    def test_gpu_operator_creation(self):
        """Test that GPU operator can be created (doesn't require GPU to run)."""
        from devito import Eq, Grid, Operator, TimeFunction

        grid = Grid(shape=(20, 20, 20))
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

        # Creating with platform='nvidiaX' should work syntactically
        # but may fail at apply() time if no GPU
        try:
            op = Operator(
                [Eq(u.forward, 2*u - u.backward + u.laplace)],
                platform='nvidiaX'
            )
            # If creation succeeds, verify it's an Operator
            assert op is not None
        except Exception:
            # Some Devito installations may not support GPU creation
            pytest.skip("GPU operator creation not supported")

    def test_gpu_fit_option(self):
        """Test that gpu-fit option can be specified."""
        from devito import Eq, Grid, Operator, TimeFunction

        grid = Grid(shape=(20, 20, 20))
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2, save=10)

        try:
            op = Operator(
                [Eq(u.forward, 2*u - u.backward + u.laplace)],
                platform='nvidiaX',
                opt=('advanced', {'gpu-fit': u})
            )
            assert op is not None
        except Exception:
            pytest.skip("GPU operator with gpu-fit not supported")
