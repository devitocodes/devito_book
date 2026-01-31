"""Performance benchmarking utilities for Devito PDE solvers.

This module provides tools for measuring and analyzing the performance
of Devito operators, including timing, FLOPS estimation, and bandwidth
measurement.

Usage:
    from src.performance import (
        benchmark_operator,
        measure_performance,
        roofline_analysis,
        compare_platforms,
    )
"""

from src.performance.benchmark import (
    BenchmarkResult,
    benchmark_operator,
    compare_platforms,
    estimate_stencil_flops,
    measure_performance,
    roofline_analysis,
)

__all__ = [
    "BenchmarkResult",
    "benchmark_operator",
    "compare_platforms",
    "estimate_stencil_flops",
    "measure_performance",
    "roofline_analysis",
]
