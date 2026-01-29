"""Systems of PDEs solvers using Devito DSL.

This module provides solvers for coupled systems of PDEs,
including the 2D Shallow Water Equations for tsunami modeling
and the 2D Elastic Wave Equations for seismic wave propagation.
"""

from src.systems.swe_devito import (
    SWEResult,
    create_swe_operator,
    solve_swe,
)
from src.systems.elastic_devito import (
    ElasticResult,
    compute_lame_parameters,
    compute_wave_velocities,
    create_elastic_operator,
    create_layered_model,
    ricker_wavelet,
    solve_elastic_2d,
    solve_elastic_2d_varying,
)

__all__ = [
    "ElasticResult",
    "SWEResult",
    "compute_lame_parameters",
    "compute_wave_velocities",
    "create_elastic_operator",
    "create_layered_model",
    "create_swe_operator",
    "ricker_wavelet",
    "solve_elastic_2d",
    "solve_elastic_2d_varying",
    "solve_swe",
]
