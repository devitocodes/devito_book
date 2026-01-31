"""Systems of PDEs solvers using Devito DSL.

This module provides solvers for coupled systems of PDEs:
- 2D Shallow Water Equations for tsunami modeling
- 2D Elastic Wave Equations for seismic wave propagation
- 2D Viscoacoustic Wave Equations with attenuation (SLS, Kelvin-Voigt, Maxwell)
- 3D Viscoelastic Wave Equations with P and S wave attenuation
"""

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
from src.systems.swe_devito import (
    SWEResult,
    create_swe_operator,
    solve_swe,
)
from src.systems.viscoacoustic_devito import (
    ViscoacousticResult,
    compute_sls_relaxation_parameters,
    create_damping_field,
    solve_viscoacoustic_kv,
    solve_viscoacoustic_maxwell,
    solve_viscoacoustic_sls,
)
from src.systems.viscoelastic_devito import (
    ViscoelasticResult,
    compute_viscoelastic_relaxation_parameters,
    create_damping_field_3d,
    create_layered_model_3d,
    solve_viscoelastic_3d,
)

__all__ = [
    "ElasticResult",
    "SWEResult",
    "ViscoacousticResult",
    "ViscoelasticResult",
    "compute_lame_parameters",
    "compute_sls_relaxation_parameters",
    "compute_viscoelastic_relaxation_parameters",
    "compute_wave_velocities",
    "create_damping_field",
    "create_damping_field_3d",
    "create_elastic_operator",
    "create_layered_model",
    "create_layered_model_3d",
    "create_swe_operator",
    "ricker_wavelet",
    "solve_elastic_2d",
    "solve_elastic_2d_varying",
    "solve_swe",
    "solve_viscoacoustic_kv",
    "solve_viscoacoustic_maxwell",
    "solve_viscoacoustic_sls",
    "solve_viscoelastic_3d",
]
