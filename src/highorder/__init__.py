"""High-Order Methods module for Finite Difference Computing with PDEs.

This module provides dispersion analysis tools and Dispersion-Relation-Preserving
(DRP) finite difference schemes for wave equation solvers.

Submodules
----------
dispersion
    Dispersion analysis utilities including Fornberg weights, dispersion
    ratio calculations, and CFL condition computations.

drp_devito
    DRP wave equation solvers using Devito, with pre-computed and custom
    optimized coefficients.

Key Functions
-------------
fornberg_weights
    Compute Fornberg (Taylor-optimal) FD weights.
drp_coefficients
    Get pre-computed DRP-optimized coefficients.
compute_drp_weights
    Compute custom DRP coefficients via optimization.
solve_wave_drp
    Solve 2D wave equation with DRP scheme.
dispersion_ratio
    Compute velocity error ratio for a FD scheme.

Examples
--------
Basic usage with pre-computed DRP coefficients:

>>> from src.highorder import drp_coefficients, solve_wave_drp
>>> weights = drp_coefficients(M=4)  # 9-point DRP stencil
>>> result = solve_wave_drp(
...     extent=(2000., 2000.),
...     shape=(201, 201),
...     velocity=1500.,
...     use_drp=True
... )

Dispersion analysis:

>>> from src.highorder import fornberg_weights, dispersion_ratio
>>> weights = fornberg_weights(M=4)
>>> ratio = dispersion_ratio(weights, h=10.0, dt=0.001, v=1500.0, k=0.1)
>>> print(f"Velocity ratio: {ratio:.4f}")
"""

from src.highorder.dispersion import (
    analytical_dispersion_relation,
    cfl_number,
    critical_dt,
    dispersion_difference,
    dispersion_error,
    dispersion_ratio,
    fornberg_weights,
    max_frequency_ricker,
    numerical_dispersion_relation,
    nyquist_spacing,
    ricker_wavelet,
)
from src.highorder.drp_devito import (
    DRP_COEFFICIENTS,
    FORNBERG_COEFFICIENTS,
    WaveDRPResult,
    compare_dispersion_wavefields,
    compute_drp_weights,
    drp_coefficients,
    drp_objective_tamwebb,
    solve_wave_drp,
    solve_wave_drp_1d,
    to_full_stencil,
)

__all__ = [
    "DRP_COEFFICIENTS",
    "FORNBERG_COEFFICIENTS",
    "WaveDRPResult",
    "analytical_dispersion_relation",
    "cfl_number",
    "compare_dispersion_wavefields",
    "compute_drp_weights",
    "critical_dt",
    "dispersion_difference",
    "dispersion_error",
    "dispersion_ratio",
    "drp_coefficients",
    "drp_objective_tamwebb",
    "fornberg_weights",
    "max_frequency_ricker",
    "numerical_dispersion_relation",
    "nyquist_spacing",
    "ricker_wavelet",
    "solve_wave_drp",
    "solve_wave_drp_1d",
    "to_full_stencil",
]
