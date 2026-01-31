"""High-Order Methods module for Finite Difference Computing with PDEs.

This module provides dispersion analysis tools, Dispersion-Relation-Preserving
(DRP) finite difference schemes, ADER time integration, and staggered grid
solvers for wave equations.

Submodules
----------
dispersion
    Dispersion analysis utilities including Fornberg weights, dispersion
    ratio calculations, and CFL condition computations.

drp_devito
    DRP wave equation solvers using Devito, with pre-computed and custom
    optimized coefficients.

ader_devito
    ADER (Arbitrary-order-accuracy via DERivatives) time integration for
    the acoustic wave equation. Enables high-order temporal accuracy and
    larger CFL numbers than standard leapfrog schemes.

staggered_devito
    Staggered grid acoustic wave solvers using the velocity-pressure
    formulation. Supports 2nd and 4th order spatial discretization.

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
solve_ader_2d
    Solve 2D acoustic wave equation with ADER time integration.
solve_staggered_acoustic_2d
    Solve 2D acoustic wave equation with staggered grid scheme.
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

ADER solver with high CFL number:

>>> from src.highorder import solve_ader_2d
>>> result = solve_ader_2d(
...     extent=(1000., 1000.),
...     shape=(101, 101),
...     c_value=1.5,
...     courant=0.85,  # Higher CFL than leapfrog
... )

Staggered grid solver:

>>> from src.highorder import solve_staggered_acoustic_2d
>>> result = solve_staggered_acoustic_2d(
...     extent=(2000., 2000.),
...     shape=(81, 81),
...     velocity=4.0,
...     space_order=4,
... )

Dispersion analysis:

>>> from src.highorder import fornberg_weights, dispersion_ratio
>>> weights = fornberg_weights(M=4)
>>> ratio = dispersion_ratio(weights, h=10.0, dt=0.001, v=1500.0, k=0.1)
>>> print(f"Velocity ratio: {ratio:.4f}")
"""

from src.highorder.ader_devito import (
    ADERResult,
    biharmonic,
    compare_ader_vs_staggered,
    graddiv,
    gradlap,
    gradlapdiv,
    lapdiv,
    solve_ader_2d,
)
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
from src.highorder.staggered_devito import (
    StaggeredResult,
    compare_space_orders,
    convergence_test_staggered,
    dgauss_wavelet,
    solve_staggered_acoustic_2d,
)

__all__ = [
    "DRP_COEFFICIENTS",
    "FORNBERG_COEFFICIENTS",
    "ADERResult",
    "StaggeredResult",
    "WaveDRPResult",
    "analytical_dispersion_relation",
    "biharmonic",
    "cfl_number",
    "compare_ader_vs_staggered",
    "compare_dispersion_wavefields",
    "compare_space_orders",
    "compute_drp_weights",
    "convergence_test_staggered",
    "critical_dt",
    "dgauss_wavelet",
    "dispersion_difference",
    "dispersion_error",
    "dispersion_ratio",
    "drp_coefficients",
    "drp_objective_tamwebb",
    "fornberg_weights",
    "graddiv",
    "gradlap",
    "gradlapdiv",
    "lapdiv",
    "max_frequency_ricker",
    "numerical_dispersion_relation",
    "nyquist_spacing",
    "ricker_wavelet",
    "solve_ader_2d",
    "solve_staggered_acoustic_2d",
    "solve_wave_drp",
    "solve_wave_drp_1d",
    "to_full_stencil",
]
