"""Electromagnetics module for FDTD Maxwell equation solvers.

This module provides implementations of the Finite-Difference Time-Domain (FDTD)
method for solving Maxwell's equations using Devito. It includes:

- 1D and 2D Maxwell solvers with staggered grid (Yee scheme)
- Perfectly Matched Layer (PML) absorbing boundary conditions
- Material models for lossy and dispersive media
- Verification utilities including Method of Manufactured Solutions
- Application examples: waveguides and ground-penetrating radar

The FDTD method discretizes Maxwell's equations on a staggered grid where
electric and magnetic field components are offset by half a grid cell in
both space and time. This naturally satisfies the divergence conditions
and provides second-order accuracy.

Example
-------
>>> from src.em import solve_maxwell_1d
>>> result = solve_maxwell_1d(
...     L=1.0,           # Domain length [m]
...     Nx=100,          # Number of grid points
...     T=3e-9,          # Simulation time [s]
...     eps_r=1.0,       # Relative permittivity
...     mu_r=1.0,        # Relative permeability
... )
>>> print(f"Wave speed: {result.c:.2e} m/s")

References
----------
.. [1] K.S. Yee, "Numerical solution of initial boundary value problems
       involving Maxwell's equations in isotropic media," IEEE Trans.
       Antennas Propag., vol. 14, no. 3, pp. 302-307, 1966.

.. [2] A. Taflove and S.C. Hagness, "Computational Electrodynamics: The
       Finite-Difference Time-Domain Method," 3rd ed., Artech House, 2005.
"""

# 1D solver
from src.em.maxwell1D_devito import (
    MaxwellResult1D,
    convergence_test_maxwell_1d,
    exact_plane_wave_1d,
    gaussian_pulse_1d,
    ricker_wavelet,
    solve_maxwell_1d,
)

# 2D solver
from src.em.maxwell2D_devito import (
    MaxwellResult2D,
    convergence_test_maxwell_2d,
    create_pml_profile,
    gaussian_source_2d,
    solve_maxwell_2d,
)

# Units and constants
from src.em.units import (
    EMConstants,
    compute_cfl_dt,
    compute_impedance,
    compute_wave_speed,
    compute_wavelength,
    reflection_coefficient,
    transmission_coefficient,
    verify_units,
)

# Materials
from src.em.materials import (
    AIR,
    COPPER,
    DebyeMaterial,
    DielectricMaterial,
    DRY_SAND,
    GLASS,
    SoilModel,
    VACUUM,
    WATER,
    WET_SAND,
    create_halfspace_model,
    create_layered_model,
)

# Waveguide
from src.em.waveguide import (
    SlabWaveguide,
    WaveguideMode,
    cutoff_wavelength,
    single_mode_condition,
)

# GPR
from src.em.gpr import (
    GPRResult,
    depth_from_travel_time,
    run_gpr_1d,
    two_way_travel_time,
    wavelet_spectrum,
)

# Verification
from src.em.verification import (
    convergence_rate,
    manufactured_solution_1d,
    verify_energy_conservation,
    verify_pec_reflection,
    verify_wave_speed,
)

# Analysis
from src.em.analysis import (
    compute_dispersion_error,
    numerical_dispersion_relation_1d,
    phase_velocity_error_1d,
)

__all__ = [
    "AIR",
    "COPPER",
    "DRY_SAND",
    "GLASS",
    "VACUUM",
    "WATER",
    "WET_SAND",
    "DebyeMaterial",
    # Materials
    "DielectricMaterial",
    # Units
    "EMConstants",
    # GPR
    "GPRResult",
    # 1D solver
    "MaxwellResult1D",
    # 2D solver
    "MaxwellResult2D",
    # Waveguide
    "SlabWaveguide",
    "SoilModel",
    "WaveguideMode",
    "compute_cfl_dt",
    "compute_dispersion_error",
    "compute_impedance",
    "compute_wave_speed",
    "compute_wavelength",
    "convergence_rate",
    "convergence_test_maxwell_1d",
    "convergence_test_maxwell_2d",
    "create_halfspace_model",
    "create_layered_model",
    "create_pml_profile",
    "cutoff_wavelength",
    "depth_from_travel_time",
    "exact_plane_wave_1d",
    "gaussian_pulse_1d",
    "gaussian_source_2d",
    # Verification
    "manufactured_solution_1d",
    # Analysis
    "numerical_dispersion_relation_1d",
    "phase_velocity_error_1d",
    "reflection_coefficient",
    "ricker_wavelet",
    "run_gpr_1d",
    "single_mode_condition",
    "solve_maxwell_1d",
    "solve_maxwell_2d",
    "transmission_coefficient",
    "two_way_travel_time",
    "verify_energy_conservation",
    "verify_pec_reflection",
    "verify_units",
    "verify_wave_speed",
    "wavelet_spectrum",
]
