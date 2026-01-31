"""Computational Electromagnetics - FDTD Maxwell's Equations Solver.

This module provides Devito-based solvers for Maxwell's equations using
the Finite-Difference Time-Domain (FDTD) method with the Yee grid.

Maxwell's equations in differential form:
    curl(E) = -μ * dH/dt       (Faraday's law)
    curl(H) = ε * dE/dt + J    (Ampère's law)

where:
    - E: electric field [V/m]
    - H: magnetic field [A/m]
    - ε: permittivity [F/m]
    - μ: permeability [H/m]
    - J: current density [A/m²]

The Yee grid staggers E and H fields in both space and time:
    - E fields at integer time steps, H fields at half-integer steps
    - E components at cell edges, H components at cell faces

Key features:
    - 1D, 2D, and 3D FDTD solvers
    - Perfectly Matched Layer (PML) absorbing boundaries
    - Multiple source types (Gaussian pulse, sinusoidal, plane wave)
    - TMz and TEz polarization modes for 2D
    - Analytical solutions for verification

References:
    - Yee, K.S. (1966). "Numerical solution of initial boundary value
      problems involving Maxwell's equations in isotropic media."
      IEEE Trans. Antennas Propagat., 14(3), 302-307.
    - Taflove, A. & Hagness, S.C. (2005). "Computational Electrodynamics:
      The Finite-Difference Time-Domain Method." Artech House.
"""

from src.maxwell.analytical import (
    cavity_resonant_frequencies,
    exact_plane_wave_1d,
    exact_plane_wave_2d,
    waveguide_cutoff_frequency,
)
from src.maxwell.maxwell_devito import (
    MaxwellResult,
    MaxwellResult2D,
    compute_energy,
    compute_energy_2d,
    solve_maxwell_1d,
    solve_maxwell_2d,
)
from src.maxwell.pml import (
    create_cpml_coefficients,
    create_pml_sigma,
)
from src.maxwell.sources import (
    gaussian_modulated_source,
    gaussian_pulse_em,
    sinusoidal_source,
)

__all__ = [
    "MaxwellResult",
    "MaxwellResult2D",
    "cavity_resonant_frequencies",
    "compute_energy",
    "compute_energy_2d",
    "create_cpml_coefficients",
    "create_pml_sigma",
    "exact_plane_wave_1d",
    "exact_plane_wave_2d",
    "gaussian_modulated_source",
    "gaussian_pulse_em",
    "sinusoidal_source",
    "solve_maxwell_1d",
    "solve_maxwell_2d",
    "waveguide_cutoff_frequency",
]
