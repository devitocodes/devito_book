"""CFD Solvers - Navier-Stokes equations using Devito.

This module provides solvers for incompressible fluid dynamics,
including the classic lid-driven cavity flow benchmark problem.

The primary solver implements the fractional step (projection) method:
1. Predict intermediate velocities (ignoring pressure)
2. Solve pressure Poisson equation to enforce incompressibility
3. Correct velocities using pressure gradient

Available functions:
- solve_cavity_2d: Complete lid-driven cavity solver
- pressure_poisson_iteration: Iterative pressure solver
- apply_velocity_bcs: Apply velocity boundary conditions
- compute_streamfunction: Post-processing utility
"""

from src.cfd.navier_stokes_devito import (
    CavityResult,
    apply_velocity_bcs,
    compute_streamfunction,
    ghia_benchmark_data,
    pressure_poisson_iteration,
    solve_cavity_2d,
)

__all__ = [
    "CavityResult",
    "apply_velocity_bcs",
    "compute_streamfunction",
    "ghia_benchmark_data",
    "pressure_poisson_iteration",
    "solve_cavity_2d",
]
