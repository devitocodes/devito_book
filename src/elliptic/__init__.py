"""Elliptic PDE solvers using Devito DSL.

This module provides solvers for steady-state elliptic PDEs
using Devito's symbolic finite difference framework.

Elliptic equations have no time derivatives and describe
equilibrium or steady-state problems. The two main equations are:

1. Laplace equation: laplace(p) = 0
   - Describes steady-state potential problems
   - Solution determined entirely by boundary conditions

2. Poisson equation: laplace(p) = b
   - Laplace equation with source term
   - Common in electrostatics, gravity, heat conduction

Both solvers use iterative methods (Jacobi iteration) with
pseudo-timestepping to converge to the steady-state solution.

Examples
--------
Solve the Laplace equation on [0, 2] x [0, 1]:

    >>> from src.elliptic import solve_laplace_2d
    >>> result = solve_laplace_2d(
    ...     Lx=2.0, Ly=1.0,
    ...     Nx=31, Ny=31,
    ...     bc_left=0.0,
    ...     bc_right=lambda y: y,
    ...     bc_bottom='neumann',
    ...     bc_top='neumann',
    ...     tol=1e-4,
    ... )
    >>> print(f"Converged in {result.iterations} iterations")

Solve the Poisson equation with point sources:

    >>> from src.elliptic import solve_poisson_2d
    >>> result = solve_poisson_2d(
    ...     Lx=2.0, Ly=1.0,
    ...     Nx=50, Ny=50,
    ...     source_points=[(0.5, 0.25, 100), (1.5, 0.75, -100)],
    ...     n_iterations=100,
    ... )
"""

from src.elliptic.laplace_devito import (
    LaplaceResult,
    convergence_test_laplace_2d,
    exact_laplace_linear,
    solve_laplace_2d,
    solve_laplace_2d_with_copy,
)
from src.elliptic.poisson_devito import (
    PoissonResult,
    convergence_test_poisson_2d,
    create_gaussian_source,
    create_point_source,
    exact_poisson_point_source,
    solve_poisson_2d,
    solve_poisson_2d_timefunction,
    solve_poisson_2d_with_copy,
)

__all__ = [
    "LaplaceResult",
    "PoissonResult",
    "convergence_test_laplace_2d",
    "convergence_test_poisson_2d",
    "create_gaussian_source",
    "create_point_source",
    "exact_laplace_linear",
    "exact_poisson_point_source",
    "solve_laplace_2d",
    "solve_laplace_2d_with_copy",
    "solve_poisson_2d",
    "solve_poisson_2d_timefunction",
    "solve_poisson_2d_with_copy",
]
