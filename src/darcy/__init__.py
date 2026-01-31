"""Darcy flow solvers for porous media using Devito DSL.

This module provides solvers for single-phase fluid flow through
porous media using Devito's symbolic finite difference framework.

The primary equation solved is the Darcy flow equation:

    -div(K * grad(p)) = f

where:
    - p is the pressure field
    - K is the permeability field (can be heterogeneous)
    - f is the source/sink term

Darcy's law relates velocity to pressure gradient:
    q = -K/mu * grad(p)

For steady-state problems, iterative methods (Jacobi/SOR) are used.
For transient problems, explicit time-stepping is employed.

Examples
--------
Solve steady-state Darcy flow with heterogeneous permeability:

    >>> from src.darcy import solve_darcy_2d, create_binary_permeability
    >>> import numpy as np
    >>>
    >>> # Create heterogeneous permeability field
    >>> K = create_binary_permeability(64, 64, K_low=4.0, K_high=12.0, seed=42)
    >>>
    >>> # Solve for pressure
    >>> result = solve_darcy_2d(
    ...     Lx=1.0, Ly=1.0, Nx=64, Ny=64,
    ...     permeability=K,
    ...     source=1.0,
    ...     bc_left=0.0, bc_right=0.0,
    ...     bc_bottom=0.0, bc_top=0.0,
    ...     tol=1e-4,
    ... )
    >>> print(f"Converged in {result.iterations} iterations")

Compute Darcy velocity from pressure:

    >>> from src.darcy import compute_darcy_velocity
    >>> dx = 1.0 / 63
    >>> qx, qy = compute_darcy_velocity(result.p, K, dx, dx)

Create various permeability fields:

    >>> from src.darcy import (
    ...     create_layered_permeability,
    ...     create_lognormal_permeability,
    ...     GaussianRandomField,
    ... )
"""

from src.darcy.darcy_devito import (
    DarcyResult,
    GaussianRandomField,
    add_fracture_to_permeability,
    add_well,
    check_mass_conservation,
    compute_darcy_velocity,
    create_binary_permeability,
    create_layered_permeability,
    create_lognormal_permeability,
    solve_darcy_2d,
    solve_darcy_transient,
    verify_linear_pressure,
)

__all__ = [
    "DarcyResult",
    "GaussianRandomField",
    "add_fracture_to_permeability",
    "add_well",
    "check_mass_conservation",
    "compute_darcy_velocity",
    "create_binary_permeability",
    "create_layered_permeability",
    "create_lognormal_permeability",
    "solve_darcy_2d",
    "solve_darcy_transient",
    "verify_linear_pressure",
]
