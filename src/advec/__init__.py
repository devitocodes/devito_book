"""Advection equation solvers using Devito DSL.

This module provides solvers for the advection equation
using Devito's symbolic finite difference framework.
"""

from src.advec.advec1D_devito import (
    AdvectionResult,
    convergence_test_advection,
    exact_advection,
    exact_advection_periodic,
    gaussian_initial_condition,
    solve_advection_lax_friedrichs,
    solve_advection_lax_wendroff,
    solve_advection_upwind,
    step_initial_condition,
)

__all__ = [
    "AdvectionResult",
    "convergence_test_advection",
    "exact_advection",
    "exact_advection_periodic",
    "gaussian_initial_condition",
    "solve_advection_lax_friedrichs",
    "solve_advection_lax_wendroff",
    "solve_advection_upwind",
    "step_initial_condition",
]
