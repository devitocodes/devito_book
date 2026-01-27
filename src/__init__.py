"""Finite Difference Computing with PDEs - Devito Edition.

This package provides:
- Symbolic mathematics utilities (symbols, operators, display)
- Verification tools for mathematical derivations
- Reproducible plotting utilities
- Devito-based PDE solvers
"""

from .display import inline_latex, show_derivation, show_eq, show_eq_aligned
from .operators import *
from .plotting import RANDOM_SEED, create_convergence_plot, create_solution_plot, set_seed
from .symbols import *
from .verification import (
    check_stencil_order,
    numerical_verify,
    verify_identity,
    verify_pde_solution,
)

__version__ = "0.1.0"

__all__ = [
    "RANDOM_SEED",
    "check_stencil_order",
    "create_convergence_plot",
    "create_solution_plot",
    "inline_latex",
    "numerical_verify",
    "set_seed",
    "show_derivation",
    "show_eq",
    "show_eq_aligned",
    "verify_identity",
    "verify_pde_solution",
]
