"""Systems of PDEs solvers using Devito DSL.

This module provides solvers for coupled systems of PDEs,
including the 2D Shallow Water Equations for tsunami modeling.
"""

from src.systems.swe_devito import (
    SWEResult,
    create_swe_operator,
    solve_swe,
)

__all__ = [
    "SWEResult",
    "create_swe_operator",
    "solve_swe",
]
