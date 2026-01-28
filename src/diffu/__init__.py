"""Diffusion equation solvers using Devito DSL.

This module provides solvers for the diffusion (heat) equation
using Devito's symbolic finite difference framework.
"""

from src.diffu.diffu1D_devito import (
    DiffusionResult,
    convergence_test_diffusion_1d,
    exact_diffusion_sine,
    gaussian_initial_condition,
    plug_initial_condition,
    solve_diffusion_1d,
)
from src.diffu.diffu2D_devito import (
    Diffusion2DResult,
    convergence_test_diffusion_2d,
    exact_diffusion_2d,
    gaussian_2d_initial_condition,
    solve_diffusion_2d,
)

__all__ = [
    "Diffusion2DResult",
    "DiffusionResult",
    "convergence_test_diffusion_1d",
    "convergence_test_diffusion_2d",
    "exact_diffusion_2d",
    "exact_diffusion_sine",
    "gaussian_2d_initial_condition",
    "gaussian_initial_condition",
    "plug_initial_condition",
    "solve_diffusion_1d",
    "solve_diffusion_2d",
]
