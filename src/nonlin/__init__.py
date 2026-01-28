"""Nonlinear PDE solvers using Devito DSL."""

from .nonlin1D_devito import (
    NonlinearResult,
    allen_cahn_reaction,
    constant_diffusion,
    fisher_reaction,
    linear_diffusion,
    logistic_reaction,
    porous_medium_diffusion,
    solve_burgers_equation,
    solve_nonlinear_diffusion_explicit,
    solve_nonlinear_diffusion_picard,
    solve_reaction_diffusion_splitting,
)

__all__ = [
    "NonlinearResult",
    "allen_cahn_reaction",
    "constant_diffusion",
    "fisher_reaction",
    "linear_diffusion",
    "logistic_reaction",
    "porous_medium_diffusion",
    "solve_burgers_equation",
    "solve_nonlinear_diffusion_explicit",
    "solve_nonlinear_diffusion_picard",
    "solve_reaction_diffusion_splitting",
]
