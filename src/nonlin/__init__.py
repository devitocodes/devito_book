"""Nonlinear PDE solvers using Devito DSL."""

from .burgers_devito import (
    Burgers2DResult,
    gaussian_initial_condition,
    init_hat,
    sinusoidal_initial_condition,
    solve_burgers_2d,
    solve_burgers_2d_vector,
)
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
    "Burgers2DResult",
    "NonlinearResult",
    "allen_cahn_reaction",
    "constant_diffusion",
    "fisher_reaction",
    "gaussian_initial_condition",
    "init_hat",
    "linear_diffusion",
    "logistic_reaction",
    "porous_medium_diffusion",
    "sinusoidal_initial_condition",
    "solve_burgers_2d",
    "solve_burgers_2d_vector",
    "solve_burgers_equation",
    "solve_nonlinear_diffusion_explicit",
    "solve_nonlinear_diffusion_picard",
    "solve_reaction_diffusion_splitting",
]
