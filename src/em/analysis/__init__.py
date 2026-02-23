"""Analysis tools for Maxwell/FDTD simulations."""

from src.em.analysis.dispersion_maxwell import (
    compute_dispersion_error,
    numerical_dispersion_relation_1d,
    numerical_dispersion_relation_2d,
    phase_velocity_error_1d,
    plot_dispersion_polar,
)

__all__ = [
    "compute_dispersion_error",
    "numerical_dispersion_relation_1d",
    "numerical_dispersion_relation_2d",
    "phase_velocity_error_1d",
    "plot_dispersion_polar",
]
