"""Wave equation solvers using Devito DSL.

This module provides Devito-based solvers for the wave equation:
u_tt = c^2 * u_xx (1D)
u_tt = c^2 * (u_xx + u_yy) (2D)
u_tt = c^2 * (u_xx + u_yy + u_zz) (3D)

All solvers use the leapfrog (central difference in time) scheme.
"""

from .wave1D_devito import (
    solve_wave_1d,
    WaveResult,
    exact_standing_wave,
    convergence_test_wave_1d,
)

__all__ = [
    'WaveResult',
    'convergence_test_wave_1d',
    'exact_standing_wave',
    'solve_wave_1d',
]
