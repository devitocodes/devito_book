"""Wave equation solvers using Devito DSL.

This module provides Devito-based solvers for the wave equation:
u_tt = c^2 * u_xx (1D)
u_tt = c^2 * (u_xx + u_yy) (2D)
u_tt = c^2 * (u_xx + u_yy + u_zz) (3D)

All solvers use the leapfrog (central difference in time) scheme.

Source wavelets are provided for seismic-style simulations:
- Ricker wavelet (Mexican hat)
- Gaussian pulse
- Derivative of Gaussian
"""

from .sources import (
    gaussian_derivative,
    gaussian_pulse,
    get_source_spectrum,
    ricker_wavelet,
    sinc_wavelet,
)
from .wave1D_devito import (
    WaveResult,
    convergence_test_wave_1d,
    exact_standing_wave,
    solve_wave_1d,
)
from .wave2D_devito import (
    Wave2DResult,
    convergence_test_wave_2d,
    exact_standing_wave_2d,
    solve_wave_2d,
)
from .abc_methods import (
    ABCResult,
    compare_abc_methods,
    create_damping_profile,
    create_directional_damping_profiles,
    create_habc_weights,
    measure_reflection,
    solve_wave_2d_abc,
)

__all__ = [
    'ABCResult',
    'Wave2DResult',
    'WaveResult',
    'compare_abc_methods',
    'convergence_test_wave_1d',
    'convergence_test_wave_2d',
    'create_damping_profile',
    'create_directional_damping_profiles',
    'create_habc_weights',
    'exact_standing_wave',
    'exact_standing_wave_2d',
    'gaussian_derivative',
    'gaussian_pulse',
    'get_source_spectrum',
    'measure_reflection',
    'ricker_wavelet',
    'sinc_wavelet',
    'solve_wave_1d',
    'solve_wave_2d',
    'solve_wave_2d_abc',
]
