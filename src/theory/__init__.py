"""
Theory module for numerical analysis fundamentals.

This module provides:
- Von Neumann stability analysis tools
- On-the-fly discrete Fourier transform using Devito
- CFL condition utilities
"""

from .stability_analysis import (
    amplification_factor_diffusion,
    amplification_factor_advection_upwind,
    amplification_factor_wave,
    compute_cfl,
    stable_timestep_diffusion,
    stable_timestep_wave,
    check_stability_diffusion,
    check_stability_wave,
)

from .fourier_dft import (
    run_otf_dft,
    run_otf_dft_multifreq,
    compare_otf_to_fft,
    ricker_wavelet,
)

__all__ = [
    "amplification_factor_advection_upwind",
    "amplification_factor_diffusion",
    "amplification_factor_wave",
    "check_stability_diffusion",
    "check_stability_wave",
    "compare_otf_to_fft",
    "compute_cfl",
    "ricker_wavelet",
    "run_otf_dft",
    "run_otf_dft_multifreq",
    "stable_timestep_diffusion",
    "stable_timestep_wave",
]
