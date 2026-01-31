"""Adjoint-state methods for seismic imaging and inversion.

This module provides solvers for:
- Forward acoustic wave modeling
- Reverse Time Migration (RTM)
- Full Waveform Inversion (FWI)
- Least-Squares Reverse Time Migration (LSRTM)
- Gradient computation via adjoint-state method

All solvers use the explicit Devito API without convenience classes
(Grid, TimeFunction, SparseTimeFunction, Function, Eq, Operator).

Usage:
    from src.adjoint import (
        solve_forward_2d,       # Forward modeling
        rtm_single_shot,        # RTM for one shot
        fwi_gradient_descent,   # FWI optimization
        lsrtm_steepest_descent, # LSRTM optimization
    )

    # Forward modeling
    result = solve_forward_2d(
        shape=(101, 101),
        extent=(1000., 1000.),
        vp=velocity_model,
        t_end=1000.0,
        f0=0.010,
        src_coords=src_coords,
        rec_coords=rec_coords,
    )

    # FWI
    result = fwi_gradient_descent(
        shape=(101, 101),
        extent=(1000., 1000.),
        vp_initial=smooth_model,
        vp_true=true_model,
        src_positions=src_positions,
        rec_coords=rec_coords,
        niter=10,
    )

    # LSRTM
    result = lsrtm_steepest_descent(
        shape=(101, 101),
        extent=(1000., 1000.),
        vp_smooth=smooth_model,
        vp_true=true_model,
        src_positions=src_positions,
        rec_coords=rec_coords,
        niter=20,
    )
"""

from .forward_devito import (
    ForwardResult,
    estimate_dt,
    ricker_wavelet,
    solve_forward_2d,
)
from .fwi_devito import (
    FWIResult,
    compute_fwi_gradient,
    compute_residual,
    create_circle_model,
    fwi_gradient_descent,
    update_with_box_constraint,
)
from .gradient import (
    compute_gradient_shot,
    compute_total_gradient,
    gradient_to_velocity_update,
)
from .lsrtm_devito import (
    LSRTMResult,
    barzilai_borwein_step,
    born_adjoint,
    born_modeling,
    create_layered_model,
    lsrtm_steepest_descent,
)
from .rtm_devito import (
    RTMResult,
    rtm_multi_shot,
    rtm_single_shot,
    solve_adjoint_2d,
)

__all__ = [
    "FWIResult",
    "ForwardResult",
    "LSRTMResult",
    "RTMResult",
    "barzilai_borwein_step",
    "born_adjoint",
    "born_modeling",
    "compute_fwi_gradient",
    "compute_gradient_shot",
    "compute_residual",
    "compute_total_gradient",
    "create_circle_model",
    "create_layered_model",
    "estimate_dt",
    "fwi_gradient_descent",
    "gradient_to_velocity_update",
    "lsrtm_steepest_descent",
    "ricker_wavelet",
    "rtm_multi_shot",
    "rtm_single_shot",
    "solve_adjoint_2d",
    "solve_forward_2d",
    "update_with_box_constraint",
]
