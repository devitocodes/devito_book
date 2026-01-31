"""Reverse Time Migration (RTM) using Devito DSL.

RTM creates images of subsurface reflectivity by correlating forward
and adjoint wavefields. The imaging condition is:

    Image(x, z) = sum_t u(x, z, t) * v(x, z, t)

where u is the forward wavefield and v is the adjoint wavefield.

This module uses the EXPLICIT Devito API:
    - Grid, Function, TimeFunction, SparseTimeFunction
    - Eq, Operator, solve

NO convenience classes are used (no SeismicModel, AcousticWaveSolver, etc.)

Usage:
    from src.adjoint import rtm_single_shot, rtm_multi_shot

    # Single shot RTM
    result = rtm_single_shot(
        shape=(101, 101),
        extent=(1000., 1000.),
        vp_true=true_velocity,
        vp_smooth=smooth_velocity,
        src_coords=np.array([[500., 20.]]),
        rec_coords=rec_coords,
        t_end=1000.0,
        f0=0.010,
    )

    # Multi-shot RTM
    result = rtm_multi_shot(
        shape=(101, 101),
        extent=(1000., 1000.),
        vp_true=true_velocity,
        vp_smooth=smooth_velocity,
        src_positions=src_positions,
        rec_coords=rec_coords,
        t_end=1000.0,
        f0=0.010,
    )
"""

import importlib.util
from dataclasses import dataclass

import numpy as np

from .forward_devito import ricker_wavelet

DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None


@dataclass
class RTMResult:
    """Results from RTM imaging.

    Attributes
    ----------
    image : np.ndarray
        RTM image, shape (nx, nz)
    x : np.ndarray
        X coordinates of grid points
    z : np.ndarray
        Z coordinates of grid points
    nshots : int
        Number of shots used
    """
    image: np.ndarray
    x: np.ndarray
    z: np.ndarray
    nshots: int


def solve_adjoint_2d(
    grid,
    model_m,
    rec_data: np.ndarray,
    rec_coords: np.ndarray,
    forward_wavefield,
    space_order: int = 4,
    dt: float = None,
) -> np.ndarray:
    """Solve adjoint wave equation and compute imaging condition.

    Parameters
    ----------
    grid : devito.Grid
        Computational grid
    model_m : devito.Function
        Squared slowness m = 1/v^2
    rec_data : np.ndarray
        Receiver data (residual), shape (nt, nrec)
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    forward_wavefield : devito.TimeFunction
        Forward wavefield u, shape (nt, nx, nz)
    space_order : int
        Spatial discretization order
    dt : float
        Time step

    Returns
    -------
    np.ndarray
        RTM image contribution from this shot
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    from devito import Eq
    from devito import Function as DevitoFunction
    from devito import Operator, SparseTimeFunction, TimeFunction, solve

    nt = rec_data.shape[0]
    nrec = rec_data.shape[1]

    # Adjoint wavefield
    v = TimeFunction(name='v', grid=grid, time_order=2, space_order=space_order)

    # Image accumulator
    image = DevitoFunction(name='image', grid=grid)

    # Residual injection at receiver locations
    residual = SparseTimeFunction(
        name='residual', grid=grid, npoint=nrec, nt=nt,
        coordinates=rec_coords
    )
    residual.data[:] = rec_data

    # Adjoint wave equation (undamped)
    # For undamped case: m * v_tt - laplace(v) = residual
    pde_adj = model_m * v.dt2 - v.laplace

    # Solve for v.backward (time reversal)
    stencil_adj = Eq(v.backward, solve(pde_adj, v.backward))

    # Inject residual into adjoint wavefield
    dt_sym = grid.stepping_dim.spacing
    res_term = residual.inject(
        field=v.backward,
        expr=residual * dt_sym**2 / model_m
    )

    # Imaging condition: Image -= u * v
    # Negative sign for correct polarity
    image_update = Eq(image, image - forward_wavefield * v)

    # Create and run operator
    op = Operator([stencil_adj] + res_term + [image_update])
    op.apply(dt=dt, time_M=nt - 2)

    return np.array(image.data[:])


def rtm_single_shot(
    shape: tuple[int, int],
    extent: tuple[float, float],
    vp_true: np.ndarray,
    vp_smooth: np.ndarray,
    src_coords: np.ndarray,
    rec_coords: np.ndarray,
    t_end: float,
    f0: float,
    space_order: int = 4,
    dt: float | None = None,
    t0: float = 0.0,
) -> RTMResult:
    """Perform RTM for a single shot.

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    extent : tuple
        Physical extent (Lx, Lz) in meters
    vp_true : np.ndarray
        True velocity model (for generating observed data)
    vp_smooth : np.ndarray
        Smooth velocity model (for migration)
    src_coords : np.ndarray
        Source coordinates, shape (1, 2) or (2,)
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    t_end : float
        End time in milliseconds
    f0 : float
        Source peak frequency in kHz
    space_order : int
        Spatial discretization order
    dt : float, optional
        Time step. If None, computed from CFL condition.
    t0 : float
        Start time

    Returns
    -------
    RTMResult
        RTM image and grid information
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    from devito import Eq
    from devito import Function as DevitoFunction
    from devito import Grid, Operator, SparseTimeFunction, TimeFunction, solve

    # Create grid
    grid = Grid(shape=shape, extent=extent, dtype=np.float32)

    # Create velocity fields
    vel_true = DevitoFunction(name='vel_true', grid=grid, space_order=space_order)
    vel_true.data[:] = vp_true

    vel_smooth = DevitoFunction(name='vel_smooth', grid=grid, space_order=space_order)
    vel_smooth.data[:] = vp_smooth

    # Squared slowness for smooth model
    model_m = DevitoFunction(name='m', grid=grid, space_order=space_order)
    model_m.data[:] = 1.0 / vp_smooth**2

    # Compute time step from CFL condition if not provided
    dx = extent[0] / (shape[0] - 1)
    dz = extent[1] / (shape[1] - 1)
    h_min = min(dx, dz)
    v_max = max(float(np.max(vp_true)), float(np.max(vp_smooth)))

    if dt is None:
        cfl_limit = h_min / (np.sqrt(2) * v_max)
        dt = 0.9 * cfl_limit

    # Compute number of time steps
    nt = int((t_end - t0) / dt) + 1
    time_values = np.linspace(t0, t_end, nt)

    # Ensure source coordinates is 2D
    src_coords = np.atleast_2d(src_coords)
    nsrc = src_coords.shape[0]
    rec_coords = np.atleast_2d(rec_coords)
    nrec = rec_coords.shape[0]

    # --- Step 1: Forward modeling with true velocity (observed data) ---
    u_true = TimeFunction(
        name='u_true', grid=grid, time_order=2, space_order=space_order
    )

    src_true = SparseTimeFunction(
        name='src_true', grid=grid, npoint=nsrc, nt=nt,
        coordinates=src_coords
    )
    wavelet = ricker_wavelet(time_values, f0)
    for i in range(nsrc):
        src_true.data[:, i] = wavelet

    rec_true = SparseTimeFunction(
        name='rec_true', grid=grid, npoint=nrec, nt=nt,
        coordinates=rec_coords
    )

    pde_true = (1.0 / vel_true**2) * u_true.dt2 - u_true.laplace
    stencil_true = Eq(u_true.forward, solve(pde_true, u_true.forward))
    dt_sym = grid.stepping_dim.spacing
    src_term_true = src_true.inject(
        field=u_true.forward,
        expr=src_true * dt_sym**2 * vel_true**2
    )
    rec_term_true = rec_true.interpolate(expr=u_true)

    op_true = Operator([stencil_true] + src_term_true + rec_term_true)
    op_true.apply(time=nt - 2, dt=dt)

    d_obs = np.array(rec_true.data[:])

    # --- Step 2: Forward modeling with smooth velocity (save wavefield) ---
    u_smooth = TimeFunction(
        name='u_smooth', grid=grid, time_order=2, space_order=space_order,
        save=nt
    )

    src_smooth = SparseTimeFunction(
        name='src_smooth', grid=grid, npoint=nsrc, nt=nt,
        coordinates=src_coords
    )
    for i in range(nsrc):
        src_smooth.data[:, i] = wavelet

    rec_smooth = SparseTimeFunction(
        name='rec_smooth', grid=grid, npoint=nrec, nt=nt,
        coordinates=rec_coords
    )

    pde_smooth = (1.0 / vel_smooth**2) * u_smooth.dt2 - u_smooth.laplace
    stencil_smooth = Eq(u_smooth.forward, solve(pde_smooth, u_smooth.forward))
    src_term_smooth = src_smooth.inject(
        field=u_smooth.forward,
        expr=src_smooth * dt_sym**2 * vel_smooth**2
    )
    rec_term_smooth = rec_smooth.interpolate(expr=u_smooth)

    op_smooth = Operator([stencil_smooth] + src_term_smooth + rec_term_smooth)
    op_smooth.apply(time=nt - 2, dt=dt)

    d_syn = np.array(rec_smooth.data[:])

    # --- Step 3: Compute residual ---
    residual_data = d_syn - d_obs

    # --- Step 4: Adjoint propagation and imaging ---
    v = TimeFunction(name='v', grid=grid, time_order=2, space_order=space_order)
    image = DevitoFunction(name='image', grid=grid)

    residual = SparseTimeFunction(
        name='residual', grid=grid, npoint=nrec, nt=nt,
        coordinates=rec_coords
    )
    residual.data[:] = residual_data

    pde_adj = model_m * v.dt2 - v.laplace
    stencil_adj = Eq(v.backward, solve(pde_adj, v.backward))
    res_term = residual.inject(
        field=v.backward,
        expr=residual * dt_sym**2 / model_m
    )
    image_update = Eq(image, image - u_smooth * v)

    op_adj = Operator([stencil_adj] + res_term + [image_update])
    op_adj.apply(u_smooth=u_smooth, v=v, dt=dt, time_M=nt - 2)

    # Extract grid coordinates
    x_coords = np.linspace(0, extent[0], shape[0])
    z_coords = np.linspace(0, extent[1], shape[1])

    return RTMResult(
        image=np.array(image.data[:]),
        x=x_coords,
        z=z_coords,
        nshots=1,
    )


def rtm_multi_shot(
    shape: tuple[int, int],
    extent: tuple[float, float],
    vp_true: np.ndarray,
    vp_smooth: np.ndarray,
    src_positions: np.ndarray,
    rec_coords: np.ndarray,
    t_end: float,
    f0: float,
    space_order: int = 4,
    dt: float | None = None,
    t0: float = 0.0,
    verbose: bool = True,
) -> RTMResult:
    """Perform multi-shot RTM.

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    extent : tuple
        Physical extent (Lx, Lz) in meters
    vp_true : np.ndarray
        True velocity model (for generating observed data)
    vp_smooth : np.ndarray
        Smooth velocity model (for migration)
    src_positions : np.ndarray
        Source positions, shape (nshots, 2)
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    t_end : float
        End time in milliseconds
    f0 : float
        Source peak frequency in kHz
    space_order : int
        Spatial discretization order
    dt : float, optional
        Time step
    t0 : float
        Start time
    verbose : bool
        Print progress

    Returns
    -------
    RTMResult
        Stacked RTM image and grid information
    """
    src_positions = np.atleast_2d(src_positions)
    nshots = src_positions.shape[0]

    # Initialize stacked image
    image_total = np.zeros(shape, dtype=np.float32)

    for i, src_pos in enumerate(src_positions):
        if verbose:
            print(f"Processing shot {i + 1}/{nshots}")

        # RTM for this shot
        result = rtm_single_shot(
            shape=shape,
            extent=extent,
            vp_true=vp_true,
            vp_smooth=vp_smooth,
            src_coords=src_pos,
            rec_coords=rec_coords,
            t_end=t_end,
            f0=f0,
            space_order=space_order,
            dt=dt,
            t0=t0,
        )

        # Stack images
        image_total += result.image

    return RTMResult(
        image=image_total,
        x=result.x,
        z=result.z,
        nshots=nshots,
    )
