"""FWI Gradient Computation using Devito DSL.

Computes the gradient of the FWI objective function:
    Phi(m) = 0.5 * ||P_r * u - d||^2

The gradient is:
    grad_m(Phi) = sum_t u[t] * v_tt[t]

where u is the forward wavefield and v is the adjoint wavefield.

This module uses the EXPLICIT Devito API:
    - Grid, Function, TimeFunction, SparseTimeFunction
    - Eq, Operator, solve

NO convenience classes are used.

Usage:
    from src.adjoint import compute_gradient_shot, compute_residual

    # Compute residual
    residual = compute_residual(d_obs, d_syn)

    # Compute gradient for one shot
    obj, grad = compute_gradient_shot(...)
"""

import importlib.util

import numpy as np

from .forward_devito import ricker_wavelet

DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None


def compute_residual(
    d_obs: np.ndarray,
    d_syn: np.ndarray,
) -> np.ndarray:
    """Compute data residual.

    Parameters
    ----------
    d_obs : np.ndarray
        Observed data, shape (nt, nrec)
    d_syn : np.ndarray
        Synthetic data, shape (nt, nrec)

    Returns
    -------
    np.ndarray
        Data residual: d_syn - d_obs
    """
    return d_syn - d_obs


def compute_gradient_shot(
    shape: tuple[int, int],
    extent: tuple[float, float],
    vp_model: np.ndarray,
    vp_true: np.ndarray,
    src_coords: np.ndarray,
    rec_coords: np.ndarray,
    t_end: float,
    f0: float,
    space_order: int = 4,
    dt: float | None = None,
    t0: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Compute FWI gradient for a single shot.

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    extent : tuple
        Physical extent (Lx, Lz) in meters
    vp_model : np.ndarray
        Current velocity model
    vp_true : np.ndarray
        True velocity model (for generating observed data)
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
        Time step
    t0 : float
        Start time

    Returns
    -------
    tuple
        (objective_value, gradient)
        - objective_value: 0.5 * ||residual||^2
        - gradient: gradient w.r.t. squared slowness, shape (nx, nz)
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

    vel_model = DevitoFunction(name='vel_model', grid=grid, space_order=space_order)
    vel_model.data[:] = vp_model

    # Squared slowness for current model
    model_m = DevitoFunction(name='m', grid=grid, space_order=space_order)
    model_m.data[:] = 1.0 / vp_model**2

    # Compute time step from CFL condition if not provided
    dx = extent[0] / (shape[0] - 1)
    dz = extent[1] / (shape[1] - 1)
    h_min = min(dx, dz)
    v_max = max(float(np.max(vp_true)), float(np.max(vp_model)))

    if dt is None:
        cfl_limit = h_min / (np.sqrt(2) * v_max)
        dt = 0.9 * cfl_limit

    # Compute number of time steps
    nt = int((t_end - t0) / dt) + 1
    time_values = np.linspace(t0, t_end, nt)

    # Ensure coordinates are 2D
    src_coords = np.atleast_2d(src_coords)
    nsrc = src_coords.shape[0]
    rec_coords = np.atleast_2d(rec_coords)
    nrec = rec_coords.shape[0]

    dt_sym = grid.stepping_dim.spacing

    # --- Forward with true model -> observed data ---
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

    rec_obs = SparseTimeFunction(
        name='rec_obs', grid=grid, npoint=nrec, nt=nt,
        coordinates=rec_coords
    )

    pde_true = (1.0 / vel_true**2) * u_true.dt2 - u_true.laplace
    stencil_true = Eq(u_true.forward, solve(pde_true, u_true.forward))
    src_term_true = src_true.inject(
        field=u_true.forward,
        expr=src_true * dt_sym**2 * vel_true**2
    )
    rec_term_true = rec_obs.interpolate(expr=u_true)

    op_true = Operator([stencil_true] + src_term_true + rec_term_true)
    op_true.apply(time=nt - 2, dt=dt)

    d_obs = np.array(rec_obs.data[:])

    # --- Forward with current model -> synthetic data and save wavefield ---
    u_syn = TimeFunction(
        name='u_syn', grid=grid, time_order=2, space_order=space_order,
        save=nt
    )

    src_syn = SparseTimeFunction(
        name='src_syn', grid=grid, npoint=nsrc, nt=nt,
        coordinates=src_coords
    )
    for i in range(nsrc):
        src_syn.data[:, i] = wavelet

    rec_syn = SparseTimeFunction(
        name='rec_syn', grid=grid, npoint=nrec, nt=nt,
        coordinates=rec_coords
    )

    pde_syn = (1.0 / vel_model**2) * u_syn.dt2 - u_syn.laplace
    stencil_syn = Eq(u_syn.forward, solve(pde_syn, u_syn.forward))
    src_term_syn = src_syn.inject(
        field=u_syn.forward,
        expr=src_syn * dt_sym**2 * vel_model**2
    )
    rec_term_syn = rec_syn.interpolate(expr=u_syn)

    op_syn = Operator([stencil_syn] + src_term_syn + rec_term_syn)
    op_syn.apply(time=nt - 2, dt=dt)

    d_syn = np.array(rec_syn.data[:])

    # --- Compute residual and objective ---
    residual_data = compute_residual(d_obs, d_syn)
    objective = 0.5 * np.sum(residual_data**2)

    # --- Adjoint propagation with gradient computation ---
    grad = DevitoFunction(name='grad', grid=grid)
    v = TimeFunction(name='v', grid=grid, time_order=2, space_order=space_order)

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

    # FWI gradient: grad += u * v.dt2
    gradient_update = Eq(grad, grad + u_syn * v.dt2)

    op_adj = Operator([stencil_adj] + res_term + [gradient_update])
    op_adj.apply(u_syn=u_syn, v=v, dt=dt, time_M=nt - 2)

    return objective, np.array(grad.data[:])


def compute_total_gradient(
    shape: tuple[int, int],
    extent: tuple[float, float],
    vp_model: np.ndarray,
    vp_true: np.ndarray,
    src_positions: np.ndarray,
    rec_coords: np.ndarray,
    t_end: float,
    f0: float,
    space_order: int = 4,
    dt: float | None = None,
    verbose: bool = True,
) -> tuple[float, np.ndarray]:
    """Compute total FWI gradient over all shots.

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    extent : tuple
        Physical extent (Lx, Lz) in meters
    vp_model : np.ndarray
        Current velocity model
    vp_true : np.ndarray
        True velocity model
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
    verbose : bool
        Print progress

    Returns
    -------
    tuple
        (total_objective, total_gradient)
    """
    src_positions = np.atleast_2d(src_positions)
    nshots = src_positions.shape[0]

    total_objective = 0.0
    total_gradient = np.zeros(shape, dtype=np.float32)

    for i, src_pos in enumerate(src_positions):
        if verbose:
            print(f"Computing gradient for shot {i + 1}/{nshots}")

        obj, grad = compute_gradient_shot(
            shape=shape,
            extent=extent,
            vp_model=vp_model,
            vp_true=vp_true,
            src_coords=src_pos,
            rec_coords=rec_coords,
            t_end=t_end,
            f0=f0,
            space_order=space_order,
            dt=dt,
        )

        total_objective += obj
        total_gradient += grad

    return total_objective, total_gradient


def gradient_to_velocity_update(
    grad_m: np.ndarray,
    vp: np.ndarray,
) -> np.ndarray:
    """Convert gradient w.r.t. m to gradient w.r.t. velocity.

    Since m = 1/v^2, we have:
        dm = -2 * v^(-3) * dv

    Therefore:
        dv = -v^3 / 2 * dm

    Parameters
    ----------
    grad_m : np.ndarray
        Gradient w.r.t. squared slowness m
    vp : np.ndarray
        Current velocity model

    Returns
    -------
    np.ndarray
        Gradient w.r.t. velocity
    """
    return -vp**3 / 2.0 * grad_m
