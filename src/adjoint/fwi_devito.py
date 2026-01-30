"""Full Waveform Inversion (FWI) using Devito DSL.

This module provides Full Waveform Inversion implementation using the
adjoint-state method for gradient computation. FWI aims to minimize
the misfit between observed and synthetic seismic data to recover
the subsurface velocity model.

The optimization problem is:

    minimize_{m} Phi(m) = 0.5 * sum_s ||P_r u_s - d_s||^2

where:
    - m is the squared slowness (1/v^2)
    - P_r is the sampling operator at receiver locations
    - u_s is the synthetic wavefield for shot s
    - d_s is the observed data for shot s

The gradient is computed via the adjoint-state method:

    nabla Phi(m) = sum_t u[t] * v_tt[t]

where u is the forward wavefield and v_tt is the second time derivative
of the adjoint wavefield.

References
----------
[1] Virieux, J. and Operto, S.: An overview of full-waveform inversion
    in exploration geophysics, GEOPHYSICS, 74, WCC1-WCC26, 2009.
[2] Plessix, R.-E.: A review of the adjoint-state method for computing
    the gradient of a functional with geophysical applications,
    Geophysical Journal International, 167, 495-503, 2006.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

try:
    from devito import (
        Eq,
        Function,
        Grid,
        Operator,
        SparseTimeFunction,
        TimeFunction,
        solve,
    )
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


__all__ = [
    "FWIResult",
    "compute_fwi_gradient",
    "compute_residual",
    "create_circle_model",
    "fwi_gradient_descent",
    "update_with_box_constraint",
]


@dataclass
class FWIResult:
    """Results from Full Waveform Inversion.

    Attributes
    ----------
    vp_final : np.ndarray
        Final recovered velocity model
    vp_initial : np.ndarray
        Initial velocity model used to start inversion
    vp_true : np.ndarray
        True velocity model (if provided)
    history : np.ndarray
        Objective function value at each iteration
    gradients : list
        List of gradient arrays at each iteration (optional)
    iterations : int
        Number of iterations performed
    """
    vp_final: np.ndarray
    vp_initial: np.ndarray
    vp_true: np.ndarray | None = None
    history: np.ndarray = field(default_factory=lambda: np.array([]))
    gradients: list = field(default_factory=list)
    iterations: int = 0


def _ricker_wavelet(t: np.ndarray, f0: float = 0.01, t0: float | None = None) -> np.ndarray:
    """Generate a Ricker (Mexican hat) wavelet.

    Parameters
    ----------
    t : np.ndarray
        Time array in milliseconds
    f0 : float
        Peak frequency in kHz (default 0.01 kHz = 10 Hz)
    t0 : float, optional
        Time of peak in milliseconds. Default is 1/f0.

    Returns
    -------
    np.ndarray
        Ricker wavelet values
    """
    if t0 is None:
        t0 = 1.0 / f0

    tau = (t - t0) * f0 * np.pi
    return (1.0 - 2.0 * tau**2) * np.exp(-tau**2)


def create_circle_model(
    shape: tuple[int, int],
    spacing: tuple[float, float],
    vp_background: float = 2.5,
    vp_circle: float = 3.0,
    circle_center: tuple[float, float] | None = None,
    circle_radius: float | None = None,
) -> np.ndarray:
    """Create a circular anomaly velocity model.

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    spacing : tuple
        Grid spacing (dx, dz) in meters
    vp_background : float
        Background velocity in km/s
    vp_circle : float
        Velocity inside the circle in km/s
    circle_center : tuple, optional
        Center of circle in meters. Default is center of domain.
    circle_radius : float, optional
        Radius of circle in meters. Default is 1/4 of domain size.

    Returns
    -------
    np.ndarray
        Velocity model with shape (nx, nz)
    """
    nx, nz = shape
    dx, dz = spacing

    # Create coordinate arrays
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz
    X, Z = np.meshgrid(x, z, indexing='ij')

    # Default center and radius
    if circle_center is None:
        circle_center = (x[-1] / 2, z[-1] / 2)
    if circle_radius is None:
        circle_radius = min(x[-1], z[-1]) / 4

    # Create model
    vp = np.full(shape, vp_background, dtype=np.float32)

    # Add circular anomaly
    dist = np.sqrt((X - circle_center[0])**2 + (Z - circle_center[1])**2)
    vp[dist <= circle_radius] = vp_circle

    return vp


def compute_residual(
    rec_syn: np.ndarray,
    rec_obs: np.ndarray,
) -> np.ndarray:
    """Compute data residual (synthetic - observed).

    Parameters
    ----------
    rec_syn : np.ndarray
        Synthetic receiver data, shape (nt, nrec)
    rec_obs : np.ndarray
        Observed receiver data, shape (nt, nrec)

    Returns
    -------
    np.ndarray
        Data residual, shape (nt, nrec)
    """
    return rec_syn - rec_obs


def update_with_box_constraint(
    vp: np.ndarray,
    alpha: float,
    gradient: np.ndarray,
    vmin: float = 1.5,
    vmax: float = 4.5,
) -> np.ndarray:
    """Apply gradient update with box constraints on velocity.

    Parameters
    ----------
    vp : np.ndarray
        Current velocity model
    alpha : float
        Step length
    gradient : np.ndarray
        Gradient of objective function
    vmin : float
        Minimum allowed velocity
    vmax : float
        Maximum allowed velocity

    Returns
    -------
    np.ndarray
        Updated velocity model with constraints applied
    """
    # Gradient descent step
    vp_new = vp - alpha * gradient

    # Apply box constraints
    vp_new = np.clip(vp_new, vmin, vmax)

    return vp_new


def _solve_forward_2d(
    grid: "Grid",
    vp: np.ndarray,
    src_coords: np.ndarray,
    rec_coords: np.ndarray,
    t_end: float,
    dt: float,
    f0: float = 0.01,
    space_order: int = 4,
    save_wavefield: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Solve 2D acoustic wave equation forward in time.

    Parameters
    ----------
    grid : Grid
        Devito grid
    vp : np.ndarray
        Velocity model in km/s
    src_coords : np.ndarray
        Source coordinates, shape (1, 2) or (nsrc, 2)
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    t_end : float
        End time in ms
    dt : float
        Time step in ms
    f0 : float
        Source peak frequency in kHz
    space_order : int
        Spatial discretization order
    save_wavefield : bool
        Whether to save the full wavefield

    Returns
    -------
    tuple
        (receiver_data, wavefield) where wavefield is None if not saved
    """
    nt = int(t_end / dt) + 1

    # Create time function
    if save_wavefield:
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order,
                        save=nt)
    else:
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order)

    # Create velocity Function
    v = Function(name='v', grid=grid, space_order=space_order)
    v.data[:] = vp

    # Wave equation: m * u_tt = laplace(u)
    # where m = 1/v^2 (squared slowness)
    m = 1.0 / (v * v)
    pde = m * u.dt2 - u.laplace
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Source
    t_vals = np.arange(nt) * dt
    src_data = _ricker_wavelet(t_vals, f0=f0)

    src = SparseTimeFunction(
        name='src', grid=grid, npoint=src_coords.shape[0], nt=nt
    )
    src.coordinates.data[:] = src_coords
    src.data[:] = src_data.reshape(-1, 1) if src_coords.shape[0] == 1 else np.tile(src_data, (src_coords.shape[0], 1)).T

    src_term = src.inject(field=u.forward, expr=src * dt**2 / m)

    # Receivers
    rec = SparseTimeFunction(
        name='rec', grid=grid, npoint=rec_coords.shape[0], nt=nt
    )
    rec.coordinates.data[:] = rec_coords

    rec_term = rec.interpolate(expr=u)

    # Create and run operator
    op = Operator([stencil] + src_term + rec_term, name='forward')
    op.apply(time_M=nt-2, dt=dt)

    # Extract results
    rec_data = rec.data.copy()
    wavefield = u.data.copy() if save_wavefield else None

    return rec_data, wavefield


def _solve_adjoint_2d(
    grid: "Grid",
    vp: np.ndarray,
    residual: np.ndarray,
    rec_coords: np.ndarray,
    t_end: float,
    dt: float,
    space_order: int = 4,
) -> np.ndarray:
    """Solve 2D acoustic wave equation adjoint (backward in time).

    The adjoint equation is the same as forward but with reversed time
    and residual injected at receiver locations.

    Parameters
    ----------
    grid : Grid
        Devito grid
    vp : np.ndarray
        Velocity model in km/s
    residual : np.ndarray
        Data residual to inject, shape (nt, nrec)
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    t_end : float
        End time in ms
    dt : float
        Time step in ms
    space_order : int
        Spatial discretization order

    Returns
    -------
    np.ndarray
        Second time derivative of adjoint wavefield (nt, nx, nz)
    """
    nt = int(t_end / dt) + 1

    # Create time function for adjoint wavefield
    v_adj = TimeFunction(name='v', grid=grid, time_order=2, space_order=space_order,
                        save=nt)

    # Create velocity Function
    vel = Function(name='vel', grid=grid, space_order=space_order)
    vel.data[:] = vp

    # Wave equation
    m = 1.0 / (vel * vel)
    pde = m * v_adj.dt2 - v_adj.laplace
    stencil = Eq(v_adj.forward, solve(pde, v_adj.forward))

    # Inject residual at receivers (time-reversed)
    rec_adj = SparseTimeFunction(
        name='rec_adj', grid=grid, npoint=rec_coords.shape[0], nt=nt
    )
    rec_adj.coordinates.data[:] = rec_coords
    # Time-reverse the residual
    rec_adj.data[:] = residual[::-1, :]

    rec_term = rec_adj.inject(field=v_adj.forward, expr=rec_adj * dt**2 / m)

    # Create and run operator
    op = Operator([stencil] + rec_term, name='adjoint')
    op.apply(time_M=nt-2, dt=dt)

    # Compute second time derivative
    v_data = v_adj.data.copy()
    v_tt = np.zeros_like(v_data)
    v_tt[1:-1] = (v_data[2:] - 2*v_data[1:-1] + v_data[:-2]) / dt**2

    return v_tt


def compute_fwi_gradient(
    shape: tuple[int, int],
    extent: tuple[float, float],
    vp_current: np.ndarray,
    vp_true: np.ndarray,
    src_positions: np.ndarray,
    rec_coords: np.ndarray,
    f0: float = 0.01,
    t_end: float = 1000.0,
    space_order: int = 4,
) -> tuple[float, np.ndarray]:
    """Compute FWI gradient using adjoint-state method.

    The gradient is computed as:
        g = sum_s sum_t u_s[t] * v_tt_s[t]

    where for each shot s:
        1. Forward with true model -> observed data
        2. Forward with current model -> synthetic data (save wavefield)
        3. Compute residual = synthetic - observed
        4. Adjoint with residual -> v_tt
        5. Correlate u and v_tt for gradient contribution

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    extent : tuple
        Domain extent (Lx, Lz) in meters
    vp_current : np.ndarray
        Current velocity model estimate
    vp_true : np.ndarray
        True velocity model (for generating observed data)
    src_positions : np.ndarray
        Source positions, shape (nshots, 2)
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    f0 : float
        Source peak frequency in kHz
    t_end : float
        Simulation end time in ms
    space_order : int
        Spatial discretization order

    Returns
    -------
    tuple
        (objective, gradient) where objective is the misfit value and
        gradient is the FWI gradient array
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required for FWI computation")

    # Create grid
    grid = Grid(shape=shape, extent=extent)

    # Compute stable time step
    dx = extent[0] / (shape[0] - 1)
    dz = extent[1] / (shape[1] - 1)
    vp_max = max(np.max(vp_current), np.max(vp_true))
    dt = 0.4 * min(dx, dz) / vp_max  # CFL condition

    nshots = src_positions.shape[0]
    objective = 0.0
    gradient = np.zeros(shape, dtype=np.float32)

    for ishot in range(nshots):
        # Get source position for this shot
        src_coords = src_positions[ishot:ishot+1, :]

        # Forward with true model -> observed data
        rec_obs, _ = _solve_forward_2d(
            grid, vp_true, src_coords, rec_coords, t_end, dt, f0, space_order,
            save_wavefield=False
        )

        # Forward with current model -> synthetic data and wavefield
        rec_syn, u_wavefield = _solve_forward_2d(
            grid, vp_current, src_coords, rec_coords, t_end, dt, f0, space_order,
            save_wavefield=True
        )

        # Compute residual
        residual = compute_residual(rec_syn, rec_obs)

        # Update objective
        objective += 0.5 * np.sum(residual**2)

        # Adjoint propagation
        v_tt = _solve_adjoint_2d(
            grid, vp_current, residual, rec_coords, t_end, dt, space_order
        )

        # Compute gradient contribution: sum_t u[t] * v_tt[t]
        # Time-reverse v_tt to align with forward wavefield
        for it in range(u_wavefield.shape[0]):
            gradient += u_wavefield[it] * v_tt[u_wavefield.shape[0] - 1 - it]

    return objective, gradient


def fwi_gradient_descent(
    shape: tuple[int, int],
    extent: tuple[float, float],
    vp_initial: np.ndarray,
    vp_true: np.ndarray,
    src_positions: np.ndarray,
    rec_coords: np.ndarray,
    f0: float = 0.01,
    t_end: float = 1000.0,
    niter: int = 10,
    vmin: float = 1.5,
    vmax: float = 4.5,
    step_length_method: str = 'simple',
    save_gradients: bool = False,
    callback: Callable[[int, float, np.ndarray], None] | None = None,
) -> FWIResult:
    """Run FWI with gradient descent optimization.

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    extent : tuple
        Domain extent (Lx, Lz) in meters
    vp_initial : np.ndarray
        Initial (smooth) velocity model
    vp_true : np.ndarray
        True velocity model (for generating observed data)
    src_positions : np.ndarray
        Source positions, shape (nshots, 2)
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    f0 : float
        Source peak frequency in kHz
    t_end : float
        Simulation end time in ms
    niter : int
        Number of FWI iterations
    vmin : float
        Minimum velocity constraint
    vmax : float
        Maximum velocity constraint
    step_length_method : str
        Step length method: 'simple' or 'backtracking'
    save_gradients : bool
        Whether to save gradient at each iteration
    callback : callable, optional
        Function called after each iteration: callback(iter, objective, vp)

    Returns
    -------
    FWIResult
        Results containing final model, history, and optionally gradients
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required for FWI")

    vp_current = vp_initial.copy()
    history = np.zeros(niter)
    gradients = [] if save_gradients else None

    for i in range(niter):
        # Compute objective and gradient
        objective, gradient = compute_fwi_gradient(
            shape, extent, vp_current, vp_true,
            src_positions, rec_coords, f0, t_end
        )

        history[i] = objective

        if save_gradients:
            gradients.append(gradient.copy())

        # Compute step length
        if step_length_method == 'simple':
            # Simple scaling by max gradient magnitude
            alpha = 0.05 / max(np.max(np.abs(gradient)), 1e-10)
        elif step_length_method == 'backtracking':
            # Backtracking line search (simplified)
            alpha = 0.1 / max(np.max(np.abs(gradient)), 1e-10)
            # Could implement actual line search here
        else:
            raise ValueError(f"Unknown step length method: {step_length_method}")

        # Update with box constraints
        vp_current = update_with_box_constraint(
            vp_current, alpha, gradient, vmin, vmax
        )

        # Call callback if provided
        if callback is not None:
            callback(i, objective, vp_current.copy())

    return FWIResult(
        vp_final=vp_current,
        vp_initial=vp_initial.copy(),
        vp_true=vp_true.copy(),
        history=history,
        gradients=gradients if save_gradients else [],
        iterations=niter,
    )
