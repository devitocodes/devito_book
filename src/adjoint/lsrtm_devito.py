"""Least-Squares Reverse Time Migration (LSRTM) using Devito DSL.

This module provides Least-Squares RTM implementation using Born modeling
and its adjoint. LSRTM iteratively improves the migrated image by minimizing
the difference between Born-modeled and observed data.

The optimization problem is:

    minimize_{m} f(m) = 0.5 * ||L*m - d||^2

where:
    - m is the reflectivity (velocity perturbation)
    - L is the Born modeling operator
    - d is the observed data

Born modeling consists of two steps:
    1. Solve background wavefield: m0 * d2p0/dt2 - laplace(p0) = source
    2. Solve scattered wavefield: m0 * d2dp/dt2 - laplace(dp) = -dm * d2p0/dt2

The adjoint (migration) operator correlates the adjoint wavefield with
the second time derivative of the forward wavefield.

References
----------
[1] Dai, W. and Schuster, G.T.: Plane-wave least-squares reverse-time
    migration, GEOPHYSICS, 78, S165-S177, 2013.
[2] Oliveira et al.: Least-squares reverse time migration (LSRTM)
    in the shot domain, Brazilian Journal of Geophysics, 34, 2016.
[3] Barzilai, J. and Borwein, J.: Two-point step size gradient method,
    IMA Journal of Numerical Analysis, 8, 141-148, 1988.
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
    "LSRTMResult",
    "barzilai_borwein_step",
    "born_adjoint",
    "born_modeling",
    "create_layered_model",
    "lsrtm_steepest_descent",
]


@dataclass
class LSRTMResult:
    """Results from Least-Squares RTM.

    Attributes
    ----------
    image_final : np.ndarray
        Final migrated image (reflectivity model)
    image_initial : np.ndarray
        Initial RTM image (first iteration)
    history : np.ndarray
        Objective function value at each iteration
    iterations : int
        Number of iterations performed
    """
    image_final: np.ndarray
    image_initial: np.ndarray
    history: np.ndarray = field(default_factory=lambda: np.array([]))
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


def create_layered_model(
    shape: tuple[int, int],
    spacing: tuple[float, float],
    vp_layers: list[float] | None = None,
    layer_depths: list[float] | None = None,
) -> np.ndarray:
    """Create a layered velocity model.

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    spacing : tuple
        Grid spacing (dx, dz) in meters
    vp_layers : list, optional
        Velocities for each layer in km/s. Default: [1.5, 2.0, 2.5, 3.0]
    layer_depths : list, optional
        Depth of layer interfaces in meters. Default: evenly spaced.

    Returns
    -------
    np.ndarray
        Velocity model with shape (nx, nz)
    """
    nx, nz = shape
    dx, dz = spacing

    if vp_layers is None:
        vp_layers = [1.5, 2.0, 2.5, 3.0]

    nlayers = len(vp_layers)

    if layer_depths is None:
        # Evenly space layers
        layer_depths = [(i + 1) * nz * dz / nlayers for i in range(nlayers - 1)]

    vp = np.full(shape, vp_layers[0], dtype=np.float32)

    for i, depth in enumerate(layer_depths):
        iz = int(depth / dz)
        if iz < nz:
            vp[:, iz:] = vp_layers[i + 1]

    return vp


def _solve_forward_background(
    grid: "Grid",
    vp_smooth: np.ndarray,
    src_coords: np.ndarray,
    t_end: float,
    dt: float,
    f0: float = 0.01,
    space_order: int = 4,
) -> np.ndarray:
    """Solve background wavefield equation.

    Parameters
    ----------
    grid : Grid
        Devito grid
    vp_smooth : np.ndarray
        Smooth background velocity model
    src_coords : np.ndarray
        Source coordinates, shape (1, 2)
    t_end : float
        End time in ms
    dt : float
        Time step in ms
    f0 : float
        Source peak frequency in kHz
    space_order : int
        Spatial discretization order

    Returns
    -------
    np.ndarray
        Background wavefield, shape (nt, nx, nz)
    """
    nt = int(t_end / dt) + 1

    # Create time function for background wavefield
    p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order,
                     save=nt)

    # Create velocity Function
    v = Function(name='v', grid=grid, space_order=space_order)
    v.data[:] = vp_smooth

    # Wave equation: m * p0_tt = laplace(p0)
    m = 1.0 / (v * v)
    pde = m * p0.dt2 - p0.laplace
    stencil = Eq(p0.forward, solve(pde, p0.forward))

    # Source
    t_vals = np.arange(nt) * dt
    src_data = _ricker_wavelet(t_vals, f0=f0)

    src = SparseTimeFunction(
        name='src', grid=grid, npoint=1, nt=nt
    )
    src.coordinates.data[:] = src_coords
    src.data[:] = src_data.reshape(-1, 1)

    src_term = src.inject(field=p0.forward, expr=src * dt**2 / m)

    # Create and run operator
    op = Operator([stencil] + src_term, name='forward_background')
    op.apply(time_M=nt-2, dt=dt)

    return p0.data.copy()


def _compute_wavefield_dt2(wavefield: np.ndarray, dt: float) -> np.ndarray:
    """Compute second time derivative of wavefield.

    Parameters
    ----------
    wavefield : np.ndarray
        Wavefield array, shape (nt, nx, nz)
    dt : float
        Time step

    Returns
    -------
    np.ndarray
        Second time derivative, shape (nt, nx, nz)
    """
    nt = wavefield.shape[0]
    dt2 = np.zeros_like(wavefield)

    dt2[1:-1] = (wavefield[2:] - 2*wavefield[1:-1] + wavefield[:-2]) / dt**2

    return dt2


def born_modeling(
    shape: tuple[int, int],
    extent: tuple[float, float],
    vp_smooth: np.ndarray,
    reflectivity: np.ndarray,
    src_coords: np.ndarray,
    rec_coords: np.ndarray,
    f0: float = 0.01,
    t_end: float = 1000.0,
    space_order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Born modeling operator: L*m -> data.

    Computes scattered wavefield using reflectivity as virtual source.
    Two-step process:
        1. Solve: m0 * d2p0/dt2 - laplace(p0) = source
        2. Solve: m0 * d2dp/dt2 - laplace(dp) = -dm * d2p0/dt2

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    extent : tuple
        Domain extent (Lx, Lz) in meters
    vp_smooth : np.ndarray
        Smooth background velocity model
    reflectivity : np.ndarray
        Reflectivity model (perturbation in squared slowness)
    src_coords : np.ndarray
        Source coordinates, shape (1, 2)
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
        (receiver_data, background_wavefield) where receiver_data is
        shape (nt, nrec) and background_wavefield is shape (nt, nx, nz)
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required for Born modeling")

    # Create grid
    grid = Grid(shape=shape, extent=extent)

    # Compute stable time step
    dx = extent[0] / (shape[0] - 1)
    dz = extent[1] / (shape[1] - 1)
    vp_max = np.max(vp_smooth)
    dt = 0.4 * min(dx, dz) / vp_max

    nt = int(t_end / dt) + 1

    # Step 1: Compute background wavefield
    p0_wavefield = _solve_forward_background(
        grid, vp_smooth, src_coords, t_end, dt, f0, space_order
    )

    # Compute second time derivative of background wavefield
    p0_tt = _compute_wavefield_dt2(p0_wavefield, dt)

    # Step 2: Solve for scattered wavefield
    dp = TimeFunction(name='dp', grid=grid, time_order=2, space_order=space_order)

    # Create velocity Function
    v = Function(name='v', grid=grid, space_order=space_order)
    v.data[:] = vp_smooth

    # Reflectivity as Function
    dm = Function(name='dm', grid=grid)
    dm.data[:] = reflectivity

    m = 1.0 / (v * v)
    pde = m * dp.dt2 - dp.laplace
    stencil = Eq(dp.forward, solve(pde, dp.forward))

    # Receivers
    rec = SparseTimeFunction(
        name='rec', grid=grid, npoint=rec_coords.shape[0], nt=nt
    )
    rec.coordinates.data[:] = rec_coords

    rec_term = rec.interpolate(expr=dp)

    # Create operator
    op = Operator([stencil] + rec_term, name='born_forward')

    # Time stepping with virtual source injection
    for it in range(1, nt - 1):
        # Inject virtual source: -dm * d2p0/dt2
        dp.data[1, :, :] += -dt**2 * dm.data * p0_tt[it, :, :]

        # Run one time step
        op.apply(time_m=1, time_M=1, dt=dt)

        # Cycle time levels
        dp.data[0, :, :] = dp.data[1, :, :]
        dp.data[1, :, :] = dp.data[2, :, :]

    return rec.data.copy(), p0_wavefield


def born_adjoint(
    shape: tuple[int, int],
    extent: tuple[float, float],
    vp_smooth: np.ndarray,
    data_residual: np.ndarray,
    forward_wavefield: np.ndarray,
    rec_coords: np.ndarray,
    dt: float,
    space_order: int = 4,
) -> np.ndarray:
    """Born adjoint operator: L^T * residual -> gradient.

    Back-propagates residual and correlates with forward wavefield
    to produce the migration image (gradient).

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    extent : tuple
        Domain extent (Lx, Lz) in meters
    vp_smooth : np.ndarray
        Smooth background velocity model
    data_residual : np.ndarray
        Data residual, shape (nt, nrec)
    forward_wavefield : np.ndarray
        Forward background wavefield, shape (nt, nx, nz)
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    dt : float
        Time step in ms
    space_order : int
        Spatial discretization order

    Returns
    -------
    np.ndarray
        Migration image (gradient), shape (nx, nz)
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required for Born adjoint")

    grid = Grid(shape=shape, extent=extent)
    nt = data_residual.shape[0]

    # Create adjoint wavefield
    v_adj = TimeFunction(name='v', grid=grid, time_order=2, space_order=space_order)

    # Create velocity Function
    vel = Function(name='vel', grid=grid, space_order=space_order)
    vel.data[:] = vp_smooth

    m = 1.0 / (vel * vel)
    pde = m * v_adj.dt2 - v_adj.laplace
    stencil = Eq(v_adj.forward, solve(pde, v_adj.forward))

    # Receivers for adjoint source (time-reversed residual)
    rec_adj = SparseTimeFunction(
        name='rec_adj', grid=grid, npoint=rec_coords.shape[0], nt=nt
    )
    rec_adj.coordinates.data[:] = rec_coords
    rec_adj.data[:] = data_residual[::-1, :]

    rec_term = rec_adj.inject(field=v_adj.forward, expr=rec_adj * dt**2 / m)

    # Create operator
    op = Operator([stencil] + rec_term, name='born_adjoint')

    # Compute second time derivative of forward wavefield
    p0_tt = _compute_wavefield_dt2(forward_wavefield, dt)

    # Initialize gradient
    gradient = np.zeros(shape, dtype=np.float32)

    # Time stepping and imaging condition
    for it in range(nt - 1):
        # Run one time step of adjoint
        op.apply(time_m=1, time_M=1, dt=dt, time=it)

        # Imaging condition: correlate v_adj with p0_tt
        # Time-reverse index for forward wavefield
        it_fwd = nt - 1 - it
        gradient += v_adj.data[1, :, :] * p0_tt[it_fwd, :, :]

        # Cycle time levels
        v_adj.data[0, :, :] = v_adj.data[1, :, :]
        v_adj.data[1, :, :] = v_adj.data[2, :, :]

    return gradient


def barzilai_borwein_step(
    s_prev: np.ndarray,
    y_prev: np.ndarray,
    iteration: int,
) -> float:
    """Compute Barzilai-Borwein step length.

    Two variants:
        alpha_BB1 = (s^T * s) / (s^T * y)
        alpha_BB2 = (s^T * y) / (y^T * y)

    where s = m_k - m_{k-1} and y = g_k - g_{k-1}

    Selects BB2 if 0 < BB2/BB1 < 1, otherwise BB1.

    Parameters
    ----------
    s_prev : np.ndarray
        Difference in model: m_k - m_{k-1}
    y_prev : np.ndarray
        Difference in gradient: g_k - g_{k-1}
    iteration : int
        Current iteration number

    Returns
    -------
    float
        Barzilai-Borwein step length
    """
    s_flat = s_prev.ravel()
    y_flat = y_prev.ravel()

    s_dot_s = np.dot(s_flat, s_flat)
    s_dot_y = np.dot(s_flat, y_flat)
    y_dot_y = np.dot(y_flat, y_flat)

    # Avoid division by zero
    eps = 1e-10

    if abs(s_dot_y) < eps:
        return 0.05 / max(np.max(np.abs(y_prev)), eps)

    alpha_bb1 = s_dot_s / s_dot_y if abs(s_dot_y) > eps else 1.0

    if abs(y_dot_y) < eps:
        return alpha_bb1

    alpha_bb2 = s_dot_y / y_dot_y

    # Selection criterion
    ratio = alpha_bb2 / alpha_bb1 if abs(alpha_bb1) > eps else 0.0

    if 0 < ratio < 1:
        return alpha_bb2
    else:
        return alpha_bb1


def lsrtm_steepest_descent(
    shape: tuple[int, int],
    extent: tuple[float, float],
    vp_smooth: np.ndarray,
    vp_true: np.ndarray,
    src_positions: np.ndarray,
    rec_coords: np.ndarray,
    f0: float = 0.01,
    t_end: float = 1000.0,
    niter: int = 20,
    space_order: int = 4,
    callback: Callable[[int, float, np.ndarray], None] | None = None,
) -> LSRTMResult:
    """Least-Squares RTM with steepest descent.

    Minimizes: f(m) = 0.5 * ||L*m - d||^2

    Uses Barzilai-Borwein step length for faster convergence.

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    extent : tuple
        Domain extent (Lx, Lz) in meters
    vp_smooth : np.ndarray
        Smooth background velocity model
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
        Number of LSRTM iterations
    space_order : int
        Spatial discretization order
    callback : callable, optional
        Function called after each iteration: callback(iter, objective, image)

    Returns
    -------
    LSRTMResult
        Results containing final image, initial image, and history
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required for LSRTM")

    nshots = src_positions.shape[0]

    # Compute stable time step
    dx = extent[0] / (shape[0] - 1)
    dz = extent[1] / (shape[1] - 1)
    vp_max = max(np.max(vp_smooth), np.max(vp_true))
    dt = 0.4 * min(dx, dz) / vp_max

    # Compute true reflectivity for generating observed data
    m0 = 1.0 / vp_smooth**2
    m_true = 1.0 / vp_true**2
    dm_true = m_true - m0

    # Initialize reflectivity model (current estimate)
    dm = np.zeros(shape, dtype=np.float32)

    history = np.zeros(niter)
    image_initial = None

    # Previous values for Barzilai-Borwein
    dm_prev = np.zeros_like(dm)
    grad_prev = np.zeros_like(dm)

    for k in range(niter):
        objective = 0.0
        gradient = np.zeros(shape, dtype=np.float32)

        for ishot in range(nshots):
            src_coords = src_positions[ishot:ishot+1, :]

            # Generate observed data using true reflectivity
            rec_obs, p0_true = born_modeling(
                shape, extent, vp_smooth, dm_true,
                src_coords, rec_coords, f0, t_end, space_order
            )

            # Generate synthetic data using current reflectivity
            rec_syn, p0_current = born_modeling(
                shape, extent, vp_smooth, dm,
                src_coords, rec_coords, f0, t_end, space_order
            )

            # Compute residual
            residual = rec_syn - rec_obs

            # Update objective
            objective += 0.5 * np.sum(residual**2)

            # Compute gradient (Born adjoint)
            grad_shot = born_adjoint(
                shape, extent, vp_smooth, residual,
                p0_current, rec_coords, dt, space_order
            )

            gradient += grad_shot

        history[k] = objective

        # Save initial image (first iteration gradient is RTM image)
        if k == 0:
            image_initial = -gradient.copy()  # Negative because we're doing descent

        # Compute step length
        if k == 0:
            # First iteration: simple scaling
            alpha = 0.05 / max(np.max(np.abs(gradient)), 1e-10)
        else:
            # Barzilai-Borwein
            s_prev = dm - dm_prev
            y_prev = gradient - grad_prev
            alpha = barzilai_borwein_step(s_prev, y_prev, k)

        # Store previous values
        dm_prev = dm.copy()
        grad_prev = gradient.copy()

        # Update reflectivity
        dm = dm - alpha * gradient

        # Call callback if provided
        if callback is not None:
            callback(k, objective, dm.copy())

    return LSRTMResult(
        image_final=dm,
        image_initial=image_initial,
        history=history,
        iterations=niter,
    )
