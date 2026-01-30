"""Dask utilities for distributed Devito workflows.

This module provides functions for running Devito computations in parallel
using Dask distributed. All functions are designed to be submitted as
Dask tasks and create Devito objects internally to avoid serialization
issues.

Functions:
    create_local_cluster: Create a LocalCluster and Client
    forward_shot: Forward modeling for a single shot (Dask-compatible)
    fwi_gradient_single_shot: FWI gradient for a single shot (Dask-compatible)
    parallel_forward_modeling: Run forward modeling for multiple shots
    parallel_fwi_gradient: Compute FWI gradient for multiple shots
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class FGPair:
    """Functional-gradient pair for reduction operations.

    This class supports addition for summing results from multiple shots.

    Attributes
    ----------
    f : float
        Objective function value
    g : np.ndarray
        Gradient array
    """

    f: float
    g: np.ndarray

    def __add__(self, other: "FGPair") -> "FGPair":
        """Add two FGPairs (for reduction)."""
        return FGPair(self.f + other.f, self.g + other.g)

    def __radd__(self, other):
        """Right addition (supports sum() starting from 0)."""
        if other == 0:
            return self
        return self.__add__(other)


def ricker_wavelet(t: np.ndarray, f0: float, t0: float | None = None) -> np.ndarray:
    """Generate a Ricker wavelet.

    Parameters
    ----------
    t : np.ndarray
        Time array
    f0 : float
        Peak frequency
    t0 : float, optional
        Time delay. Default is 1.5/f0

    Returns
    -------
    np.ndarray
        Ricker wavelet values
    """
    if t0 is None:
        t0 = 1.5 / f0
    pi_f0_t = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * pi_f0_t**2) * np.exp(-pi_f0_t**2)


def create_local_cluster(
    n_workers: int = 4,
    threads_per_worker: int = 1,
    death_timeout: int = 600,
) -> tuple:
    """Create a Dask LocalCluster and Client.

    Parameters
    ----------
    n_workers : int, optional
        Number of workers. Default is 4
    threads_per_worker : int, optional
        Threads per worker. Default is 1
    death_timeout : int, optional
        Timeout for worker death in seconds. Default is 600

    Returns
    -------
    tuple
        (cluster, client)

    Raises
    ------
    ImportError
        If dask.distributed is not available
    """
    try:
        from dask.distributed import Client, LocalCluster
    except ImportError as e:
        raise ImportError(
            "dask.distributed is required for parallel execution. "
            "Install with: pip install dask[distributed]"
        ) from e

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        death_timeout=death_timeout,
    )
    client = Client(cluster)

    return cluster, client


def forward_shot(
    shot_id: int,
    velocity: np.ndarray,
    src_coord: np.ndarray,
    rec_coords: np.ndarray,
    nt: int,
    dt: float,
    f0: float,
    extent: tuple[float, float],
    space_order: int = 4,
) -> np.ndarray:
    """Run forward modeling for a single shot.

    This function is designed to be submitted as a Dask task.
    All Devito objects are created inside the function to avoid
    serialization issues.

    Parameters
    ----------
    shot_id : int
        Shot identifier (for logging)
    velocity : np.ndarray
        Velocity model (2D array)
    src_coord : np.ndarray
        Source coordinates [x, z]
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    nt : int
        Number of time steps
    dt : float
        Time step
    f0 : float
        Source peak frequency
    extent : tuple
        Domain extent (Lx, Lz)
    space_order : int, optional
        Spatial discretization order. Default is 4

    Returns
    -------
    np.ndarray
        Receiver data, shape (nt, nrec)
    """
    # Import Devito inside function to ensure fresh compilation on worker
    from devito import (
        Eq,
        Function,
        Grid,
        Operator,
        SparseTimeFunction,
        TimeFunction,
        solve,
    )

    shape = velocity.shape
    grid = Grid(shape=shape, extent=extent, dtype=np.float32)

    # Velocity field
    vel = Function(name="vel", grid=grid, space_order=space_order)
    vel.data[:] = velocity

    # Wavefield
    u = TimeFunction(name="u", grid=grid, time_order=2, space_order=space_order)

    # Source
    src_coords_arr = np.array([src_coord])
    src = SparseTimeFunction(
        name="src", grid=grid, npoint=1, nt=nt, coordinates=src_coords_arr
    )
    time_values = np.arange(nt) * dt
    src.data[:, 0] = ricker_wavelet(time_values, f0)

    # Receivers
    nrec = len(rec_coords)
    rec = SparseTimeFunction(
        name="rec", grid=grid, npoint=nrec, nt=nt, coordinates=rec_coords
    )

    # Build operator
    pde = (1.0 / vel**2) * u.dt2 - u.laplace
    stencil = Eq(u.forward, solve(pde, u.forward))
    src_term = src.inject(
        field=u.forward, expr=src * grid.stepping_dim.spacing**2 * vel**2
    )
    rec_term = rec.interpolate(expr=u)

    op = Operator([stencil] + src_term + rec_term)
    op.apply(time=nt - 2, dt=dt)

    return rec.data.copy()


def fwi_gradient_single_shot(
    velocity: np.ndarray,
    src_coord: np.ndarray,
    rec_coords: np.ndarray,
    d_obs: np.ndarray,
    shape: tuple[int, int],
    extent: tuple[float, float],
    nt: int,
    dt: float,
    f0: float,
    space_order: int = 4,
) -> tuple[float, np.ndarray]:
    """Compute FWI gradient for a single shot.

    This function is designed to be submitted as a Dask task.
    All Devito objects are created inside the function to avoid
    serialization issues.

    Parameters
    ----------
    velocity : np.ndarray
        Current velocity model
    src_coord : np.ndarray
        Source coordinates [x, z]
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    d_obs : np.ndarray
        Observed data for this shot, shape (nt, nrec)
    shape : tuple
        Grid shape
    extent : tuple
        Domain extent
    nt : int
        Number of time steps
    dt : float
        Time step
    f0 : float
        Source peak frequency
    space_order : int, optional
        Spatial discretization order. Default is 4

    Returns
    -------
    tuple
        (objective_value, gradient)
    """
    # Import Devito inside function
    from devito import (
        Eq,
        Function,
        Grid,
        Operator,
        SparseTimeFunction,
        TimeFunction,
        solve,
    )

    grid = Grid(shape=shape, extent=extent, dtype=np.float32)

    # Velocity and squared slowness
    vel = Function(name="vel", grid=grid, space_order=space_order)
    vel.data[:] = velocity
    m = Function(name="m", grid=grid, space_order=space_order)
    m.data[:] = 1.0 / velocity**2

    # Forward wavefield (save all time steps for adjoint correlation)
    u = TimeFunction(
        name="u", grid=grid, time_order=2, space_order=space_order, save=nt
    )

    # Source
    src_coords_arr = np.array([src_coord])
    src = SparseTimeFunction(
        name="src", grid=grid, npoint=1, nt=nt, coordinates=src_coords_arr
    )
    time_values = np.arange(nt) * dt
    src.data[:, 0] = ricker_wavelet(time_values, f0)

    # Receivers
    nrec = len(rec_coords)
    rec = SparseTimeFunction(
        name="rec", grid=grid, npoint=nrec, nt=nt, coordinates=rec_coords
    )

    # Forward operator
    pde = m * u.dt2 - u.laplace
    stencil = Eq(u.forward, solve(pde, u.forward))
    src_term = src.inject(
        field=u.forward, expr=src * grid.stepping_dim.spacing**2 / m
    )
    rec_term = rec.interpolate(expr=u)

    op_fwd = Operator([stencil] + src_term + rec_term)
    op_fwd.apply(time=nt - 2, dt=dt)

    # Compute residual and objective
    n_timesteps = min(rec.data.shape[0], d_obs.shape[0])
    residual_data = rec.data[:n_timesteps, :] - d_obs[:n_timesteps, :]
    objective = 0.5 * np.sum(residual_data**2)

    # Adjoint wavefield
    v = TimeFunction(name="v", grid=grid, time_order=2, space_order=space_order)

    # Gradient
    grad = Function(name="grad", grid=grid)

    # Residual injection
    residual = SparseTimeFunction(
        name="residual", grid=grid, npoint=nrec, nt=nt, coordinates=rec_coords
    )
    residual.data[:n_timesteps, :] = residual_data

    # Adjoint operator
    pde_adj = m * v.dt2 - v.laplace
    stencil_adj = Eq(v.backward, solve(pde_adj, v.backward))
    res_term = residual.inject(
        field=v.backward, expr=residual * grid.stepping_dim.spacing**2 / m
    )

    # Gradient update: grad += u * v.dt2
    gradient_update = Eq(grad, grad + u * v.dt2)

    op_adj = Operator([stencil_adj] + res_term + [gradient_update])
    op_adj.apply(u=u, v=v, dt=dt, time_M=nt - 2)

    return objective, grad.data.copy()


def fwi_gradient_single_shot_fg_pair(
    velocity: np.ndarray,
    src_coord: np.ndarray,
    rec_coords: np.ndarray,
    d_obs: np.ndarray,
    shape: tuple[int, int],
    extent: tuple[float, float],
    nt: int,
    dt: float,
    f0: float,
    space_order: int = 4,
) -> FGPair:
    """Compute FWI gradient for a single shot, returning FGPair.

    Same as fwi_gradient_single_shot but returns FGPair for
    Dask reduction operations.

    Parameters
    ----------
    (same as fwi_gradient_single_shot)

    Returns
    -------
    FGPair
        Objective and gradient pair
    """
    objective, gradient = fwi_gradient_single_shot(
        velocity,
        src_coord,
        rec_coords,
        d_obs,
        shape,
        extent,
        nt,
        dt,
        f0,
        space_order,
    )
    return FGPair(objective, gradient)


def sum_fg_pairs(fg_pairs: list[FGPair]) -> FGPair:
    """Sum a list of FGPairs.

    Parameters
    ----------
    fg_pairs : list
        List of FGPair objects

    Returns
    -------
    FGPair
        Sum of all pairs
    """
    return sum(fg_pairs)


def parallel_forward_modeling(
    client,
    velocity: np.ndarray,
    src_positions: np.ndarray,
    rec_coords: np.ndarray,
    nt: int,
    dt: float,
    f0: float,
    extent: tuple[float, float],
    space_order: int = 4,
) -> list[np.ndarray]:
    """Run forward modeling for multiple shots in parallel.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask client
    velocity : np.ndarray
        Velocity model
    src_positions : np.ndarray
        Source positions, shape (nshots, 2)
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    nt : int
        Number of time steps
    dt : float
        Time step
    f0 : float
        Source peak frequency
    extent : tuple
        Domain extent
    space_order : int, optional
        Spatial discretization order. Default is 4

    Returns
    -------
    list
        List of shot records (numpy arrays)
    """
    from dask.distributed import wait

    nshots = len(src_positions)

    # Submit tasks
    futures = []
    for i in range(nshots):
        future = client.submit(
            forward_shot,
            i,
            velocity,
            src_positions[i],
            rec_coords,
            nt,
            dt,
            f0,
            extent,
            space_order,
        )
        futures.append(future)

    # Wait and gather
    wait(futures)
    return client.gather(futures)


def parallel_fwi_gradient(
    client,
    velocity: np.ndarray,
    src_positions: np.ndarray,
    rec_coords: np.ndarray,
    observed_data: list[np.ndarray],
    shape: tuple[int, int],
    extent: tuple[float, float],
    nt: int,
    dt: float,
    f0: float,
    space_order: int = 4,
    use_reduction: bool = False,
) -> tuple[float, np.ndarray]:
    """Compute FWI gradient for multiple shots in parallel.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask client
    velocity : np.ndarray
        Current velocity model
    src_positions : np.ndarray
        Source positions, shape (nshots, 2)
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2)
    observed_data : list
        List of observed data arrays, one per shot
    shape : tuple
        Grid shape
    extent : tuple
        Domain extent
    nt : int
        Number of time steps
    dt : float
        Time step
    f0 : float
        Source peak frequency
    space_order : int, optional
        Spatial discretization order. Default is 4
    use_reduction : bool, optional
        Use Dask reduction (sum) instead of gather. Default is False

    Returns
    -------
    tuple
        (total_objective, total_gradient)
    """
    from dask.distributed import wait

    nshots = len(src_positions)

    if use_reduction:
        # Use FGPair for Dask reduction
        futures = []
        for i in range(nshots):
            future = client.submit(
                fwi_gradient_single_shot_fg_pair,
                velocity,
                src_positions[i],
                rec_coords,
                observed_data[i],
                shape,
                extent,
                nt,
                dt,
                f0,
                space_order,
            )
            futures.append(future)

        # Reduce using sum
        total_fg = client.submit(sum, futures)
        result = total_fg.result()
        return result.f, result.g

    else:
        # Standard gather and sum
        futures = []
        for i in range(nshots):
            future = client.submit(
                fwi_gradient_single_shot,
                velocity,
                src_positions[i],
                rec_coords,
                observed_data[i],
                shape,
                extent,
                nt,
                dt,
                f0,
                space_order,
            )
            futures.append(future)

        wait(futures)

        # Gather and reduce
        total_objective = 0.0
        total_gradient = np.zeros(shape)

        for future in futures:
            obj, grad = future.result()
            total_objective += obj
            total_gradient += grad

        return total_objective, total_gradient


def create_scipy_loss_function(
    client,
    shape: tuple[int, int],
    extent: tuple[float, float],
    src_positions: np.ndarray,
    rec_coords: np.ndarray,
    observed_data: list[np.ndarray],
    nt: int,
    dt: float,
    f0: float,
    vmin: float = 1.4,
    vmax: float = 4.0,
    space_order: int = 4,
) -> Callable:
    """Create a loss function compatible with scipy.optimize.

    The returned function takes squared slowness as input and returns
    (objective, gradient) for use with scipy.optimize.minimize.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask client
    shape : tuple
        Grid shape
    extent : tuple
        Domain extent
    src_positions : np.ndarray
        Source positions, shape (nshots, 2)
    rec_coords : np.ndarray
        Receiver coordinates
    observed_data : list
        List of observed data arrays
    nt : int
        Number of time steps
    dt : float
        Time step
    f0 : float
        Source peak frequency
    vmin : float, optional
        Minimum velocity for clipping. Default is 1.4
    vmax : float, optional
        Maximum velocity for clipping. Default is 4.0
    space_order : int, optional
        Spatial discretization order. Default is 4

    Returns
    -------
    callable
        Function with signature f(m_flat) -> (objective, gradient_flat)
    """

    def loss(m_flat: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute FWI loss and gradient.

        Parameters
        ----------
        m_flat : np.ndarray
            Squared slowness, flattened (1D array)

        Returns
        -------
        tuple
            (objective, gradient) where gradient is 1D float64
        """
        # Convert squared-slowness to velocity
        m = m_flat.reshape(shape)
        velocity = 1.0 / np.sqrt(m)
        velocity = np.clip(velocity, vmin, vmax).astype(np.float32)

        # Compute objective and gradient in parallel
        objective, gradient = parallel_fwi_gradient(
            client,
            velocity,
            src_positions,
            rec_coords,
            observed_data,
            shape,
            extent,
            nt,
            dt,
            f0,
            space_order,
        )

        # Convert gradient to flat float64 (required by scipy)
        grad_flat = gradient.flatten().astype(np.float64)

        return objective, grad_flat

    return loss
