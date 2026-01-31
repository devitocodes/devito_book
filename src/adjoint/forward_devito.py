"""2D Acoustic Forward Modeling using Devito DSL.

Solves the 2D acoustic wave equation:
    (1/v^2) * u_tt - laplace(u) = s(x, z, t)

on domain [0, Lx] x [0, Lz] with:
    - Velocity model v(x, z)
    - Point source with Ricker wavelet
    - Point receivers recording wavefield

This module uses the EXPLICIT Devito API:
    - Grid, Function, TimeFunction, SparseTimeFunction
    - Eq, Operator, solve

NO convenience classes are used (no SeismicModel, AcousticWaveSolver, etc.)

Usage:
    from src.adjoint import solve_forward_2d, ricker_wavelet

    result = solve_forward_2d(
        shape=(101, 101),
        extent=(1000., 1000.),
        vp=velocity_model,
        t_end=1000.0,
        f0=0.010,
        src_coords=np.array([[500., 20.]]),
        rec_coords=rec_coords,
    )
"""

import importlib.util
from dataclasses import dataclass

import numpy as np

DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None


def ricker_wavelet(
    t: np.ndarray,
    f0: float,
    t0: float | None = None,
    amp: float = 1.0,
) -> np.ndarray:
    """Generate a Ricker wavelet (Mexican hat wavelet).

    The Ricker wavelet is the negative normalized second derivative of a
    Gaussian. It is commonly used in seismic modeling due to its compact
    support in both time and frequency domains.

    r(t) = amp * (1 - 2*(pi*f0*(t-t0))^2) * exp(-(pi*f0*(t-t0))^2)

    Parameters
    ----------
    t : np.ndarray
        Time array
    f0 : float
        Peak frequency in Hz (or kHz depending on time units)
    t0 : float, optional
        Time shift (delay). If None, defaults to 1.5/f0 to ensure
        the wavelet starts near zero.
    amp : float
        Amplitude scaling factor

    Returns
    -------
    np.ndarray
        Ricker wavelet values at times t

    Notes
    -----
    The wavelet has zero mean and is bandlimited. The frequency spectrum
    has a peak at f0 and falls off on both sides. The wavelet is
    essentially zero outside |t - t0| > 1.5/f0.

    Examples
    --------
    >>> t = np.linspace(0, 1000, 2001)  # Time in ms
    >>> src = ricker_wavelet(t, f0=0.010)  # 10 Hz
    >>> plt.plot(t, src)
    """
    if t0 is None:
        t0 = 1.5 / f0

    # Normalized time
    pi_f0_t = np.pi * f0 * (t - t0)
    pi_f0_t_sq = pi_f0_t ** 2

    return amp * (1.0 - 2.0 * pi_f0_t_sq) * np.exp(-pi_f0_t_sq)


@dataclass
class ForwardResult:
    """Results from 2D acoustic forward modeling.

    Attributes
    ----------
    u : np.ndarray
        Wavefield at final time or full wavefield if save_wavefield=True.
        Shape: (nt, nx, nz) if saved, (3, nx, nz) otherwise.
    rec : np.ndarray
        Receiver recordings (shot record), shape (nt, nrec)
    x : np.ndarray
        X coordinates of grid points
    z : np.ndarray
        Z coordinates of grid points
    t : np.ndarray
        Time array
    dt : float
        Time step used
    src_coords : np.ndarray
        Source coordinates used
    rec_coords : np.ndarray
        Receiver coordinates used
    """
    u: np.ndarray
    rec: np.ndarray
    x: np.ndarray
    z: np.ndarray
    t: np.ndarray
    dt: float
    src_coords: np.ndarray
    rec_coords: np.ndarray


def solve_forward_2d(
    shape: tuple[int, int],
    extent: tuple[float, float],
    vp: np.ndarray | float,
    t_end: float,
    f0: float,
    src_coords: np.ndarray,
    rec_coords: np.ndarray,
    space_order: int = 4,
    dt: float | None = None,
    save_wavefield: bool = False,
    t0: float = 0.0,
) -> ForwardResult:
    """2D acoustic forward modeling with explicit Devito API.

    Solves the acoustic wave equation:
        (1/v^2) * u_tt - laplace(u) = s

    using second-order time stepping and configurable spatial order.

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, nz)
    extent : tuple
        Physical extent (Lx, Lz) in meters
    vp : np.ndarray or float
        P-wave velocity model. If float, creates homogeneous model.
        Array should have shape (nx, nz).
    t_end : float
        End time in milliseconds
    f0 : float
        Source peak frequency in kHz (e.g., 0.010 for 10 Hz)
    src_coords : np.ndarray
        Source coordinates, shape (nsrc, 2) where columns are (x, z)
    rec_coords : np.ndarray
        Receiver coordinates, shape (nrec, 2) where columns are (x, z)
    space_order : int
        Spatial discretization order (default: 4)
    dt : float, optional
        Time step. If None, computed from CFL condition.
    save_wavefield : bool
        If True, save full wavefield for all time steps.
        WARNING: This requires significant memory for large problems.
    t0 : float
        Start time (default: 0.0)

    Returns
    -------
    ForwardResult
        Results including wavefield, receiver data, and grid information.

    Raises
    ------
    ImportError
        If Devito is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> # Create simple velocity model
    >>> vp = np.ones((101, 101)) * 2.0  # 2 km/s
    >>> vp[:, 50:] = 2.5  # Layer at depth
    >>>
    >>> # Source and receivers
    >>> src = np.array([[500., 20.]])
    >>> rec = np.zeros((101, 2))
    >>> rec[:, 0] = np.linspace(0, 1000, 101)
    >>> rec[:, 1] = 30.
    >>>
    >>> result = solve_forward_2d(
    ...     shape=(101, 101),
    ...     extent=(1000., 1000.),
    ...     vp=vp,
    ...     t_end=1000.0,
    ...     f0=0.010,
    ...     src_coords=src,
    ...     rec_coords=rec,
    ... )
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

    # Create velocity field
    vel = DevitoFunction(name='vel', grid=grid, space_order=space_order)
    if isinstance(vp, (int, float)):
        vel.data[:] = float(vp)
    else:
        vel.data[:] = vp

    # Compute time step from CFL condition if not provided
    dx = extent[0] / (shape[0] - 1)
    dz = extent[1] / (shape[1] - 1)
    h_min = min(dx, dz)
    v_max = float(np.max(vel.data))

    if dt is None:
        # CFL condition: dt <= h / (sqrt(2) * v_max) for 2D
        cfl_limit = h_min / (np.sqrt(2) * v_max)
        dt = 0.9 * cfl_limit  # Use 90% of CFL limit

    # Compute number of time steps
    nt = int((t_end - t0) / dt) + 1
    time_values = np.linspace(t0, t_end, nt)

    # Ensure source coordinates is 2D
    src_coords = np.atleast_2d(src_coords)
    nsrc = src_coords.shape[0]

    # Create wavefield
    if save_wavefield:
        u = TimeFunction(
            name='u', grid=grid, time_order=2, space_order=space_order,
            save=nt
        )
    else:
        u = TimeFunction(
            name='u', grid=grid, time_order=2, space_order=space_order
        )

    # Create source using SparseTimeFunction
    src = SparseTimeFunction(
        name='src', grid=grid, npoint=nsrc, nt=nt,
        coordinates=src_coords
    )

    # Set source wavelet
    wavelet = ricker_wavelet(time_values, f0)
    for i in range(nsrc):
        src.data[:, i] = wavelet

    # Create receivers using SparseTimeFunction
    rec_coords = np.atleast_2d(rec_coords)
    nrec = rec_coords.shape[0]

    rec = SparseTimeFunction(
        name='rec', grid=grid, npoint=nrec, nt=nt,
        coordinates=rec_coords
    )

    # Build wave equation
    # PDE: (1/v^2) * u_tt - laplace(u) = 0
    pde = (1.0 / vel**2) * u.dt2 - u.laplace

    # Solve for u.forward
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Source injection: add scaled source to wavefield
    dt_sym = grid.stepping_dim.spacing
    src_term = src.inject(
        field=u.forward,
        expr=src * dt_sym**2 * vel**2
    )

    # Receiver interpolation: sample wavefield at receiver locations
    rec_term = rec.interpolate(expr=u)

    # Create and run operator
    op = Operator([stencil] + src_term + rec_term)
    op.apply(time=nt - 2, dt=dt)

    # Extract results
    x_coords = np.linspace(0, extent[0], shape[0])
    z_coords = np.linspace(0, extent[1], shape[1])

    if save_wavefield:
        u_data = np.array(u.data[:])
    else:
        u_data = np.array(u.data[:])

    return ForwardResult(
        u=u_data,
        rec=np.array(rec.data[:]),
        x=x_coords,
        z=z_coords,
        t=time_values,
        dt=dt,
        src_coords=src_coords,
        rec_coords=rec_coords,
    )


def estimate_dt(vp: np.ndarray | float, extent: tuple, shape: tuple) -> float:
    """Estimate stable time step from CFL condition.

    Parameters
    ----------
    vp : np.ndarray or float
        Velocity model or constant velocity
    extent : tuple
        Physical extent (Lx, Lz)
    shape : tuple
        Grid shape (nx, nz)

    Returns
    -------
    float
        Recommended time step
    """
    dx = extent[0] / (shape[0] - 1)
    dz = extent[1] / (shape[1] - 1)
    h_min = min(dx, dz)

    if isinstance(vp, (int, float)):
        v_max = float(vp)
    else:
        v_max = float(np.max(vp))

    # CFL condition for 2D: dt <= h / (sqrt(2) * v_max)
    return 0.9 * h_min / (np.sqrt(2) * v_max)
