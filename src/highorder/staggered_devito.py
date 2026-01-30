"""Staggered Grid Acoustic Wave Equation Solver using Devito.

This module implements staggered grid finite difference schemes for solving
the first-order acoustic wave equation in velocity-pressure form. Staggered
grids place different variables at different grid locations, naturally
capturing the physics of wave propagation.

The velocity-pressure formulation:
    dp/dt = lambda * div(v)
    dv/dt = (1/rho) * grad(p)

where p is pressure, v is velocity, rho is density, and lambda = rho * c^2.

Usage:
    from src.highorder.staggered_devito import (
        solve_staggered_acoustic_2d,
        StaggeredResult,
    )

    result = solve_staggered_acoustic_2d(
        extent=(2000., 2000.),
        shape=(81, 81),
        velocity=4.0,
        t_end=200.,
        space_order=4,
    )

References:
    [1] Virieux, J. (1986). "P-SV wave propagation in heterogeneous media:
        Velocity-stress finite-difference method." GEOPHYSICS, 51(4).
    [2] Lavender, A.R. (1988). "Fourth-order finite-difference P-SV
        seismograms." GEOPHYSICS, 53(11).
"""

from dataclasses import dataclass

import numpy as np

try:
    from devito import (
        NODE,
        Eq,
        Function,
        Grid,
        Operator,
        SparseTimeFunction,
        TimeFunction,
        VectorTimeFunction,
        div,
        grad,
        solve,
    )
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False

__all__ = [
    "StaggeredResult",
    "dgauss_wavelet",
    "ricker_wavelet",
    "solve_staggered_acoustic_2d",
]


def ricker_wavelet(t: np.ndarray, f0: float = 0.01, A: float = 1.0) -> np.ndarray:
    """Generate a Ricker wavelet (Mexican hat wavelet).

    Parameters
    ----------
    t : np.ndarray
        Time values.
    f0 : float, optional
        Peak frequency in kHz. Default is 0.01 kHz (10 Hz).
    A : float, optional
        Amplitude. Default is 1.0.

    Returns
    -------
    np.ndarray
        Wavelet values at times t.

    Notes
    -----
    The wavelet is centered at t = 1/f0.
    """
    tau = (np.pi * f0 * (t - 1.0 / f0)) ** 2
    return A * (1 - 2 * tau) * np.exp(-tau)


def dgauss_wavelet(
    t: np.ndarray,
    f0: float = 0.01,
    A: float = 0.004,
) -> np.ndarray:
    """Generate a derivative of Gaussian wavelet.

    This is commonly used as a source wavelet in seismic applications.
    It is the first derivative of a Gaussian.

    Parameters
    ----------
    t : np.ndarray
        Time values.
    f0 : float, optional
        Peak frequency in kHz. Default is 0.01 kHz (10 Hz).
    A : float, optional
        Amplitude scaling factor. Default is 0.004.

    Returns
    -------
    np.ndarray
        Wavelet values at times t.
    """
    t0 = 1.0 / f0  # Center time
    tau = t - t0
    sigma = 1.0 / (2 * np.pi * f0)
    return -A * tau / (sigma ** 3 * np.sqrt(2 * np.pi)) * np.exp(-tau ** 2 / (2 * sigma ** 2))


@dataclass
class StaggeredResult:
    """Results from the staggered grid acoustic solver.

    Attributes
    ----------
    p : np.ndarray
        Final pressure field, shape (Nx, Nz).
    vx : np.ndarray
        Final x-velocity field, shape (Nx, Nz).
    vz : np.ndarray
        Final z-velocity field, shape (Nx, Nz).
    x : np.ndarray
        x-coordinate array.
    z : np.ndarray
        z-coordinate array.
    t_final : float
        Final simulation time.
    dt : float
        Time step used.
    nt : int
        Number of time steps.
    space_order : int
        Spatial discretization order used.
    p_norm : float
        L2 norm of final pressure field.
    """
    p: np.ndarray
    vx: np.ndarray
    vz: np.ndarray
    x: np.ndarray
    z: np.ndarray
    t_final: float
    dt: float
    nt: int
    space_order: int
    p_norm: float


def solve_staggered_acoustic_2d(
    extent: tuple[float, float] = (2000.0, 2000.0),
    shape: tuple[int, int] = (81, 81),
    velocity: float | np.ndarray = 4.0,
    density: float | np.ndarray = 1.0,
    t_end: float = 200.0,
    dt: float | None = None,
    courant: float = 0.5,
    f0: float = 0.01,
    source_location: tuple[float, float] | None = None,
    space_order: int = 2,
    wavelet: str = "dgauss",
) -> StaggeredResult:
    """Solve 2D acoustic wave equation with staggered grid scheme.

    This solver uses a staggered grid (Arakawa C-grid) where pressure is
    defined at cell centers and velocity components at cell faces. The
    time integration uses leapfrog (staggered in time).

    Parameters
    ----------
    extent : tuple, optional
        Domain size (Lx, Lz) in meters/km. Default is (2000, 2000).
    shape : tuple, optional
        Grid shape (Nx, Nz). Default is (81, 81).
    velocity : float or np.ndarray, optional
        Wave velocity. Can be scalar (uniform) or 2D array.
        Default is 4.0 (km/s for typical seismic).
    density : float or np.ndarray, optional
        Material density. Can be scalar or 2D array. Default is 1.0.
    t_end : float, optional
        Simulation end time. Default is 200.
    dt : float, optional
        Time step. If None, computed from CFL condition.
    courant : float, optional
        Courant number for stability. Typical range is 0.4-0.5 for staggered
        schemes. Default is 0.5.
    f0 : float, optional
        Source peak frequency in kHz. Default is 0.01 kHz (10 Hz).
    source_location : tuple, optional
        Source (x, z) coordinates. Default is center of domain.
    space_order : int, optional
        Spatial discretization order (2 or 4). Higher order uses wider
        stencils but reduces numerical dispersion. Default is 2.
    wavelet : str, optional
        Source wavelet type: "dgauss" or "ricker". Default is "dgauss".

    Returns
    -------
    StaggeredResult
        Solution data including pressure, velocity fields, and metadata.

    Raises
    ------
    ImportError
        If Devito is not installed.
    ValueError
        If invalid wavelet type is specified.

    Notes
    -----
    The staggered grid discretization:
    - Pressure p: at cell centers (NODE staggering)
    - Velocity vx: at x-faces (half-integer in x)
    - Velocity vz: at z-faces (half-integer in z)

    The update equations are:
    - v^{n+1} = v^n + dt * (1/rho) * grad(p^n)
    - p^{n+1} = p^n + dt * lambda * div(v^{n+1})

    Note the leapfrog structure where the new velocity is used in the
    pressure update.

    Examples
    --------
    >>> result = solve_staggered_acoustic_2d(
    ...     extent=(2000., 2000.),
    ...     shape=(81, 81),
    ...     velocity=4.0,
    ...     t_end=200.,
    ...     space_order=4
    ... )
    >>> print(f"Pressure norm: {result.p_norm:.5f}")

    Compare 2nd and 4th order schemes:

    >>> result_2 = solve_staggered_acoustic_2d(space_order=2)
    >>> result_4 = solve_staggered_acoustic_2d(space_order=4)
    >>> print(f"2nd order norm: {result_2.p_norm:.5f}")
    >>> print(f"4th order norm: {result_4.p_norm:.5f}")
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    if wavelet not in ("dgauss", "ricker"):
        raise ValueError(
            f"Unknown wavelet type: {wavelet}. Use 'dgauss' or 'ricker'."
        )

    # Create grid
    grid = Grid(extent=extent, shape=shape)

    # Compute time step from CFL if not provided
    if dt is None:
        hx = extent[0] / (shape[0] - 1)
        hz = extent[1] / (shape[1] - 1)
        h_min = min(hx, hz)

        if np.isscalar(velocity):
            v_max = velocity
        else:
            v_max = np.amax(velocity)

        # CFL condition for staggered leapfrog: dt <= h / (sqrt(2) * c)
        # With safety factor (courant)
        dt = courant * h_min / (np.sqrt(2) * v_max)

    nt = int(t_end / dt) + 1

    # Create staggered fields
    # Pressure at cell centers (NODE)
    p = TimeFunction(
        name='p', grid=grid, staggered=NODE,
        space_order=space_order, time_order=1
    )

    # Velocity at staggered locations (default for VectorTimeFunction)
    v = VectorTimeFunction(
        name='v', grid=grid,
        space_order=space_order, time_order=1
    )

    # Material properties
    if np.isscalar(velocity):
        V_p = velocity
    else:
        V_p = Function(name='V_p', grid=grid)
        V_p.data[:] = velocity

    if np.isscalar(density):
        rho = density
        ro = 1.0 / density  # 1/rho
    else:
        rho_func = Function(name='rho', grid=grid)
        rho_func.data[:] = density
        rho = rho_func
        ro = 1.0 / rho

    # lambda = rho * c^2
    if np.isscalar(velocity) and np.isscalar(density):
        l2m = V_p ** 2 * density
    else:
        # For heterogeneous case, use Functions
        l2m = V_p ** 2 * rho if not np.isscalar(rho) else V_p ** 2 * density

    # Update equations (leapfrog staggered in time)
    # First update velocity using current pressure
    u_v = Eq(v.forward, solve(v.dt - ro * grad(p), v.forward))

    # Then update pressure using new velocity
    u_p = Eq(p.forward, solve(p.dt - l2m * div(v.forward), p.forward))

    # Source setup
    t_values = np.linspace(0, t_end, nt)

    if wavelet == "dgauss":
        src_data = dgauss_wavelet(t_values, f0=f0)
    else:
        src_data = ricker_wavelet(t_values, f0=f0)

    if source_location is None:
        source_location = (extent[0] / 2, extent[1] / 2)

    source = SparseTimeFunction(
        name='src',
        grid=grid,
        npoint=1,
        nt=nt,
        coordinates=[source_location]
    )
    source.data[:, 0] = src_data

    # Source injection into pressure field
    src_term = source.inject(field=p.forward, expr=source)

    # Build and run operator
    op = Operator([u_v, u_p] + src_term)
    op.apply(time=nt - 1, dt=dt)

    # Extract results
    x_coords = np.linspace(0, extent[0], shape[0])
    z_coords = np.linspace(0, extent[1], shape[1])

    # Compute norm of final pressure field
    p_norm = float(np.linalg.norm(p.data[0]))

    return StaggeredResult(
        p=p.data[0].copy(),
        vx=v[0].data[0].copy(),
        vz=v[1].data[0].copy(),
        x=x_coords,
        z=z_coords,
        t_final=t_end,
        dt=dt,
        nt=nt,
        space_order=space_order,
        p_norm=p_norm,
    )


def compare_space_orders(
    extent: tuple[float, float] = (2000.0, 2000.0),
    shape: tuple[int, int] = (81, 81),
    velocity: float = 4.0,
    t_end: float = 200.0,
) -> tuple[StaggeredResult, StaggeredResult]:
    """Compare 2nd and 4th order staggered grid schemes.

    Parameters
    ----------
    extent : tuple
        Domain size (Lx, Lz).
    shape : tuple
        Grid shape (Nx, Nz).
    velocity : float
        Wave velocity.
    t_end : float
        Simulation end time.

    Returns
    -------
    tuple
        (result_2and, result_4th) - Results for 2nd and 4th order schemes.

    Notes
    -----
    The 4th order scheme uses wider stencils (5 points vs 3 points)
    but has reduced numerical dispersion for the same grid spacing.
    """
    # Second order scheme
    result_2and = solve_staggered_acoustic_2d(
        extent=extent,
        shape=shape,
        velocity=velocity,
        t_end=t_end,
        space_order=2,
    )

    # Fourth order scheme
    result_4th = solve_staggered_acoustic_2d(
        extent=extent,
        shape=shape,
        velocity=velocity,
        t_end=t_end,
        space_order=4,
    )

    return result_2and, result_4th


def convergence_test_staggered(
    grid_sizes: list | None = None,
    extent: tuple[float, float] = (2000.0, 2000.0),
    velocity: float = 4.0,
    t_end: float = 50.0,
    space_order: int = 2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run convergence test for staggered grid solver.

    Uses successively refined grids to estimate the convergence rate.

    Parameters
    ----------
    grid_sizes : list, optional
        List of grid sizes to test. Default: [21, 41, 81, 161].
    extent : tuple
        Domain size (Lx, Lz).
    velocity : float
        Wave velocity.
    t_end : float
        Simulation end time. Keep short for convergence test.
    space_order : int
        Spatial discretization order.

    Returns
    -------
    tuple
        (grid_sizes, norms, observed_order) where norms are the L2 norms
        of pressure fields at each resolution.

    Notes
    -----
    Since we don't have an exact solution, we compare against the finest
    grid solution to estimate the convergence rate.
    """
    if grid_sizes is None:
        grid_sizes = [21, 41, 81, 161]

    norms = []

    for n in grid_sizes:
        result = solve_staggered_acoustic_2d(
            extent=extent,
            shape=(n, n),
            velocity=velocity,
            t_end=t_end,
            space_order=space_order,
        )
        norms.append(result.p_norm)

    grid_sizes = np.array(grid_sizes)
    norms = np.array(norms)

    # Estimate convergence rate from consecutive norms
    # This is a rough estimate since we don't have exact solution
    log_h = np.log(1.0 / grid_sizes[:-1])
    log_diff = np.log(np.abs(norms[:-1] - norms[1:]) + 1e-15)

    if len(log_h) >= 2:
        observed_order = np.polyfit(log_h, log_diff, 1)[0]
    else:
        observed_order = float('nan')

    return grid_sizes, norms, observed_order
