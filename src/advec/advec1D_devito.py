"""1D Advection Equation Solvers using Devito DSL.

Solves the linear advection equation:
    u_t + c * u_x = 0

where c is the advection velocity. The solution propagates the initial
condition I(x) to the right (if c > 0) without change in shape.

Exact solution: u(x, t) = I(x - c*t)

Schemes implemented:
- Upwind: First-order accurate, stable for 0 < C <= 1
- Lax-Wendroff: Second-order accurate, stable for |C| <= 1
- Lax-Friedrichs: First-order accurate, stable for |C| <= 1

where C = c*dt/dx is the Courant number.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from devito import Constant, Eq, Grid, Operator, TimeFunction


@dataclass
class AdvectionResult:
    """Result container for 1D advection solver."""

    u: np.ndarray  # Final solution
    x: np.ndarray  # Spatial coordinates
    t: float  # Final time
    dt: float  # Time step used
    C: float  # Courant number
    u_history: list | None = None  # Solution history (if save_history=True)
    t_history: list | None = None  # Time values (if save_history=True)


def solve_advection_upwind(
    L: float = 1.0,
    c: float = 1.0,
    Nx: int = 100,
    T: float = 1.0,
    C: float = 0.8,
    I: Callable | None = None,
    periodic_bc: bool = True,
    save_history: bool = False,
) -> AdvectionResult:
    """
    Solve 1D advection equation using upwind scheme.

    The upwind scheme uses a backward difference for u_x when c > 0:
        u^{n+1}_i = u^n_i - C*(u^n_i - u^n_{i-1})

    This is first-order accurate in both space and time.
    Stable for 0 < C <= 1.

    Parameters
    ----------
    L : float
        Domain length [0, L]
    c : float
        Advection velocity (must be positive for this scheme)
    Nx : int
        Number of spatial intervals
    T : float
        Final time
    C : float
        Target Courant number (must be <= 1 for stability)
    I : callable
        Initial condition function I(x)
    periodic_bc : bool
        If True, use periodic boundary conditions
    save_history : bool
        If True, save solution at each time step

    Returns
    -------
    AdvectionResult
        Solution data container
    """
    if C > 1.0:
        raise ValueError(
            f"Courant number C = {C} > 1 violates stability condition. "
            "Upwind scheme requires C <= 1."
        )

    if c <= 0:
        raise ValueError(f"Advection velocity c = {c} must be positive for upwind.")

    # Default initial condition: Gaussian pulse
    if I is None:
        sigma = L / 20

        def I(x):
            return np.exp(-0.5 * ((x - L / 4) / sigma) ** 2)

    # Grid setup
    dx = L / Nx
    dt = C * dx / c
    Nt = int(round(T / dt))
    actual_T = Nt * dt

    # Create Devito grid and function
    grid = Grid(shape=(Nx + 1,), extent=(L,))
    (x_dim,) = grid.dimensions
    t_dim = grid.stepping_dim

    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=1)

    # Set initial condition
    x_coords = np.linspace(0, L, Nx + 1)
    u.data[0, :] = I(x_coords)
    u.data[1, :] = I(x_coords)

    # Courant number as Devito Constant
    courant = Constant(name="C", value=C)

    # Upwind stencil: u^{n+1}_i = u^n_i - C*(u^n_i - u^n_{i-1})
    # For interior points (i = 1, ..., Nx-1)
    stencil = u - courant * (u - u.subs(x_dim, x_dim - x_dim.spacing))
    update = Eq(u.forward, stencil)

    # Boundary conditions
    if periodic_bc:
        # Periodic: u[0] = u[Nx], handled by copying
        bc_left = Eq(u[t_dim + 1, 0], u[t_dim, Nx])
        bc_right = Eq(u[t_dim + 1, Nx], u[t_dim + 1, 0])
        op = Operator([update, bc_left, bc_right])
    else:
        # Inflow BC on left: u[0] = I(0 - c*t) for a traveling wave
        # For simplicity, keep u[0] = I(0) constant
        bc_left = Eq(u[t_dim + 1, 0], I(0))
        op = Operator([update, bc_left])

    # Time stepping
    u_history = []
    t_history = []

    if save_history:
        u_history.append(u.data[0, :].copy())
        t_history.append(0.0)

    for n in range(Nt):
        op.apply(time_m=n, time_M=n, dt=dt)
        if save_history:
            u_history.append(u.data[(n + 1) % 2, :].copy())
            t_history.append((n + 1) * dt)

    # Get final solution
    final_idx = Nt % 2
    u_final = u.data[final_idx, :].copy()

    return AdvectionResult(
        u=u_final,
        x=x_coords,
        t=actual_T,
        dt=dt,
        C=C,
        u_history=u_history if save_history else None,
        t_history=t_history if save_history else None,
    )


def solve_advection_lax_wendroff(
    L: float = 1.0,
    c: float = 1.0,
    Nx: int = 100,
    T: float = 1.0,
    C: float = 0.8,
    I: Callable | None = None,
    periodic_bc: bool = True,
    save_history: bool = False,
) -> AdvectionResult:
    """
    Solve 1D advection equation using Lax-Wendroff scheme.

    The Lax-Wendroff scheme is second-order accurate:
        u^{n+1}_i = u^n_i - (C/2)*(u^n_{i+1} - u^n_{i-1})
                          + (C²/2)*(u^n_{i+1} - 2*u^n_i + u^n_{i-1})

    This combines a centered difference for advection with an
    artificial diffusion term for stability.
    Stable for |C| <= 1.

    Parameters
    ----------
    L : float
        Domain length [0, L]
    c : float
        Advection velocity
    Nx : int
        Number of spatial intervals
    T : float
        Final time
    C : float
        Target Courant number (must be <= 1 for stability)
    I : callable
        Initial condition function I(x)
    periodic_bc : bool
        If True, use periodic boundary conditions
    save_history : bool
        If True, save solution at each time step

    Returns
    -------
    AdvectionResult
        Solution data container
    """
    if abs(C) > 1.0:
        raise ValueError(
            f"Courant number |C| = {abs(C)} > 1 violates stability condition. "
            "Lax-Wendroff scheme requires |C| <= 1."
        )

    # Default initial condition: Gaussian pulse
    if I is None:
        sigma = L / 20

        def I(x):
            return np.exp(-0.5 * ((x - L / 4) / sigma) ** 2)

    # Grid setup
    dx = L / Nx
    dt = abs(C) * dx / abs(c)
    Nt = int(round(T / dt))
    actual_T = Nt * dt

    # Create Devito grid and function
    grid = Grid(shape=(Nx + 1,), extent=(L,))
    (x_dim,) = grid.dimensions
    t_dim = grid.stepping_dim

    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=1)

    # Set initial condition
    x_coords = np.linspace(0, L, Nx + 1)
    u.data[0, :] = I(x_coords)
    u.data[1, :] = I(x_coords)

    # Courant number as Devito Constant
    courant = Constant(name="C", value=C)

    # Lax-Wendroff stencil using explicit shifted indexing:
    # u^{n+1} = u - (C/2)*(u_{i+1} - u_{i-1}) + (C²/2)*(u_{i+1} - 2*u + u_{i-1})
    u_plus = u.subs(x_dim, x_dim + x_dim.spacing)
    u_minus = u.subs(x_dim, x_dim - x_dim.spacing)
    stencil = (
        u
        - 0.5 * courant * (u_plus - u_minus)
        + 0.5 * courant**2 * (u_plus - 2 * u + u_minus)
    )
    update = Eq(u.forward, stencil)

    # Boundary conditions
    if periodic_bc:
        # Periodic: u[0] wraps to u[Nx], u[Nx] wraps to u[0]
        bc_left = Eq(u[t_dim + 1, 0], u[t_dim, Nx])
        bc_right = Eq(u[t_dim + 1, Nx], u[t_dim + 1, 0])
        op = Operator([update, bc_left, bc_right])
    else:
        # Simple extrapolation for non-periodic
        bc_left = Eq(u[t_dim + 1, 0], u[t_dim, 0])
        bc_right = Eq(u[t_dim + 1, Nx], u[t_dim, Nx])
        op = Operator([update, bc_left, bc_right])

    # Time stepping
    u_history = []
    t_history = []

    if save_history:
        u_history.append(u.data[0, :].copy())
        t_history.append(0.0)

    for n in range(Nt):
        op.apply(time_m=n, time_M=n, dt=dt)
        if save_history:
            u_history.append(u.data[(n + 1) % 2, :].copy())
            t_history.append((n + 1) * dt)

    # Get final solution
    final_idx = Nt % 2
    u_final = u.data[final_idx, :].copy()

    return AdvectionResult(
        u=u_final,
        x=x_coords,
        t=actual_T,
        dt=dt,
        C=C,
        u_history=u_history if save_history else None,
        t_history=t_history if save_history else None,
    )


def solve_advection_lax_friedrichs(
    L: float = 1.0,
    c: float = 1.0,
    Nx: int = 100,
    T: float = 1.0,
    C: float = 0.8,
    I: Callable | None = None,
    periodic_bc: bool = True,
    save_history: bool = False,
) -> AdvectionResult:
    """
    Solve 1D advection equation using Lax-Friedrichs scheme.

    The Lax-Friedrichs scheme:
        u^{n+1}_i = 0.5*(u^n_{i+1} + u^n_{i-1}) - (C/2)*(u^n_{i+1} - u^n_{i-1})

    This is first-order accurate but unconditionally stable for |C| <= 1.
    It introduces significant numerical diffusion.

    Parameters
    ----------
    L : float
        Domain length [0, L]
    c : float
        Advection velocity
    Nx : int
        Number of spatial intervals
    T : float
        Final time
    C : float
        Target Courant number (must be <= 1 for stability)
    I : callable
        Initial condition function I(x)
    periodic_bc : bool
        If True, use periodic boundary conditions
    save_history : bool
        If True, save solution at each time step

    Returns
    -------
    AdvectionResult
        Solution data container
    """
    if abs(C) > 1.0:
        raise ValueError(
            f"Courant number |C| = {abs(C)} > 1 violates stability condition. "
            "Lax-Friedrichs scheme requires |C| <= 1."
        )

    # Default initial condition: Gaussian pulse
    if I is None:
        sigma = L / 20

        def I(x):
            return np.exp(-0.5 * ((x - L / 4) / sigma) ** 2)

    # Grid setup
    dx = L / Nx
    dt = abs(C) * dx / abs(c)
    Nt = int(round(T / dt))
    actual_T = Nt * dt

    # Create Devito grid and function
    grid = Grid(shape=(Nx + 1,), extent=(L,))
    (x_dim,) = grid.dimensions
    t_dim = grid.stepping_dim

    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=1)

    # Set initial condition
    x_coords = np.linspace(0, L, Nx + 1)
    u.data[0, :] = I(x_coords)
    u.data[1, :] = I(x_coords)

    # Courant number as Devito Constant
    courant = Constant(name="C", value=C)

    # Lax-Friedrichs stencil:
    # u^{n+1}_i = 0.5*(u_{i+1} + u_{i-1}) - (C/2)*(u_{i+1} - u_{i-1})
    u_plus = u.subs(x_dim, x_dim + x_dim.spacing)
    u_minus = u.subs(x_dim, x_dim - x_dim.spacing)
    stencil = 0.5 * (u_plus + u_minus) - 0.5 * courant * (u_plus - u_minus)
    update = Eq(u.forward, stencil)

    # Boundary conditions
    if periodic_bc:
        bc_left = Eq(u[t_dim + 1, 0], u[t_dim, Nx])
        bc_right = Eq(u[t_dim + 1, Nx], u[t_dim + 1, 0])
        op = Operator([update, bc_left, bc_right])
    else:
        bc_left = Eq(u[t_dim + 1, 0], u[t_dim, 0])
        bc_right = Eq(u[t_dim + 1, Nx], u[t_dim, Nx])
        op = Operator([update, bc_left, bc_right])

    # Time stepping
    u_history = []
    t_history = []

    if save_history:
        u_history.append(u.data[0, :].copy())
        t_history.append(0.0)

    for n in range(Nt):
        op.apply(time_m=n, time_M=n, dt=dt)
        if save_history:
            u_history.append(u.data[(n + 1) % 2, :].copy())
            t_history.append((n + 1) * dt)

    # Get final solution
    final_idx = Nt % 2
    u_final = u.data[final_idx, :].copy()

    return AdvectionResult(
        u=u_final,
        x=x_coords,
        t=actual_T,
        dt=dt,
        C=C,
        u_history=u_history if save_history else None,
        t_history=t_history if save_history else None,
    )


def exact_advection(x: np.ndarray, t: float, c: float, I: Callable) -> np.ndarray:
    """
    Compute exact solution of advection equation.

    The exact solution is u(x, t) = I(x - c*t), i.e., the initial
    condition translated by c*t.

    Parameters
    ----------
    x : ndarray
        Spatial coordinates
    t : float
        Time
    c : float
        Advection velocity
    I : callable
        Initial condition function I(x)

    Returns
    -------
    ndarray
        Exact solution at time t
    """
    return I(x - c * t)


def exact_advection_periodic(
    x: np.ndarray, t: float, c: float, L: float, I: Callable
) -> np.ndarray:
    """
    Compute exact solution with periodic boundary conditions.

    Parameters
    ----------
    x : ndarray
        Spatial coordinates
    t : float
        Time
    c : float
        Advection velocity
    L : float
        Domain length
    I : callable
        Initial condition function I(x)

    Returns
    -------
    ndarray
        Exact solution at time t with periodicity
    """
    # Shift and wrap around domain [0, L]
    x_shifted = (x - c * t) % L
    return I(x_shifted)


def gaussian_initial_condition(
    x: np.ndarray, L: float = 1.0, sigma: float = 0.05, x0: float | None = None
) -> np.ndarray:
    """
    Gaussian initial condition.

    Parameters
    ----------
    x : ndarray
        Spatial coordinates
    L : float
        Domain length
    sigma : float
        Width of Gaussian
    x0 : float or None
        Center of Gaussian (default: L/4)

    Returns
    -------
    ndarray
        Gaussian pulse values
    """
    if x0 is None:
        x0 = L / 4
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def step_initial_condition(
    x: np.ndarray, L: float = 1.0, x_step: float | None = None
) -> np.ndarray:
    """
    Step (Heaviside) initial condition.

    Parameters
    ----------
    x : ndarray
        Spatial coordinates
    L : float
        Domain length
    x_step : float or None
        Location of step (default: L/4)

    Returns
    -------
    ndarray
        Step function values
    """
    if x_step is None:
        x_step = L / 4
    return np.where(x < x_step, 1.0, 0.0)


def convergence_test_advection(
    solver_func: Callable,
    grid_sizes: list[int] | None = None,
    T: float = 0.5,
    C: float = 0.8,
    L: float = 1.0,
    c: float = 1.0,
) -> tuple[list[int], list[float], float]:
    """
    Test convergence rate for an advection solver.

    Parameters
    ----------
    solver_func : callable
        Solver function (solve_advection_upwind, solve_advection_lax_wendroff, etc.)
    grid_sizes : list of int
        Grid sizes to test
    T : float
        Final time
    C : float
        Courant number
    L : float
        Domain length
    c : float
        Advection velocity

    Returns
    -------
    tuple
        (grid_sizes, errors, observed_rate)
    """
    if grid_sizes is None:
        grid_sizes = [25, 50, 100, 200]

    sigma = L / 20
    x0 = L / 4

    def I(x):
        return np.exp(-0.5 * ((x - x0) / sigma) ** 2)

    errors = []

    for Nx in grid_sizes:
        result = solver_func(L=L, c=c, Nx=Nx, T=T, C=C, I=I, periodic_bc=True)

        # Compute exact solution with periodicity
        u_exact = exact_advection_periodic(result.x, result.t, c, L, I)

        # L2 error
        dx = L / Nx
        error = np.sqrt(dx * np.sum((result.u - u_exact) ** 2))
        errors.append(error)

    # Compute observed convergence rate
    rates = []
    for i in range(1, len(errors)):
        if errors[i] > 1e-15 and errors[i - 1] > 1e-15:
            rate = np.log(errors[i - 1] / errors[i]) / np.log(
                grid_sizes[i] / grid_sizes[i - 1]
            )
            rates.append(rate)

    avg_rate = np.mean(rates) if rates else 0.0

    return grid_sizes, errors, avg_rate
