"""2D Wave Equation Solver using Devito DSL.

Solves the 2D wave equation:
    u_tt = c^2 * (u_xx + u_yy) = c^2 * laplace(u)

on domain [0, Lx] x [0, Ly] with:
    - Initial conditions: u(x, y, 0) = I(x, y), u_t(x, y, 0) = V(x, y)
    - Boundary conditions: u = 0 on all boundaries (Dirichlet)

The discretization uses:
    - Time: Central difference (leapfrog) - O(dt^2)
    - Space: Central difference - O(dx^2, dy^2)

Update formula:
    u^{n+1} = 2*u^n - u^{n-1} + dt^2 * c^2 * laplace(u^n)

CFL stability condition: C = c*dt*sqrt(1/dx^2 + 1/dy^2) <= 1

Usage:
    from src.wave import solve_wave_2d

    result = solve_wave_2d(
        Lx=1.0, Ly=1.0,  # Domain size
        c=1.0,            # Wave speed
        Nx=50, Ny=50,     # Grid points
        T=1.0,            # Final time
        C=0.5,            # Courant number
    )
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    from devito import Constant, Eq, Grid, Operator, TimeFunction, solve
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@dataclass
class Wave2DResult:
    """Results from the 2D wave equation solver.

    Attributes
    ----------
    u : np.ndarray
        Solution at final time, shape (Nx+1, Ny+1)
    x : np.ndarray
        Spatial grid points in x, shape (Nx+1,)
    y : np.ndarray
        Spatial grid points in y, shape (Ny+1,)
    t : float
        Final time
    dt : float
        Time step used
    u_history : np.ndarray, optional
        Full solution history, shape (Nt+1, Nx+1, Ny+1)
    t_history : np.ndarray, optional
        Time points for history
    C : float
        Effective Courant number used
    """
    u: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t: float
    dt: float
    u_history: np.ndarray | None = None
    t_history: np.ndarray | None = None
    C: float = 0.0


def solve_wave_2d(
    Lx: float = 1.0,
    Ly: float = 1.0,
    c: float = 1.0,
    Nx: int = 50,
    Ny: int = 50,
    T: float = 1.0,
    C: float = 0.5,
    I: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    V: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    save_history: bool = False,
) -> Wave2DResult:
    """Solve the 2D wave equation using Devito.

    Parameters
    ----------
    Lx : float
        Domain length in x direction [0, Lx]
    Ly : float
        Domain length in y direction [0, Ly]
    c : float
        Wave speed
    Nx : int
        Number of spatial grid intervals in x
    Ny : int
        Number of spatial grid intervals in y
    T : float
        Final simulation time
    C : float
        Target Courant number. Must be <= 1 for stability.
        Actual dt computed as: dt = C / (c * sqrt(1/dx^2 + 1/dy^2))
    I : callable, optional
        Initial displacement: I(X, Y) -> u(x, y, 0)
        X, Y are 2D meshgrid arrays
        Default: sin(pi*x/Lx) * sin(pi*y/Ly)
    V : callable, optional
        Initial velocity: V(X, Y) -> u_t(x, y, 0)
        Default: 0
    save_history : bool
        If True, save full solution history

    Returns
    -------
    Wave2DResult
        Solution data including final solution, grids, and optionally history

    Raises
    ------
    ImportError
        If Devito is not installed
    ValueError
        If Courant number > 1 (unstable)
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    if C > 1.0:
        raise ValueError(
            f"Courant number C={C} > 1 violates CFL stability condition"
        )

    # Compute grid spacing
    dx = Lx / Nx
    dy = Ly / Ny

    # Compute time step from CFL condition
    # C = c * dt * sqrt(1/dx^2 + 1/dy^2) <= 1
    stability_factor = np.sqrt(1/dx**2 + 1/dy**2)
    dt = C / (c * stability_factor)

    # Default initial conditions
    if I is None:
        def I(X, Y):
            return np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
    if V is None:
        def V(X, Y):
            return np.zeros_like(X)

    # Handle T=0 case
    x_coords = np.linspace(0, Lx, Nx + 1)
    y_coords = np.linspace(0, Ly, Ny + 1)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    if T <= 0:
        u0 = I(X, Y)
        return Wave2DResult(
            u=u0,
            x=x_coords,
            y=y_coords,
            t=0.0,
            dt=dt,
            u_history=u0.reshape(1, Nx+1, Ny+1) if save_history else None,
            t_history=np.array([0.0]) if save_history else None,
            C=C,
        )

    Nt = int(round(T / dt))
    dt = T / Nt  # Adjust dt to hit T exactly

    # Recalculate actual Courant number
    C_actual = c * dt * stability_factor

    # Create Devito grid - Note: Devito uses (y, x) ordering internally
    # but we use extent and shape consistently
    grid = Grid(shape=(Nx + 1, Ny + 1), extent=(Lx, Ly))

    # Create time function with time_order=2 for wave equation
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

    # Set initial conditions
    u0_vals = I(X, Y)
    u.data[0, :, :] = u0_vals
    u.data[1, :, :] = u0_vals  # Will be corrected below

    # Wave equation using laplace: u_tt = c^2 * laplace(u)
    c_sq = Constant(name='c_sq')
    pde = u.dt2 - c_sq * u.laplace
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Boundary conditions (Dirichlet: u = 0 on all boundaries)
    t_dim = grid.stepping_dim
    x_dim, y_dim = grid.dimensions

    # Set boundaries to zero
    bc_x0 = Eq(u[t_dim + 1, 0, y_dim], 0)
    bc_xN = Eq(u[t_dim + 1, Nx, y_dim], 0)
    bc_y0 = Eq(u[t_dim + 1, x_dim, 0], 0)
    bc_yN = Eq(u[t_dim + 1, x_dim, Ny], 0)

    # Create operator
    op = Operator([stencil, bc_x0, bc_xN, bc_y0, bc_yN])

    # Special first step to incorporate initial velocity V
    # u^1 = u^0 + dt*V(X,Y) + 0.5*dt^2*c^2*laplace(u^0)
    v0 = V(X, Y)

    # Compute Laplacian of initial condition
    laplace_u0 = np.zeros_like(u0_vals)
    laplace_u0[1:-1, 1:-1] = (
        (u0_vals[2:, 1:-1] - 2*u0_vals[1:-1, 1:-1] + u0_vals[:-2, 1:-1]) / dx**2 +
        (u0_vals[1:-1, 2:] - 2*u0_vals[1:-1, 1:-1] + u0_vals[1:-1, :-2]) / dy**2
    )

    u1 = u0_vals + dt * v0 + 0.5 * dt**2 * c**2 * laplace_u0
    # Apply boundary conditions
    u1[0, :] = 0
    u1[-1, :] = 0
    u1[:, 0] = 0
    u1[:, -1] = 0

    u.data[1, :, :] = u1

    # Storage for history
    if save_history:
        u_history = np.zeros((Nt + 1, Nx + 1, Ny + 1))
        u_history[0, :, :] = u.data[0, :, :]
        u_history[1, :, :] = u.data[1, :, :]
        t_history = np.linspace(0, T, Nt + 1)
    else:
        u_history = None
        t_history = None

    # Time stepping
    for n in range(2, Nt + 1):
        # Run one time step
        op.apply(time_m=1, time_M=1, dt=dt, c_sq=c**2)

        # Copy to next time level (modular time indexing)
        u.data[0, :, :] = u.data[1, :, :]
        u.data[1, :, :] = u.data[2, :, :]

        # Save to history if requested
        if save_history:
            u_history[n, :, :] = u.data[1, :, :]

    # Extract final solution
    u_final = u.data[1, :, :].copy()

    return Wave2DResult(
        u=u_final,
        x=x_coords,
        y=y_coords,
        t=T,
        dt=dt,
        u_history=u_history,
        t_history=t_history,
        C=C_actual,
    )


def exact_standing_wave_2d(
    X: np.ndarray,
    Y: np.ndarray,
    t: float,
    Lx: float,
    Ly: float,
    c: float,
) -> np.ndarray:
    """Exact solution for 2D standing wave.

    Initial condition: I(x,y) = sin(pi*x/Lx) * sin(pi*y/Ly), V=0

    Solution: u(x, y, t) = sin(pi*x/Lx) * sin(pi*y/Ly) * cos(omega*t)
    where omega = c * pi * sqrt(1/Lx^2 + 1/Ly^2)

    Parameters
    ----------
    X : np.ndarray
        X coordinates (2D meshgrid)
    Y : np.ndarray
        Y coordinates (2D meshgrid)
    t : float
        Time
    Lx : float
        Domain length in x
    Ly : float
        Domain length in y
    c : float
        Wave speed

    Returns
    -------
    np.ndarray
        Exact solution at (X, Y, t)
    """
    omega = c * np.pi * np.sqrt(1/Lx**2 + 1/Ly**2)
    return np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly) * np.cos(omega * t)


def convergence_test_wave_2d(
    grid_sizes: list | None = None,
    T: float = 0.25,
    C: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run convergence test for 2D wave solver.

    Uses the exact standing wave solution for error computation.

    Parameters
    ----------
    grid_sizes : list, optional
        List of N values to test (same for Nx and Ny).
        Default: [10, 20, 40, 80]
    T : float
        Final time
    C : float
        Courant number

    Returns
    -------
    tuple
        (grid_sizes, errors, observed_order)
    """
    if grid_sizes is None:
        grid_sizes = [10, 20, 40, 80]

    errors = []
    Lx = Ly = 1.0
    c = 1.0

    for N in grid_sizes:
        result = solve_wave_2d(
            Lx=Lx, Ly=Ly, c=c, Nx=N, Ny=N, T=T, C=C
        )

        # Create meshgrid for exact solution
        X, Y = np.meshgrid(result.x, result.y, indexing='ij')

        # Exact solution at final time
        u_exact = exact_standing_wave_2d(X, Y, result.t, Lx, Ly, c)

        # L2 error
        error = np.sqrt(np.mean((result.u - u_exact)**2))
        errors.append(error)

    errors = np.array(errors)
    grid_sizes = np.array(grid_sizes)

    # Compute observed order
    log_h = np.log(1.0 / grid_sizes)
    log_err = np.log(errors)
    observed_order = np.polyfit(log_h, log_err, 1)[0]

    return grid_sizes, errors, observed_order
