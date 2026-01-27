"""1D Wave Equation Solver using Devito DSL.

Solves the 1D wave equation:
    u_tt = c^2 * u_xx

on domain [0, L] with:
    - Initial conditions: u(x, 0) = I(x), u_t(x, 0) = V(x)
    - Boundary conditions: u(0, t) = u(L, t) = 0 (Dirichlet)

The discretization uses:
    - Time: Central difference (leapfrog) - O(dt^2)
    - Space: Central difference - O(dx^2)

Update formula:
    u^{n+1} = 2*u^n - u^{n-1} + C^2 * (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)

where C = c*dt/dx is the Courant number.

Usage:
    from src.wave import solve_wave_1d

    result = solve_wave_1d(
        L=1.0,           # Domain length
        c=1.0,           # Wave speed
        Nx=100,          # Grid points
        T=1.0,           # Final time
        C=0.9,           # Courant number
        I=lambda x: np.sin(np.pi * x),  # Initial displacement
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
class WaveResult:
    """Results from the wave equation solver.

    Attributes
    ----------
    u : np.ndarray
        Solution at final time, shape (Nx+1,)
    x : np.ndarray
        Spatial grid points
    t : float
        Final time
    dt : float
        Time step used
    u_history : np.ndarray, optional
        Full solution history, shape (Nt+1, Nx+1)
    t_history : np.ndarray, optional
        Time points for history
    C : float
        Courant number used
    """
    u: np.ndarray
    x: np.ndarray
    t: float
    dt: float
    u_history: np.ndarray | None = None
    t_history: np.ndarray | None = None
    C: float = 0.0


def solve_wave_1d(
    L: float = 1.0,
    c: float = 1.0,
    Nx: int = 100,
    T: float = 1.0,
    C: float = 0.9,
    I: Callable[[np.ndarray], np.ndarray] | None = None,
    V: Callable[[np.ndarray], np.ndarray] | None = None,
    save_history: bool = False,
) -> WaveResult:
    """Solve the 1D wave equation using Devito.

    Parameters
    ----------
    L : float
        Domain length [0, L]
    c : float
        Wave speed
    Nx : int
        Number of spatial grid intervals
    T : float
        Final simulation time
    C : float
        Courant number (c*dt/dx). Must be <= 1 for stability.
    I : callable, optional
        Initial displacement: I(x) -> u(x, 0)
        Default: sin(pi * x / L)
    V : callable, optional
        Initial velocity: V(x) -> u_t(x, 0)
        Default: 0
    save_history : bool
        If True, save full solution history

    Returns
    -------
    WaveResult
        Solution data including final solution, grid, and optionally history

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

    # Default initial conditions
    if I is None:
        I = lambda x: np.sin(np.pi * x / L)
    if V is None:
        V = lambda x: np.zeros_like(x)

    # Compute grid spacing and time step
    dx = L / Nx
    dt = C * dx / c

    # Handle T=0 case (just return initial condition)
    if T <= 0:
        x_coords = np.linspace(0, L, Nx + 1)
        u0 = I(x_coords)
        return WaveResult(
            u=u0,
            x=x_coords,
            t=0.0,
            dt=dt,
            u_history=u0.reshape(1, -1) if save_history else None,
            t_history=np.array([0.0]) if save_history else None,
            C=C,
        )

    Nt = int(round(T / dt))
    dt = T / Nt  # Adjust dt to hit T exactly

    # Recalculate actual Courant number
    C_actual = c * dt / dx

    # Create Devito grid
    grid = Grid(shape=(Nx + 1,), extent=(L,))
    x_dim = grid.dimensions[0]

    # Create time function with time_order=2 for wave equation
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

    # Get spatial coordinate array
    x_coords = np.linspace(0, L, Nx + 1)

    # Set initial conditions
    # u(x, 0) = I(x)
    u.data[0, :] = I(x_coords)
    u.data[1, :] = I(x_coords)  # Will be corrected below

    # For the first time step, use a special formula incorporating V:
    # u^1 = u^0 + dt*V + 0.5*C^2*(u^0_{i+1} - 2*u^0_i + u^0_{i-1})
    # This is done after setting up the operator

    # Wave equation: u_tt = c^2 * u_xx
    # Using solve() to get the update formula
    c_sq = Constant(name='c_sq')
    pde = u.dt2 - c_sq * u.dx2
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Boundary conditions (Dirichlet: u = 0 at boundaries)
    # We'll handle these by resetting after each step
    bc_left = Eq(u[grid.stepping_dim + 1, 0], 0)
    bc_right = Eq(u[grid.stepping_dim + 1, Nx], 0)

    # Create operator
    op = Operator([stencil, bc_left, bc_right])

    # Special first step to incorporate initial velocity V
    # u^1 = u^0 + dt*V(x) + 0.5*dt^2*c^2*u_xx^0
    u0 = I(x_coords)
    v0 = V(x_coords)
    u_xx_0 = np.zeros_like(u0)
    u_xx_0[1:-1] = (u0[2:] - 2*u0[1:-1] + u0[:-2]) / dx**2

    u1 = u0 + dt * v0 + 0.5 * dt**2 * c**2 * u_xx_0
    u1[0] = 0  # Boundary conditions
    u1[-1] = 0

    # Set the corrected u^1
    u.data[1, :] = u1

    # Storage for history
    if save_history:
        u_history = np.zeros((Nt + 1, Nx + 1))
        u_history[0, :] = u.data[0, :]
        u_history[1, :] = u.data[1, :]
        t_history = np.linspace(0, T, Nt + 1)
    else:
        u_history = None
        t_history = None

    # Time stepping - always use manual loop to properly handle time buffers
    for n in range(2, Nt + 1):
        # Run one time step
        op.apply(time_m=1, time_M=1, dt=dt, c_sq=c**2)

        # Copy to next time level (modular time indexing)
        u.data[0, :] = u.data[1, :]
        u.data[1, :] = u.data[2, :]

        # Save to history if requested
        if save_history:
            u_history[n, :] = u.data[1, :]

    # Extract final solution
    u_final = u.data[1, :].copy()

    return WaveResult(
        u=u_final,
        x=x_coords,
        t=T,
        dt=dt,
        u_history=u_history,
        t_history=t_history,
        C=C_actual,
    )


def exact_standing_wave(x: np.ndarray, t: float, L: float, c: float) -> np.ndarray:
    """Exact solution for standing wave with I(x) = sin(pi*x/L), V=0.

    Solution: u(x, t) = sin(pi*x/L) * cos(pi*c*t/L)

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates
    t : float
        Time
    L : float
        Domain length
    c : float
        Wave speed

    Returns
    -------
    np.ndarray
        Exact solution at (x, t)
    """
    return np.sin(np.pi * x / L) * np.cos(np.pi * c * t / L)


def convergence_test_wave_1d(
    grid_sizes: list = None,
    T: float = 0.5,
    C: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run convergence test for 1D wave solver.

    Uses the exact standing wave solution for error computation.

    Parameters
    ----------
    grid_sizes : list, optional
        List of Nx values to test. Default: [20, 40, 80, 160]
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
        grid_sizes = [20, 40, 80, 160]

    errors = []
    L = 1.0
    c = 1.0

    for Nx in grid_sizes:
        result = solve_wave_1d(L=L, c=c, Nx=Nx, T=T, C=C)

        # Exact solution at final time
        u_exact = exact_standing_wave(result.x, result.t, L, c)

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
