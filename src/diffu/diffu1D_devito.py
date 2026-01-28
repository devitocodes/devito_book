"""1D Diffusion Equation Solver using Devito DSL.

Solves the 1D diffusion equation (heat equation):
    u_t = a * u_xx

on domain [0, L] with:
    - Initial condition: u(x, 0) = I(x)
    - Boundary conditions: u(0, t) = u(L, t) = 0 (Dirichlet)

The discretization uses:
    - Time: Forward Euler (explicit) - O(dt)
    - Space: Central difference - O(dx^2)

Update formula:
    u^{n+1} = u^n + F * (u_{i-1}^n - 2*u_i^n + u_{i+1}^n)

where F = a*dt/dx^2 is the Fourier number (mesh Fourier number).

Stability requires: F <= 0.5

Usage:
    from src.diffu import solve_diffusion_1d

    result = solve_diffusion_1d(
        L=1.0,           # Domain length
        a=1.0,           # Diffusion coefficient
        Nx=100,          # Grid points
        T=0.1,           # Final time
        F=0.5,           # Fourier number
        I=lambda x: np.sin(np.pi * x),  # Initial condition
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
class DiffusionResult:
    """Results from the diffusion equation solver.

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
    F : float
        Fourier number used
    """
    u: np.ndarray
    x: np.ndarray
    t: float
    dt: float
    u_history: np.ndarray | None = None
    t_history: np.ndarray | None = None
    F: float = 0.0


def solve_diffusion_1d(
    L: float = 1.0,
    a: float = 1.0,
    Nx: int = 100,
    T: float = 0.1,
    F: float = 0.5,
    I: Callable[[np.ndarray], np.ndarray] | None = None,
    f: Callable[[np.ndarray, float], np.ndarray] | None = None,
    save_history: bool = False,
) -> DiffusionResult:
    """Solve the 1D diffusion equation using Devito (Forward Euler).

    Solves: u_t = a * u_xx + f(x, t)
    with u(0,t) = u(L,t) = 0 and u(x,0) = I(x)

    Parameters
    ----------
    L : float
        Domain length [0, L]
    a : float
        Diffusion coefficient (thermal diffusivity)
    Nx : int
        Number of spatial grid intervals
    T : float
        Final simulation time
    F : float
        Fourier number (a*dt/dx^2). Must be <= 0.5 for stability.
    I : callable, optional
        Initial condition: I(x) -> u(x, 0)
        Default: sin(pi * x / L)
    f : callable, optional
        Source term: f(x, t) -> source value
        Default: 0 (no source)
    save_history : bool
        If True, save full solution history

    Returns
    -------
    DiffusionResult
        Solution data including final solution, grid, and optionally history

    Raises
    ------
    ImportError
        If Devito is not installed
    ValueError
        If Fourier number > 0.5 (unstable)
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    if F > 0.5:
        raise ValueError(
            f"Fourier number F={F} > 0.5 violates stability condition. "
            "Forward Euler requires F <= 0.5."
        )

    # Default initial condition
    if I is None:
        I = lambda x: np.sin(np.pi * x / L)

    # Default source term (no source)
    if f is None:
        f = lambda x, t: np.zeros_like(x)

    # Compute grid spacing and time step from Fourier number
    dx = L / Nx
    dt = F * dx**2 / a

    # Handle T=0 case (just return initial condition)
    if T <= 0:
        x_coords = np.linspace(0, L, Nx + 1)
        u0 = I(x_coords)
        return DiffusionResult(
            u=u0,
            x=x_coords,
            t=0.0,
            dt=dt,
            u_history=u0.reshape(1, -1) if save_history else None,
            t_history=np.array([0.0]) if save_history else None,
            F=F,
        )

    Nt = int(round(T / dt))
    dt = T / Nt  # Adjust dt to hit T exactly

    # Recalculate actual Fourier number
    F_actual = a * dt / dx**2

    # Create Devito grid
    grid = Grid(shape=(Nx + 1,), extent=(L,))

    # Create time function with time_order=1 for diffusion equation
    # (first-order time derivative, second-order spatial)
    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)

    # Get spatial coordinate array
    x_coords = np.linspace(0, L, Nx + 1)

    # Set initial condition
    u.data[0, :] = I(x_coords)

    # Diffusion equation: u_t = a * u_xx
    # Using solve() to get the update formula
    a_const = Constant(name='a_const')
    pde = u.dt - a_const * u.dx2
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Boundary conditions (Dirichlet: u = 0 at boundaries)
    bc_left = Eq(u[grid.stepping_dim + 1, 0], 0)
    bc_right = Eq(u[grid.stepping_dim + 1, Nx], 0)

    # Create operator
    op = Operator([stencil, bc_left, bc_right])

    # Storage for history
    if save_history:
        u_history = np.zeros((Nt + 1, Nx + 1))
        u_history[0, :] = u.data[0, :]
        t_history = np.linspace(0, T, Nt + 1)
    else:
        u_history = None
        t_history = None

    # Time stepping
    for n in range(Nt):
        # Run one time step
        op.apply(time_m=0, time_M=0, dt=dt, a_const=a)

        # Copy to next time level (modular time indexing)
        u.data[0, :] = u.data[1, :]

        # Save to history if requested
        if save_history:
            u_history[n + 1, :] = u.data[0, :]

    # Extract final solution
    u_final = u.data[0, :].copy()

    return DiffusionResult(
        u=u_final,
        x=x_coords,
        t=T,
        dt=dt,
        u_history=u_history,
        t_history=t_history,
        F=F_actual,
    )


def exact_diffusion_sine(
    x: np.ndarray, t: float, L: float, a: float, m: int = 1
) -> np.ndarray:
    """Exact solution for diffusion with I(x) = sin(m*pi*x/L).

    Solution: u(x, t) = exp(-a * (m*pi/L)^2 * t) * sin(m*pi*x/L)

    This is the decaying eigenmode solution for the heat equation
    with homogeneous Dirichlet boundary conditions.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates
    t : float
        Time
    L : float
        Domain length
    a : float
        Diffusion coefficient
    m : int
        Mode number (m=1 is the fundamental mode)

    Returns
    -------
    np.ndarray
        Exact solution at (x, t)
    """
    kappa = (m * np.pi / L)**2
    return np.exp(-a * kappa * t) * np.sin(m * np.pi * x / L)


def convergence_test_diffusion_1d(
    grid_sizes: list = None,
    T: float = 0.1,
    F: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run convergence test for 1D diffusion solver.

    Uses the exact sinusoidal solution for error computation.
    With Forward Euler (first order in time, second in space),
    the expected convergence rate depends on how dt and dx
    are coupled through F.

    Parameters
    ----------
    grid_sizes : list, optional
        List of Nx values to test. Default: [10, 20, 40, 80]
    T : float
        Final time
    F : float
        Fourier number (fixed for all runs)

    Returns
    -------
    tuple
        (grid_sizes, errors, observed_order)
    """
    if grid_sizes is None:
        grid_sizes = [10, 20, 40, 80]

    errors = []
    L = 1.0
    a = 1.0

    for Nx in grid_sizes:
        result = solve_diffusion_1d(L=L, a=a, Nx=Nx, T=T, F=F)

        # Exact solution at final time
        u_exact = exact_diffusion_sine(result.x, result.t, L, a)

        # L2 error
        error = np.sqrt(np.mean((result.u - u_exact)**2))
        errors.append(error)

    errors = np.array(errors)
    grid_sizes = np.array(grid_sizes)

    # Compute observed order
    # Note: With F fixed, dx decreases and dt = F*dx^2/a decreases as dx^2
    # So the spatial error O(dx^2) dominates and we expect ~2nd order
    log_h = np.log(1.0 / grid_sizes)
    log_err = np.log(errors)
    observed_order = np.polyfit(log_h, log_err, 1)[0]

    return grid_sizes, errors, observed_order


def gaussian_initial_condition(x: np.ndarray, L: float, sigma: float = 0.05) -> np.ndarray:
    """Gaussian initial condition centered in the domain.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates
    L : float
        Domain length
    sigma : float
        Width of the Gaussian

    Returns
    -------
    np.ndarray
        Gaussian profile
    """
    return np.exp(-0.5 * ((x - L / 2) / sigma)**2)


def plug_initial_condition(x: np.ndarray, L: float, width: float = 0.1) -> np.ndarray:
    """Plug (discontinuous) initial condition.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates
    L : float
        Domain length
    width : float
        Half-width of the plug

    Returns
    -------
    np.ndarray
        Plug profile (1 inside, 0 outside)
    """
    return np.where(np.abs(x - L / 2) <= width, 1.0, 0.0)
