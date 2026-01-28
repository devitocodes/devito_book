"""2D Diffusion Equation Solver using Devito DSL.

Solves the 2D diffusion equation (heat equation):
    u_t = a * (u_xx + u_yy) = a * laplace(u)

on domain [0, Lx] x [0, Ly] with:
    - Initial condition: u(x, y, 0) = I(x, y)
    - Boundary conditions: u = 0 on all boundaries (Dirichlet)

The discretization uses:
    - Time: Forward Euler (explicit) - O(dt)
    - Space: Central differences - O(dx^2, dy^2)

Update formula (uniform grid, dx = dy = h):
    u^{n+1} = u^n + F * (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4*u_{i,j})

where F = a*dt/h^2 is the Fourier number.

Stability requires: F <= 0.25 (in 2D with equal spacing)

Usage:
    from src.diffu import solve_diffusion_2d

    result = solve_diffusion_2d(
        Lx=1.0, Ly=1.0,  # Domain size
        a=1.0,           # Diffusion coefficient
        Nx=50, Ny=50,    # Grid points
        T=0.1,           # Final time
        F=0.25,          # Fourier number
        I=lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y),
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
class Diffusion2DResult:
    """Results from the 2D diffusion equation solver.

    Attributes
    ----------
    u : np.ndarray
        Solution at final time, shape (Nx+1, Ny+1)
    x : np.ndarray
        x-coordinate grid points
    y : np.ndarray
        y-coordinate grid points
    t : float
        Final time
    dt : float
        Time step used
    u_history : np.ndarray, optional
        Full solution history, shape (Nt+1, Nx+1, Ny+1)
    t_history : np.ndarray, optional
        Time points for history
    F : float
        Fourier number used
    """
    u: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t: float
    dt: float
    u_history: np.ndarray | None = None
    t_history: np.ndarray | None = None
    F: float = 0.0


def solve_diffusion_2d(
    Lx: float = 1.0,
    Ly: float = 1.0,
    a: float = 1.0,
    Nx: int = 50,
    Ny: int = 50,
    T: float = 0.1,
    F: float = 0.25,
    I: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    save_history: bool = False,
) -> Diffusion2DResult:
    """Solve the 2D diffusion equation using Devito (Forward Euler).

    Solves: u_t = a * (u_xx + u_yy)
    with u = 0 on all boundaries and u(x, y, 0) = I(x, y)

    Parameters
    ----------
    Lx : float
        Domain length in x direction [0, Lx]
    Ly : float
        Domain length in y direction [0, Ly]
    a : float
        Diffusion coefficient (thermal diffusivity)
    Nx : int
        Number of spatial grid intervals in x
    Ny : int
        Number of spatial grid intervals in y
    T : float
        Final simulation time
    F : float
        Fourier number. For 2D with dx=dy, requires F <= 0.25 for stability.
    I : callable, optional
        Initial condition: I(X, Y) -> u(x, y, 0) where X, Y are meshgrid arrays
        Default: sin(pi*x/Lx) * sin(pi*y/Ly)
    save_history : bool
        If True, save full solution history

    Returns
    -------
    Diffusion2DResult
        Solution data including final solution, grids, and optionally history

    Raises
    ------
    ImportError
        If Devito is not installed
    ValueError
        If Fourier number violates stability condition
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    # 2D stability condition with equal spacing: F <= 1/(2*d) = 0.25
    dx = Lx / Nx
    dy = Ly / Ny

    # General 2D stability: a*dt*(1/dx^2 + 1/dy^2) <= 0.5
    max_F = 0.5 / (dx**2 * (1/dx**2 + 1/dy**2))  # This simplifies for equal spacing

    if dx == dy and F > 0.25:
        raise ValueError(
            f"Fourier number F={F} > 0.25 violates 2D stability condition. "
            "Forward Euler in 2D with equal spacing requires F <= 0.25."
        )

    # Default initial condition: 2D standing mode
    if I is None:
        def I(X, Y):
            return np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

    # Compute time step from Fourier number (using smaller spacing)
    h = min(dx, dy)
    dt = F * h**2 / a

    # Handle T=0 case
    if T <= 0:
        x_coords = np.linspace(0, Lx, Nx + 1)
        y_coords = np.linspace(0, Ly, Ny + 1)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        u0 = I(X, Y)
        return Diffusion2DResult(
            u=u0,
            x=x_coords,
            y=y_coords,
            t=0.0,
            dt=dt,
            u_history=u0.reshape(1, Nx + 1, Ny + 1) if save_history else None,
            t_history=np.array([0.0]) if save_history else None,
            F=F,
        )

    Nt = int(round(T / dt))
    dt = T / Nt  # Adjust dt to hit T exactly

    # Recalculate actual Fourier number
    F_actual = a * dt / h**2

    # Create Devito 2D grid
    grid = Grid(shape=(Nx + 1, Ny + 1), extent=(Lx, Ly))
    x_dim, y_dim = grid.dimensions

    # Create time function
    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)

    # Get coordinate arrays
    x_coords = np.linspace(0, Lx, Nx + 1)
    y_coords = np.linspace(0, Ly, Ny + 1)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Set initial condition
    u.data[0, :, :] = I(X, Y)

    # Diffusion equation: u_t = a * laplace(u)
    # Using Devito's .laplace attribute for dimension-agnostic Laplacian
    a_const = Constant(name='a_const')
    pde = u.dt - a_const * u.laplace
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Boundary conditions (Dirichlet: u = 0 on all boundaries)
    t_step = grid.stepping_dim

    bc_x0 = Eq(u[t_step + 1, 0, y_dim], 0)           # Left
    bc_xN = Eq(u[t_step + 1, Nx, y_dim], 0)          # Right
    bc_y0 = Eq(u[t_step + 1, x_dim, 0], 0)           # Bottom
    bc_yN = Eq(u[t_step + 1, x_dim, Ny], 0)          # Top

    # Create operator
    op = Operator([stencil, bc_x0, bc_xN, bc_y0, bc_yN])

    # Storage for history
    if save_history:
        u_history = np.zeros((Nt + 1, Nx + 1, Ny + 1))
        u_history[0, :, :] = u.data[0, :, :]
        t_history = np.linspace(0, T, Nt + 1)
    else:
        u_history = None
        t_history = None

    # Time stepping
    for n in range(Nt):
        # Run one time step
        op.apply(time_m=0, time_M=0, dt=dt, a_const=a)

        # Copy to next time level
        u.data[0, :, :] = u.data[1, :, :]

        # Save to history if requested
        if save_history:
            u_history[n + 1, :, :] = u.data[0, :, :]

    # Extract final solution
    u_final = u.data[0, :, :].copy()

    return Diffusion2DResult(
        u=u_final,
        x=x_coords,
        y=y_coords,
        t=T,
        dt=dt,
        u_history=u_history,
        t_history=t_history,
        F=F_actual,
    )


def exact_diffusion_2d(
    X: np.ndarray,
    Y: np.ndarray,
    t: float,
    Lx: float,
    Ly: float,
    a: float,
    m: int = 1,
    n: int = 1,
) -> np.ndarray:
    """Exact solution for 2D diffusion with sinusoidal initial condition.

    Solution: u(x, y, t) = exp(-a * kappa * t) * sin(m*pi*x/Lx) * sin(n*pi*y/Ly)

    where kappa = (m*pi/Lx)^2 + (n*pi/Ly)^2

    Parameters
    ----------
    X : np.ndarray
        x-coordinates (meshgrid)
    Y : np.ndarray
        y-coordinates (meshgrid)
    t : float
        Time
    Lx : float
        Domain length in x
    Ly : float
        Domain length in y
    a : float
        Diffusion coefficient
    m : int
        Mode number in x direction
    n : int
        Mode number in y direction

    Returns
    -------
    np.ndarray
        Exact solution at (x, y, t)
    """
    kappa = (m * np.pi / Lx)**2 + (n * np.pi / Ly)**2
    return np.exp(-a * kappa * t) * np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)


def convergence_test_diffusion_2d(
    grid_sizes: list = None,
    T: float = 0.05,
    F: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run convergence test for 2D diffusion solver.

    Uses the exact sinusoidal solution for error computation.

    Parameters
    ----------
    grid_sizes : list, optional
        List of Nx=Ny values to test. Default: [10, 20, 40, 80]
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
    Lx = Ly = 1.0
    a = 1.0

    for Nx in grid_sizes:
        result = solve_diffusion_2d(Lx=Lx, Ly=Ly, a=a, Nx=Nx, Ny=Nx, T=T, F=F)

        # Create meshgrid for exact solution
        X, Y = np.meshgrid(result.x, result.y, indexing='ij')

        # Exact solution at final time
        u_exact = exact_diffusion_2d(X, Y, result.t, Lx, Ly, a)

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


def gaussian_2d_initial_condition(
    X: np.ndarray, Y: np.ndarray, Lx: float, Ly: float, sigma: float = 0.1
) -> np.ndarray:
    """2D Gaussian initial condition centered in the domain.

    Parameters
    ----------
    X : np.ndarray
        x-coordinates (meshgrid)
    Y : np.ndarray
        y-coordinates (meshgrid)
    Lx : float
        Domain length in x
    Ly : float
        Domain length in y
    sigma : float
        Width of the Gaussian

    Returns
    -------
    np.ndarray
        2D Gaussian profile
    """
    r2 = (X - Lx / 2)**2 + (Y - Ly / 2)**2
    return np.exp(-r2 / (2 * sigma**2))
