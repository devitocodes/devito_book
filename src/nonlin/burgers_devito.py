"""2D Coupled Burgers Equations Solver using Devito DSL.

Solves the 2D coupled Burgers equations:
    u_t + u * u_x + v * u_y = nu * (u_xx + u_yy)
    v_t + u * v_x + v * v_y = nu * (v_xx + v_yy)

This combines nonlinear advection with viscous diffusion.
The equations model various physical phenomena including:
- Simplified fluid flow without pressure
- Traffic flow modeling
- Shock wave formation and propagation

Key implementation features:
- Uses first_derivative() with explicit fd_order=1 for advection terms
- Uses .laplace for diffusion terms (second-order)
- Supports both scalar TimeFunction and VectorTimeFunction approaches
- Applies Dirichlet boundary conditions

Stability requires satisfying both:
- CFL condition: C = |u|_max * dt / dx <= 1
- Diffusion condition: F = nu * dt / dx^2 <= 0.25

Usage:
    from src.nonlin.burgers_devito import solve_burgers_2d

    result = solve_burgers_2d(
        Lx=2.0, Ly=2.0,   # Domain size
        nu=0.01,           # Viscosity
        Nx=41, Ny=41,      # Grid points
        T=0.5,             # Final time
    )
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    from devito import (
        Constant,
        Eq,
        Grid,
        Operator,
        TimeFunction,
        VectorTimeFunction,
        first_derivative,
        grad,
        left,
        solve,
    )

    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@dataclass
class Burgers2DResult:
    """Result container for 2D Burgers equation solver.

    Attributes
    ----------
    u : np.ndarray
        x-velocity component at final time, shape (Nx+1, Ny+1)
    v : np.ndarray
        y-velocity component at final time, shape (Nx+1, Ny+1)
    x : np.ndarray
        x-coordinate grid points
    y : np.ndarray
        y-coordinate grid points
    t : float
        Final time
    dt : float
        Time step used
    u_history : list or None
        Solution history for u (if save_history=True)
    v_history : list or None
        Solution history for v (if save_history=True)
    t_history : list or None
        Time values (if save_history=True)
    """

    u: np.ndarray
    v: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t: float
    dt: float
    u_history: list | None = None
    v_history: list | None = None
    t_history: list | None = None


def init_hat(
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float = 2.0,
    Ly: float = 2.0,
    value: float = 2.0,
) -> np.ndarray:
    """Initialize with a 'hat' function (square pulse).

    Creates a pulse with given value in the region
    [0.5, 1] x [0.5, 1] and 1.0 elsewhere.

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
    value : float
        Value inside the hat region

    Returns
    -------
    np.ndarray
        Initial condition array
    """
    result = np.ones_like(X)
    # Region where the 'hat' is elevated
    mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
    result[mask] = value
    return result


def solve_burgers_2d(
    Lx: float = 2.0,
    Ly: float = 2.0,
    nu: float = 0.01,
    Nx: int = 41,
    Ny: int = 41,
    T: float = 0.5,
    sigma: float = 0.0009,
    I_u: Callable | None = None,
    I_v: Callable | None = None,
    bc_value: float = 1.0,
    save_history: bool = False,
    save_every: int = 100,
) -> Burgers2DResult:
    """Solve 2D coupled Burgers equations using Devito.

    Solves:
        u_t + u * u_x + v * u_y = nu * laplace(u)
        v_t + u * v_x + v * v_y = nu * laplace(v)

    Uses backward (upwind) differences for advection terms and
    centered differences for diffusion terms.

    Parameters
    ----------
    Lx : float
        Domain length in x direction [0, Lx]
    Ly : float
        Domain length in y direction [0, Ly]
    nu : float
        Viscosity (diffusion coefficient)
    Nx : int
        Number of grid points in x
    Ny : int
        Number of grid points in y
    T : float
        Final simulation time
    sigma : float
        Stability parameter: dt = sigma * dx * dy / nu
    I_u : callable or None
        Initial condition for u: I_u(X, Y) -> array
        Default: hat function with value 2 in [0.5, 1] x [0.5, 1]
    I_v : callable or None
        Initial condition for v: I_v(X, Y) -> array
        Default: hat function with value 2 in [0.5, 1] x [0.5, 1]
    bc_value : float
        Dirichlet boundary condition value (default: 1.0)
    save_history : bool
        If True, save solution history
    save_every : int
        Save every N time steps (if save_history=True)

    Returns
    -------
    Burgers2DResult
        Solution data container with u, v fields and metadata

    Raises
    ------
    ImportError
        If Devito is not installed
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. Install with: pip install devito"
        )

    # Grid setup
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = sigma * dx * dy / nu

    # Handle T=0 case
    if T <= 0:
        x_coords = np.linspace(0, Lx, Nx)
        y_coords = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        if I_u is None:
            u0 = init_hat(X, Y, Lx, Ly, value=2.0)
        else:
            u0 = I_u(X, Y)
        if I_v is None:
            v0 = init_hat(X, Y, Lx, Ly, value=2.0)
        else:
            v0 = I_v(X, Y)
        return Burgers2DResult(
            u=u0,
            v=v0,
            x=x_coords,
            y=y_coords,
            t=0.0,
            dt=dt,
        )

    Nt = int(round(T / dt))
    actual_T = Nt * dt

    # Create Devito grid
    grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
    x_dim, y_dim = grid.dimensions
    t_dim = grid.stepping_dim

    # Create time functions with space_order=2 for diffusion
    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)
    v = TimeFunction(name="v", grid=grid, time_order=1, space_order=2)

    # Get coordinate arrays
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

    # Set initial conditions
    if I_u is None:
        u.data[0, :, :] = init_hat(X, Y, Lx, Ly, value=2.0)
    else:
        u.data[0, :, :] = I_u(X, Y)

    if I_v is None:
        v.data[0, :, :] = init_hat(X, Y, Lx, Ly, value=2.0)
    else:
        v.data[0, :, :] = I_v(X, Y)

    # Viscosity as Devito Constant
    a = Constant(name="a")

    # Create explicit first-order backward derivatives for advection
    # Using first_derivative() with side=left and fd_order=1
    # This gives: (u[x] - u[x-dx]) / dx (backward/upwind difference)
    u_dx = first_derivative(u, dim=x_dim, side=left, fd_order=1)
    u_dy = first_derivative(u, dim=y_dim, side=left, fd_order=1)
    v_dx = first_derivative(v, dim=x_dim, side=left, fd_order=1)
    v_dy = first_derivative(v, dim=y_dim, side=left, fd_order=1)

    # Write down the equations:
    # u_t + u * u_x + v * u_y = nu * laplace(u)
    # v_t + u * v_x + v * v_y = nu * laplace(v)
    # Apply only in interior using subdomain
    eq_u = Eq(u.dt + u * u_dx + v * u_dy, a * u.laplace, subdomain=grid.interior)
    eq_v = Eq(v.dt + u * v_dx + v * v_dy, a * v.laplace, subdomain=grid.interior)

    # Let SymPy solve for the update expressions
    stencil_u = solve(eq_u, u.forward)
    stencil_v = solve(eq_v, v.forward)
    update_u = Eq(u.forward, stencil_u)
    update_v = Eq(v.forward, stencil_v)

    # Dirichlet boundary conditions using low-level API
    # u boundary conditions
    bc_u = [Eq(u[t_dim + 1, 0, y_dim], bc_value)]  # left
    bc_u += [Eq(u[t_dim + 1, Nx - 1, y_dim], bc_value)]  # right
    bc_u += [Eq(u[t_dim + 1, x_dim, 0], bc_value)]  # bottom
    bc_u += [Eq(u[t_dim + 1, x_dim, Ny - 1], bc_value)]  # top

    # v boundary conditions
    bc_v = [Eq(v[t_dim + 1, 0, y_dim], bc_value)]  # left
    bc_v += [Eq(v[t_dim + 1, Nx - 1, y_dim], bc_value)]  # right
    bc_v += [Eq(v[t_dim + 1, x_dim, 0], bc_value)]  # bottom
    bc_v += [Eq(v[t_dim + 1, x_dim, Ny - 1], bc_value)]  # top

    # Create operator
    op = Operator([update_u, update_v] + bc_u + bc_v)

    # Storage for history
    u_history = []
    v_history = []
    t_history = []

    if save_history:
        u_history.append(u.data[0, :, :].copy())
        v_history.append(v.data[0, :, :].copy())
        t_history.append(0.0)

    # Time stepping
    for n in range(Nt):
        op.apply(time_m=n, time_M=n, dt=dt, a=nu)

        if save_history and (n + 1) % save_every == 0:
            u_history.append(u.data[(n + 1) % 2, :, :].copy())
            v_history.append(v.data[(n + 1) % 2, :, :].copy())
            t_history.append((n + 1) * dt)

    # Get final solution
    final_idx = Nt % 2
    u_final = u.data[final_idx, :, :].copy()
    v_final = v.data[final_idx, :, :].copy()

    return Burgers2DResult(
        u=u_final,
        v=v_final,
        x=x_coords,
        y=y_coords,
        t=actual_T,
        dt=dt,
        u_history=u_history if save_history else None,
        v_history=v_history if save_history else None,
        t_history=t_history if save_history else None,
    )


def solve_burgers_2d_vector(
    Lx: float = 2.0,
    Ly: float = 2.0,
    nu: float = 0.01,
    Nx: int = 41,
    Ny: int = 41,
    T: float = 0.5,
    sigma: float = 0.0009,
    I_u: Callable | None = None,
    I_v: Callable | None = None,
    bc_value: float = 1.0,
    save_history: bool = False,
    save_every: int = 100,
) -> Burgers2DResult:
    """Solve 2D Burgers equations using VectorTimeFunction.

    This is an alternative implementation using Devito's
    VectorTimeFunction to represent the velocity field as
    a single vector U = (u, v).

    The vector form of Burgers' equation:
        U_t + (grad(U) * U) = nu * laplace(U)

    Parameters
    ----------
    Lx : float
        Domain length in x direction [0, Lx]
    Ly : float
        Domain length in y direction [0, Ly]
    nu : float
        Viscosity (diffusion coefficient)
    Nx : int
        Number of grid points in x
    Ny : int
        Number of grid points in y
    T : float
        Final simulation time
    sigma : float
        Stability parameter: dt = sigma * dx * dy / nu
    I_u : callable or None
        Initial condition for u component
    I_v : callable or None
        Initial condition for v component
    bc_value : float
        Dirichlet boundary condition value
    save_history : bool
        If True, save solution history
    save_every : int
        Save every N time steps (if save_history=True)

    Returns
    -------
    Burgers2DResult
        Solution data container
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. Install with: pip install devito"
        )

    # Grid setup
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = sigma * dx * dy / nu

    # Handle T=0 case
    if T <= 0:
        x_coords = np.linspace(0, Lx, Nx)
        y_coords = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        if I_u is None:
            u0 = init_hat(X, Y, Lx, Ly, value=2.0)
        else:
            u0 = I_u(X, Y)
        if I_v is None:
            v0 = init_hat(X, Y, Lx, Ly, value=2.0)
        else:
            v0 = I_v(X, Y)
        return Burgers2DResult(
            u=u0,
            v=v0,
            x=x_coords,
            y=y_coords,
            t=0.0,
            dt=dt,
        )

    Nt = int(round(T / dt))
    actual_T = Nt * dt

    # Create Devito grid
    grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
    x_dim, y_dim = grid.dimensions
    t_dim = grid.stepping_dim
    s = grid.time_dim.spacing  # dt symbol

    # Create VectorTimeFunction
    U = VectorTimeFunction(name="U", grid=grid, space_order=2)

    # Get coordinate arrays
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

    # Set initial conditions
    # U[0] is the x-component (u), U[1] is the y-component (v)
    if I_u is None:
        U[0].data[0, :, :] = init_hat(X, Y, Lx, Ly, value=2.0)
    else:
        U[0].data[0, :, :] = I_u(X, Y)

    if I_v is None:
        U[1].data[0, :, :] = init_hat(X, Y, Lx, Ly, value=2.0)
    else:
        U[1].data[0, :, :] = I_v(X, Y)

    # Viscosity as Devito Constant
    a = Constant(name="a")

    # Vector form of Burgers equation:
    # U_t + grad(U) * U = nu * laplace(U)
    # Rearranged: U_forward = U - dt * (grad(U) * U - nu * laplace(U))
    update_U = Eq(
        U.forward,
        U - s * (grad(U) * U - a * U.laplace),
        subdomain=grid.interior,
    )

    # Boundary conditions for both components
    bc_U = [Eq(U[0][t_dim + 1, 0, y_dim], bc_value)]  # u left
    bc_U += [Eq(U[0][t_dim + 1, Nx - 1, y_dim], bc_value)]  # u right
    bc_U += [Eq(U[0][t_dim + 1, x_dim, 0], bc_value)]  # u bottom
    bc_U += [Eq(U[0][t_dim + 1, x_dim, Ny - 1], bc_value)]  # u top
    bc_U += [Eq(U[1][t_dim + 1, 0, y_dim], bc_value)]  # v left
    bc_U += [Eq(U[1][t_dim + 1, Nx - 1, y_dim], bc_value)]  # v right
    bc_U += [Eq(U[1][t_dim + 1, x_dim, 0], bc_value)]  # v bottom
    bc_U += [Eq(U[1][t_dim + 1, x_dim, Ny - 1], bc_value)]  # v top

    # Create operator
    op = Operator([update_U] + bc_U)

    # Storage for history
    u_history = []
    v_history = []
    t_history = []

    if save_history:
        u_history.append(U[0].data[0, :, :].copy())
        v_history.append(U[1].data[0, :, :].copy())
        t_history.append(0.0)

    # Time stepping
    for n in range(Nt):
        op.apply(time_m=n, time_M=n, dt=dt, a=nu)

        if save_history and (n + 1) % save_every == 0:
            u_history.append(U[0].data[(n + 1) % 2, :, :].copy())
            v_history.append(U[1].data[(n + 1) % 2, :, :].copy())
            t_history.append((n + 1) * dt)

    # Get final solution
    final_idx = Nt % 2
    u_final = U[0].data[final_idx, :, :].copy()
    v_final = U[1].data[final_idx, :, :].copy()

    return Burgers2DResult(
        u=u_final,
        v=v_final,
        x=x_coords,
        y=y_coords,
        t=actual_T,
        dt=dt,
        u_history=u_history if save_history else None,
        v_history=v_history if save_history else None,
        t_history=t_history if save_history else None,
    )


def sinusoidal_initial_condition(
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float = 2.0,
    Ly: float = 2.0,
) -> np.ndarray:
    """Sinusoidal initial condition.

    Creates sin(pi * x / Lx) * sin(pi * y / Ly).

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

    Returns
    -------
    np.ndarray
        Initial condition array
    """
    return np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)


def gaussian_initial_condition(
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float = 2.0,
    Ly: float = 2.0,
    sigma: float = 0.2,
    amplitude: float = 2.0,
) -> np.ndarray:
    """2D Gaussian initial condition centered in domain.

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
    amplitude : float
        Peak amplitude

    Returns
    -------
    np.ndarray
        Gaussian profile + 1.0 (background)
    """
    x0, y0 = Lx / 2, Ly / 2
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    return 1.0 + amplitude * np.exp(-r2 / (2 * sigma**2))
