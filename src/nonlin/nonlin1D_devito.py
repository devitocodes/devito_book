"""1D Nonlinear PDE Solvers using Devito DSL.

Solves nonlinear PDEs including:
1. Nonlinear diffusion: u_t = div(D(u) * grad(u))
2. Reaction-diffusion: u_t = a * u_xx + R(u)
3. Burgers' equation: u_t + u * u_x = nu * u_xx

Key techniques:
- Explicit time stepping with lagged coefficients
- Operator splitting (Lie and Strang splitting)
- Picard iteration for implicit schemes
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from devito import Constant, Eq, Function, Grid, Operator, TimeFunction


@dataclass
class NonlinearResult:
    """Result container for nonlinear PDE solver."""

    u: np.ndarray  # Final solution
    x: np.ndarray  # Spatial coordinates
    t: float  # Final time
    dt: float  # Time step used
    u_history: list | None = None  # Solution history (if save_history=True)
    t_history: list | None = None  # Time values (if save_history=True)


def solve_nonlinear_diffusion_explicit(
    L: float = 1.0,
    Nx: int = 100,
    T: float = 0.1,
    F: float = 0.4,
    I: Callable | None = None,
    D_func: Callable | None = None,
    save_history: bool = False,
) -> NonlinearResult:
    """
    Solve nonlinear diffusion equation with explicit time stepping.

    Solves: u_t = (D(u) * u_x)_x  on [0, L] with Dirichlet BCs u(0,t) = u(L,t) = 0

    Uses Forward Euler with lagged coefficient evaluation:
        u^{n+1} = u^n + dt * D(u^n) * u_xx^n

    Parameters
    ----------
    L : float
        Domain length [0, L]
    Nx : int
        Number of spatial intervals
    T : float
        Final time
    F : float
        Target mesh Fourier number (F = D*dt/dx^2, should be <= 0.5)
    I : callable
        Initial condition function I(x)
    D_func : callable
        Diffusion coefficient function D(u)
    save_history : bool
        If True, save solution at each time step

    Returns
    -------
    NonlinearResult
        Solution data container
    """
    # Default initial condition: sine wave
    if I is None:

        def I(x):
            return np.sin(np.pi * x / L)

    # Default diffusion coefficient: D(u) = 1 + u (nonlinear)
    if D_func is None:

        def D_func(u):
            return 1.0 + u

    # Grid setup
    dx = L / Nx
    x_coords = np.linspace(0, L, Nx + 1)
    u_init = I(x_coords)

    # Estimate max D for stability using initial condition
    D_init = D_func(u_init)
    D_max = max(np.max(np.abs(D_init)), 1e-10)  # Avoid division by zero
    dt = F * dx**2 / D_max
    Nt = int(round(T / dt))
    if Nt == 0:
        Nt = 1
    actual_T = Nt * dt

    # Create Devito grid and functions
    grid = Grid(shape=(Nx + 1,), extent=(L,))
    (x_dim,) = grid.dimensions
    t_dim = grid.stepping_dim

    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=1)
    D = Function(name="D", grid=grid)

    # Set initial condition
    u.data[0, :] = u_init
    u.data[1, :] = u_init
    D.data[:] = D_init

    # Time step as Devito Constant
    dt_const = Constant(name="dt", value=dt)

    # Explicit shifted indexing for second derivative
    u_plus = u.subs(x_dim, x_dim + x_dim.spacing)
    u_minus = u.subs(x_dim, x_dim - x_dim.spacing)

    # Explicit update: u^{n+1} = u + dt * D(u^n) * u_xx^n
    stencil = u + dt_const * D / (dx**2) * (u_plus - 2 * u + u_minus)
    update = Eq(u.forward, stencil)

    # Boundary conditions: u(0) = u(L) = 0
    bc_left = Eq(u[t_dim + 1, 0], 0.0)
    bc_right = Eq(u[t_dim + 1, Nx], 0.0)

    op = Operator([update, bc_left, bc_right])

    # Time stepping
    u_history = []
    t_history = []

    if save_history:
        u_history.append(u.data[0, :].copy())
        t_history.append(0.0)

    for n in range(Nt):
        # Update diffusion coefficient based on current solution
        curr_idx = n % 2
        D.data[:] = D_func(u.data[curr_idx, :])

        op.apply(time_m=n, time_M=n, dt=dt)

        if save_history:
            u_history.append(u.data[(n + 1) % 2, :].copy())
            t_history.append((n + 1) * dt)

    # Get final solution
    final_idx = Nt % 2
    u_final = u.data[final_idx, :].copy()

    return NonlinearResult(
        u=u_final,
        x=x_coords,
        t=actual_T,
        dt=dt,
        u_history=u_history if save_history else None,
        t_history=t_history if save_history else None,
    )


def solve_reaction_diffusion_splitting(
    L: float = 1.0,
    a: float = 1.0,
    Nx: int = 100,
    T: float = 0.1,
    F: float = 0.4,
    I: Callable | None = None,
    R_func: Callable | None = None,
    splitting: str = "strang",
    save_history: bool = False,
) -> NonlinearResult:
    """
    Solve reaction-diffusion equation using operator splitting.

    Solves: u_t = a * u_xx + R(u)  on [0, L] with Dirichlet BCs

    Uses operator splitting to separate diffusion and reaction:
    - Lie splitting: O(dt) accuracy
    - Strang splitting: O(dt^2) accuracy

    Parameters
    ----------
    L : float
        Domain length [0, L]
    a : float
        Diffusion coefficient
    Nx : int
        Number of spatial intervals
    T : float
        Final time
    F : float
        Target mesh Fourier number (F = a*dt/dx^2, should be <= 0.5)
    I : callable
        Initial condition function I(x)
    R_func : callable
        Reaction term function R(u)
    splitting : str
        Splitting method: "lie" (first-order) or "strang" (second-order)
    save_history : bool
        If True, save solution at each time step

    Returns
    -------
    NonlinearResult
        Solution data container
    """
    if splitting not in ("lie", "strang"):
        raise ValueError(f"splitting must be 'lie' or 'strang', got '{splitting}'")

    # Default initial condition: Gaussian pulse
    if I is None:

        def I(x):
            return np.exp(-0.5 * ((x - L / 2) / (L / 10)) ** 2)

    # Default reaction term: logistic growth R(u) = u*(1-u)
    if R_func is None:

        def R_func(u):
            return u * (1 - u)

    # Grid setup
    dx = L / Nx
    dt = F * dx**2 / a
    Nt = int(round(T / dt))
    if Nt == 0:
        Nt = 1
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

    # Fourier number as Devito Constant
    fourier = Constant(name="F", value=F)

    # Diffusion update using explicit shifted indexing:
    # u^{n+1} = u + F * (u[x+dx] - 2*u + u[x-dx])
    u_plus = u.subs(x_dim, x_dim + x_dim.spacing)
    u_minus = u.subs(x_dim, x_dim - x_dim.spacing)
    diff_stencil = u + fourier * (u_plus - 2 * u + u_minus)
    diff_update = Eq(u.forward, diff_stencil)

    # Boundary conditions
    bc_left = Eq(u[t_dim + 1, 0], 0.0)
    bc_right = Eq(u[t_dim + 1, Nx], 0.0)

    op_diff = Operator([diff_update, bc_left, bc_right])

    # Time stepping
    u_history = []
    t_history = []

    if save_history:
        u_history.append(u.data[0, :].copy())
        t_history.append(0.0)

    for n in range(Nt):
        curr_idx = n % 2
        next_idx = (n + 1) % 2

        if splitting == "strang":
            # Strang splitting: R(dt/2) -> D(dt) -> R(dt/2)
            # Half step of reaction (interior only)
            u.data[curr_idx, 1:-1] += 0.5 * dt * R_func(u.data[curr_idx, 1:-1])

            # Full step of diffusion
            op_diff.apply(time_m=n, time_M=n, dt=dt)

            # Half step of reaction (interior only)
            u.data[next_idx, 1:-1] += 0.5 * dt * R_func(u.data[next_idx, 1:-1])

        else:  # Lie splitting
            # Lie splitting: D(dt) -> R(dt)
            # Full step of diffusion
            op_diff.apply(time_m=n, time_M=n, dt=dt)

            # Full step of reaction (interior only)
            u.data[next_idx, 1:-1] += dt * R_func(u.data[next_idx, 1:-1])

        if save_history:
            u_history.append(u.data[next_idx, :].copy())
            t_history.append((n + 1) * dt)

    # Get final solution
    final_idx = Nt % 2
    u_final = u.data[final_idx, :].copy()

    return NonlinearResult(
        u=u_final,
        x=x_coords,
        t=actual_T,
        dt=dt,
        u_history=u_history if save_history else None,
        t_history=t_history if save_history else None,
    )


def solve_burgers_equation(
    L: float = 2.0,
    nu: float = 0.01,
    Nx: int = 100,
    T: float = 0.5,
    C: float = 0.5,
    I: Callable | None = None,
    save_history: bool = False,
) -> NonlinearResult:
    """
    Solve 1D viscous Burgers' equation using explicit time stepping.

    Solves: u_t + u * u_x = nu * u_xx  on [0, L]

    Uses conservative form (u^2/2)_x with centered differences and
    centered difference for the diffusion term.

    Parameters
    ----------
    L : float
        Domain length [0, L]
    nu : float
        Viscosity (diffusion coefficient)
    Nx : int
        Number of spatial intervals
    T : float
        Final time
    C : float
        Target CFL number (C = u_max * dt / dx)
    I : callable
        Initial condition function I(x)
    save_history : bool
        If True, save solution at each time step

    Returns
    -------
    NonlinearResult
        Solution data container
    """
    # Default initial condition: smooth bump that satisfies BCs
    if I is None:

        def I(x):
            return np.sin(np.pi * x / L)

    # Grid setup
    dx = L / Nx
    x_coords = np.linspace(0, L, Nx + 1)
    u_init = I(x_coords)
    u_max = max(abs(u_init.max()), abs(u_init.min()), 0.1)

    # Time step: use more conservative stability criteria
    # CFL for advection and Fourier for diffusion, with safety factor
    dt_advec = 0.5 * C * dx / u_max  # Conservative CFL
    dt_diff = 0.25 * dx**2 / nu if nu > 0 else float("inf")  # F=0.25 for stability
    dt = min(dt_advec, dt_diff)
    Nt = int(round(T / dt))
    if Nt == 0:
        Nt = 1
    actual_T = Nt * dt

    # Create Devito grid and functions
    grid = Grid(shape=(Nx + 1,), extent=(L,))
    (x_dim,) = grid.dimensions
    t_dim = grid.stepping_dim

    # Use space_order=2 to allocate halo points for boundary stencil access
    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)

    # Set initial condition
    u.data[0, :] = u_init
    u.data[1, :] = u_init

    # Time step and viscosity as Devito Constants
    dt_const = Constant(name="dt", value=np.float32(dt))
    nu_const = Constant(name="nu", value=np.float32(nu))

    # Neighbor values using explicit shifted indexing
    u_plus = u.subs(x_dim, x_dim + x_dim.spacing)
    u_minus = u.subs(x_dim, x_dim - x_dim.spacing)

    # Conservative form: u_t + (u^2/2)_x = nu * u_xx
    # Advection: use centered difference for flux derivative
    # (u^2/2)_x ≈ (u_{i+1}^2 - u_{i-1}^2) / (4*dx)
    advection_term = 0.25 * dt_const / dx * (u_plus**2 - u_minus**2)
    diffusion_term = nu_const * dt_const / (dx**2) * (u_plus - 2 * u + u_minus)
    stencil = u - advection_term + diffusion_term

    # Apply stencil only to interior points using subdomain
    update = Eq(u.forward, stencil, subdomain=grid.interior)

    # Dirichlet boundary conditions: u(0) = u(L) = 0
    bc_left = Eq(u[t_dim + 1, 0], 0.0)
    bc_right = Eq(u[t_dim + 1, Nx], 0.0)

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

    return NonlinearResult(
        u=u_final,
        x=x_coords,
        t=actual_T,
        dt=dt,
        u_history=u_history if save_history else None,
        t_history=t_history if save_history else None,
    )


def solve_nonlinear_diffusion_picard(
    L: float = 1.0,
    Nx: int = 100,
    T: float = 0.1,
    dt: float = 0.01,
    I: Callable | None = None,
    D_func: Callable | None = None,
    picard_tol: float = 1e-6,
    picard_max_iter: int = 100,
    save_history: bool = False,
) -> NonlinearResult:
    """
    Solve nonlinear diffusion with Picard iteration (implicit).

    Solves: u_t = (D(u) * u_x)_x  using Backward Euler with Picard iteration.

    At each time step, solves the nonlinear system iteratively:
        (u^{n+1,k+1} - u^n) / dt = (D(u^{n+1,k}) * u_xx^{n+1,k+1})

    Parameters
    ----------
    L : float
        Domain length [0, L]
    Nx : int
        Number of spatial intervals
    T : float
        Final time
    dt : float
        Time step (no stability restriction for implicit)
    I : callable
        Initial condition function I(x)
    D_func : callable
        Diffusion coefficient function D(u)
    picard_tol : float
        Convergence tolerance for Picard iteration
    picard_max_iter : int
        Maximum Picard iterations per time step
    save_history : bool
        If True, save solution at each time step

    Returns
    -------
    NonlinearResult
        Solution data container

    Note
    ----
    This implementation uses explicit Forward Euler for the inner Picard
    iteration, which is a simplified approach. A full implicit scheme would
    require solving a linear system at each Picard iteration.
    """
    # Default initial condition
    if I is None:

        def I(x):
            return np.sin(np.pi * x / L)

    # Default diffusion coefficient
    if D_func is None:

        def D_func(u):
            return 1.0 + u

    # Grid setup
    dx = L / Nx
    Nt = int(round(T / dt))
    actual_T = Nt * dt

    # Create Devito grid and functions
    grid = Grid(shape=(Nx + 1,), extent=(L,))
    t_dim = grid.stepping_dim

    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)
    u_old = Function(name="u_old", grid=grid)  # Previous time level
    D = Function(name="D", grid=grid)

    # Set initial condition
    x_coords = np.linspace(0, L, Nx + 1)
    u.data[0, :] = I(x_coords)
    u.data[1, :] = I(x_coords)

    # Picard iteration operator
    # Simplified: use lagged D but still explicit in time
    # u^{k+1} = u^n + dt * D(u^k) * u_xx^k
    dt_const = Constant(name="dt", value=dt)
    stencil = u_old + dt_const * D * u.dx2
    update = Eq(u.forward, stencil, subdomain=grid.interior)

    bc_left = Eq(u[t_dim + 1, 0], 0.0)
    bc_right = Eq(u[t_dim + 1, Nx], 0.0)

    op = Operator([update, bc_left, bc_right])

    # Time stepping
    u_history = []
    t_history = []

    if save_history:
        u_history.append(u.data[0, :].copy())
        t_history.append(0.0)

    for n in range(Nt):
        curr_idx = n % 2
        next_idx = (n + 1) % 2

        # Store previous time level
        u_old.data[:] = u.data[curr_idx, :]

        # Picard iteration
        for k in range(picard_max_iter):
            # Update diffusion coefficient
            D.data[:] = D_func(u.data[curr_idx, :])

            # Store current iterate for convergence check
            u_prev = u.data[curr_idx, :].copy()

            # Apply one iteration
            op.apply(time_m=n, time_M=n, dt=dt)

            # Copy result back for next iteration
            u.data[curr_idx, :] = u.data[next_idx, :]

            # Check convergence
            diff = np.max(np.abs(u.data[curr_idx, :] - u_prev))
            if diff < picard_tol:
                break

        # Final result is in curr_idx, copy to next_idx for proper indexing
        u.data[next_idx, :] = u.data[curr_idx, :]

        if save_history:
            u_history.append(u.data[next_idx, :].copy())
            t_history.append((n + 1) * dt)

    # Get final solution
    final_idx = Nt % 2
    u_final = u.data[final_idx, :].copy()

    return NonlinearResult(
        u=u_final,
        x=x_coords,
        t=actual_T,
        dt=dt,
        u_history=u_history if save_history else None,
        t_history=t_history if save_history else None,
    )


# Common reaction functions
def logistic_reaction(u: np.ndarray, r: float = 1.0, K: float = 1.0) -> np.ndarray:
    """Logistic growth reaction term: R(u) = r * u * (1 - u/K)."""
    return r * u * (1 - u / K)


def fisher_reaction(u: np.ndarray, r: float = 1.0) -> np.ndarray:
    """Fisher-KPP reaction term: R(u) = r * u * (1 - u)."""
    return r * u * (1 - u)


def allen_cahn_reaction(u: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """Allen-Cahn reaction term: R(u) = (u - u^3) / epsilon^2."""
    return (u - u**3) / epsilon**2


# Common diffusion coefficient functions
def constant_diffusion(u: np.ndarray, D0: float = 1.0) -> np.ndarray:
    """Constant diffusion coefficient (linear case)."""
    return np.full_like(u, D0)


def linear_diffusion(u: np.ndarray, D0: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    """Linear diffusion coefficient: D(u) = D0 + alpha * u."""
    return D0 + alpha * u


def porous_medium_diffusion(
    u: np.ndarray, m: float = 2.0, D0: float = 1.0
) -> np.ndarray:
    """Porous medium diffusion: D(u) = D0 * m * u^(m-1)."""
    return D0 * m * np.maximum(u, 0) ** (m - 1)
