"""2D Incompressible Navier-Stokes Solver using Devito DSL.

Solves the incompressible Navier-Stokes equations:

    du/dt + u*du/dx + v*du/dy = -1/rho * dp/dx + nu * laplace(u)
    dv/dt + u*dv/dx + v*dv/dy = -1/rho * dp/dy + nu * laplace(v)
    div(u, v) = du/dx + dv/dy = 0  (incompressibility)

The solver uses the fractional step (projection) method:
1. Predict intermediate velocities ignoring pressure
2. Solve pressure Poisson equation to enforce divergence-free velocity
3. Correct velocities using pressure gradient

The primary application is the lid-driven cavity flow benchmark problem.

Boundary conditions for lid-driven cavity:
    - u = U_lid, v = 0 on top wall (moving lid)
    - u = v = 0 on other walls (no-slip)
    - dp/dn = 0 on all walls (Neumann for pressure)

The Reynolds number is Re = U_lid * L / nu, where:
    - U_lid is the lid velocity
    - L is the cavity size
    - nu is the kinematic viscosity

Usage:
    from src.cfd import solve_cavity_2d

    result = solve_cavity_2d(
        N=41,               # Grid points
        Re=100.0,           # Reynolds number
        nt=1000,            # Time steps
        nit=50,             # Pressure iterations per step
    )

References:
    - Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions
      for incompressible flow using the Navier-Stokes equations and
      a multigrid method. Journal of Computational Physics, 48(3), 387-411.
"""

from dataclasses import dataclass

import numpy as np

try:
    from devito import Eq, Function, Grid, Operator, TimeFunction, configuration, solve
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@dataclass
class CavityResult:
    """Results from the lid-driven cavity flow solver.

    Attributes
    ----------
    u : np.ndarray
        Final x-velocity field, shape (N, N)
    v : np.ndarray
        Final y-velocity field, shape (N, N)
    p : np.ndarray
        Final pressure field, shape (N, N)
    x : np.ndarray
        x-coordinate array
    y : np.ndarray
        y-coordinate array
    Re : float
        Reynolds number
    nt : int
        Number of time steps performed
    converged : bool
        Whether steady state was reached
    u_history : list, optional
        History of u-velocity at specified intervals
    v_history : list, optional
        History of v-velocity at specified intervals
    """
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    x: np.ndarray
    y: np.ndarray
    Re: float
    nt: int
    converged: bool
    u_history: list | None = None
    v_history: list | None = None


def ghia_benchmark_data(Re: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Return Ghia et al. (1982) benchmark data for lid-driven cavity.

    The benchmark provides centerline velocity profiles for validation:
    - u-velocity along vertical centerline (x = 0.5)
    - v-velocity along horizontal centerline (y = 0.5)

    Parameters
    ----------
    Re : float
        Reynolds number. Available: 100, 400, 1000, 3200

    Returns
    -------
    tuple
        (u_data, v_data) where each is an array with columns [y/L or x/L, velocity]

    References
    ----------
    Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for
    incompressible flow using the Navier-Stokes equations and a multigrid
    method. Journal of Computational Physics, 48(3), 387-411.
    """
    # y-coordinates and u-velocity at x = 0.5 (vertical centerline)
    # First column: y/L, Second column: u/U_lid
    y_coords = np.array([
        0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
        0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
        0.9688, 0.9766, 1.0000
    ])

    # x-coordinates and v-velocity at y = 0.5 (horizontal centerline)
    x_coords = np.array([
        0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
        0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
        0.9609, 0.9688, 1.0000
    ])

    if Re == 100:
        u_values = np.array([
            0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
            -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
            0.68717, 0.73722, 0.78871, 0.84123, 1.00000
        ])
        v_values = np.array([
            0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
            0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
            -0.10313, -0.08864, -0.07391, -0.05906, 0.00000
        ])
    elif Re == 400:
        u_values = np.array([
            0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
            -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
            0.55892, 0.61756, 0.68439, 0.75837, 1.00000
        ])
        v_values = np.array([
            0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124,
            0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827,
            -0.22847, -0.19254, -0.15663, -0.12146, 0.00000
        ])
    elif Re == 1000:
        u_values = np.array([
            0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
            -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
            0.46604, 0.51117, 0.57492, 0.65928, 1.00000
        ])
        v_values = np.array([
            0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
            0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
            -0.39188, -0.33714, -0.27669, -0.21388, 0.00000
        ])
    elif Re == 3200:
        u_values = np.array([
            0.00000, -0.32407, -0.35344, -0.37827, -0.41933, -0.34323,
            -0.24427, -0.86636, -0.04272, 0.07156, 0.19791, 0.34682,
            0.46101, 0.46547, 0.48296, 0.53236, 1.00000
        ])
        v_values = np.array([
            0.00000, 0.39560, 0.40917, 0.41906, 0.42768, 0.37119,
            0.29030, 0.28188, 0.00999, -0.31184, -0.37401, -0.44307,
            -0.54053, -0.52357, -0.47425, -0.39017, 0.00000
        ])
    else:
        raise ValueError(
            f"Benchmark data not available for Re={Re}. "
            "Available: 100, 400, 1000, 3200"
        )

    # Stack into arrays with [coordinate, velocity]
    u_data = np.column_stack([y_coords, u_values])
    v_data = np.column_stack([x_coords, v_values])

    return u_data, v_data


def apply_velocity_bcs(
    u_data: np.ndarray,
    v_data: np.ndarray,
    N: int,
    U_lid: float = 1.0,
) -> None:
    """Apply velocity boundary conditions for lid-driven cavity.

    In-place modification of velocity arrays.

    Parameters
    ----------
    u_data : np.ndarray
        x-velocity field, shape (N, N), modified in place
    v_data : np.ndarray
        y-velocity field, shape (N, N), modified in place
    N : int
        Grid size
    U_lid : float
        Lid velocity (default 1.0)
    """
    # Bottom wall (y = 0): no-slip
    u_data[:, 0] = 0.0
    v_data[:, 0] = 0.0

    # Top wall (y = 1): moving lid
    u_data[:, -1] = U_lid
    v_data[:, -1] = 0.0

    # Left wall (x = 0): no-slip
    u_data[0, :] = 0.0
    v_data[0, :] = 0.0

    # Right wall (x = 1): no-slip
    u_data[-1, :] = 0.0
    v_data[-1, :] = 0.0


def pressure_poisson_iteration(
    p: np.ndarray,
    b: np.ndarray,
    dx: float,
    dy: float,
    nit: int = 50,
) -> np.ndarray:
    """Solve pressure Poisson equation iteratively (NumPy reference).

    Solves: laplace(p) = b
    with Neumann boundary conditions dp/dn = 0 on all walls,
    plus p = 0 at one point for uniqueness.

    Parameters
    ----------
    p : np.ndarray
        Initial pressure guess and output, shape (N, N)
    b : np.ndarray
        Right-hand side source term, shape (N, N)
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y
    nit : int
        Number of Jacobi iterations

    Returns
    -------
    np.ndarray
        Updated pressure field
    """
    pn = np.empty_like(p)

    for _ in range(nit):
        pn[:] = p[:]

        # Jacobi update for interior points
        p[1:-1, 1:-1] = (
            ((pn[2:, 1:-1] + pn[:-2, 1:-1]) * dy**2 +
             (pn[1:-1, 2:] + pn[1:-1, :-2]) * dx**2) /
            (2 * (dx**2 + dy**2)) -
            dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]
        )

        # Neumann BCs: dp/dn = 0
        p[0, :] = p[1, :]      # dp/dx = 0 at x = 0
        p[-1, :] = p[-2, :]    # dp/dx = 0 at x = 1
        p[:, 0] = p[:, 1]      # dp/dy = 0 at y = 0
        p[:, -1] = p[:, -2]    # dp/dy = 0 at y = 1

        # Fix pressure at one point for uniqueness
        p[0, 0] = 0.0

    return p


def compute_streamfunction(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the stream function from velocity field.

    The stream function psi satisfies:
        u = dpsi/dy
        v = -dpsi/dx

    Computed by integrating v along x, then correcting with u.

    Parameters
    ----------
    u : np.ndarray
        x-velocity field, shape (N, N)
    v : np.ndarray
        y-velocity field, shape (N, N)
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y

    Returns
    -------
    np.ndarray
        Stream function field, shape (N, N)
    """
    N = u.shape[0]
    psi = np.zeros((N, N))

    # Integrate -v along x for each y
    for j in range(N):
        for i in range(1, N):
            psi[i, j] = psi[i-1, j] - v[i, j] * dx

    return psi


def solve_cavity_2d(
    N: int = 41,
    Re: float = 100.0,
    nt: int = 1000,
    nit: int = 50,
    dt: float | None = None,
    U_lid: float = 1.0,
    L: float = 1.0,
    rho: float = 1.0,
    steady_tol: float = 1e-6,
    check_steady: int = 100,
    save_interval: int | None = None,
) -> CavityResult:
    """Solve the 2D lid-driven cavity flow using Devito.

    Uses the fractional step (projection) method:
    1. Advance velocity with convection and diffusion (ignoring pressure)
    2. Solve pressure Poisson equation for divergence-free correction
    3. Correct velocities with pressure gradient

    Parameters
    ----------
    N : int
        Number of grid points in each direction (N x N grid)
    Re : float
        Reynolds number. Re = U_lid * L / nu
    nt : int
        Maximum number of time steps
    nit : int
        Number of pressure Poisson iterations per time step
    dt : float, optional
        Time step. If None, computed from stability requirements.
    U_lid : float
        Lid velocity (default 1.0 for unit normalization)
    L : float
        Cavity size (default 1.0 for unit square)
    rho : float
        Fluid density (default 1.0)
    steady_tol : float
        Tolerance for steady state detection
    check_steady : int
        Check for steady state every this many steps
    save_interval : int, optional
        Save velocity history every this many steps

    Returns
    -------
    CavityResult
        Solution data including final velocity, pressure, and optional history

    Raises
    ------
    ImportError
        If Devito is not installed
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    # Suppress Devito logging
    configuration['log-level'] = 'ERROR'

    # Compute kinematic viscosity from Reynolds number
    nu = U_lid * L / Re

    # Grid spacing
    dx = L / (N - 1)
    dy = L / (N - 1)

    # Time step from stability (CFL and diffusion)
    if dt is None:
        # CFL condition: dt < dx / U_lid
        # Diffusion stability: dt < 0.25 * dx^2 / nu
        dt_cfl = 0.5 * dx / U_lid
        dt_diff = 0.25 * dx**2 / nu
        dt = min(dt_cfl, dt_diff, 0.001)

    # Coordinate arrays
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)

    # Create Devito grid
    grid = Grid(shape=(N, N), extent=(L, L))
    x_dim, y_dim = grid.dimensions
    t = grid.stepping_dim

    # Create TimeFunction fields for velocities
    u = TimeFunction(name='u', grid=grid, space_order=2)
    v = TimeFunction(name='v', grid=grid, space_order=2)

    # Create TimeFunction for pressure (use pseudo-time for iteration)
    p = TimeFunction(name='p', grid=grid, space_order=2)

    # Initialize fields to zero
    u.data[:] = 0.0
    v.data[:] = 0.0
    p.data[:] = 0.0

    # -------------------------------------------------------------------------
    # Build the pressure Poisson operator
    # -------------------------------------------------------------------------
    # The RHS of pressure Poisson is computed from velocity divergence
    # We compute this in Python and pass it as a Function

    b = Function(name='b', grid=grid)

    # Pressure Poisson equation: laplace(p) = b
    # Using p.forward to alternate buffers in pseudo-time iteration
    eq_p = Eq(p.laplace, b, subdomain=grid.interior)
    stencil_p = solve(eq_p, p)
    update_p = Eq(p.forward, stencil_p)

    # Pressure boundary conditions (Neumann: dp/dn = 0)
    bc_p = [
        Eq(p[t+1, 0, y_dim], p[t+1, 1, y_dim]),       # dp/dx = 0 at x = 0
        Eq(p[t+1, N-1, y_dim], p[t+1, N-2, y_dim]),   # dp/dx = 0 at x = 1
        Eq(p[t+1, x_dim, 0], p[t+1, x_dim, 1]),       # dp/dy = 0 at y = 0
        Eq(p[t+1, x_dim, N-1], p[t+1, x_dim, N-2]),   # dp/dy = 0 at y = 1
        Eq(p[t+1, 0, 0], 0),                           # Fix p at corner
    ]

    op_pressure = Operator([update_p] + bc_p)

    # -------------------------------------------------------------------------
    # Build the velocity update operator
    # -------------------------------------------------------------------------
    # Momentum equations (using first-order upwind for advection)
    # du/dt + u*du/dx + v*du/dy = -1/rho * dp/dx + nu * laplace(u)

    eq_u = Eq(
        u.dt + u*u.dx + v*u.dy,
        -1.0/rho * p.dxc + nu * u.laplace,
        subdomain=grid.interior
    )
    eq_v = Eq(
        v.dt + u*v.dx + v*v.dy,
        -1.0/rho * p.dyc + nu * v.laplace,
        subdomain=grid.interior
    )

    stencil_u = solve(eq_u, u.forward)
    stencil_v = solve(eq_v, v.forward)

    update_u = Eq(u.forward, stencil_u)
    update_v = Eq(v.forward, stencil_v)

    # Velocity boundary conditions
    bc_u = [
        Eq(u[t+1, x_dim, 0], 0),                # Bottom: u = 0
        Eq(u[t+1, x_dim, N-1], U_lid),          # Top: u = U_lid
        Eq(u[t+1, 0, y_dim], 0),                # Left: u = 0
        Eq(u[t+1, N-1, y_dim], 0),              # Right: u = 0
    ]
    bc_v = [
        Eq(v[t+1, x_dim, 0], 0),                # Bottom: v = 0
        Eq(v[t+1, x_dim, N-1], 0),              # Top: v = 0
        Eq(v[t+1, 0, y_dim], 0),                # Left: v = 0
        Eq(v[t+1, N-1, y_dim], 0),              # Right: v = 0
    ]

    op_velocity = Operator([update_u, update_v] + bc_u + bc_v)

    # -------------------------------------------------------------------------
    # Time-stepping loop
    # -------------------------------------------------------------------------
    u_history = [] if save_interval is not None else None
    v_history = [] if save_interval is not None else None
    converged = False

    for step in range(nt):
        # Save history if requested
        if save_interval is not None and step % save_interval == 0:
            u_history.append(u.data[0].copy())
            v_history.append(v.data[0].copy())

        # Compute pressure Poisson RHS: b = rho * (div(u)/dt - nonlinear terms)
        # This enforces incompressibility in the corrected velocity
        u_curr = u.data[0]
        v_curr = v.data[0]

        b.data[1:-1, 1:-1] = rho * (
            # Divergence rate term
            1.0 / dt * (
                (u_curr[2:, 1:-1] - u_curr[:-2, 1:-1]) / (2*dx) +
                (v_curr[1:-1, 2:] - v_curr[1:-1, :-2]) / (2*dy)
            ) -
            # Nonlinear terms
            ((u_curr[2:, 1:-1] - u_curr[:-2, 1:-1]) / (2*dx))**2 -
            2 * ((u_curr[1:-1, 2:] - u_curr[1:-1, :-2]) / (2*dy) *
                 (v_curr[2:, 1:-1] - v_curr[:-2, 1:-1]) / (2*dx)) -
            ((v_curr[1:-1, 2:] - v_curr[1:-1, :-2]) / (2*dy))**2
        )

        # Solve pressure Poisson (pseudo-timestepping)
        if step > 0:
            op_pressure(time_M=nit)

        # Update velocities
        op_velocity(time_m=step, time_M=step, dt=dt)

        # Check for steady state
        if step > 0 and step % check_steady == 0:
            # Compare current and previous velocity fields
            u_diff = np.max(np.abs(u.data[0] - u.data[1]))
            v_diff = np.max(np.abs(v.data[0] - v.data[1]))

            if max(u_diff, v_diff) < steady_tol:
                converged = True
                break

    # Extract final results
    u_final = u.data[0].copy()
    v_final = v.data[0].copy()
    p_final = p.data[0].copy()

    return CavityResult(
        u=u_final,
        v=v_final,
        p=p_final,
        x=x,
        y=y,
        Re=Re,
        nt=step + 1,
        converged=converged,
        u_history=u_history,
        v_history=v_history,
    )


def solve_cavity_numpy(
    N: int = 41,
    Re: float = 100.0,
    nt: int = 1000,
    nit: int = 50,
    dt: float | None = None,
    U_lid: float = 1.0,
    L: float = 1.0,
    rho: float = 1.0,
) -> CavityResult:
    """Solve lid-driven cavity using pure NumPy (reference implementation).

    This provides a baseline for comparison with the Devito solver.

    Parameters
    ----------
    N : int
        Number of grid points in each direction
    Re : float
        Reynolds number
    nt : int
        Number of time steps
    nit : int
        Number of pressure iterations per step
    dt : float, optional
        Time step
    U_lid : float
        Lid velocity
    L : float
        Cavity size
    rho : float
        Fluid density

    Returns
    -------
    CavityResult
        Solution data
    """
    # Compute kinematic viscosity from Reynolds number
    nu = U_lid * L / Re

    # Grid spacing
    dx = L / (N - 1)
    dy = L / (N - 1)

    # Time step
    if dt is None:
        dt_cfl = 0.5 * dx / U_lid
        dt_diff = 0.25 * dx**2 / nu
        dt = min(dt_cfl, dt_diff, 0.001)

    # Coordinates
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)

    # Initialize fields
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    p = np.zeros((N, N))
    b = np.zeros((N, N))

    # Time stepping
    for _ in range(nt):
        un = u.copy()
        vn = v.copy()

        # Build pressure RHS
        b[1:-1, 1:-1] = rho * (
            1.0 / dt * (
                (un[2:, 1:-1] - un[:-2, 1:-1]) / (2*dx) +
                (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2*dy)
            ) -
            ((un[2:, 1:-1] - un[:-2, 1:-1]) / (2*dx))**2 -
            2 * ((un[1:-1, 2:] - un[1:-1, :-2]) / (2*dy) *
                 (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2*dx)) -
            ((vn[1:-1, 2:] - vn[1:-1, :-2]) / (2*dy))**2
        )

        # Solve pressure Poisson
        p = pressure_poisson_iteration(p, b, dx, dy, nit)

        # Update u-velocity
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1] -
            un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
            vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
            dt / (2 * rho * dx) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
            nu * (
                dt / dx**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]) +
                dt / dy**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])
            )
        )

        # Update v-velocity
        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1] -
            un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
            vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
            dt / (2 * rho * dy) * (p[1:-1, 2:] - p[1:-1, :-2]) +
            nu * (
                dt / dx**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1]) +
                dt / dy**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2])
            )
        )

        # Apply boundary conditions
        apply_velocity_bcs(u, v, N, U_lid)

    return CavityResult(
        u=u,
        v=v,
        p=p,
        x=x,
        y=y,
        Re=Re,
        nt=nt,
        converged=False,
    )


def extract_centerline_velocities(
    result: CavityResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract centerline velocity profiles from cavity solution.

    Parameters
    ----------
    result : CavityResult
        Solution from solve_cavity_2d

    Returns
    -------
    tuple
        (y, u_centerline, x, v_centerline) where:
        - y: y-coordinates
        - u_centerline: u-velocity along x = 0.5
        - x: x-coordinates
        - v_centerline: v-velocity along y = 0.5
    """
    N = len(result.x)
    mid = N // 2

    y = result.y
    u_centerline = result.u[mid, :]

    x = result.x
    v_centerline = result.v[:, mid]

    return y, u_centerline, x, v_centerline


def compute_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute vorticity field from velocity.

    Vorticity omega = dv/dx - du/dy

    Parameters
    ----------
    u : np.ndarray
        x-velocity field
    v : np.ndarray
        y-velocity field
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y

    Returns
    -------
    np.ndarray
        Vorticity field (interior points only, padded with zeros)
    """
    N = u.shape[0]
    omega = np.zeros((N, N))

    # Central differences for interior
    omega[1:-1, 1:-1] = (
        (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dx) -
        (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dy)
    )

    return omega
