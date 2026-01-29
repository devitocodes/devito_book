"""2D Laplace Equation Solver using Devito DSL.

Solves the steady-state Laplace equation:
    laplace(p) = p_xx + p_yy = 0

on domain [0, Lx] x [0, Ly] with:
    - Dirichlet boundary conditions: prescribed values on boundaries
    - Neumann boundary conditions: prescribed derivatives on boundaries

The discretization uses central differences for the Laplacian:
    p_{i,j} = (dx^2*(p_{i,j+1} + p_{i,j-1}) + dy^2*(p_{i+1,j} + p_{i-1,j}))
              / (2*(dx^2 + dy^2))

This is an iterative (pseudo-timestepping) solver that converges to
the steady-state solution. Convergence is measured using the L1 norm.

The solver uses a dual-buffer approach with two Function objects,
alternating between them to avoid data copies during iteration.

Usage:
    from src.elliptic import solve_laplace_2d

    result = solve_laplace_2d(
        Lx=2.0, Ly=1.0,           # Domain size
        Nx=31, Ny=31,             # Grid points
        bc_left=0.0,              # p = 0 at x = 0
        bc_right=lambda y: y,     # p = y at x = Lx
        bc_bottom='neumann',      # dp/dy = 0 at y = 0
        bc_top='neumann',         # dp/dy = 0 at y = Ly
        tol=1e-4,                 # Convergence tolerance
    )
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    from devito import Eq, Function, Grid, Operator, solve
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@dataclass
class LaplaceResult:
    """Results from the 2D Laplace equation solver.

    Attributes
    ----------
    p : np.ndarray
        Solution at convergence, shape (Nx+1, Ny+1)
    x : np.ndarray
        x-coordinate grid points
    y : np.ndarray
        y-coordinate grid points
    iterations : int
        Number of iterations to convergence
    final_l1norm : float
        Final L1 norm (convergence measure)
    converged : bool
        Whether the solver converged within max_iterations
    p_history : list, optional
        Solution history at specified intervals
    """
    p: np.ndarray
    x: np.ndarray
    y: np.ndarray
    iterations: int
    final_l1norm: float
    converged: bool
    p_history: list | None = None


def solve_laplace_2d(
    Lx: float = 2.0,
    Ly: float = 1.0,
    Nx: int = 31,
    Ny: int = 31,
    bc_left: float | Callable[[np.ndarray], np.ndarray] | str = 0.0,
    bc_right: float | Callable[[np.ndarray], np.ndarray] | str = "neumann",
    bc_bottom: float | Callable[[np.ndarray], np.ndarray] | str = "neumann",
    bc_top: float | Callable[[np.ndarray], np.ndarray] | str = "neumann",
    tol: float = 1e-4,
    max_iterations: int = 10000,
    save_interval: int | None = None,
) -> LaplaceResult:
    """Solve the 2D Laplace equation using Devito (iterative method).

    Solves: laplace(p) = p_xx + p_yy = 0
    using an iterative pseudo-timestepping approach with dual buffers.

    Parameters
    ----------
    Lx : float
        Domain length in x direction [0, Lx]
    Ly : float
        Domain length in y direction [0, Ly]
    Nx : int
        Number of grid points in x (including boundaries)
    Ny : int
        Number of grid points in y (including boundaries)
    bc_left : float, callable, or 'neumann'
        Boundary condition at x=0:
        - float: Dirichlet with constant value
        - callable: Dirichlet with f(y) profile
        - 'neumann': Zero-gradient (dp/dx = 0)
    bc_right : float, callable, or 'neumann'
        Boundary condition at x=Lx (same options as bc_left)
    bc_bottom : float, callable, or 'neumann'
        Boundary condition at y=0:
        - float: Dirichlet with constant value
        - callable: Dirichlet with f(x) profile
        - 'neumann': Zero-gradient (dp/dy = 0)
    bc_top : float, callable, or 'neumann'
        Boundary condition at y=Ly (same options as bc_bottom)
    tol : float
        Convergence tolerance for L1 norm
    max_iterations : int
        Maximum number of iterations
    save_interval : int, optional
        If specified, save solution every save_interval iterations

    Returns
    -------
    LaplaceResult
        Solution data including converged solution, grids, and iteration info

    Raises
    ------
    ImportError
        If Devito is not installed

    Notes
    -----
    The solver uses a dual-buffer approach where two Function objects
    alternate roles as source and target. This avoids data copies and
    provides good performance.

    Neumann boundary conditions are implemented by copying the
    second-to-last row/column to the boundary (numerical approximation
    of zero gradient).
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    # Create Devito 2D grid
    grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
    x_dim, y_dim = grid.dimensions

    # Create two explicit buffers for pseudo-timestepping
    p = Function(name='p', grid=grid, space_order=2)
    pn = Function(name='pn', grid=grid, space_order=2)

    # Get coordinate arrays
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)

    # Create boundary condition profiles
    bc_left_vals = _process_bc(bc_left, y_coords, "left")
    bc_right_vals = _process_bc(bc_right, y_coords, "right")
    bc_bottom_vals = _process_bc(bc_bottom, x_coords, "bottom")
    bc_top_vals = _process_bc(bc_top, x_coords, "top")

    # Create boundary condition functions for prescribed profiles
    if isinstance(bc_right_vals, np.ndarray):
        bc_right_func = Function(name='bc_right', shape=(Ny,), dimensions=(y_dim,))
        bc_right_func.data[:] = bc_right_vals

    if isinstance(bc_left_vals, np.ndarray):
        bc_left_func = Function(name='bc_left', shape=(Ny,), dimensions=(y_dim,))
        bc_left_func.data[:] = bc_left_vals

    if isinstance(bc_bottom_vals, np.ndarray):
        bc_bottom_func = Function(name='bc_bottom', shape=(Nx,), dimensions=(x_dim,))
        bc_bottom_func.data[:] = bc_bottom_vals

    if isinstance(bc_top_vals, np.ndarray):
        bc_top_func = Function(name='bc_top', shape=(Nx,), dimensions=(x_dim,))
        bc_top_func.data[:] = bc_top_vals

    # Create Laplace equation based on pn
    # laplace(pn) = 0, solve for central point
    eqn = Eq(pn.laplace, subdomain=grid.interior)
    stencil = solve(eqn, pn)

    # Create update expression: p gets the stencil from pn
    eq_stencil = Eq(p, stencil)

    # Create boundary condition expressions
    bc_exprs = []

    # Left boundary (x = 0)
    if isinstance(bc_left_vals, str) and bc_left_vals == "neumann":
        # dp/dx = 0: copy second column to first
        bc_exprs.append(Eq(p[0, y_dim], p[1, y_dim]))
    elif isinstance(bc_left_vals, np.ndarray):
        bc_exprs.append(Eq(p[0, y_dim], bc_left_func[y_dim]))
    else:
        bc_exprs.append(Eq(p[0, y_dim], float(bc_left_vals)))

    # Right boundary (x = Lx)
    if isinstance(bc_right_vals, str) and bc_right_vals == "neumann":
        # dp/dx = 0: copy second-to-last column to last
        bc_exprs.append(Eq(p[Nx - 1, y_dim], p[Nx - 2, y_dim]))
    elif isinstance(bc_right_vals, np.ndarray):
        bc_exprs.append(Eq(p[Nx - 1, y_dim], bc_right_func[y_dim]))
    else:
        bc_exprs.append(Eq(p[Nx - 1, y_dim], float(bc_right_vals)))

    # Bottom boundary (y = 0)
    if isinstance(bc_bottom_vals, str) and bc_bottom_vals == "neumann":
        # dp/dy = 0: copy second row to first
        bc_exprs.append(Eq(p[x_dim, 0], p[x_dim, 1]))
    elif isinstance(bc_bottom_vals, np.ndarray):
        bc_exprs.append(Eq(p[x_dim, 0], bc_bottom_func[x_dim]))
    else:
        bc_exprs.append(Eq(p[x_dim, 0], float(bc_bottom_vals)))

    # Top boundary (y = Ly)
    if isinstance(bc_top_vals, str) and bc_top_vals == "neumann":
        # dp/dy = 0: copy second-to-last row to last
        bc_exprs.append(Eq(p[x_dim, Ny - 1], p[x_dim, Ny - 2]))
    elif isinstance(bc_top_vals, np.ndarray):
        bc_exprs.append(Eq(p[x_dim, Ny - 1], bc_top_func[x_dim]))
    else:
        bc_exprs.append(Eq(p[x_dim, Ny - 1], float(bc_top_vals)))

    # Create operator
    op = Operator([eq_stencil] + bc_exprs)

    # Initialize both buffers
    p.data[:] = 0.0
    pn.data[:] = 0.0

    # Apply initial boundary conditions to both buffers
    _apply_initial_bc(p.data, bc_left_vals, bc_right_vals,
                      bc_bottom_vals, bc_top_vals, Nx, Ny)
    _apply_initial_bc(pn.data, bc_left_vals, bc_right_vals,
                      bc_bottom_vals, bc_top_vals, Nx, Ny)

    # Storage for history
    p_history = [] if save_interval is not None else None
    if save_interval is not None:
        p_history.append(p.data[:].copy())

    # Run convergence loop by explicitly flipping buffers
    l1norm = 1.0
    iteration = 0

    while l1norm > tol and iteration < max_iterations:
        # Determine buffer order based on iteration parity
        if iteration % 2 == 0:
            _p = p
            _pn = pn
        else:
            _p = pn
            _pn = p

        # Apply operator
        op(p=_p, pn=_pn)

        # Compute L1 norm for convergence check
        denom = np.sum(np.abs(_pn.data[:]))
        if denom > 1e-15:
            l1norm = np.sum(np.abs(_p.data[:]) - np.abs(_pn.data[:])) / denom
        else:
            l1norm = np.sum(np.abs(_p.data[:]) - np.abs(_pn.data[:]))

        l1norm = abs(l1norm)
        iteration += 1

        # Save history if requested
        if save_interval is not None and iteration % save_interval == 0:
            p_history.append(_p.data[:].copy())

    # Get the final result from the correct buffer
    if iteration % 2 == 1:
        p_final = p.data[:].copy()
    else:
        p_final = pn.data[:].copy()

    converged = l1norm <= tol

    return LaplaceResult(
        p=p_final,
        x=x_coords,
        y=y_coords,
        iterations=iteration,
        final_l1norm=l1norm,
        converged=converged,
        p_history=p_history,
    )


def _process_bc(bc, coords, name):
    """Process boundary condition specification.

    Parameters
    ----------
    bc : float, callable, or 'neumann'
        Boundary condition specification
    coords : np.ndarray
        Coordinate array along the boundary
    name : str
        Name of the boundary for error messages

    Returns
    -------
    float, np.ndarray, or 'neumann'
        Processed boundary condition value(s)
    """
    if isinstance(bc, str):
        if bc.lower() == "neumann":
            return "neumann"
        else:
            raise ValueError(f"Unknown boundary condition type for {name}: {bc}")
    elif callable(bc):
        return bc(coords)
    else:
        return float(bc)


def _apply_initial_bc(data, bc_left, bc_right, bc_bottom, bc_top, Nx, Ny):
    """Apply initial boundary conditions to a data array.

    Parameters
    ----------
    data : np.ndarray
        Data array to modify (shape Nx x Ny)
    bc_left, bc_right, bc_bottom, bc_top : various
        Boundary condition specifications
    Nx, Ny : int
        Grid dimensions
    """
    def _is_neumann(bc):
        return isinstance(bc, str) and bc == "neumann"

    # Left (x = 0)
    if isinstance(bc_left, np.ndarray):
        data[0, :] = bc_left
    elif not _is_neumann(bc_left):
        data[0, :] = float(bc_left)

    # Right (x = Lx)
    if isinstance(bc_right, np.ndarray):
        data[Nx - 1, :] = bc_right
    elif not _is_neumann(bc_right):
        data[Nx - 1, :] = float(bc_right)

    # Bottom (y = 0)
    if isinstance(bc_bottom, np.ndarray):
        data[:, 0] = bc_bottom
    elif not _is_neumann(bc_bottom):
        data[:, 0] = float(bc_bottom)

    # Top (y = Ly)
    if isinstance(bc_top, np.ndarray):
        data[:, Ny - 1] = bc_top
    elif not _is_neumann(bc_top):
        data[:, Ny - 1] = float(bc_top)

    # Handle Neumann BCs by copying adjacent values
    if _is_neumann(bc_left):
        data[0, :] = data[1, :]
    if _is_neumann(bc_right):
        data[Nx - 1, :] = data[Nx - 2, :]
    if _is_neumann(bc_bottom):
        data[:, 0] = data[:, 1]
    if _is_neumann(bc_top):
        data[:, Ny - 1] = data[:, Ny - 2]


def solve_laplace_2d_with_copy(
    Lx: float = 2.0,
    Ly: float = 1.0,
    Nx: int = 31,
    Ny: int = 31,
    bc_left: float | Callable[[np.ndarray], np.ndarray] | str = 0.0,
    bc_right: float | Callable[[np.ndarray], np.ndarray] | str = "neumann",
    bc_bottom: float | Callable[[np.ndarray], np.ndarray] | str = "neumann",
    bc_top: float | Callable[[np.ndarray], np.ndarray] | str = "neumann",
    tol: float = 1e-4,
    max_iterations: int = 10000,
) -> LaplaceResult:
    """Solve 2D Laplace equation using data copies (for comparison).

    This is the straightforward implementation that copies data between
    buffers on each iteration. The buffer-swapping version
    (solve_laplace_2d) is more efficient for large grids.

    Parameters are identical to solve_laplace_2d.
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    # Create Devito 2D grid
    grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
    x_dim, y_dim = grid.dimensions

    # Create two explicit buffers for pseudo-timestepping
    p = Function(name='p', grid=grid, space_order=2)
    pn = Function(name='pn', grid=grid, space_order=2)

    # Get coordinate arrays
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)

    # Create boundary condition profiles
    bc_left_vals = _process_bc(bc_left, y_coords, "left")
    bc_right_vals = _process_bc(bc_right, y_coords, "right")
    bc_bottom_vals = _process_bc(bc_bottom, x_coords, "bottom")
    bc_top_vals = _process_bc(bc_top, x_coords, "top")

    # Create boundary condition functions for prescribed profiles
    if isinstance(bc_right_vals, np.ndarray):
        bc_right_func = Function(name='bc_right', shape=(Ny,), dimensions=(y_dim,))
        bc_right_func.data[:] = bc_right_vals

    if isinstance(bc_left_vals, np.ndarray):
        bc_left_func = Function(name='bc_left', shape=(Ny,), dimensions=(y_dim,))
        bc_left_func.data[:] = bc_left_vals

    if isinstance(bc_bottom_vals, np.ndarray):
        bc_bottom_func = Function(name='bc_bottom', shape=(Nx,), dimensions=(x_dim,))
        bc_bottom_func.data[:] = bc_bottom_vals

    if isinstance(bc_top_vals, np.ndarray):
        bc_top_func = Function(name='bc_top', shape=(Nx,), dimensions=(x_dim,))
        bc_top_func.data[:] = bc_top_vals

    # Create Laplace equation based on pn
    eqn = Eq(pn.laplace, subdomain=grid.interior)
    stencil = solve(eqn, pn)
    eq_stencil = Eq(p, stencil)

    # Create boundary condition expressions
    bc_exprs = []

    # Left boundary
    if isinstance(bc_left_vals, str) and bc_left_vals == "neumann":
        bc_exprs.append(Eq(p[0, y_dim], p[1, y_dim]))
    elif isinstance(bc_left_vals, np.ndarray):
        bc_exprs.append(Eq(p[0, y_dim], bc_left_func[y_dim]))
    else:
        bc_exprs.append(Eq(p[0, y_dim], float(bc_left_vals)))

    # Right boundary
    if isinstance(bc_right_vals, str) and bc_right_vals == "neumann":
        bc_exprs.append(Eq(p[Nx - 1, y_dim], p[Nx - 2, y_dim]))
    elif isinstance(bc_right_vals, np.ndarray):
        bc_exprs.append(Eq(p[Nx - 1, y_dim], bc_right_func[y_dim]))
    else:
        bc_exprs.append(Eq(p[Nx - 1, y_dim], float(bc_right_vals)))

    # Bottom boundary
    if isinstance(bc_bottom_vals, str) and bc_bottom_vals == "neumann":
        bc_exprs.append(Eq(p[x_dim, 0], p[x_dim, 1]))
    elif isinstance(bc_bottom_vals, np.ndarray):
        bc_exprs.append(Eq(p[x_dim, 0], bc_bottom_func[x_dim]))
    else:
        bc_exprs.append(Eq(p[x_dim, 0], float(bc_bottom_vals)))

    # Top boundary
    if isinstance(bc_top_vals, str) and bc_top_vals == "neumann":
        bc_exprs.append(Eq(p[x_dim, Ny - 1], p[x_dim, Ny - 2]))
    elif isinstance(bc_top_vals, np.ndarray):
        bc_exprs.append(Eq(p[x_dim, Ny - 1], bc_top_func[x_dim]))
    else:
        bc_exprs.append(Eq(p[x_dim, Ny - 1], float(bc_top_vals)))

    # Create operator
    op = Operator([eq_stencil] + bc_exprs)

    # Initialize both buffers
    p.data[:] = 0.0
    pn.data[:] = 0.0

    # Apply initial boundary conditions
    _apply_initial_bc(p.data, bc_left_vals, bc_right_vals,
                      bc_bottom_vals, bc_top_vals, Nx, Ny)
    _apply_initial_bc(pn.data, bc_left_vals, bc_right_vals,
                      bc_bottom_vals, bc_top_vals, Nx, Ny)

    # Run convergence loop with deep data copies
    l1norm = 1.0
    iteration = 0

    while l1norm > tol and iteration < max_iterations:
        # Deep copy (this is what we want to avoid in production)
        pn.data[:] = p.data[:]

        # Apply operator
        op(p=p, pn=pn)

        # Compute L1 norm
        denom = np.sum(np.abs(pn.data[:]))
        if denom > 1e-15:
            l1norm = np.sum(np.abs(p.data[:]) - np.abs(pn.data[:])) / denom
        else:
            l1norm = np.sum(np.abs(p.data[:]) - np.abs(pn.data[:]))

        l1norm = abs(l1norm)
        iteration += 1

    converged = l1norm <= tol

    return LaplaceResult(
        p=p.data[:].copy(),
        x=x_coords,
        y=y_coords,
        iterations=iteration,
        final_l1norm=l1norm,
        converged=converged,
    )


def exact_laplace_linear(
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float = 2.0,
    Ly: float = 1.0,
) -> np.ndarray:
    """Exact solution for Laplace equation with linear boundary conditions.

    For the boundary conditions:
        p = 0 at x = 0
        p = y at x = Lx
        dp/dy = 0 at y = 0 and y = Ly

    The exact solution is p(x, y) = x * y / Lx

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
        Exact solution at (x, y)
    """
    return X * Y / Lx


def convergence_test_laplace_2d(
    grid_sizes: list | None = None,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run convergence test for 2D Laplace solver.

    Uses the linear solution test case for error computation.

    Parameters
    ----------
    grid_sizes : list, optional
        List of N values to test (same for Nx and Ny).
        Default: [11, 21, 41, 81]
    tol : float
        Convergence tolerance for the solver

    Returns
    -------
    tuple
        (grid_sizes, errors, observed_order)
    """
    if grid_sizes is None:
        grid_sizes = [11, 21, 41, 81]

    errors = []
    Lx = 2.0
    Ly = 1.0

    for N in grid_sizes:
        result = solve_laplace_2d(
            Lx=Lx, Ly=Ly,
            Nx=N, Ny=N,
            bc_left=0.0,
            bc_right=lambda y: y,
            bc_bottom="neumann",
            bc_top="neumann",
            tol=tol,
        )

        # Create meshgrid for exact solution
        X, Y = np.meshgrid(result.x, result.y, indexing='ij')

        # Exact solution
        p_exact = exact_laplace_linear(X, Y, Lx, Ly)

        # L2 error
        error = np.sqrt(np.mean((result.p - p_exact) ** 2))
        errors.append(error)

    errors = np.array(errors)
    grid_sizes = np.array(grid_sizes)

    # Compute observed order
    log_h = np.log(1.0 / grid_sizes)
    log_err = np.log(errors + 1e-15)  # Avoid log(0)
    observed_order = np.polyfit(log_h, log_err, 1)[0]

    return grid_sizes, errors, observed_order
