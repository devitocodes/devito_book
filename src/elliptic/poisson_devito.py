"""2D Poisson Equation Solver using Devito DSL.

Solves the Poisson equation with source term:
    laplace(p) = p_xx + p_yy = b

on domain [0, Lx] x [0, Ly] with:
    - Dirichlet boundary conditions (default: p = 0 on all boundaries)
    - Source term b(x, y)

The discretization uses central differences:
    p_{i,j} = (dy^2*(p_{i+1,j} + p_{i-1,j}) + dx^2*(p_{i,j+1} + p_{i,j-1})
              - b_{i,j}*dx^2*dy^2) / (2*(dx^2 + dy^2))

Two solver approaches are provided:
1. Dual-buffer (manual loop): Uses two Function objects with explicit
   buffer swapping and Python convergence loop. Good for understanding
   the algorithm and adding custom convergence criteria.

2. TimeFunction (internal loop): Uses Devito's TimeFunction with
   internal time stepping. More efficient for many iterations.

Usage:
    from src.elliptic import solve_poisson_2d

    # Define source term with point sources
    result = solve_poisson_2d(
        Lx=2.0, Ly=1.0,
        Nx=50, Ny=50,
        source_points=[(0.5, 0.25, 100), (1.5, 0.75, -100)],
        n_iterations=100,
    )
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    from devito import Eq, Function, Grid, Operator, TimeFunction, solve
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@dataclass
class PoissonResult:
    """Results from the 2D Poisson equation solver.

    Attributes
    ----------
    p : np.ndarray
        Solution at final iteration, shape (Nx, Ny)
    x : np.ndarray
        x-coordinate grid points
    y : np.ndarray
        y-coordinate grid points
    b : np.ndarray
        Source term used
    iterations : int
        Number of iterations performed
    p_history : list, optional
        Solution history at specified intervals
    """
    p: np.ndarray
    x: np.ndarray
    y: np.ndarray
    b: np.ndarray
    iterations: int
    p_history: list | None = None


def solve_poisson_2d(
    Lx: float = 2.0,
    Ly: float = 1.0,
    Nx: int = 50,
    Ny: int = 50,
    b: Callable[[np.ndarray, np.ndarray], np.ndarray] | np.ndarray | None = None,
    source_points: list[tuple[float, float, float]] | None = None,
    n_iterations: int = 100,
    bc_value: float = 0.0,
    save_interval: int | None = None,
) -> PoissonResult:
    """Solve the 2D Poisson equation using Devito (dual-buffer approach).

    Solves: laplace(p) = p_xx + p_yy = b
    with p = bc_value on all boundaries (Dirichlet).

    Uses a dual-buffer approach with two Function objects and explicit
    buffer swapping for efficiency. The Python loop allows custom
    convergence criteria if needed.

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
    b : callable, np.ndarray, or None
        Source term specification:
        - callable: b(X, Y) where X, Y are meshgrid arrays
        - np.ndarray: explicit source array of shape (Nx, Ny)
        - None: use source_points or default to zero
    source_points : list of tuples, optional
        List of (x, y, value) tuples for point sources.
        Each tuple places a source of given value at (x, y).
    n_iterations : int
        Number of pseudo-timestep iterations
    bc_value : float
        Dirichlet boundary condition value (same on all boundaries)
    save_interval : int, optional
        If specified, save solution every save_interval iterations

    Returns
    -------
    PoissonResult
        Solution data including final solution, grids, and source term

    Raises
    ------
    ImportError
        If Devito is not installed

    Notes
    -----
    The dual-buffer approach alternates between two Function objects
    to avoid data copies. On even iterations, pn -> p; on odd
    iterations, p -> pn. The operator is called with swapped arguments.

    This is more efficient than copying data on each iteration,
    especially for large grids.
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
    pd = Function(name='pd', grid=grid, space_order=2)

    # Initialize source term function
    b_func = Function(name='b', grid=grid)

    # Get coordinate arrays
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Set source term
    b_func.data[:] = 0.0

    if b is not None:
        if callable(b):
            b_func.data[:] = b(X, Y)
        elif isinstance(b, np.ndarray):
            if b.shape != (Nx, Ny):
                raise ValueError(
                    f"Source array shape {b.shape} does not match grid ({Nx}, {Ny})"
                )
            b_func.data[:] = b
    elif source_points is not None:
        # Add point sources
        for x_src, y_src, value in source_points:
            # Find nearest grid indices
            i = int(round(x_src * (Nx - 1) / Lx))
            j = int(round(y_src * (Ny - 1) / Ly))
            i = max(0, min(Nx - 1, i))
            j = max(0, min(Ny - 1, j))
            b_func.data[i, j] = value

    # Create Poisson equation based on pd: laplace(pd) = b
    eq = Eq(pd.laplace, b_func, subdomain=grid.interior)
    stencil = solve(eq, pd)

    # Create update expression: p gets the stencil from pd
    eq_stencil = Eq(p, stencil)

    # Boundary condition expressions (Dirichlet: p = bc_value)
    bc_exprs = [
        Eq(p[x_dim, 0], bc_value),           # Bottom (y = 0)
        Eq(p[x_dim, Ny - 1], bc_value),      # Top (y = Ly)
        Eq(p[0, y_dim], bc_value),           # Left (x = 0)
        Eq(p[Nx - 1, y_dim], bc_value),      # Right (x = Lx)
    ]

    # Create operator
    op = Operator([eq_stencil] + bc_exprs)

    # Initialize buffers
    p.data[:] = 0.0
    pd.data[:] = 0.0

    # Storage for history
    p_history = [] if save_interval is not None else None
    if save_interval is not None:
        p_history.append(p.data[:].copy())

    # Run the outer loop with buffer swapping
    for i in range(n_iterations):
        # Determine buffer order based on iteration parity
        if i % 2 == 0:
            _p = p
            _pd = pd
        else:
            _p = pd
            _pd = p

        # Apply operator
        op(p=_p, pd=_pd)

        # Save history if requested
        if save_interval is not None and (i + 1) % save_interval == 0:
            p_history.append(_p.data[:].copy())

    # Get the final result from the correct buffer
    if n_iterations % 2 == 1:
        p_final = p.data[:].copy()
    else:
        p_final = pd.data[:].copy()

    return PoissonResult(
        p=p_final,
        x=x_coords,
        y=y_coords,
        b=b_func.data[:].copy(),
        iterations=n_iterations,
        p_history=p_history,
    )


def solve_poisson_2d_timefunction(
    Lx: float = 2.0,
    Ly: float = 1.0,
    Nx: int = 50,
    Ny: int = 50,
    b: Callable[[np.ndarray, np.ndarray], np.ndarray] | np.ndarray | None = None,
    source_points: list[tuple[float, float, float]] | None = None,
    n_iterations: int = 100,
    bc_value: float = 0.0,
) -> PoissonResult:
    """Solve 2D Poisson equation using TimeFunction (internal loop).

    This version uses Devito's TimeFunction to internalize the
    pseudo-timestepping loop, which is more efficient for large
    numbers of iterations.

    Parameters are identical to solve_poisson_2d.

    Notes
    -----
    The TimeFunction approach lets Devito handle buffer management
    internally. This results in a compiled kernel with an internal
    time loop, avoiding Python overhead for each iteration.

    The tradeoff is less flexibility for custom convergence criteria
    during iteration.
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    # Create Devito 2D grid
    grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
    x_dim, y_dim = grid.dimensions
    t_dim = grid.stepping_dim

    # Create TimeFunction for implicit buffer management
    p = TimeFunction(name='p', grid=grid, space_order=2)

    # Initialize source term function
    b_func = Function(name='b', grid=grid)

    # Get coordinate arrays
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Set source term
    b_func.data[:] = 0.0

    if b is not None:
        if callable(b):
            b_func.data[:] = b(X, Y)
        elif isinstance(b, np.ndarray):
            if b.shape != (Nx, Ny):
                raise ValueError(
                    f"Source array shape {b.shape} does not match grid ({Nx}, {Ny})"
                )
            b_func.data[:] = b
    elif source_points is not None:
        # Add point sources
        for x_src, y_src, value in source_points:
            # Find nearest grid indices
            i = int(round(x_src * (Nx - 1) / Lx))
            j = int(round(y_src * (Ny - 1) / Ly))
            i = max(0, min(Nx - 1, i))
            j = max(0, min(Ny - 1, j))
            b_func.data[i, j] = value

    # Create Poisson equation: laplace(p) = b
    # Let SymPy solve for the central stencil point
    eq = Eq(p.laplace, b_func)
    stencil = solve(eq, p)

    # Create update to populate p.forward
    eq_stencil = Eq(p.forward, stencil)

    # Boundary condition expressions
    # Note: with TimeFunction we need explicit time index t + 1
    bc_exprs = [
        Eq(p[t_dim + 1, x_dim, 0], bc_value),           # Bottom
        Eq(p[t_dim + 1, x_dim, Ny - 1], bc_value),      # Top
        Eq(p[t_dim + 1, 0, y_dim], bc_value),           # Left
        Eq(p[t_dim + 1, Nx - 1, y_dim], bc_value),      # Right
    ]

    # Create operator
    op = Operator([eq_stencil] + bc_exprs)

    # Initialize
    p.data[:] = 0.0

    # Execute operator with internal time loop
    op(time=n_iterations)

    # Get final solution (from buffer 0 due to modular indexing)
    p_final = p.data[0, :, :].copy()

    return PoissonResult(
        p=p_final,
        x=x_coords,
        y=y_coords,
        b=b_func.data[:].copy(),
        iterations=n_iterations,
    )


def solve_poisson_2d_with_copy(
    Lx: float = 2.0,
    Ly: float = 1.0,
    Nx: int = 50,
    Ny: int = 50,
    b: Callable[[np.ndarray, np.ndarray], np.ndarray] | np.ndarray | None = None,
    source_points: list[tuple[float, float, float]] | None = None,
    n_iterations: int = 100,
    bc_value: float = 0.0,
) -> PoissonResult:
    """Solve 2D Poisson equation using data copies (for comparison).

    This is the straightforward implementation that copies data between
    buffers on each iteration. The buffer-swapping version
    (solve_poisson_2d) is more efficient for large grids.

    Parameters are identical to solve_poisson_2d.

    Notes
    -----
    This function is provided for educational purposes to demonstrate
    the performance difference between copying data and swapping buffers.
    For production use, prefer solve_poisson_2d or
    solve_poisson_2d_timefunction.
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    # Create Devito 2D grid
    grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
    x_dim, y_dim = grid.dimensions

    # Create two explicit buffers
    p = Function(name='p', grid=grid, space_order=2)
    pd = Function(name='pd', grid=grid, space_order=2)

    # Initialize source term function
    b_func = Function(name='b', grid=grid)

    # Get coordinate arrays
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Set source term
    b_func.data[:] = 0.0

    if b is not None:
        if callable(b):
            b_func.data[:] = b(X, Y)
        elif isinstance(b, np.ndarray):
            b_func.data[:] = b
    elif source_points is not None:
        for x_src, y_src, value in source_points:
            i = int(round(x_src * (Nx - 1) / Lx))
            j = int(round(y_src * (Ny - 1) / Ly))
            i = max(0, min(Nx - 1, i))
            j = max(0, min(Ny - 1, j))
            b_func.data[i, j] = value

    # Create Poisson equation
    eq = Eq(pd.laplace, b_func, subdomain=grid.interior)
    stencil = solve(eq, pd)
    eq_stencil = Eq(p, stencil)

    # Boundary conditions
    bc_exprs = [
        Eq(p[x_dim, 0], bc_value),
        Eq(p[x_dim, Ny - 1], bc_value),
        Eq(p[0, y_dim], bc_value),
        Eq(p[Nx - 1, y_dim], bc_value),
    ]

    # Create operator
    op = Operator([eq_stencil] + bc_exprs)

    # Initialize
    p.data[:] = 0.0
    pd.data[:] = 0.0

    # Run with data copies (less efficient)
    for _ in range(n_iterations):
        pd.data[:] = p.data[:]  # Deep copy
        op(p=p, pd=pd)

    return PoissonResult(
        p=p.data[:].copy(),
        x=x_coords,
        y=y_coords,
        b=b_func.data[:].copy(),
        iterations=n_iterations,
    )


def create_point_source(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    x_src: float,
    y_src: float,
    value: float,
) -> np.ndarray:
    """Create a point source array for the Poisson equation.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions
    Lx, Ly : float
        Domain extents
    x_src, y_src : float
        Source location
    value : float
        Source strength

    Returns
    -------
    np.ndarray
        Source array with single point source
    """
    b = np.zeros((Nx, Ny))
    i = int(round(x_src * (Nx - 1) / Lx))
    j = int(round(y_src * (Ny - 1) / Ly))
    i = max(0, min(Nx - 1, i))
    j = max(0, min(Ny - 1, j))
    b[i, j] = value
    return b


def create_gaussian_source(
    X: np.ndarray,
    Y: np.ndarray,
    x0: float,
    y0: float,
    sigma: float = 0.1,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Create a Gaussian source term for the Poisson equation.

    Parameters
    ----------
    X, Y : np.ndarray
        Meshgrid coordinate arrays
    x0, y0 : float
        Center of the Gaussian
    sigma : float
        Width of the Gaussian
    amplitude : float
        Peak amplitude

    Returns
    -------
    np.ndarray
        Gaussian source distribution
    """
    r2 = (X - x0)**2 + (Y - y0)**2
    return amplitude * np.exp(-r2 / (2 * sigma**2))


def exact_poisson_point_source(
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float,
    Ly: float,
    x_src: float,
    y_src: float,
    strength: float,
    n_terms: int = 20,
) -> np.ndarray:
    """Analytical solution for Poisson equation with point source.

    Uses Fourier series solution for a point source in a rectangular
    domain with homogeneous Dirichlet boundary conditions.

    The solution is:
        p(x, y) = sum_{m,n} A_{mn} * sin(m*pi*x/Lx) * sin(n*pi*y/Ly)

    where the coefficients A_{mn} are determined by the point source.

    Parameters
    ----------
    X, Y : np.ndarray
        Meshgrid coordinate arrays
    Lx, Ly : float
        Domain dimensions
    x_src, y_src : float
        Source location
    strength : float
        Source strength
    n_terms : int
        Number of terms in Fourier series

    Returns
    -------
    np.ndarray
        Analytical solution
    """
    p = np.zeros_like(X)

    for m in range(1, n_terms + 1):
        for n in range(1, n_terms + 1):
            # Eigenvalue
            lambda_mn = (m * np.pi / Lx)**2 + (n * np.pi / Ly)**2

            # Source coefficient
            f_mn = (4 / (Lx * Ly)) * strength * \
                   np.sin(m * np.pi * x_src / Lx) * \
                   np.sin(n * np.pi * y_src / Ly)

            # Solution coefficient
            A_mn = f_mn / lambda_mn

            # Add term
            p += A_mn * np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)

    return p


def convergence_test_poisson_2d(
    grid_sizes: list | None = None,
    n_iterations: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Run convergence test for 2D Poisson solver.

    Uses a manufactured solution to test convergence.

    Parameters
    ----------
    grid_sizes : list, optional
        List of N values to test (same for Nx and Ny).
        Default: [20, 40, 80]
    n_iterations : int
        Number of iterations for each grid size

    Returns
    -------
    tuple
        (grid_sizes, errors)

    Notes
    -----
    Uses manufactured solution:
        p_exact(x, y) = sin(pi*x) * sin(pi*y)
    which satisfies:
        laplace(p) = -2*pi^2 * sin(pi*x) * sin(pi*y)
    with p = 0 on all boundaries of [0, 1] x [0, 1].
    """
    if grid_sizes is None:
        grid_sizes = [20, 40, 80]

    errors = []
    Lx = Ly = 1.0

    # Source term for manufactured solution
    def b_mms(X, Y):
        return -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    for N in grid_sizes:
        result = solve_poisson_2d(
            Lx=Lx, Ly=Ly,
            Nx=N, Ny=N,
            b=b_mms,
            n_iterations=n_iterations,
            bc_value=0.0,
        )

        # Create meshgrid for exact solution
        X, Y = np.meshgrid(result.x, result.y, indexing='ij')

        # Exact solution
        p_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

        # L2 error
        error = np.sqrt(np.mean((result.p - p_exact) ** 2))
        errors.append(error)

    return np.array(grid_sizes), np.array(errors)
