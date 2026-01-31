"""Darcy flow solvers using Devito DSL.

This module provides solvers for porous media flow governed by
Darcy's law and the continuity equation.

Darcy's law:
    q = -K/mu * grad(p)

where:
    q = Darcy velocity (volumetric flux)
    K = permeability
    mu = dynamic viscosity
    p = pressure

Combined with mass conservation (div(q) = f) gives:
    -div(K * grad(p)) = f

The module provides:
1. Steady-state Darcy flow solver (iterative)
2. Transient single-phase flow solver
3. Heterogeneous permeability field generators
4. Velocity computation from pressure
5. Verification utilities
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.fft as fft

try:
    from devito import Eq, Function, Grid, Operator, TimeFunction, solve

    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass
class DarcyResult:
    """Results from Darcy flow solver.

    Attributes
    ----------
    p : np.ndarray
        Pressure field, shape (Nx, Ny)
    x : np.ndarray
        x-coordinate grid points
    y : np.ndarray
        y-coordinate grid points
    qx : np.ndarray, optional
        x-component of Darcy velocity
    qy : np.ndarray, optional
        y-component of Darcy velocity
    K : np.ndarray or float
        Permeability field or constant
    iterations : int
        Number of iterations to convergence (steady-state)
        or time steps (transient)
    converged : bool
        Whether solver converged (steady-state only)
    final_l1norm : float, optional
        Final L1 norm at convergence
    p_history : list, optional
        Pressure history for transient problems
    """

    p: np.ndarray
    x: np.ndarray
    y: np.ndarray
    qx: np.ndarray | None = None
    qy: np.ndarray | None = None
    K: np.ndarray | float = 1.0
    iterations: int = 0
    converged: bool = True
    final_l1norm: float | None = None
    p_history: list | None = None


# =============================================================================
# Permeability Field Generators
# =============================================================================


class GaussianRandomField:
    """Generate Gaussian random fields for permeability modeling.

    The covariance structure follows a Matern-like spectrum with
    parameters controlling correlation length and smoothness.

    Parameters
    ----------
    size : int
        Grid size (size x size)
    alpha : float
        Smoothness parameter (higher = smoother fields)
    tau : float
        Inverse correlation length (higher = shorter correlation)
    sigma : float, optional
        Amplitude (computed from alpha, tau if not provided)

    Examples
    --------
    >>> grf = GaussianRandomField(64, alpha=2, tau=3)
    >>> fields = grf.sample(5)  # Generate 5 random fields
    >>> print(fields.shape)  # (5, 64, 64)
    """

    def __init__(
        self, size: int, alpha: float = 2.0, tau: float = 3.0, sigma: float | None = None
    ):
        self.size = size
        self.dim = 2

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))

        k_max = size // 2
        wavenumbers = np.concatenate([np.arange(0, k_max), np.arange(-k_max, 0)])
        wavenumbers = np.tile(wavenumbers, (size, 1))

        k_x = wavenumbers.T
        k_y = wavenumbers

        # Spectral density (Matern-like covariance)
        self.sqrt_eig = (
            size**2
            * math.sqrt(2.0)
            * sigma
            * ((4 * math.pi**2 * (k_x**2 + k_y**2) + tau**2) ** (-alpha / 2.0))
        )
        self.sqrt_eig[0, 0] = 0.0  # Zero mean

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate n_samples random fields.

        Parameters
        ----------
        n_samples : int
            Number of fields to generate

        Returns
        -------
        np.ndarray
            Random fields of shape (n_samples, size, size)
        """
        coeff = np.random.randn(n_samples, self.size, self.size)
        coeff = self.sqrt_eig * coeff
        return fft.ifftn(coeff, axes=(1, 2)).real


def create_layered_permeability(
    nx: int,
    ny: int,
    layers: list[tuple[float, float]],
) -> np.ndarray:
    """Create a layered permeability field.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    layers : list of tuples
        Each tuple is (y_fraction, K_value) specifying the layer
        boundary as a fraction of domain height and its permeability.
        Layers are applied from bottom (y=0) upward.

    Returns
    -------
    K : np.ndarray
        Permeability field, shape (nx, ny)

    Examples
    --------
    >>> # Three-layer system
    >>> layers = [
    ...     (0.33, 1e-12),   # Bottom: low permeability
    ...     (0.67, 1e-10),   # Middle: high permeability
    ...     (1.0,  1e-13),   # Top: medium-low permeability
    ... ]
    >>> K = create_layered_permeability(64, 64, layers)
    """
    K = np.zeros((nx, ny))

    # Sort layers by y_fraction
    layers = sorted(layers, key=lambda x: x[0])

    for j in range(ny):
        y_frac = j / (ny - 1) if ny > 1 else 0.0
        # Find which layer this y-coordinate belongs to
        K_val = layers[-1][1]  # Default to top layer
        for y_bound, K_layer in layers:
            if y_frac < y_bound:
                K_val = K_layer
                break
        K[:, j] = K_val

    return K


def create_binary_permeability(
    nx: int,
    ny: int,
    K_low: float = 4.0,
    K_high: float = 12.0,
    seed: int | None = None,
    alpha: float = 2.0,
    tau: float = 3.0,
) -> np.ndarray:
    """Create a binary permeability field using threshold method.

    Generates a Gaussian random field and thresholds it to create
    a binary distribution of permeability values, representing
    channelized flow paths in low-permeability matrix.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    K_low, K_high : float
        Permeability values for low and high regions
    seed : int, optional
        Random seed for reproducibility
    alpha : float
        Smoothness parameter for random field
    tau : float
        Inverse correlation length

    Returns
    -------
    K : np.ndarray
        Binary permeability field
    """
    if seed is not None:
        np.random.seed(seed)

    size = max(nx, ny)
    grf = GaussianRandomField(size, alpha=alpha, tau=tau)
    field = grf.sample(1)[0, :nx, :ny]

    # Apply threshold at zero
    K = np.where(field >= 0, K_high, K_low)

    return K


def create_lognormal_permeability(
    nx: int,
    ny: int,
    K_ref: float = 1.0,
    sigma_logK: float = 1.0,
    seed: int | None = None,
    alpha: float = 2.5,
    tau: float = 4.0,
) -> np.ndarray:
    """Create a log-normal permeability field.

    Generates permeability following K = K_ref * exp(sigma * Z)
    where Z is a zero-mean, unit-variance Gaussian random field.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    K_ref : float
        Reference (geometric mean) permeability
    sigma_logK : float
        Standard deviation of log(K)
    seed : int, optional
        Random seed for reproducibility
    alpha : float
        Smoothness parameter
    tau : float
        Inverse correlation length

    Returns
    -------
    K : np.ndarray
        Log-normal permeability field
    """
    if seed is not None:
        np.random.seed(seed)

    size = max(nx, ny)
    grf = GaussianRandomField(size, alpha=alpha, tau=tau)
    Z = grf.sample(1)[0, :nx, :ny]

    # Normalize to unit variance
    std = np.std(Z)
    if std > 1e-10:
        Z = Z / std

    K = K_ref * np.exp(sigma_logK * Z)
    return K


def add_fracture_to_permeability(
    K: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    K_fracture: float,
    width: int = 1,
) -> np.ndarray:
    """Add a line fracture to permeability field.

    Uses Bresenham's line algorithm to identify cells along
    the fracture path.

    Parameters
    ----------
    K : np.ndarray
        Permeability field to modify (modified in-place)
    x0, y0, x1, y1 : int
        Fracture endpoints (grid indices)
    K_fracture : float
        Fracture permeability
    width : int
        Fracture width in grid cells

    Returns
    -------
    K : np.ndarray
        Modified permeability field
    """
    nx, ny = K.shape

    # Bresenham's line algorithm
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        # Set permeability in fracture cell and neighbors
        half_width = width // 2
        for di in range(-half_width, half_width + 1):
            for dj in range(-half_width, half_width + 1):
                xi, yj = x + di, y + dj
                if 0 <= xi < nx and 0 <= yj < ny:
                    K[xi, yj] = K_fracture

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return K


def add_well(
    source: np.ndarray,
    i_well: int,
    j_well: int,
    rate: float,
    well_radius: int = 1,
) -> np.ndarray:
    """Add a well to source term array.

    Parameters
    ----------
    source : np.ndarray
        Source array to modify (modified in-place)
    i_well, j_well : int
        Well location (grid indices)
    rate : float
        Injection rate (positive) or production rate (negative)
    well_radius : int
        Radius of well in grid cells for distribution

    Returns
    -------
    source : np.ndarray
        Modified source array
    """
    nx, ny = source.shape

    # Count cells in well footprint
    cells = 0
    for di in range(-well_radius, well_radius + 1):
        for dj in range(-well_radius, well_radius + 1):
            if di * di + dj * dj <= well_radius * well_radius:
                i, j = i_well + di, j_well + dj
                if 0 <= i < nx and 0 <= j < ny:
                    cells += 1

    rate_per_cell = rate / max(cells, 1)

    # Distribute rate
    for di in range(-well_radius, well_radius + 1):
        for dj in range(-well_radius, well_radius + 1):
            if di * di + dj * dj <= well_radius * well_radius:
                i, j = i_well + di, j_well + dj
                if 0 <= i < nx and 0 <= j < ny:
                    source[i, j] += rate_per_cell

    return source


# =============================================================================
# Velocity Computation
# =============================================================================


def compute_darcy_velocity(
    p: np.ndarray,
    K: np.ndarray | float,
    dx: float,
    dy: float,
    mu: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Darcy velocity from pressure field.

    Implements q = -K/mu * grad(p) using central differences.

    Parameters
    ----------
    p : np.ndarray
        Pressure field, shape (Nx, Ny)
    K : np.ndarray or float
        Permeability field or constant
    dx, dy : float
        Grid spacing
    mu : float
        Dynamic viscosity

    Returns
    -------
    qx, qy : np.ndarray
        Velocity components at cell centers
    """
    nx, ny = p.shape

    # Compute pressure gradients using central differences
    dp_dx = np.zeros_like(p)
    dp_dy = np.zeros_like(p)

    # Central differences for interior
    dp_dx[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * dx)
    dp_dy[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dy)

    # One-sided differences at boundaries
    dp_dx[0, :] = (p[1, :] - p[0, :]) / dx
    dp_dx[-1, :] = (p[-1, :] - p[-2, :]) / dx
    dp_dy[:, 0] = (p[:, 1] - p[:, 0]) / dy
    dp_dy[:, -1] = (p[:, -1] - p[:, -2]) / dy

    # Darcy velocity: q = -K/mu * grad(p)
    if np.isscalar(K):
        qx = -K / mu * dp_dx
        qy = -K / mu * dp_dy
    else:
        qx = -K / mu * dp_dx
        qy = -K / mu * dp_dy

    return qx, qy


# =============================================================================
# Steady-State Solver
# =============================================================================


def solve_darcy_2d(
    Lx: float = 1.0,
    Ly: float = 1.0,
    Nx: int = 64,
    Ny: int = 64,
    permeability: np.ndarray | float = 1.0,
    source: np.ndarray | float | None = None,
    bc_left: float | str = 0.0,
    bc_right: float | str = 1.0,
    bc_bottom: float | str = "neumann",
    bc_top: float | str = "neumann",
    tol: float = 1e-4,
    max_iterations: int = 10000,
    omega: float = 1.0,
    compute_velocity: bool = True,
) -> DarcyResult:
    """Solve steady-state 2D Darcy flow equation.

    Solves: -div(K * grad(p)) = f

    using an iterative (Jacobi/SOR) method with the dual-buffer pattern.

    Parameters
    ----------
    Lx, Ly : float
        Domain extent [0, Lx] x [0, Ly]
    Nx, Ny : int
        Number of grid points in each direction
    permeability : np.ndarray or float
        Permeability field K(x,y), shape (Nx, Ny), or constant value
    source : np.ndarray, float, or None
        Source term f(x,y), shape (Nx, Ny), constant, or None (zero)
    bc_left, bc_right : float or 'neumann'
        Boundary conditions at x=0 and x=Lx
    bc_bottom, bc_top : float or 'neumann'
        Boundary conditions at y=0 and y=Ly
    tol : float
        Convergence tolerance for L1 norm
    max_iterations : int
        Maximum number of iterations
    omega : float
        Relaxation parameter (1.0 = Jacobi, >1 = SOR)
    compute_velocity : bool
        Whether to compute velocity field after solving

    Returns
    -------
    DarcyResult
        Solution containing pressure, coordinates, and optional velocity

    Raises
    ------
    ImportError
        If Devito is not installed

    Notes
    -----
    The solver uses a dual-buffer approach where two Function objects
    alternate roles as source and target, avoiding data copies during
    iteration.

    For variable permeability, the equation is discretized using the
    product rule:
        div(K * grad(p)) = K * laplacian(p) + grad(K) . grad(p)
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required. Install with: pip install devito")

    # Create grid
    grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
    x, y = grid.dimensions

    # Create solution buffers (dual-buffer pattern)
    p = Function(name="p", grid=grid, space_order=2)
    pn = Function(name="pn", grid=grid, space_order=2)

    # Permeability field
    K = Function(name="K", grid=grid, space_order=2)
    if np.isscalar(permeability):
        K.data[:] = permeability
        K_array = permeability
    else:
        K.data[:] = permeability
        K_array = permeability

    # Source term
    f = Function(name="f", grid=grid)
    if source is None:
        f.data[:] = 0.0
    elif np.isscalar(source):
        f.data[:] = source
    else:
        f.data[:] = source

    # The Darcy equation: -div(K * grad(p)) = f
    # Expanded using product rule: -(K * laplacian(p) + grad(K) . grad(p)) = f
    # Rearranged: K * laplacian(pn) + grad(K) . grad(pn) = -f

    laplacian_term = K * pn.laplace
    gradient_coupling = K.dx * pn.dx + K.dy * pn.dy

    eqn = Eq(laplacian_term + gradient_coupling, -f, subdomain=grid.interior)
    stencil = solve(eqn, pn)

    # Apply relaxation if omega != 1
    if omega != 1.0:
        update_expr = (1 - omega) * pn + omega * stencil
    else:
        update_expr = stencil

    eq_update = Eq(p, update_expr)

    # Build boundary condition equations
    bc_exprs = []

    # Left boundary (x = 0)
    if bc_left == "neumann":
        bc_exprs.append(Eq(p[0, y], p[1, y]))
    else:
        bc_exprs.append(Eq(p[0, y], float(bc_left)))

    # Right boundary (x = Lx)
    if bc_right == "neumann":
        bc_exprs.append(Eq(p[Nx - 1, y], p[Nx - 2, y]))
    else:
        bc_exprs.append(Eq(p[Nx - 1, y], float(bc_right)))

    # Bottom boundary (y = 0)
    if bc_bottom == "neumann":
        bc_exprs.append(Eq(p[x, 0], p[x, 1]))
    else:
        bc_exprs.append(Eq(p[x, 0], float(bc_bottom)))

    # Top boundary (y = Ly)
    if bc_top == "neumann":
        bc_exprs.append(Eq(p[x, Ny - 1], p[x, Ny - 2]))
    else:
        bc_exprs.append(Eq(p[x, Ny - 1], float(bc_top)))

    # Build operator
    op = Operator([eq_update] + bc_exprs)

    # Initialize
    p.data[:] = 0.0
    pn.data[:] = 0.0

    # Set initial Dirichlet boundary values
    if bc_left != "neumann":
        p.data[0, :] = float(bc_left)
        pn.data[0, :] = float(bc_left)
    if bc_right != "neumann":
        p.data[-1, :] = float(bc_right)
        pn.data[-1, :] = float(bc_right)
    if bc_bottom != "neumann":
        p.data[:, 0] = float(bc_bottom)
        pn.data[:, 0] = float(bc_bottom)
    if bc_top != "neumann":
        p.data[:, -1] = float(bc_top)
        pn.data[:, -1] = float(bc_top)

    # Convergence loop with buffer swapping
    l1norm = 1.0
    iteration = 0

    while l1norm > tol and iteration < max_iterations:
        if iteration % 2 == 0:
            _p, _pn = p, pn
        else:
            _p, _pn = pn, p

        op(p=_p, pn=_pn)

        # L1 convergence measure
        denom = np.sum(np.abs(_pn.data[:]))
        if denom > 1e-15:
            l1norm = abs(np.sum(np.abs(_p.data[:]) - np.abs(_pn.data[:])) / denom)
        else:
            l1norm = abs(np.sum(np.abs(_p.data[:]) - np.abs(_pn.data[:])))

        iteration += 1

    # Get result from correct buffer
    if iteration % 2 == 1:
        p_final = p.data[:].copy()
    else:
        p_final = pn.data[:].copy()

    # Coordinate arrays
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)

    # Compute velocity if requested
    qx, qy = None, None
    if compute_velocity:
        dx = Lx / (Nx - 1) if Nx > 1 else Lx
        dy = Ly / (Ny - 1) if Ny > 1 else Ly
        qx, qy = compute_darcy_velocity(p_final, K_array, dx, dy)

    return DarcyResult(
        p=p_final,
        x=x_coords,
        y=y_coords,
        qx=qx,
        qy=qy,
        K=K_array,
        iterations=iteration,
        converged=l1norm <= tol,
        final_l1norm=l1norm,
    )


# =============================================================================
# Transient Solver
# =============================================================================


def solve_darcy_transient(
    Lx: float = 1.0,
    Ly: float = 1.0,
    Nx: int = 64,
    Ny: int = 64,
    permeability: np.ndarray | float = 1.0,
    porosity: float = 0.2,
    source: np.ndarray | float | None = None,
    bc_left: float | str = 0.0,
    bc_right: float | str = 1.0,
    bc_bottom: float | str = "neumann",
    bc_top: float | str = "neumann",
    p_initial: np.ndarray | float = 0.5,
    T: float = 1.0,
    nt: int = 100,
    save_interval: int | None = None,
    compute_velocity: bool = True,
) -> DarcyResult:
    """Solve transient single-phase Darcy flow.

    Solves: phi * dp/dt = div(K * grad(p)) + f

    where phi is the porosity (storage coefficient).

    Parameters
    ----------
    Lx, Ly : float
        Domain extent
    Nx, Ny : int
        Number of grid points
    permeability : np.ndarray or float
        Permeability field
    porosity : float
        Porosity (storage coefficient)
    source : np.ndarray, float, or None
        Source term
    bc_left, bc_right : float or 'neumann'
        Boundary conditions at x=0 and x=Lx
    bc_bottom, bc_top : float or 'neumann'
        Boundary conditions at y=0 and y=Ly
    p_initial : np.ndarray or float
        Initial pressure field
    T : float
        Final simulation time
    nt : int
        Number of time steps
    save_interval : int, optional
        Save pressure every save_interval steps
    compute_velocity : bool
        Whether to compute final velocity field

    Returns
    -------
    DarcyResult
        Solution at final time with optional history

    Raises
    ------
    ValueError
        If time step exceeds stability limit
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required. Install with: pip install devito")

    dt = T / nt
    dx = Lx / (Nx - 1) if Nx > 1 else Lx
    dy = Ly / (Ny - 1) if Ny > 1 else Ly

    # Get maximum permeability for stability check
    K_max = np.max(permeability) if not np.isscalar(permeability) else permeability

    # Stability check for explicit scheme
    max_dt = porosity * min(dx, dy) ** 2 / (4 * K_max)
    if dt > max_dt:
        raise ValueError(
            f"Time step dt={dt:.6e} exceeds stability limit {max_dt:.6e}. "
            f"Increase nt or decrease K/porosity ratio."
        )

    # Create grid
    grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
    x, y = grid.dimensions
    t = grid.stepping_dim

    # TimeFunction for automatic buffer management
    p = TimeFunction(name="p", grid=grid, time_order=1, space_order=2)

    # Permeability
    K = Function(name="K", grid=grid, space_order=2)
    if np.isscalar(permeability):
        K.data[:] = permeability
        K_array = permeability
    else:
        K.data[:] = permeability
        K_array = permeability

    # Source term
    f = Function(name="f", grid=grid)
    if source is None:
        f.data[:] = 0.0
    elif np.isscalar(source):
        f.data[:] = source
    else:
        f.data[:] = source

    # Initial condition
    if np.isscalar(p_initial):
        p.data[0, :, :] = p_initial
        p.data[1, :, :] = p_initial
    else:
        p.data[0, :, :] = p_initial
        p.data[1, :, :] = p_initial

    # Transient equation: phi * dp/dt = div(K * grad(p)) + f
    # Forward Euler: p^{n+1} = p^n + dt/phi * (div(K*grad(p^n)) + f)
    # Using product rule: div(K*grad(p)) = K*laplacian(p) + grad(K).grad(p)

    laplacian_term = K * p.laplace
    gradient_coupling = K.dx * p.dx + K.dy * p.dy
    rhs = laplacian_term + gradient_coupling + f

    eq_update = Eq(p.forward, p + (dt / porosity) * rhs, subdomain=grid.interior)

    # Boundary conditions with time index
    bc_exprs = []

    if bc_left == "neumann":
        bc_exprs.append(Eq(p[t + 1, 0, y], p[t + 1, 1, y]))
    else:
        bc_exprs.append(Eq(p[t + 1, 0, y], float(bc_left)))

    if bc_right == "neumann":
        bc_exprs.append(Eq(p[t + 1, Nx - 1, y], p[t + 1, Nx - 2, y]))
    else:
        bc_exprs.append(Eq(p[t + 1, Nx - 1, y], float(bc_right)))

    if bc_bottom == "neumann":
        bc_exprs.append(Eq(p[t + 1, x, 0], p[t + 1, x, 1]))
    else:
        bc_exprs.append(Eq(p[t + 1, x, 0], float(bc_bottom)))

    if bc_top == "neumann":
        bc_exprs.append(Eq(p[t + 1, x, Ny - 1], p[t + 1, x, Ny - 2]))
    else:
        bc_exprs.append(Eq(p[t + 1, x, Ny - 1], float(bc_top)))

    op = Operator([eq_update] + bc_exprs)

    # Apply initial Dirichlet BCs
    if bc_left != "neumann":
        p.data[:, 0, :] = float(bc_left)
    if bc_right != "neumann":
        p.data[:, -1, :] = float(bc_right)
    if bc_bottom != "neumann":
        p.data[:, :, 0] = float(bc_bottom)
    if bc_top != "neumann":
        p.data[:, :, -1] = float(bc_top)

    # Time stepping
    p_history = []
    if save_interval is not None:
        p_history.append(p.data[0, :, :].copy())

    for step in range(nt):
        op.apply(time_m=0, time_M=0)
        # Swap buffers
        p.data[0, :, :] = p.data[1, :, :]

        if save_interval is not None and (step + 1) % save_interval == 0:
            p_history.append(p.data[0, :, :].copy())

    # Final solution
    p_final = p.data[0, :, :].copy()

    # Coordinates
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)

    # Compute velocity
    qx, qy = None, None
    if compute_velocity:
        qx, qy = compute_darcy_velocity(p_final, K_array, dx, dy)

    return DarcyResult(
        p=p_final,
        x=x_coords,
        y=y_coords,
        qx=qx,
        qy=qy,
        K=K_array,
        iterations=nt,
        converged=True,
        p_history=p_history if save_interval else None,
    )


# =============================================================================
# Verification Utilities
# =============================================================================


def check_mass_conservation(
    p: np.ndarray,
    K: np.ndarray | float,
    source: np.ndarray | float,
    Lx: float,
    Ly: float,
) -> float:
    """Check mass conservation for Darcy flow solution.

    For steady-state flow, total boundary flux should equal total source.

    Parameters
    ----------
    p : np.ndarray
        Pressure field
    K : np.ndarray or float
        Permeability field
    source : np.ndarray or float
        Source term
    Lx, Ly : float
        Domain extent

    Returns
    -------
    imbalance : float
        Relative mass imbalance (should be near zero for good solutions)
    """
    Nx, Ny = p.shape
    dx = Lx / (Nx - 1) if Nx > 1 else Lx
    dy = Ly / (Ny - 1) if Ny > 1 else Ly

    # Handle scalar permeability
    if np.isscalar(K):
        K = np.full_like(p, K)

    # Compute fluxes at boundaries (outward positive)
    # Left boundary (x = 0): flux = -K * dp/dx (inward if dp/dx > 0)
    flux_left = -np.sum(K[0, :] * (p[1, :] - p[0, :]) / dx) * dy

    # Right boundary (x = Lx): flux = K * dp/dx (outward if dp/dx > 0)
    flux_right = np.sum(K[-1, :] * (p[-1, :] - p[-2, :]) / dx) * dy

    # Bottom boundary (y = 0)
    flux_bottom = -np.sum(K[:, 0] * (p[:, 1] - p[:, 0]) / dy) * dx

    # Top boundary (y = Ly)
    flux_top = np.sum(K[:, -1] * (p[:, -1] - p[:, -2]) / dy) * dx

    # Total boundary flux (net outward)
    boundary_flux = flux_left + flux_right + flux_bottom + flux_top

    # Total source
    if np.isscalar(source):
        total_source = source * Lx * Ly
    else:
        total_source = np.sum(source) * dx * dy

    # Relative imbalance
    reference = max(abs(total_source), abs(boundary_flux), 1e-15)
    imbalance = abs(boundary_flux - total_source) / reference

    return imbalance


def verify_linear_pressure(tol: float = 1e-4) -> float:
    """Verify solver against linear analytical solution.

    For constant K, no source, and linear pressure BCs,
    the exact solution is p(x) = p0 + (p1 - p0) * x / L.

    Parameters
    ----------
    tol : float
        Solver convergence tolerance

    Returns
    -------
    error : float
        Maximum error between numerical and analytical solutions
    """
    Lx, Ly = 1.0, 0.1  # Thin domain approximates 1D
    Nx, Ny = 64, 8
    p0, p1 = 1.0, 0.0

    result = solve_darcy_2d(
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        permeability=1.0,
        source=0.0,
        bc_left=p0,
        bc_right=p1,
        bc_bottom="neumann",
        bc_top="neumann",
        tol=tol,
        compute_velocity=False,
    )

    # Analytical solution
    p_exact = p0 + (p1 - p0) * result.x / Lx

    # Compare at centerline
    j_mid = Ny // 2
    p_numerical = result.p[:, j_mid]

    error = np.max(np.abs(p_numerical - p_exact))
    return error
