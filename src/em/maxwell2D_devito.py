"""2D Maxwell Equation Solver (FDTD Method).

Solves the 2D Maxwell's equations in the **TM polarization** (out-of-plane
electric field) using the Yee/FDTD scheme:

TM mode (E_z, H_x, H_y):
    dH_x/dt = -(1/mu) * dE_z/dy
    dH_y/dt = (1/mu) * dE_z/dx
    dE_z/dt = (1/eps) * (dH_y/dx - dH_x/dy)

The Yee scheme uses a staggered grid (Yee cell) where:
    - E_z is at cell centers: (i, j)
    - H_x is at cell edges: (i, j+1/2)
    - H_y is at cell edges: (i+1/2, j)

This module includes a simple graded-conductivity absorbing layer that can be
used as a basic "PML-like" boundary treatment for pedagogical purposes.

References
----------
.. [1] K.S. Yee, "Numerical solution of initial boundary value problems
       involving Maxwell's equations in isotropic media," IEEE Trans.
       Antennas Propag., vol. 14, no. 3, pp. 302-307, 1966.

.. [2] J.-P. Berenger, "A perfectly matched layer for the absorption of
       electromagnetic waves," J. Compute. Phys., vol. 114, pp. 185-200, 1994.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.em.units import EMConstants

try:
    DEVITO_AVAILABLE = True
except Exception:
    DEVITO_AVAILABLE = False


@dataclass
class MaxwellResult2D:
    """Results from the 2D Maxwell FDTD solver.

    Attributes
    ----------
    E_z : np.ndarray
        Electric field (z-component) at final time, shape (Nx+1, Ny+1)
    H_x : np.ndarray
        Magnetic field (x-component) at final time, shape (Nx+1, Ny)
    H_y : np.ndarray
        Magnetic field (y-component) at final time, shape (Nx, Ny+1)
    x : np.ndarray
        x-coordinates for E_z
    y : np.ndarray
        y-coordinates for E_z
    t : float
        Final simulation time [s]
    dt : float
        Time step used [s]
    dx : float
        Grid spacing in x [m]
    dy : float
        Grid spacing in y [m]
    c : float
        Wave speed [m/s]
    C : float
        Courant number used
    E_history : list, optional
        Full E_z history at selected times
    t_history : np.ndarray, optional
        Time points for history
    """
    E_z: np.ndarray
    H_x: np.ndarray
    H_y: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t: float
    dt: float
    dx: float
    dy: float
    c: float
    C: float
    E_history: list | None = None
    t_history: np.ndarray | None = None


def create_pml_profile(
    N: int,
    pml_width: int,
    sigma_max: float = 1.0,
    order: int = 3,
) -> np.ndarray:
    """Create PML conductivity profile.

    Uses polynomial grading: sigma(d) = sigma_max * (d/pml_width)^order
    where d is the distance into the PML.

    Parameters
    ----------
    N : int
        Total number of grid points
    pml_width : int
        Width of PML region in grid points
    sigma_max : float
        Maximum conductivity at PML edge
    order : int
        Polynomial order for grading (typically 3-4)

    Returns
    -------
    np.ndarray
        Conductivity profile array, shape (N,)
    """
    sigma = np.zeros(N)

    # Left PML
    for i in range(pml_width):
        d = (pml_width - i) / pml_width
        sigma[i] = sigma_max * (d ** order)

    # Right PML
    for i in range(N - pml_width, N):
        d = (i - (N - pml_width - 1)) / pml_width
        sigma[i] = sigma_max * (d ** order)

    return sigma


def solve_maxwell_2d(
    Lx: float = 1.0,
    Ly: float = 1.0,
    Nx: int = 100,
    Ny: int = 100,
    T: float = 5e-9,
    CFL: float = 0.9,
    eps_r: float | np.ndarray = 1.0,
    mu_r: float | np.ndarray = 1.0,
    sigma: float | np.ndarray = 0.0,
    E_init: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    source_func: Callable[[float], float] | None = None,
    source_position: tuple | None = None,
    pml_width: int = 0,
    pml_sigma_max: float = None,
    save_history: bool = False,
    save_every: int = 10,
    dtype: np.dtype = np.float64,
) -> MaxwellResult2D:
    """Solve 2D Maxwell's equations (TE mode) using FDTD.

    Parameters
    ----------
    Lx : float
        Domain length in x [m]
    Ly : float
        Domain length in y [m]
    Nx : int
        Number of grid intervals in x
    Ny : int
        Number of grid intervals in y
    T : float
        Final simulation time [s]
    CFL : float
        Courant number. Must satisfy CFL <= 1/sqrt(2) for 2D stability.
    eps_r : float or np.ndarray
        Relative permittivity. Can be 2D array of shape (Nx+1, Ny+1).
    mu_r : float or np.ndarray
        Relative permeability.
    sigma : float or np.ndarray
        Conductivity [S/m] for lossy media.
    E_init : callable, optional
        Initial E_z field: E_init(X, Y) -> E_z(x, y, 0)
        X, Y are 2D meshgrid arrays.
    source_func : callable, optional
        Time-dependent source: source_func(t) -> amplitude
    source_position : tuple, optional
        (x, y) coordinates for source injection [m]
    pml_width : int
        Width of PML region in grid cells. 0 for no PML (PEC boundaries).
    pml_sigma_max : float, optional
        Maximum PML conductivity. Default: computed from optimal formula.
    save_history : bool
        If True, save solution history.
    save_every : int
        Save history every this many time steps.
    dtype : np.dtype
        Floating-point precision.

    Returns
    -------
    MaxwellResult2D
        Solution data including final fields and optionally history
    """
    # 2D CFL stability requires C <= 1/sqrt(2) ≈ 0.707
    if CFL > 1.0 / np.sqrt(2):
        raise ValueError(
            f"CFL={CFL} > 1/sqrt(2) ≈ 0.707 violates 2D stability condition"
        )

    # Physical constants
    const = EMConstants()

    # Grid spacing
    dx = Lx / Nx
    dy = Ly / Ny

    # Wave speed
    if isinstance(eps_r, np.ndarray):
        eps_r_max = np.max(eps_r)
    else:
        eps_r_max = eps_r
    if isinstance(mu_r, np.ndarray):
        mu_r_max = np.max(mu_r)
    else:
        mu_r_max = mu_r
    c_min = const.c0 / np.sqrt(eps_r_max * mu_r_max)

    # Time step from 2D CFL
    dt = CFL / (c_min * np.sqrt(1/dx**2 + 1/dy**2))

    # Number of time steps
    Nt = int(np.ceil(T / dt))
    dt = T / Nt

    # Coordinate arrays
    x = np.linspace(0, Lx, Nx + 1, dtype=dtype)
    y = np.linspace(0, Ly, Ny + 1, dtype=dtype)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initialize material arrays
    if isinstance(eps_r, np.ndarray):
        eps_arr = eps_r * const.eps0
    else:
        eps_arr = np.full((Nx + 1, Ny + 1), eps_r * const.eps0, dtype=dtype)

    if isinstance(mu_r, np.ndarray):
        mu_arr = mu_r * const.mu0
    else:
        mu_arr = np.full((Nx + 1, Ny + 1), mu_r * const.mu0, dtype=dtype)

    if isinstance(sigma, np.ndarray):
        sigma_arr = sigma
    else:
        sigma_arr = np.full((Nx + 1, Ny + 1), sigma, dtype=dtype)

    # PML setup
    if pml_width > 0:
        if pml_sigma_max is None:
            # Optimal sigma_max formula
            pml_sigma_max = 0.8 * (3 + 1) / (const.eta0 * dx)

        sigma_x = create_pml_profile(Nx + 1, pml_width, pml_sigma_max)
        sigma_y = create_pml_profile(Ny + 1, pml_width, pml_sigma_max)

        # Add PML conductivity to material conductivity
        for i in range(Nx + 1):
            sigma_arr[i, :] += sigma_x[i]
        for j in range(Ny + 1):
            sigma_arr[:, j] += sigma_y[j]

    # Update coefficients for E_z
    # E_z^{n+1} = Ca * E_z^n + Cb * (dH_y/dx - dH_x/dy)
    denom = 1.0 + sigma_arr * dt / (2 * eps_arr)
    Ca = (1.0 - sigma_arr * dt / (2 * eps_arr)) / denom
    Cb_x = (dt / eps_arr / dx) / denom
    Cb_y = (dt / eps_arr / dy) / denom

    # Update coefficients for H fields
    # Simple lossless update for H (can be extended for magnetic losses)
    Ch_x = dt / (mu_arr[:, :-1] * dy)  # For H_x update
    Ch_y = dt / (mu_arr[:-1, :] * dx)  # For H_y update

    # Initialize fields
    E_z = np.zeros((Nx + 1, Ny + 1), dtype=dtype)
    H_x = np.zeros((Nx + 1, Ny), dtype=dtype)  # At (i, j+1/2)
    H_y = np.zeros((Nx, Ny + 1), dtype=dtype)  # At (i+1/2, j)

    if E_init is not None:
        E_z[:, :] = E_init(X, Y)

    # Source setup
    if source_func is not None:
        if source_position is None:
            raise ValueError("source_position required when source_func given")
        src_i = int(round(source_position[0] / dx))
        src_j = int(round(source_position[1] / dy))
        src_i = max(pml_width, min(src_i, Nx - pml_width))
        src_j = max(pml_width, min(src_j, Ny - pml_width))
    else:
        src_i = src_j = None

    # History storage
    if save_history:
        E_history = []
        t_history = []
    else:
        E_history = None
        t_history = None

    # Main time-stepping loop
    for n in range(Nt):
        t_n = n * dt

        # Update H_x: H_x -= Ch * dE_z/dy
        # H_x(i, j+1/2) uses E_z(i, j+1) and E_z(i, j)
        H_x[:, :] = H_x[:, :] - Ch_x * (E_z[:, 1:] - E_z[:, :-1])

        # Update H_y: H_y += Ch * dE_z/dx
        # H_y(i+1/2, j) uses E_z(i+1, j) and E_z(i, j)
        H_y[:, :] = H_y[:, :] + Ch_y * (E_z[1:, :] - E_z[:-1, :])

        # Update E_z: E_z = Ca*E_z + Cb*(dH_y/dx - dH_x/dy)
        E_z[1:-1, 1:-1] = (
            Ca[1:-1, 1:-1] * E_z[1:-1, 1:-1]
            + Cb_x[1:-1, 1:-1] * (H_y[1:, 1:-1] - H_y[:-1, 1:-1])
            - Cb_y[1:-1, 1:-1] * (H_x[1:-1, 1:] - H_x[1:-1, :-1])
        )

        # Inject source
        if source_func is not None:
            E_z[src_i, src_j] += source_func(t_n + dt)

        # PEC boundary conditions (E_z = 0 at boundaries)
        if pml_width == 0:
            E_z[0, :] = 0.0
            E_z[-1, :] = 0.0
            E_z[:, 0] = 0.0
            E_z[:, -1] = 0.0

        # Save history
        if save_history and (n % save_every == 0 or n == Nt - 1):
            E_history.append(E_z.copy())
            t_history.append(t_n + dt)

    if save_history:
        t_history = np.array(t_history, dtype=dtype)

    return MaxwellResult2D(
        E_z=E_z.copy(),
        H_x=H_x.copy(),
        H_y=H_y.copy(),
        x=x,
        y=y,
        t=T,
        dt=dt,
        dx=dx,
        dy=dy,
        c=c_min,
        C=CFL,
        E_history=E_history,
        t_history=t_history,
    )


def gaussian_source_2d(
    X: np.ndarray,
    Y: np.ndarray,
    x0: float,
    y0: float,
    sigma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """2D Gaussian initial condition for E_z.

    Parameters
    ----------
    X, Y : np.ndarray
        Meshgrid coordinate arrays
    x0, y0 : float
        Center position [m]
    sigma : float
        Gaussian width [m]
    amplitude : float
        Peak amplitude [V/m]

    Returns
    -------
    np.ndarray
        E_z initial condition
    """
    r2 = (X - x0)**2 + (Y - y0)**2
    return amplitude * np.exp(-r2 / (2 * sigma**2))


def line_source_2d(
    X: np.ndarray,
    Y: np.ndarray,
    x0: float,
    sigma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Line source (infinite in y) initial condition for E_z.

    Parameters
    ----------
    X, Y : np.ndarray
        Meshgrid coordinate arrays
    x0 : float
        x-position of line [m]
    sigma : float
        Line width [m]
    amplitude : float
        Peak amplitude [V/m]

    Returns
    -------
    np.ndarray
        E_z initial condition
    """
    return amplitude * np.exp(-((X - x0)**2) / (2 * sigma**2))


def convergence_test_maxwell_2d(
    grid_sizes: list = None,
    T: float = 1e-9,
    CFL: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run convergence test for 2D Maxwell solver.

    Uses a Gaussian pulse with known analytical behavior for short times.

    Parameters
    ----------
    grid_sizes : list, optional
        List of N values (Nx=Ny=N). Default: [25, 50, 100, 200]
    T : float
        Final time [s]
    CFL : float
        Courant number

    Returns
    -------
    tuple
        (grid_sizes, errors, observed_order)
    """
    if grid_sizes is None:
        grid_sizes = [25, 50, 100, 200]

    L = 1.0
    x0, y0 = 0.5, 0.5  # Center of domain
    sigma = 0.05  # Gaussian width

    errors = []

    for N in grid_sizes:
        # Reference solution at double resolution
        N_ref = 2 * N

        result = solve_maxwell_2d(
            Lx=L, Ly=L, Nx=N, Ny=N, T=T, CFL=CFL,
            E_init=lambda X, Y: gaussian_source_2d(X, Y, x0, y0, sigma),
            pml_width=10,
        )

        result_ref = solve_maxwell_2d(
            Lx=L, Ly=L, Nx=N_ref, Ny=N_ref, T=T, CFL=CFL,
            E_init=lambda X, Y: gaussian_source_2d(X, Y, x0, y0, sigma),
            pml_width=20,
        )

        # Interpolate reference to coarse grid for comparison
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (result_ref.x, result_ref.y), result_ref.E_z,
            method='linear', bounds_error=False, fill_value=0.0
        )

        X, Y = np.meshgrid(result.x, result.y, indexing='ij')
        E_ref_interp = interp((X, Y))

        # L2 error (excluding PML region)
        inner = slice(15, -15)
        error = np.sqrt(np.mean((result.E_z[inner, inner] - E_ref_interp[inner, inner])**2))
        errors.append(error)

    errors = np.array(errors)
    grid_sizes = np.array(grid_sizes)

    # Compute observed order
    dx_vals = L / grid_sizes
    log_dx = np.log(dx_vals)
    log_err = np.log(errors)
    coeffs = np.polyfit(log_dx[-3:], log_err[-3:], 1)
    observed_order = coeffs[0]

    return grid_sizes, errors, observed_order


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MaxwellResult2D",
    "convergence_test_maxwell_2d",
    "create_pml_profile",
    "gaussian_source_2d",
    "line_source_2d",
    "solve_maxwell_2d",
]
