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

Two absorbing boundary implementations are available:
    - 'conductivity': Graded-conductivity absorbing layer (simple, pedagogical)
    - 'cpml': Convolutional PML with recursive convolution (production-quality)

References
----------
.. [1] K.S. Yee, "Numerical solution of initial boundary value problems
       involving Maxwell's equations in isotropic media," IEEE Trans.
       Antennas Propag., vol. 14, no. 3, pp. 302-307, 1966.

.. [2] J.-P. Berenger, "A perfectly matched layer for the absorption of
       electromagnetic waves," J. Compute. Phys., vol. 114, pp. 185-200, 1994.

.. [3] J. A. Roden and S. D. Gedney, "Convolution PML (CPML): An efficient
       FDTD implementation of the CFS-PML for arbitrary media," Microwave
       Opt. Technol. Lett., vol. 27, no. 5, pp. 334-339, 2000.
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


def _create_cpml_coefficients(
    N: int,
    pml_width: int,
    dt: float,
    sigma_max: float,
    order: int = 3,
    kappa_max: float = 1.0,
    alpha_max: float = 0.0,
    dtype: np.dtype = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CPML recursive convolution coefficients.

    Uses the CFS-PML (Complex Frequency-Shifted PML) formulation
    from Roden & Gedney (2000). The stretching function is:
        s(d) = kappa(d) + sigma(d) / (alpha(d) + j*omega)

    Returns b and a coefficients for the recursive update:
        Psi^{n+1} = b * Psi^n + a * (spatial derivative)

    Parameters
    ----------
    N : int
        Total number of grid points in this dimension
    pml_width : int
        Width of PML region in grid points
    dt : float
        Time step [s]
    sigma_max : float
        Maximum PML conductivity
    order : int
        Polynomial grading order
    kappa_max : float
        Maximum kappa (coordinate stretching factor). 1.0 = no stretching.
    alpha_max : float
        Maximum alpha for CFS (improves absorption of evanescent waves)
    dtype : np.dtype
        Floating-point precision

    Returns
    -------
    b : np.ndarray
        Recursive coefficient b, shape (N,)
    a : np.ndarray
        Recursive coefficient a, shape (N,)
    kappa : np.ndarray
        Coordinate stretching factor, shape (N,)
    """
    sigma = np.zeros(N, dtype=dtype)
    kappa = np.ones(N, dtype=dtype)
    alpha = np.zeros(N, dtype=dtype)

    for i in range(pml_width):
        d = (pml_width - i) / pml_width
        sigma[i] = sigma_max * (d ** order)
        kappa[i] = 1.0 + (kappa_max - 1.0) * (d ** order)
        alpha[i] = alpha_max * (1.0 - d)

    for i in range(N - pml_width, N):
        d = (i - (N - pml_width - 1)) / pml_width
        sigma[i] = sigma_max * (d ** order)
        kappa[i] = 1.0 + (kappa_max - 1.0) * (d ** order)
        alpha[i] = alpha_max * (1.0 - d)

    # CPML coefficients: b = exp(-(sigma/kappa + alpha)*dt)
    denom = kappa * (sigma + kappa * alpha)
    b = np.exp(-(sigma / kappa + alpha) * dt)
    a = np.zeros_like(denom)
    np.divide(sigma * (b - 1.0), denom, out=a, where=denom != 0)

    return b, a, kappa


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
    pml_type: str = 'cpml',
    save_history: bool = False,
    save_every: int = 10,
    dtype: np.dtype = np.float64,
) -> MaxwellResult2D:
    """Solve 2D Maxwell's equations (TM mode) using FDTD.

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
    pml_type : str
        PML implementation: 'conductivity' (graded sigma, simple) or
        'cpml' (Convolutional PML with recursive convolution, default).
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

    # Validate pml_type
    if pml_type not in ('conductivity', 'cpml'):
        raise ValueError(
            f"pml_type must be 'conductivity' or 'cpml', got '{pml_type}'"
        )

    # PML setup
    use_cpml = pml_width > 0 and pml_type == 'cpml'

    if pml_width > 0 and pml_type == 'conductivity':
        if pml_sigma_max is None:
            pml_sigma_max = 0.8 * (3 + 1) / (const.eta0 * dx)

        sigma_x = create_pml_profile(Nx + 1, pml_width, pml_sigma_max)
        sigma_y = create_pml_profile(Ny + 1, pml_width, pml_sigma_max)

        # Add PML conductivity to material conductivity
        for i in range(Nx + 1):
            sigma_arr[i, :] += sigma_x[i]
        for j in range(Ny + 1):
            sigma_arr[:, j] += sigma_y[j]

    # CPML setup: compute recursive convolution coefficients
    if use_cpml:
        if pml_sigma_max is None:
            pml_sigma_max = 0.8 * (3 + 1) / (const.eta0 * dx)

        # Coefficients for E-field updates (at integer grid points)
        bx_e, ax_e, kx_e = _create_cpml_coefficients(
            Nx + 1, pml_width, dt, pml_sigma_max, dtype=dtype)
        by_e, ay_e, ky_e = _create_cpml_coefficients(
            Ny + 1, pml_width, dt, pml_sigma_max, dtype=dtype)

        # Coefficients for H-field updates (at half-integer points)
        # Use averaged values between integer grid points
        bx_h = np.zeros(Nx, dtype=dtype)
        ax_h = np.zeros(Nx, dtype=dtype)
        kx_h = np.ones(Nx, dtype=dtype)
        for i in range(Nx):
            bx_h[i] = 0.5 * (bx_e[i] + bx_e[i + 1])
            ax_h[i] = 0.5 * (ax_e[i] + ax_e[i + 1])
            kx_h[i] = 0.5 * (kx_e[i] + kx_e[i + 1])

        by_h = np.zeros(Ny, dtype=dtype)
        ay_h = np.zeros(Ny, dtype=dtype)
        ky_h = np.ones(Ny, dtype=dtype)
        for j in range(Ny):
            by_h[j] = 0.5 * (by_e[j] + by_e[j + 1])
            ay_h[j] = 0.5 * (ay_e[j] + ay_e[j + 1])
            ky_h[j] = 0.5 * (ky_e[j] + ky_e[j + 1])

        # CPML auxiliary (Psi) fields
        # For H_x update: Psi_Hx_y at (i, j+1/2)
        Psi_Hx_y = np.zeros((Nx + 1, Ny), dtype=dtype)
        # For H_y update: Psi_Hy_x at (i+1/2, j)
        Psi_Hy_x = np.zeros((Nx, Ny + 1), dtype=dtype)
        # For E_z update: Psi_Ez_x at (i, j), Psi_Ez_y at (i, j)
        Psi_Ez_x = np.zeros((Nx + 1, Ny + 1), dtype=dtype)
        Psi_Ez_y = np.zeros((Nx + 1, Ny + 1), dtype=dtype)

    # Update coefficients for E_z
    # E_z^{n+1} = Ca * E_z^n + Cb * (dH_y/dx - dH_x/dy)
    denom = 1.0 + sigma_arr * dt / (2 * eps_arr)
    Ca = (1.0 - sigma_arr * dt / (2 * eps_arr)) / denom
    Cb_x = (dt / eps_arr / dx) / denom
    Cb_y = (dt / eps_arr / dy) / denom

    # Update coefficients for H fields
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

        # ---- H-field updates ----
        # dE_z/dy for H_x update
        dEz_dy = (E_z[:, 1:] - E_z[:, :-1]) / dy
        # dE_z/dx for H_y update
        dEz_dx = (E_z[1:, :] - E_z[:-1, :]) / dx

        if use_cpml:
            # Update CPML Psi for H fields
            for j in range(Ny):
                Psi_Hx_y[:, j] = by_h[j] * Psi_Hx_y[:, j] + ay_h[j] * dEz_dy[:, j]
            for i in range(Nx):
                Psi_Hy_x[i, :] = bx_h[i] * Psi_Hy_x[i, :] + ax_h[i] * dEz_dx[i, :]

            # H_x update with CPML correction
            for j in range(Ny):
                H_x[:, j] -= Ch_x[:, j] * dy * (dEz_dy[:, j] / ky_h[j] + Psi_Hx_y[:, j])
            # H_y update with CPML correction
            for i in range(Nx):
                H_y[i, :] += Ch_y[i, :] * dx * (dEz_dx[i, :] / kx_h[i] + Psi_Hy_x[i, :])
        else:
            # Standard H update (with conductivity-based PML if active)
            H_x[:, :] -= Ch_x * (E_z[:, 1:] - E_z[:, :-1])
            H_y[:, :] += Ch_y * (E_z[1:, :] - E_z[:-1, :])

        # ---- E-field update ----
        # Curl H components
        dHy_dx = (H_y[1:, 1:-1] - H_y[:-1, 1:-1]) / dx
        dHx_dy = (H_x[1:-1, 1:] - H_x[1:-1, :-1]) / dy

        if use_cpml:
            # Update CPML Psi for E field
            for i in range(1, Nx):
                Psi_Ez_x[i, 1:-1] = (bx_e[i] * Psi_Ez_x[i, 1:-1]
                                      + ax_e[i] * dHy_dx[i - 1, :])
            for j in range(1, Ny):
                Psi_Ez_y[1:-1, j] = (by_e[j] * Psi_Ez_y[1:-1, j]
                                      + ay_e[j] * dHx_dy[:, j - 1])

            # E_z update with CPML correction
            curl_H_x = np.zeros_like(E_z[1:-1, 1:-1])
            curl_H_y = np.zeros_like(E_z[1:-1, 1:-1])
            for i in range(1, Nx):
                curl_H_x[i - 1, :] = dHy_dx[i - 1, :] / kx_e[i] + Psi_Ez_x[i, 1:-1]
            for j in range(1, Ny):
                curl_H_y[:, j - 1] = dHx_dy[:, j - 1] / ky_e[j] + Psi_Ez_y[1:-1, j]

            E_z[1:-1, 1:-1] = (
                Ca[1:-1, 1:-1] * E_z[1:-1, 1:-1]
                + Cb_x[1:-1, 1:-1] * dx * curl_H_x
                - Cb_y[1:-1, 1:-1] * dy * curl_H_y
            )
        else:
            # Standard E update
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
    "_create_cpml_coefficients",
    "convergence_test_maxwell_2d",
    "create_pml_profile",
    "gaussian_source_2d",
    "line_source_2d",
    "solve_maxwell_2d",
]
