"""Absorbing Boundary Condition methods for the 2D wave equation.

Implements several ABC techniques for the acoustic wave equation:
    u_tt = c^2 * (u_xx + u_yy)

Methods available:
    - 'dirichlet': Simple u=0 boundaries (strong reflections)
    - 'first_order': Clayton-Engquist first-order ABC
    - 'damping': Sponge layer with polynomial damping profile
    - 'pml': Split-field Perfectly Matched Layer (Grote-Sim)
    - 'higdon': Second-order Higdon ABC (P=2, angles 0 and pi/4)
    - 'habc': Hybrid ABC combining Higdon with weighted absorption layer

References
----------
.. [1] B. Engquist and A. Majda, "Absorbing boundary conditions for the
       numerical simulation of waves," Math. Comp., 1977.
.. [2] C. Cerjan et al., "A nonreflecting boundary condition for discrete
       acoustic and elastic wave equations," Geophysics, 1985.
.. [3] J.-P. Berenger, "A perfectly matched layer for the absorption of
       electromagnetic waves," J. Comput. Phys., 1994.
.. [4] D. I. Dolci et al., "Effectiveness and computational efficiency of
       absorbing boundary conditions for full-waveform inversion,"
       Geosci. Model Dev., 2022.
.. [5] M. J. Grote and I. Sim, "Efficient PML for the wave equation,"
       arXiv:1001.0319, 2010.
.. [6] R. L. Higdon, "Absorbing boundary conditions for difference
       approximations to the multidimensional wave equation,"
       Math. Comp., 1986.
.. [7] Y. Liu and M. K. Sen, "An improved hybrid absorbing boundary
       condition for wave equation modeling," J. Geophys. Eng., 2018.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    from devito import (
        Constant,
        Eq,
        Function,
        Grid,
        Operator,
        TimeFunction,
        solve,
    )
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@dataclass
class ABCResult:
    """Results from the 2D wave equation solver with ABCs.

    Attributes
    ----------
    u : np.ndarray
        Solution at final time, shape (Nx+1, Ny+1)
    x : np.ndarray
        Spatial grid points in x
    y : np.ndarray
        Spatial grid points in y
    t : float
        Final time
    dt : float
        Time step used
    abc_type : str
        ABC method used
    pad_width : int
        Width of absorbing layer (0 for dirichlet/first_order/higdon)
    u_history : np.ndarray, optional
        Full solution history
    t_history : np.ndarray, optional
        Time points for history
    C : float
        Effective Courant number
    """
    u: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t: float
    dt: float
    abc_type: str
    pad_width: int = 0
    u_history: np.ndarray | None = None
    t_history: np.ndarray | None = None
    C: float = 0.0


def create_damping_profile(
    grid_shape: tuple[int, int],
    pad_width: int,
    sigma_max: float | None = None,
    order: int = 3,
    c: float = 1.0,
    dx: float | None = None,
) -> np.ndarray:
    """Create 2D polynomial damping profile for sponge layer.

    The damping coefficient is zero in the interior and ramps
    polynomially in the absorbing region:
        gamma(d) = sigma_max * (d / pad_width)^order

    Parameters
    ----------
    grid_shape : tuple of int
        Shape of the grid (Nx+1, Ny+1)
    pad_width : int
        Width of damping region in grid points on each side
    sigma_max : float or None
        Maximum damping coefficient at the outer boundary.
        If None, computed as 3*c/W where W = pad_width*dx.
    order : int
        Polynomial order for the ramp (typically 2-3)
    c : float
        Wave speed (used when sigma_max is None)
    dx : float or None
        Grid spacing (used when sigma_max is None).
        If None, estimated from grid_shape assuming unit domain.

    Returns
    -------
    np.ndarray
        2D damping profile array, shape grid_shape
    """
    if sigma_max is None:
        if dx is None:
            dx = 1.0 / (grid_shape[0] - 1)
        W = pad_width * dx
        sigma_max = 3.0 * c / W
    Nx_plus1, Ny_plus1 = grid_shape
    gamma = np.zeros(grid_shape)

    # Build 1D profiles for x and y directions
    gamma_x = np.zeros(Nx_plus1)
    gamma_y = np.zeros(Ny_plus1)

    for i in range(pad_width):
        d = (pad_width - i) / pad_width
        gamma_x[i] = sigma_max * (d ** order)

    for i in range(Nx_plus1 - pad_width, Nx_plus1):
        d = (i - (Nx_plus1 - pad_width - 1)) / pad_width
        gamma_x[i] = sigma_max * (d ** order)

    for j in range(pad_width):
        d = (pad_width - j) / pad_width
        gamma_y[j] = sigma_max * (d ** order)

    for j in range(Ny_plus1 - pad_width, Ny_plus1):
        d = (j - (Ny_plus1 - pad_width - 1)) / pad_width
        gamma_y[j] = sigma_max * (d ** order)

    # Combine: take maximum of x and y damping at each point
    for i in range(Nx_plus1):
        for j in range(Ny_plus1):
            gamma[i, j] = max(gamma_x[i], gamma_y[j])

    return gamma


def create_directional_damping_profiles(
    grid_shape: tuple[int, int],
    pad_width: int,
    sigma_max: float,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create separate x- and y-direction damping profiles for PML.

    Unlike `create_damping_profile` which takes the maximum, this
    returns independent profiles: sigma_x ramps only near x-boundaries,
    sigma_y ramps only near y-boundaries.

    Parameters
    ----------
    grid_shape : tuple of int
        Shape of the grid (Nx+1, Ny+1)
    pad_width : int
        Width of PML region in grid points on each side
    sigma_max : float
        Maximum damping coefficient
    order : int
        Polynomial order for the ramp

    Returns
    -------
    sigma_x, sigma_y : tuple of np.ndarray
        Directional damping profiles, each of shape grid_shape
    """
    Nx_plus1, Ny_plus1 = grid_shape
    sigma_x = np.zeros(grid_shape)
    sigma_y = np.zeros(grid_shape)

    # sigma_x: ramps near left (x=0) and right (x=Lx) boundaries
    for i in range(pad_width):
        d = (pad_width - i) / pad_width
        sigma_x[i, :] = sigma_max * (d ** order)
    for i in range(Nx_plus1 - pad_width, Nx_plus1):
        d = (i - (Nx_plus1 - pad_width - 1)) / pad_width
        sigma_x[i, :] = sigma_max * (d ** order)

    # sigma_y: ramps near bottom (y=0) and top (y=Ly) boundaries
    for j in range(pad_width):
        d = (pad_width - j) / pad_width
        sigma_y[:, j] = sigma_max * (d ** order)
    for j in range(Ny_plus1 - pad_width, Ny_plus1):
        d = (j - (Ny_plus1 - pad_width - 1)) / pad_width
        sigma_y[:, j] = sigma_max * (d ** order)

    return sigma_x, sigma_y


def create_habc_weights(
    pad_width: int,
    P: int = 2,
    alpha: float | None = None,
) -> np.ndarray:
    """Create non-linear HABC weight function.

    The weight controls blending between the standard wave equation
    solution and the Higdon ABC correction in the absorption layer.

    Parameters
    ----------
    pad_width : int
        Width of absorption layer in grid points
    P : int
        Number of Higdon angles minus 1 (controls flat region width)
    alpha : float or None
        Exponent for the polynomial decay. If None, uses the formula
        alpha = 1.0 + 0.15 * (pad_width - P) from Dolci et al.

    Returns
    -------
    np.ndarray
        Weight array of length pad_width, from outer boundary (index 0)
        to inner boundary (index pad_width-1)
    """
    if alpha is None:
        alpha = 1.0 + 0.15 * (pad_width - P)

    weights = np.zeros(pad_width)
    for k in range(pad_width):
        if k <= P:
            weights[k] = 1.0
        elif k < pad_width - 1:
            weights[k] = ((pad_width - k) / (pad_width - P)) ** alpha
        else:
            weights[k] = 0.0

    return weights


def _higdon_coefficients(c, dt, dh, alpha1=0.0, alpha2=np.pi/4, a=0.5, b=0.5):
    """Compute Higdon P=2 stencil coefficients for one spatial direction.

    Parameters
    ----------
    c : float
        Wave speed
    dt : float
        Time step
    dh : float
        Grid spacing in the normal direction
    alpha1, alpha2 : float
        Incidence angles for the two Higdon operators
    a, b : float
        Time and space averaging parameters (0.5 for centered)

    Returns
    -------
    tuple
        (c1_coeffs, c2_coeffs, denom) where c1_coeffs and c2_coeffs
        are 4-tuples of stencil coefficients and denom = c11 * c21.
    """
    ca1 = np.cos(alpha1)
    ca2 = np.cos(alpha2)

    g11 = ca1 * (1 - a) / dt
    g12 = ca1 * a / dt
    g13 = ca1 * (1 - b) * c / dh
    g14 = ca1 * b * c / dh

    c11 = g11 + g13
    c12 = -g11 + g14
    c13 = g12 - g13
    c14 = -g12 - g14

    g21 = ca2 * (1 - a) / dt
    g22 = ca2 * a / dt
    g23 = ca2 * (1 - b) * c / dh
    g24 = ca2 * b * c / dh

    c21 = g21 + g23
    c22 = -g21 + g24
    c23 = g22 - g23
    c24 = -g22 - g24

    denom = c11 * c21

    return (c11, c12, c13, c14), (c21, c22, c23, c24), denom


def _higdon_update(u1_bnd, u1_p1, u1_p2, u2_bnd, u2_p1, u2_p2,
                   u3_p1, u3_p2, c1, c2, denom):
    """Compute Higdon P=2 boundary value (vectorized over boundary points).

    Parameters
    ----------
    u1_bnd, u1_p1, u1_p2 : array
        u at t-1 at boundary, +1 interior, +2 interior
    u2_bnd, u2_p1, u2_p2 : array
        u at t at boundary, +1 interior, +2 interior
    u3_p1, u3_p2 : array
        u at t+1 (wave eq) at +1 interior, +2 interior
    c1, c2 : tuple
        Higdon coefficients (c11,c12,c13,c14), (c21,c22,c23,c24)
    denom : float
        c11 * c21

    Returns
    -------
    array
        Higdon boundary values at t+1
    """
    c11, c12, c13, c14 = c1
    c21, c22, c23, c24 = c2

    return (
        u2_bnd * (-c11*c22 - c12*c21)
        + u3_p1 * (-c11*c23 - c13*c21)
        + u2_p1 * (-c11*c24 - c12*c23 - c14*c21 - c13*c22)
        + u1_bnd * (-c12*c22)
        + u1_p1 * (-c12*c24 - c14*c22)
        + u3_p2 * (-c13*c23)
        + u2_p2 * (-c13*c24 - c14*c23)
        + u1_p2 * (-c14*c24)
    ) / denom


def _apply_higdon_bc(u_data, Nx, Ny, c, dt, dx, dy):
    """Apply second-order Higdon ABC at all four boundaries.

    Modifies u_data[2] (forward time level) at boundary grid lines.
    Uses P=2 with angles 0 and pi/4.

    Parameters
    ----------
    u_data : array
        TimeFunction data with shape (3, Nx+1, Ny+1)
    Nx, Ny : int
        Number of grid intervals
    c, dt, dx, dy : float
        Wave speed, time step, grid spacings
    """
    u1 = u_data[0]  # t - 1
    u2 = u_data[1]  # t
    u3 = u_data[2]  # t + 1 (from wave equation)

    # Coefficients for x-boundaries (use dx)
    cx1, cx2, denom_x = _higdon_coefficients(c, dt, dx)
    # Coefficients for y-boundaries (use dy)
    cy1, cy2, denom_y = _higdon_coefficients(c, dt, dy)

    # Left boundary (x=0): interior direction is +x
    u3[0, :] = _higdon_update(
        u1[0, :], u1[1, :], u1[2, :],
        u2[0, :], u2[1, :], u2[2, :],
        u3[1, :], u3[2, :],
        cx1, cx2, denom_x,
    )

    # Right boundary (x=Nx): interior direction is -x
    u3[Nx, :] = _higdon_update(
        u1[Nx, :], u1[Nx-1, :], u1[Nx-2, :],
        u2[Nx, :], u2[Nx-1, :], u2[Nx-2, :],
        u3[Nx-1, :], u3[Nx-2, :],
        cx1, cx2, denom_x,
    )

    # Bottom boundary (y=0): interior direction is +y
    u3[:, 0] = _higdon_update(
        u1[:, 0], u1[:, 1], u1[:, 2],
        u2[:, 0], u2[:, 1], u2[:, 2],
        u3[:, 1], u3[:, 2],
        cy1, cy2, denom_y,
    )

    # Top boundary (y=Ny): interior direction is -y
    u3[:, Ny] = _higdon_update(
        u1[:, Ny], u1[:, Ny-1], u1[:, Ny-2],
        u2[:, Ny], u2[:, Ny-1], u2[:, Ny-2],
        u3[:, Ny-1], u3[:, Ny-2],
        cy1, cy2, denom_y,
    )


def _apply_habc_correction(u_data, Nx, Ny, c, dt, dx, dy,
                           pad_width, weights):
    """Apply HABC correction in the absorption layer.

    Blends the wave equation solution with Higdon corrections using
    the weight function at each layer point.

    Parameters
    ----------
    u_data : array
        TimeFunction data with shape (3, Nx+1, Ny+1)
    Nx, Ny : int
        Number of grid intervals
    c, dt, dx, dy : float
        Wave speed, time step, grid spacings
    pad_width : int
        Width of absorption layer
    weights : np.ndarray
        1D weight array of length pad_width
    """
    u1 = u_data[0]
    u2 = u_data[1]
    u3_wave = u_data[2].copy()  # Snapshot of wave equation solution

    cx1, cx2, denom_x = _higdon_coefficients(c, dt, dx)
    cy1, cy2, denom_y = _higdon_coefficients(c, dt, dy)

    # Left layer (x = 0 to pad_width-1)
    for k in range(min(pad_width, Nx - 1)):
        i = k
        w = weights[k]
        if w == 0 or i + 2 > Nx:
            continue
        u_hig = _higdon_update(
            u1[i, :], u1[i+1, :], u1[i+2, :],
            u2[i, :], u2[i+1, :], u2[i+2, :],
            u3_wave[i+1, :], u3_wave[i+2, :],
            cx1, cx2, denom_x,
        )
        u_data[2][i, :] = (1 - w) * u3_wave[i, :] + w * u_hig

    # Right layer (x = Nx down to Nx-pad_width+1)
    for k in range(min(pad_width, Nx - 1)):
        i = Nx - k
        w = weights[k]
        if w == 0 or i - 2 < 0:
            continue
        u_hig = _higdon_update(
            u1[i, :], u1[i-1, :], u1[i-2, :],
            u2[i, :], u2[i-1, :], u2[i-2, :],
            u3_wave[i-1, :], u3_wave[i-2, :],
            cx1, cx2, denom_x,
        )
        u_data[2][i, :] = (1 - w) * u3_wave[i, :] + w * u_hig

    # Bottom layer (y = 0 to pad_width-1)
    for k in range(min(pad_width, Ny - 1)):
        j = k
        w = weights[k]
        if w == 0 or j + 2 > Ny:
            continue
        u_hig = _higdon_update(
            u1[:, j], u1[:, j+1], u1[:, j+2],
            u2[:, j], u2[:, j+1], u2[:, j+2],
            u3_wave[:, j+1], u3_wave[:, j+2],
            cy1, cy2, denom_y,
        )
        # Blend (use max weight from x and y for corners)
        u_data[2][:, j] = np.where(
            u_data[2][:, j] != u3_wave[:, j],
            # Already modified by x-layer: take the more absorptive
            np.minimum(np.abs(u_data[2][:, j]),
                       np.abs((1 - w) * u3_wave[:, j] + w * u_hig))
            * np.sign(u_data[2][:, j]),
            (1 - w) * u3_wave[:, j] + w * u_hig,
        )

    # Top layer (y = Ny down to Ny-pad_width+1)
    for k in range(min(pad_width, Ny - 1)):
        j = Ny - k
        w = weights[k]
        if w == 0 or j - 2 < 0:
            continue
        u_hig = _higdon_update(
            u1[:, j], u1[:, j-1], u1[:, j-2],
            u2[:, j], u2[:, j-1], u2[:, j-2],
            u3_wave[:, j-1], u3_wave[:, j-2],
            cy1, cy2, denom_y,
        )
        u_data[2][:, j] = np.where(
            u_data[2][:, j] != u3_wave[:, j],
            np.minimum(np.abs(u_data[2][:, j]),
                       np.abs((1 - w) * u3_wave[:, j] + w * u_hig))
            * np.sign(u_data[2][:, j]),
            (1 - w) * u3_wave[:, j] + w * u_hig,
        )


def solve_wave_2d_abc(
    Lx: float = 2.0,
    Ly: float = 2.0,
    c: float = 1.0,
    Nx: int = 100,
    Ny: int = 100,
    T: float = 1.0,
    CFL: float = 0.5,
    I: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    abc_type: str = 'first_order',
    pad_width: int = 20,
    sigma_max: float | None = None,
    damping_order: int = 3,
    save_history: bool = False,
) -> ABCResult:
    """Solve the 2D wave equation with selectable ABC method.

    Parameters
    ----------
    Lx, Ly : float
        Domain extent in x and y
    c : float
        Wave speed
    Nx, Ny : int
        Number of grid intervals
    T : float
        Final simulation time
    CFL : float
        Target Courant number (must be <= 1)
    I : callable, optional
        Initial displacement: I(X, Y) -> u(x, y, 0).
        Default: Gaussian point source at center.
    abc_type : str
        ABC method: 'dirichlet', 'first_order', 'damping', 'pml',
        'higdon', 'habc'
    pad_width : int
        Width of absorbing layer in grid cells (for 'damping', 'pml', 'habc')
    sigma_max : float or None
        Maximum damping coefficient. If None, computed from theory.
    damping_order : int
        Polynomial order for damping ramp
    save_history : bool
        If True, save full solution history

    Returns
    -------
    ABCResult
        Solution data
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    valid_types = ('dirichlet', 'first_order', 'damping', 'pml',
                   'higdon', 'habc')
    if abc_type not in valid_types:
        raise ValueError(f"abc_type must be one of {valid_types}, got '{abc_type}'")

    if CFL > 1.0:
        raise ValueError(f"CFL={CFL} > 1 violates stability condition")

    dx = Lx / Nx
    dy = Ly / Ny
    stability_factor = np.sqrt(1/dx**2 + 1/dy**2)
    dt = CFL / (c * stability_factor)

    # Default initial condition: Gaussian point source at center
    if I is None:
        x0, y0 = Lx / 2, Ly / 2
        sigma_src = min(Lx, Ly) / 20
        def I(X, Y):
            return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma_src**2))

    x_coords = np.linspace(0, Lx, Nx + 1)
    y_coords = np.linspace(0, Ly, Ny + 1)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    Nt = int(round(T / dt))
    if Nt == 0:
        Nt = 1
    dt = T / Nt
    C_actual = c * dt * stability_factor

    # Create Devito grid
    grid = Grid(shape=(Nx + 1, Ny + 1), extent=(Lx, Ly))
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

    # Set initial conditions
    u0_vals = I(X, Y)
    u.data[0, :, :] = u0_vals
    u.data[1, :, :] = u0_vals  # V=0 assumed

    # Laplacian for first time step
    laplace_u0 = np.zeros_like(u0_vals)
    laplace_u0[1:-1, 1:-1] = (
        (u0_vals[2:, 1:-1] - 2*u0_vals[1:-1, 1:-1] + u0_vals[:-2, 1:-1]) / dx**2 +
        (u0_vals[1:-1, 2:] - 2*u0_vals[1:-1, 1:-1] + u0_vals[1:-1, :-2]) / dy**2
    )
    u1 = u0_vals + 0.5 * dt**2 * c**2 * laplace_u0
    u1[0, :] = 0; u1[-1, :] = 0; u1[:, 0] = 0; u1[:, -1] = 0
    u.data[1, :, :] = u1

    # Build equations based on abc_type
    c_sq = Constant(name='c_sq')
    t_dim = grid.stepping_dim
    x_dim, y_dim = grid.dimensions

    # Auxiliary state for PML
    phi_x = None
    phi_y = None
    # HABC weights
    habc_weights = None

    if abc_type == 'damping':
        # Sponge layer: u_tt + gamma * u_t = c^2 * laplace(u)
        damp = Function(name='damp', grid=grid)
        damp_profile = create_damping_profile(
            (Nx + 1, Ny + 1), pad_width, sigma_max, damping_order,
            c=c, dx=dx,
        )
        damp.data[:] = damp_profile

        pde = u.dt2 + damp * u.dt - c_sq * u.laplace
        stencil = Eq(u.forward, solve(pde, u.forward))

        bc_x0 = Eq(u[t_dim + 1, 0, y_dim], 0)
        bc_xN = Eq(u[t_dim + 1, Nx, y_dim], 0)
        bc_y0 = Eq(u[t_dim + 1, x_dim, 0], 0)
        bc_yN = Eq(u[t_dim + 1, x_dim, Ny], 0)
        eqs = [stencil, bc_x0, bc_xN, bc_y0, bc_yN]

    elif abc_type == 'pml':
        # Split-field PML (Grote-Sim formulation).
        # Uses separate directional damping profiles sigma_x, sigma_y
        # and auxiliary fields phi_x, phi_y for angle-independent absorption.
        R_target = 1e-3
        W = pad_width * dx
        pml_sigma_max = (damping_order + 1) * c * np.log(1.0 / R_target) / (2 * W)

        sigma_x_arr, sigma_y_arr = create_directional_damping_profiles(
            (Nx + 1, Ny + 1), pad_width, pml_sigma_max, damping_order)

        sigma_x_fn = Function(name='sigma_x', grid=grid)
        sigma_y_fn = Function(name='sigma_y', grid=grid)
        sigma_x_fn.data[:] = sigma_x_arr
        sigma_y_fn.data[:] = sigma_y_arr

        # Auxiliary fields for the split-field PML
        phi_x = TimeFunction(name='phi_x', grid=grid, time_order=1, space_order=2)
        phi_y = TimeFunction(name='phi_y', grid=grid, time_order=1, space_order=2)
        phi_x.data[:] = 0.0
        phi_y.data[:] = 0.0

        # Grote-Sim PML equation:
        # u_tt + (sigma_x + sigma_y)*u_t + sigma_x*sigma_y*u
        #   = c^2*laplace(u) + d(phi_x)/dx + d(phi_y)/dy
        pde = (u.dt2
               + (sigma_x_fn + sigma_y_fn) * u.dt
               + sigma_x_fn * sigma_y_fn * u
               - c_sq * u.laplace
               - phi_x.dx - phi_y.dy)
        stencil_u = Eq(u.forward, solve(pde, u.forward))

        # Auxiliary field updates (forward Euler):
        # phi_x_t = -sigma_x*phi_x + c^2*(sigma_y - sigma_x)*u_x
        # phi_y_t = -sigma_y*phi_y + c^2*(sigma_x - sigma_y)*u_y
        dt_sym = grid.stepping_dim.spacing
        eq_phi_x = Eq(phi_x.forward,
                      phi_x + dt_sym * (
                          -sigma_x_fn * phi_x
                          + c_sq * (sigma_y_fn - sigma_x_fn) * u.dx))
        eq_phi_y = Eq(phi_y.forward,
                      phi_y + dt_sym * (
                          -sigma_y_fn * phi_y
                          + c_sq * (sigma_x_fn - sigma_y_fn) * u.dy))

        bc_x0 = Eq(u[t_dim + 1, 0, y_dim], 0)
        bc_xN = Eq(u[t_dim + 1, Nx, y_dim], 0)
        bc_y0 = Eq(u[t_dim + 1, x_dim, 0], 0)
        bc_yN = Eq(u[t_dim + 1, x_dim, Ny], 0)
        eqs = [stencil_u, eq_phi_x, eq_phi_y,
               bc_x0, bc_xN, bc_y0, bc_yN]

    elif abc_type == 'first_order':
        # Clayton-Engquist first-order ABC: u_t + c * u_n = 0
        pde = u.dt2 - c_sq * u.laplace
        stencil = Eq(u.forward, solve(pde, u.forward),
                      subdomain=grid.interior)

        dx_sym = grid.spacing[0]
        dy_sym = grid.spacing[1]
        c_val = Constant(name='c_val')

        bc_x0 = Eq(u[t_dim + 1, 0, y_dim],
                    u[t_dim, 0, y_dim]
                    + c_val * dt / dx_sym * (u[t_dim, 1, y_dim] - u[t_dim, 0, y_dim]))
        bc_xN = Eq(u[t_dim + 1, Nx, y_dim],
                    u[t_dim, Nx, y_dim]
                    - c_val * dt / dx_sym * (u[t_dim, Nx, y_dim] - u[t_dim, Nx - 1, y_dim]))
        bc_y0 = Eq(u[t_dim + 1, x_dim, 0],
                    u[t_dim, x_dim, 0]
                    + c_val * dt / dy_sym * (u[t_dim, x_dim, 1] - u[t_dim, x_dim, 0]))
        bc_yN = Eq(u[t_dim + 1, x_dim, Ny],
                    u[t_dim, x_dim, Ny]
                    - c_val * dt / dy_sym * (u[t_dim, x_dim, Ny] - u[t_dim, x_dim, Ny - 1]))

        eqs = [stencil, bc_x0, bc_xN, bc_y0, bc_yN]

    elif abc_type in ('higdon', 'habc'):
        # For Higdon and HABC: solve wave equation in interior,
        # then apply Higdon corrections as a post-processing step.
        pde = u.dt2 - c_sq * u.laplace
        stencil = Eq(u.forward, solve(pde, u.forward),
                      subdomain=grid.interior)

        # Dirichlet at boundaries (will be overwritten by Higdon)
        bc_x0 = Eq(u[t_dim + 1, 0, y_dim], 0)
        bc_xN = Eq(u[t_dim + 1, Nx, y_dim], 0)
        bc_y0 = Eq(u[t_dim + 1, x_dim, 0], 0)
        bc_yN = Eq(u[t_dim + 1, x_dim, Ny], 0)
        eqs = [stencil, bc_x0, bc_xN, bc_y0, bc_yN]

        if abc_type == 'habc':
            habc_weights = create_habc_weights(pad_width)

    else:  # dirichlet
        pde = u.dt2 - c_sq * u.laplace
        stencil = Eq(u.forward, solve(pde, u.forward))

        bc_x0 = Eq(u[t_dim + 1, 0, y_dim], 0)
        bc_xN = Eq(u[t_dim + 1, Nx, y_dim], 0)
        bc_y0 = Eq(u[t_dim + 1, x_dim, 0], 0)
        bc_yN = Eq(u[t_dim + 1, x_dim, Ny], 0)
        eqs = [stencil, bc_x0, bc_xN, bc_y0, bc_yN]

    # Create operator
    op = Operator(eqs)

    # Build operator kwargs
    op_kwargs = {'time_m': 1, 'time_M': 1, 'dt': dt, 'c_sq': c**2}
    if abc_type == 'first_order':
        op_kwargs['c_val'] = c

    # History storage
    if save_history:
        u_history = np.zeros((Nt + 1, Nx + 1, Ny + 1))
        u_history[0, :, :] = u.data[0, :, :]
        u_history[1, :, :] = u.data[1, :, :]
        t_history = np.linspace(0, T, Nt + 1)
    else:
        u_history = None
        t_history = None

    # Time stepping
    for n in range(2, Nt + 1):
        op.apply(**op_kwargs)

        # Post-processing for Higdon/HABC (before buffer swap)
        if abc_type == 'higdon':
            _apply_higdon_bc(u.data, Nx, Ny, c, dt, dx, dy)
        elif abc_type == 'habc':
            _apply_habc_correction(u.data, Nx, Ny, c, dt, dx, dy,
                                   pad_width, habc_weights)

        # Buffer swap for u
        u.data[0, :, :] = u.data[1, :, :]
        u.data[1, :, :] = u.data[2, :, :]

        # PML: swap auxiliary field buffers
        if phi_x is not None:
            phi_x.data[1, :, :] = phi_x.data[0, :, :]
            phi_y.data[1, :, :] = phi_y.data[0, :, :]

        if save_history:
            u_history[n, :, :] = u.data[1, :, :]

    u_final = u.data[1, :, :].copy()

    has_layer = abc_type in ('damping', 'pml', 'habc')
    return ABCResult(
        u=u_final,
        x=x_coords,
        y=y_coords,
        t=T,
        dt=dt,
        abc_type=abc_type,
        pad_width=pad_width if has_layer else 0,
        u_history=u_history,
        t_history=t_history,
        C=C_actual,
    )


def measure_reflection(
    result_abc: ABCResult,
    result_ref: ABCResult | None = None,
    inner_fraction: float = 0.3,
) -> float:
    """Compute reflection coefficient from ABC result.

    Measures the energy remaining in the interior after the wavefront
    has had time to reach the boundaries. If a reference solution on a
    larger domain is provided, computes the relative error.

    Parameters
    ----------
    result_abc : ABCResult
        Solution with ABC applied
    result_ref : ABCResult, optional
        Reference solution (e.g., on a much larger domain). If None,
        uses the energy in the interior as a proxy.
    inner_fraction : float
        Fraction of domain to consider as "interior" for measurement

    Returns
    -------
    float
        Reflection coefficient between 0 and 1
    """
    Nx = len(result_abc.x) - 1
    Ny = len(result_abc.y) - 1

    # Define interior region
    margin_x = int(Nx * (1 - inner_fraction) / 2)
    margin_y = int(Ny * (1 - inner_fraction) / 2)
    inner_x = slice(margin_x, Nx - margin_x + 1)
    inner_y = slice(margin_y, Ny - margin_y + 1)

    u_inner = result_abc.u[inner_x, inner_y]

    if result_ref is not None:
        # Compare to reference solution
        u_ref_inner = result_ref.u[inner_x, inner_y]
        energy_error = np.sqrt(np.sum((u_inner - u_ref_inner)**2))
        energy_ref = np.sqrt(np.sum(u_ref_inner**2))
        if energy_ref > 0:
            return float(min(energy_error / energy_ref, 1.0))
        return 0.0
    else:
        # Use energy ratio: reflected energy / total initial energy
        energy_inner = np.sum(u_inner**2)
        energy_total = np.sum(result_abc.u**2)
        if energy_total > 0:
            return float(min(energy_inner / energy_total, 1.0))
        return 0.0


def compare_abc_methods(
    Lx: float = 2.0,
    Ly: float = 2.0,
    c: float = 1.0,
    Nx: int = 100,
    Ny: int = 100,
    T: float = 1.5,
    CFL: float = 0.5,
    I: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    methods: list[str] | None = None,
    pad_width: int = 20,
) -> dict[str, ABCResult]:
    """Run comparison across ABC methods on the same test problem.

    Parameters
    ----------
    Lx, Ly, c, Nx, Ny, T, CFL : float/int
        Problem parameters (see solve_wave_2d_abc)
    I : callable, optional
        Initial condition
    methods : list of str, optional
        ABC methods to compare. Default: all six methods.
    pad_width : int
        Width of absorbing layer for damping, PML, and HABC

    Returns
    -------
    dict
        Mapping from method name to ABCResult
    """
    if methods is None:
        methods = ['dirichlet', 'first_order', 'damping', 'pml',
                   'higdon', 'habc']

    results = {}
    for method in methods:
        results[method] = solve_wave_2d_abc(
            Lx=Lx, Ly=Ly, c=c, Nx=Nx, Ny=Ny, T=T, CFL=CFL,
            I=I, abc_type=method, pad_width=pad_width,
        )

    return results
