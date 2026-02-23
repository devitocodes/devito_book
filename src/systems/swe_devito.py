"""2D Shallow Water Equations Solver using Devito DSL.

Solves the 2D Shallow Water Equations (SWE):

    deta/dt + dM/dx + dN/dy = 0                                    (continuity)
    dM/dt + d(M^2/D)/dx + d(MN/D)/dy + gD*deta/dx + friction*M = 0 (x-momentum)
    dN/dt + d(MN/D)/dx + d(N^2/D)/dy + gD*deta/dy + friction*N = 0 (y-momentum)

where:
    - eta: wave height (surface elevation above mean sea level)
    - M, N: discharge fluxes in x and y directions (M = u*D, N = v*D)
    - D = h + eta: total water column depth
    - h: bathymetry (depth from mean sea level to seafloor)
    - g: gravitational acceleration
    - friction = g * alpha^2 * sqrt(M^2 + N^2) / D^(7/3)
    - alpha: Manning's roughness coefficient

The equations are discretized using the FTCS (Forward Time, Centered Space)
scheme with the solve() function to isolate forward time terms.

Applications:
    - Tsunami propagation modeling
    - Storm surge prediction
    - Dam break simulations
    - Coastal engineering

Usage:
    from src.systems import solve_swe

    result = solve_swe(
        Lx=100.0, Ly=100.0,  # Domain size [m]
        Nx=401, Ny=401,       # Grid points
        T=3.0,                # Final time [s]
        dt=1/4500,            # Time step [s]
        g=9.81,               # Gravity [m/s^2]
        alpha=0.025,          # Manning's roughness
        h0=50.0,              # Constant depth [m]
    )
"""

from dataclasses import dataclass

import numpy as np

try:
    from devito import (
        ConditionalDimension,
        Eq,
        Function,
        Grid,
        Operator,
        TimeFunction,
        solve,
        sqrt,
    )
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@dataclass
class SWEResult:
    """Results from the Shallow Water Equations solver.

    Attributes
    ----------
    eta : np.ndarray
        Final wave height field, shape (Ny, Nx)
    M : np.ndarray
        Final x-discharge flux, shape (Ny, Nx)
    N : np.ndarray
        Final y-discharge flux, shape (Ny, Nx)
    x : np.ndarray
        x-coordinates, shape (Nx,)
    y : np.ndarray
        y-coordinates, shape (Ny,)
    t : float
        Final simulation time
    dt : float
        Time step used
    eta_snapshots : np.ndarray or None
        Saved snapshots of eta, shape (nsnaps, Ny, Nx)
    t_snapshots : np.ndarray or None
        Time values for snapshots
    """
    eta: np.ndarray
    M: np.ndarray
    N: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t: float
    dt: float
    eta_snapshots: np.ndarray | None = None
    t_snapshots: np.ndarray | None = None


def create_swe_operator(
    eta: "TimeFunction",
    M: "TimeFunction",
    N: "TimeFunction",
    h: "Function",
    D: "Function",
    g: float,
    alpha: float,
    grid: "Grid",
    eta_save: "TimeFunction | None" = None,
) -> "Operator":
    """Create the Devito operator for the Shallow Water Equations.

    This function constructs the finite difference operator that solves
    the coupled system of three PDEs (continuity + two momentum equations).

    Parameters
    ----------
    eta : TimeFunction
        Wave height field (surface elevation)
    M : TimeFunction
        Discharge flux in x-direction
    N : TimeFunction
        Discharge flux in y-direction
    h : Function
        Bathymetry (static field, depth to seafloor)
    D : Function
        Total water depth (D = h + eta)
    g : float
        Gravitational acceleration [m/s^2]
    alpha : float
        Manning's roughness coefficient
    grid : Grid
        Devito computational grid
    eta_save : TimeFunction, optional
        TimeFunction for saving snapshots at reduced frequency

    Returns
    -------
    Operator
        Devito operator that advances the solution by one time step
    """
    # Friction term: represents energy loss due to seafloor interaction
    # friction = g * alpha^2 * sqrt(M^2 + N^2) / D^(7/3)
    friction_M = g * alpha**2 * sqrt(M**2 + N**2) / D**(7.0/3.0)

    # Continuity equation: deta/dt + dM/dx + dN/dy = 0
    # Using centered differences for spatial derivatives
    pde_eta = Eq(eta.dt + M.dxc + N.dyc)

    # x-Momentum equation:
    # dM/dt + d(M^2/D)/dx + d(MN/D)/dy + gD*deta/dx + friction*M = 0
    # Note: We use eta.forward for the pressure gradient term to improve stability
    pde_M = Eq(
        M.dt
        + (M**2 / D).dxc
        + (M * N / D).dyc
        + g * D * eta.forward.dxc
        + friction_M * M
    )

    # y-Momentum equation:
    # dN/dt + d(MN/D)/dx + d(N^2/D)/dy + gD*deta/dy + friction*N = 0
    # Note: Uses M.forward to maintain temporal consistency
    friction_N = g * alpha**2 * sqrt(M.forward**2 + N**2) / D**(7.0/3.0)
    pde_N = Eq(
        N.dt
        + (M.forward * N / D).dxc
        + (N**2 / D).dyc
        + g * D * eta.forward.dyc
        + friction_N * N
    )

    # Use solve() to isolate the forward time terms
    stencil_eta = solve(pde_eta, eta.forward)
    stencil_M = solve(pde_M, M.forward)
    stencil_N = solve(pde_N, N.forward)

    # Update equations for interior points only (avoiding boundaries)
    update_eta = Eq(eta.forward, stencil_eta, subdomain=grid.interior)
    update_M = Eq(M.forward, stencil_M, subdomain=grid.interior)
    update_N = Eq(N.forward, stencil_N, subdomain=grid.interior)

    # Update total water depth D = h + eta
    eq_D = Eq(D, eta.forward + h)

    # Build equation list
    equations = [update_eta, update_M, update_N, eq_D]

    # Add snapshot saving if eta_save is provided
    if eta_save is not None:
        equations.append(Eq(eta_save, eta))

    return Operator(equations)


def solve_swe(
    Lx: float = 100.0,
    Ly: float = 100.0,
    Nx: int = 401,
    Ny: int = 401,
    T: float = 3.0,
    dt: float = 1/4500,
    g: float = 9.81,
    alpha: float = 0.025,
    h0: float | np.ndarray = 50.0,
    eta0: np.ndarray | None = None,
    M0: np.ndarray | None = None,
    N0: np.ndarray | None = None,
    nsnaps: int = 0,
) -> SWEResult:
    """Solve the 2D Shallow Water Equations using Devito.

    Parameters
    ----------
    Lx : float
        Domain extent in x-direction [m]
    Ly : float
        Domain extent in y-direction [m]
    Nx : int
        Number of grid points in x-direction
    Ny : int
        Number of grid points in y-direction
    T : float
        Final simulation time [s]
    dt : float
        Time step [s]
    g : float
        Gravitational acceleration [m/s^2]
    alpha : float
        Manning's roughness coefficient
    h0 : float or ndarray
        Bathymetry: either constant depth or 2D array (Ny, Nx)
    eta0 : ndarray, optional
        Initial wave height, shape (Ny, Nx). Default: Gaussian at center.
    M0 : ndarray, optional
        Initial x-discharge flux, shape (Ny, Nx). Default: 100 * eta0.
    N0 : ndarray, optional
        Initial y-discharge flux, shape (Ny, Nx). Default: zeros.
    nsnaps : int
        Number of snapshots to save (0 = no snapshots)

    Returns
    -------
    SWEResult
        Solution data including final fields and optional snapshots

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

    # Compute number of time steps
    Nt = int(T / dt)

    # Create coordinate arrays
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Set up bathymetry
    if isinstance(h0, (int, float)):
        h_array = h0 * np.ones((Ny, Nx), dtype=np.float32)
    else:
        h_array = np.asarray(h0, dtype=np.float32)

    # Default initial conditions
    if eta0 is None:
        # Gaussian pulse at center
        eta0 = 0.5 * np.exp(-((X - Lx/2)**2 / 10) - ((Y - Ly/2)**2 / 10))
    eta0 = np.asarray(eta0, dtype=np.float32)

    if M0 is None:
        M0 = 100.0 * eta0
    M0 = np.asarray(M0, dtype=np.float32)

    if N0 is None:
        N0 = np.zeros_like(M0)
    N0 = np.asarray(N0, dtype=np.float32)

    # Create Devito grid
    grid = Grid(shape=(Ny, Nx), extent=(Ly, Lx), dtype=np.float32)

    # Create TimeFunction fields for the three unknowns
    eta = TimeFunction(name='eta', grid=grid, space_order=2)
    M = TimeFunction(name='M', grid=grid, space_order=2)
    N = TimeFunction(name='N', grid=grid, space_order=2)

    # Create static Functions for bathymetry and total depth
    h = Function(name='h', grid=grid)
    D = Function(name='D', grid=grid)

    # Set initial conditions
    eta.data[0, :, :] = eta0
    M.data[0, :, :] = M0
    N.data[0, :, :] = N0
    h.data[:] = h_array
    D.data[:] = eta0 + h_array

    # Set up snapshot saving with ConditionalDimension
    eta_save = None
    if nsnaps > 0:
        factor = max(1, round(Nt / nsnaps))
        time_subsampled = ConditionalDimension(
            't_sub', parent=grid.time_dim, factor=factor
        )
        eta_save = TimeFunction(
            name='eta_save', grid=grid, space_order=2,
            save=nsnaps, time_dim=time_subsampled
        )

    # Create the operator
    op = create_swe_operator(eta, M, N, h, D, g, alpha, grid, eta_save)

    # Apply the operator
    op.apply(
        eta=eta, M=M, N=N, D=D, h=h,
        time=Nt - 2, dt=dt,
        **({"eta_save": eta_save} if eta_save is not None else {})
    )

    # Extract results
    eta_final = eta.data[0, :, :].copy()
    M_final = M.data[0, :, :].copy()
    N_final = N.data[0, :, :].copy()

    # Extract snapshots if saved
    eta_snapshots = None
    t_snapshots = None
    if eta_save is not None:
        eta_snapshots = eta_save.data.copy()
        t_snapshots = np.linspace(0, T, nsnaps)

    return SWEResult(
        eta=eta_final,
        M=M_final,
        N=N_final,
        x=x,
        y=y,
        t=T,
        dt=dt,
        eta_snapshots=eta_snapshots,
        t_snapshots=t_snapshots,
    )


def gaussian_tsunami_source(
    X: np.ndarray,
    Y: np.ndarray,
    x0: float,
    y0: float,
    amplitude: float = 0.5,
    sigma_x: float = 10.0,
    sigma_y: float = 10.0,
) -> np.ndarray:
    """Create a Gaussian tsunami source.

    Parameters
    ----------
    X : ndarray
        X-coordinate meshgrid
    Y : ndarray
        Y-coordinate meshgrid
    x0 : float
        Source center x-coordinate
    y0 : float
        Source center y-coordinate
    amplitude : float
        Peak amplitude [m]
    sigma_x : float
        Width parameter in x-direction
    sigma_y : float
        Width parameter in y-direction

    Returns
    -------
    ndarray
        Initial wave height field
    """
    return amplitude * np.exp(
        -((X - x0)**2 / sigma_x) - ((Y - y0)**2 / sigma_y)
    )


def seamount_bathymetry(
    X: np.ndarray,
    Y: np.ndarray,
    h_base: float = 50.0,
    x0: float = None,
    y0: float = None,
    height: float = 45.0,
    sigma: float = 20.0,
) -> np.ndarray:
    """Create bathymetry with a seamount.

    Parameters
    ----------
    X : ndarray
        X-coordinate meshgrid
    Y : ndarray
        Y-coordinate meshgrid
    h_base : float
        Base ocean depth [m]
    x0 : float
        Seamount center x-coordinate (default: domain center)
    y0 : float
        Seamount center y-coordinate (default: domain center)
    height : float
        Seamount height above seafloor [m]
    sigma : float
        Width parameter for Gaussian seamount

    Returns
    -------
    ndarray
        Bathymetry array
    """
    if x0 is None:
        x0 = (X.max() + X.min()) / 2
    if y0 is None:
        y0 = (Y.max() + Y.min()) / 2

    h = h_base * np.ones_like(X)
    h -= height * np.exp(-((X - x0)**2 / sigma) - ((Y - y0)**2 / sigma))
    return h


def tanh_bathymetry(
    X: np.ndarray,
    Y: np.ndarray,
    h_deep: float = 50.0,
    h_shallow: float = 5.0,
    x_transition: float = 70.0,
    width: float = 8.0,
) -> np.ndarray:
    """Create bathymetry with tanh transition (coastal profile).

    Parameters
    ----------
    X : ndarray
        X-coordinate meshgrid
    Y : ndarray
        Y-coordinate meshgrid
    h_deep : float
        Deep water depth [m]
    h_shallow : float
        Shallow water depth [m]
    x_transition : float
        Location of transition
    width : float
        Width parameter for transition

    Returns
    -------
    ndarray
        Bathymetry array
    """
    return h_deep - (h_deep - h_shallow) * (
        0.5 * (1 + np.tanh((X - x_transition) / width))
    )
