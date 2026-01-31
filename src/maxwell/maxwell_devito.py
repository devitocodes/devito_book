"""FDTD Maxwell's Equations Solver using Devito DSL.

This module implements the Finite-Difference Time-Domain (FDTD) method
for solving Maxwell's equations using the Yee grid scheme.

Maxwell's curl equations (time-harmonic free space):
    curl(E) = -μ₀ * ∂H/∂t      (Faraday's law)
    curl(H) = ε₀ * ∂E/∂t       (Ampère's law, no sources)

The Yee scheme staggers E and H in both space and time:
    - E fields at integer time steps (n*dt)
    - H fields at half-integer steps ((n+1/2)*dt)
    - E components at cell edges
    - H components at cell faces

Update equations (leapfrog):
    H^{n+1/2} = H^{n-1/2} - (dt/μ) * curl(E^n)
    E^{n+1} = E^n + (dt/ε) * curl(H^{n+1/2})

Stability condition (CFL):
    dt ≤ 1 / (c * sqrt(1/dx² + 1/dy² + 1/dz²))

In 1D: dt ≤ dx / c
In 2D: dt ≤ dx / (c * sqrt(2))  [assuming dx = dy]

References:
    - Yee, K.S. (1966). IEEE Trans. Antennas Propagat., 14(3), 302-307.
    - Taflove, A. & Hagness, S.C. (2005). "Computational Electrodynamics."

Usage:
    from src.maxwell import solve_maxwell_1d, solve_maxwell_2d

    # 1D plane wave propagation
    result = solve_maxwell_1d(L=1.0, Nx=200, T=3e-9, source_type='gaussian')

    # 2D cavity simulation
    result = solve_maxwell_2d(Lx=0.1, Ly=0.1, Nx=101, Ny=101, T=1e-9)
"""

from dataclasses import dataclass

import numpy as np

try:
    from devito import (
        Constant,
        Eq,
        Function,
        Grid,
        Operator,
        SpaceDimension,
        TimeFunction,
        solve,
    )
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False

# Physical constants
C0 = 299792458.0  # Speed of light in vacuum [m/s]
MU0 = 4.0 * np.pi * 1e-7  # Permeability of free space [H/m]
EPS0 = 8.854187817e-12  # Permittivity of free space [F/m]
ETA0 = np.sqrt(MU0 / EPS0)  # Impedance of free space ≈ 377 Ω


@dataclass
class MaxwellResult:
    """Results from 1D FDTD Maxwell solver.

    Attributes
    ----------
    Ey : np.ndarray
        Final electric field (y-component), shape (Nx,)
    Hz : np.ndarray
        Final magnetic field (z-component), shape (Nx,)
    x : np.ndarray
        Spatial coordinates, shape (Nx,)
    t : float
        Final simulation time [s]
    dt : float
        Time step used [s]
    Ey_history : np.ndarray or None
        Time history of Ey, shape (Nt, Nx)
    Hz_history : np.ndarray or None
        Time history of Hz, shape (Nt, Nx)
    t_history : np.ndarray or None
        Time values for history snapshots
    c : float
        Wave speed used [m/s]
    """
    Ey: np.ndarray
    Hz: np.ndarray
    x: np.ndarray
    t: float
    dt: float
    Ey_history: np.ndarray | None = None
    Hz_history: np.ndarray | None = None
    t_history: np.ndarray | None = None
    c: float = C0


@dataclass
class MaxwellResult2D:
    """Results from 2D FDTD Maxwell solver.

    Attributes
    ----------
    Ez : np.ndarray
        Final electric field (z-component for TMz), shape (Nx, Ny)
    Hx : np.ndarray
        Final magnetic field (x-component for TMz), shape (Nx, Ny)
    Hy : np.ndarray
        Final magnetic field (y-component for TMz), shape (Nx, Ny)
    x : np.ndarray
        x-coordinates, shape (Nx,)
    y : np.ndarray
        y-coordinates, shape (Ny,)
    t : float
        Final simulation time [s]
    dt : float
        Time step used [s]
    Ez_history : np.ndarray or None
        Time history of Ez, shape (nsnaps, Nx, Ny)
    t_history : np.ndarray or None
        Time values for history snapshots
    c : float
        Wave speed used [m/s]
    """
    Ez: np.ndarray
    Hx: np.ndarray
    Hy: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t: float
    dt: float
    Ez_history: np.ndarray | None = None
    t_history: np.ndarray | None = None
    c: float = C0


def solve_maxwell_1d(
    L: float = 1.0,
    Nx: int = 200,
    T: float = 3e-9,
    dt: float | None = None,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
    source_type: str = "gaussian",
    source_position: float | None = None,
    f0: float = 1e9,
    bc_left: str = "pec",
    bc_right: str = "pec",
    save_history: bool = False,
    save_every: int = 1,
) -> MaxwellResult:
    """Solve 1D Maxwell's equations using FDTD.

    Solves for Ey and Hz fields with propagation in the x-direction.
    The update equations are:

        Hz^{n+1/2}_i = Hz^{n-1/2}_i - (dt/μ*dx) * (Ey^n_{i+1} - Ey^n_i)
        Ey^{n+1}_i = Ey^n_i + (dt/ε*dx) * (Hz^{n+1/2}_i - Hz^{n+1/2}_{i-1})

    Parameters
    ----------
    L : float
        Domain length [m]
    Nx : int
        Number of grid points
    T : float
        Final simulation time [s]
    dt : float, optional
        Time step [s]. If None, computed from CFL condition.
    eps_r : float
        Relative permittivity (can be array for inhomogeneous media)
    mu_r : float
        Relative permeability (can be array for inhomogeneous media)
    source_type : str
        Source type: "gaussian", "sinusoidal", or "ricker"
    source_position : float, optional
        Source location [m]. Default: L/4
    f0 : float
        Source frequency [Hz]
    bc_left : str
        Left boundary condition: "pec" (E=0), "pmc" (H=0), or "abc"
    bc_right : str
        Right boundary condition: "pec", "pmc", or "abc"
    save_history : bool
        If True, save field history
    save_every : int
        Save history every N time steps

    Returns
    -------
    MaxwellResult
        Solution data including final fields and optional history

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

    # Grid spacing
    dx = L / (Nx - 1)

    # Wave speed in medium
    c = C0 / np.sqrt(eps_r * mu_r)

    # Time step from CFL condition
    if dt is None:
        dt = 0.99 * dx / c  # 99% of CFL limit

    # Number of time steps
    Nt = int(T / dt)

    # Material parameters (normalized)
    eps = EPS0 * eps_r
    mu = MU0 * mu_r

    # Update coefficients
    ce = dt / (eps * dx)  # E-field update coefficient
    ch = dt / (mu * dx)   # H-field update coefficient

    # Create Devito grid
    x_dim = SpaceDimension(name='x', spacing=Constant(name='h_x', value=dx))
    grid = Grid(extent=(L,), shape=(Nx,), dimensions=(x_dim,))

    # Create field functions
    # Ey at integer grid points, Hz at half-integer points
    Ey = TimeFunction(name='Ey', grid=grid, time_order=1, space_order=2)
    Hz = TimeFunction(name='Hz', grid=grid, time_order=1, space_order=2)

    # Initialize to zero
    Ey.data.fill(0.0)
    Hz.data.fill(0.0)

    # Set up source
    if source_position is None:
        source_position = L / 4
    src_idx = int(source_position / dx)

    # Create source waveform
    t_vals = np.arange(Nt) * dt

    if source_type == "gaussian":
        sigma = 1.0 / (4.0 * f0)
        t0 = 4.0 * sigma
        source = np.exp(-((t_vals - t0) / sigma) ** 2)
    elif source_type == "sinusoidal":
        omega = 2.0 * np.pi * f0
        t_ramp = 2.0 / f0
        ramp = np.minimum(t_vals / t_ramp, 1.0)
        source = ramp * np.sin(omega * t_vals)
    elif source_type == "ricker":
        t0 = 1.0 / f0
        pi_f0_t = np.pi * f0 * (t_vals - t0)
        source = (1.0 - 2.0 * pi_f0_t**2) * np.exp(-pi_f0_t**2)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    # Build update equations using central differences
    # H update: Hz^{n+1} = Hz^n - ch * (Ey^{n+1/2}[i+1] - Ey^{n+1/2}[i])
    # For leapfrog, we use forward difference on Ey for H update
    pde_H = Hz.dt + ch * Ey.dxr  # dxr is right-sided derivative

    # E update: Ey^{n+1} = Ey^n + ce * (Hz^{n+1/2}[i] - Hz^{n+1/2}[i-1])
    # Use left-sided derivative
    pde_E = Ey.dt - ce * Hz.forward.dxl  # dxl is left-sided derivative

    # Solve for forward values
    update_H = Eq(Hz.forward, solve(pde_H, Hz.forward))
    update_E = Eq(Ey.forward, solve(pde_E, Ey.forward))

    # Create operator
    op = Operator([update_H, update_E])

    # Storage for history
    if save_history:
        n_saves = Nt // save_every + 1
        Ey_history = np.zeros((n_saves, Nx))
        Hz_history = np.zeros((n_saves, Nx))
        t_history = np.zeros(n_saves)
        save_idx = 0
    else:
        Ey_history = None
        Hz_history = None
        t_history = None

    # Time stepping loop
    for n in range(Nt):
        # Inject source (soft source - add to existing field)
        Ey.data[0, src_idx] += source[n]

        # Apply operator for one time step
        op.apply(time_m=0, time_M=0, dt=dt)

        # Apply boundary conditions
        if bc_left == "pec":
            Ey.data[1, 0] = 0.0
        elif bc_left == "pmc":
            Hz.data[1, 0] = 0.0
        elif bc_left == "abc":
            # Simple first-order ABC
            Ey.data[1, 0] = Ey.data[0, 1]

        if bc_right == "pec":
            Ey.data[1, -1] = 0.0
        elif bc_right == "pmc":
            Hz.data[1, -1] = 0.0
        elif bc_right == "abc":
            Ey.data[1, -1] = Ey.data[0, -2]

        # Save history
        if save_history and n % save_every == 0:
            Ey_history[save_idx, :] = Ey.data[1, :].copy()
            Hz_history[save_idx, :] = Hz.data[1, :].copy()
            t_history[save_idx] = n * dt
            save_idx += 1

    # Extract final solution
    x_coords = np.linspace(0, L, Nx)

    return MaxwellResult(
        Ey=Ey.data[1, :].copy(),
        Hz=Hz.data[1, :].copy(),
        x=x_coords,
        t=T,
        dt=dt,
        Ey_history=Ey_history,
        Hz_history=Hz_history,
        t_history=t_history,
        c=c,
    )


def solve_maxwell_2d(
    Lx: float = 0.1,
    Ly: float = 0.1,
    Nx: int = 101,
    Ny: int = 101,
    T: float = 1e-9,
    dt: float | None = None,
    eps_r: float | np.ndarray = 1.0,
    mu_r: float | np.ndarray = 1.0,
    source_type: str = "gaussian",
    source_position: tuple[float, float] | None = None,
    f0: float = 3e9,
    bc_type: str = "pec",
    polarization: str = "TMz",
    nsnaps: int = 0,
) -> MaxwellResult2D:
    """Solve 2D Maxwell's equations using FDTD with TMz or TEz modes.

    For TMz mode (Ez, Hx, Hy):
        dHz/dt = 0  (no z-variation)
        dHx/dt = -(1/μ) * dEz/dy
        dHy/dt = (1/μ) * dEz/dx
        dEz/dt = (1/ε) * (dHy/dx - dHx/dy)

    Parameters
    ----------
    Lx : float
        Domain extent in x [m]
    Ly : float
        Domain extent in y [m]
    Nx : int
        Number of grid points in x
    Ny : int
        Number of grid points in y
    T : float
        Final simulation time [s]
    dt : float, optional
        Time step [s]. If None, computed from CFL.
    eps_r : float or np.ndarray
        Relative permittivity (scalar or field)
    mu_r : float or np.ndarray
        Relative permeability (scalar or field)
    source_type : str
        Source type: "gaussian", "sinusoidal", or "ricker"
    source_position : tuple, optional
        Source location (x, y) [m]. Default: center of domain.
    f0 : float
        Source frequency [Hz]
    bc_type : str
        Boundary condition: "pec" (perfect electric conductor),
        "pmc" (perfect magnetic conductor), or "abc" (absorbing)
    polarization : str
        "TMz" (Ez polarization) or "TEz" (Hz polarization)
    nsnaps : int
        Number of snapshots to save (0 = none, -1 = all)

    Returns
    -------
    MaxwellResult2D
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

    if polarization.upper() != "TMZ":
        raise NotImplementedError("Only TMz polarization implemented")

    # Grid spacing
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    # Wave speed
    if np.isscalar(eps_r) and np.isscalar(mu_r):
        c = C0 / np.sqrt(eps_r * mu_r)
    else:
        # Use minimum for CFL
        c = C0 / np.sqrt(np.max(eps_r) * np.max(mu_r))

    # Time step from CFL condition
    if dt is None:
        dt = 0.99 / (c * np.sqrt(1/dx**2 + 1/dy**2))

    # Number of time steps
    Nt = int(T / dt)

    # Create Devito grid
    x_dim = SpaceDimension(name='x', spacing=Constant(name='h_x', value=dx))
    y_dim = SpaceDimension(name='y', spacing=Constant(name='h_y', value=dy))
    grid = Grid(extent=(Lx, Ly), shape=(Nx, Ny), dimensions=(x_dim, y_dim))

    # Material parameters
    if np.isscalar(eps_r):
        eps = EPS0 * eps_r
    else:
        eps_field = Function(name='eps', grid=grid)
        eps_field.data[:] = EPS0 * eps_r
        eps = eps_field

    if np.isscalar(mu_r):
        mu = MU0 * mu_r
    else:
        mu_field = Function(name='mu', grid=grid)
        mu_field.data[:] = MU0 * mu_r
        mu = mu_field

    # Create field functions for TMz mode
    # Ez at integer grid points, Hx and Hy at staggered positions
    Ez = TimeFunction(name='Ez', grid=grid, time_order=1, space_order=2)
    Hx = TimeFunction(name='Hx', grid=grid, time_order=1, space_order=2)
    Hy = TimeFunction(name='Hy', grid=grid, time_order=1, space_order=2)

    # Initialize to zero
    Ez.data.fill(0.0)
    Hx.data.fill(0.0)
    Hy.data.fill(0.0)

    # Set up source
    if source_position is None:
        source_position = (Lx / 2, Ly / 2)
    src_ix = int(source_position[0] / dx)
    src_it = int(source_position[1] / dy)

    # Create source waveform
    t_vals = np.arange(Nt) * dt

    if source_type == "gaussian":
        sigma = 1.0 / (4.0 * f0)
        t0 = 4.0 * sigma
        source = np.exp(-((t_vals - t0) / sigma) ** 2)
    elif source_type == "sinusoidal":
        omega = 2.0 * np.pi * f0
        t_ramp = 2.0 / f0
        ramp = np.minimum(t_vals / t_ramp, 1.0)
        source = ramp * np.sin(omega * t_vals)
    elif source_type == "ricker":
        t0 = 1.0 / f0
        pi_f0_t = np.pi * f0 * (t_vals - t0)
        source = (1.0 - 2.0 * pi_f0_t**2) * np.exp(-pi_f0_t**2)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    # TMz update equations:
    # dHx/dt = -(1/μ) * dEz/dy
    # dHy/dt = (1/μ) * dEz/dx
    # dEz/dt = (1/ε) * (dHy/dx - dHx/dy)

    # Build PDEs
    pde_Hx = Hx.dt + (1/mu) * Ez.dyr  # Right-sided y derivative
    pde_Hy = Hy.dt - (1/mu) * Ez.dxr  # Right-sided x derivative
    pde_Ez = Ez.dt - (1/eps) * (Hy.forward.dxl - Hx.forward.dyl)

    # Solve for forward values
    update_Hx = Eq(Hx.forward, solve(pde_Hx, Hx.forward))
    update_Hy = Eq(Hy.forward, solve(pde_Hy, Hy.forward))
    update_Ez = Eq(Ez.forward, solve(pde_Ez, Ez.forward))

    # Create operator
    op = Operator([update_Hx, update_Hy, update_Ez])

    # Storage for snapshots
    if nsnaps > 0:
        snap_interval = max(1, Nt // nsnaps)
        Ez_history = []
        t_history = []
    elif nsnaps == -1:
        snap_interval = 1
        Ez_history = []
        t_history = []
    else:
        Ez_history = None
        t_history = None

    # Time stepping loop
    for n in range(Nt):
        # Inject source (soft source)
        Ez.data[0, src_ix, src_it] += source[n]

        # Apply operator
        op.apply(time_m=0, time_M=0, dt=dt)

        # Apply boundary conditions
        if bc_type == "pec":
            # Ez = 0 at boundaries
            Ez.data[1, 0, :] = 0.0
            Ez.data[1, -1, :] = 0.0
            Ez.data[1, :, 0] = 0.0
            Ez.data[1, :, -1] = 0.0
        elif bc_type == "pmc":
            # Tangential H = 0 at boundaries
            Hx.data[1, 0, :] = 0.0
            Hx.data[1, -1, :] = 0.0
            Hy.data[1, :, 0] = 0.0
            Hy.data[1, :, -1] = 0.0
        elif bc_type == "abc":
            # Simple first-order Mur ABC
            Ez.data[1, 0, :] = Ez.data[0, 1, :]
            Ez.data[1, -1, :] = Ez.data[0, -2, :]
            Ez.data[1, :, 0] = Ez.data[0, :, 1]
            Ez.data[1, :, -1] = Ez.data[0, :, -2]

        # Save snapshot
        if nsnaps != 0 and n % snap_interval == 0:
            Ez_history.append(Ez.data[1, :, :].copy())
            t_history.append(n * dt)

    # Convert history to arrays
    if Ez_history is not None:
        Ez_history = np.array(Ez_history)
        t_history = np.array(t_history)

    # Extract final solution
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)

    return MaxwellResult2D(
        Ez=Ez.data[1, :, :].copy(),
        Hx=Hx.data[1, :, :].copy(),
        Hy=Hy.data[1, :, :].copy(),
        x=x_coords,
        y=y_coords,
        t=T,
        dt=dt,
        Ez_history=Ez_history,
        t_history=t_history,
        c=c,
    )


def compute_energy(
    Ey: np.ndarray,
    Hz: np.ndarray,
    dx: float,
    eps: float = EPS0,
    mu: float = MU0,
) -> float:
    """Compute total electromagnetic energy in 1D.

    The energy density is:
        u = (1/2) * ε * E² + (1/2) * μ * H²

    Parameters
    ----------
    Ey : np.ndarray
        Electric field, shape (Nx,)
    Hz : np.ndarray
        Magnetic field, shape (Nx,)
    dx : float
        Grid spacing [m]
    eps : float
        Permittivity [F/m]
    mu : float
        Permeability [H/m]

    Returns
    -------
    float
        Total electromagnetic energy [J/m²] (energy per unit area)
    """
    energy_E = 0.5 * eps * np.sum(Ey**2) * dx
    energy_H = 0.5 * mu * np.sum(Hz**2) * dx
    return energy_E + energy_H


def compute_energy_2d(
    Ez: np.ndarray,
    Hx: np.ndarray,
    Hy: np.ndarray,
    dx: float,
    dy: float,
    eps: float = EPS0,
    mu: float = MU0,
) -> float:
    """Compute total electromagnetic energy in 2D TMz mode.

    The energy density is:
        u = (1/2) * ε * Ez² + (1/2) * μ * (Hx² + Hy²)

    Parameters
    ----------
    Ez : np.ndarray
        Electric field, shape (Nx, Ny)
    Hx : np.ndarray
        Magnetic field x-component, shape (Nx, Ny)
    Hy : np.ndarray
        Magnetic field y-component, shape (Nx, Ny)
    dx : float
        Grid spacing in x [m]
    dy : float
        Grid spacing in y [m]
    eps : float
        Permittivity [F/m]
    mu : float
        Permeability [H/m]

    Returns
    -------
    float
        Total electromagnetic energy [J/m] (energy per unit length in z)
    """
    dA = dx * dy
    energy_E = 0.5 * eps * np.sum(Ez**2) * dA
    energy_H = 0.5 * mu * np.sum(Hx**2 + Hy**2) * dA
    return energy_E + energy_H


def compute_poynting_vector_1d(
    Ey: np.ndarray,
    Hz: np.ndarray,
) -> np.ndarray:
    """Compute Poynting vector (power flow) in 1D.

    For 1D TMz mode with Ey and Hz:
        S_x = Ey * Hz

    Parameters
    ----------
    Ey : np.ndarray
        Electric field, shape (Nx,)
    Hz : np.ndarray
        Magnetic field, shape (Nx,)

    Returns
    -------
    np.ndarray
        Poynting vector (x-component), shape (Nx,)
    """
    return Ey * Hz


def compute_poynting_vector_2d(
    Ez: np.ndarray,
    Hx: np.ndarray,
    Hy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Poynting vector components in 2D TMz mode.

    S = E × H
    For TMz (Ez, Hx, Hy):
        Sx = -Ez * Hy
        Sy = Ez * Hx

    Parameters
    ----------
    Ez : np.ndarray
        Electric field, shape (Nx, Ny)
    Hx : np.ndarray
        Magnetic field x-component, shape (Nx, Ny)
    Hy : np.ndarray
        Magnetic field y-component, shape (Nx, Ny)

    Returns
    -------
    Sx : np.ndarray
        Poynting vector x-component, shape (Nx, Ny)
    Sy : np.ndarray
        Poynting vector y-component, shape (Nx, Ny)
    """
    Sx = -Ez * Hy
    Sy = Ez * Hx
    return Sx, Sy
