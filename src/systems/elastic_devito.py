"""2D Elastic Wave Equations Solver using Devito DSL.

Solves the 2D Elastic Wave Equations in velocity-stress formulation:

    rho * dv/dt = div(tau)                                          (momentum)
    dtau/dt = lam * div(v) * I + mu * (grad(v) + grad(v)^T)        (stress)

where:
    - v: velocity vector (vx, vz) [m/s]
    - tau: stress tensor [[tau_xx, tau_xz], [tau_xz, tau_zz]] [Pa]
    - rho: density [kg/m^3]
    - lam: first Lame parameter [Pa]
    - mu: shear modulus (second Lame parameter) [Pa]

The P-wave and S-wave velocities are related to Lame parameters by:
    V_p = sqrt((lam + 2*mu) / rho)
    V_s = sqrt(mu / rho)

Applications:
    - Seismic wave propagation
    - Full waveform inversion (FWI)
    - Earthquake simulation
    - Non-destructive testing

Usage:
    from src.systems import solve_elastic_2d

    result = solve_elastic_2d(
        Lx=1500.0, Lz=1500.0,  # Domain size [m]
        Nx=201, Nz=201,         # Grid points
        T=300.0,                # Final time [ms]
        V_p=2.0, V_s=1.0,       # Wave velocities [km/s]
        rho=1.8,                # Density [g/cm^3]
    )
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
        TensorTimeFunction,
        VectorTimeFunction,
        diag,
        div,
        grad,
        solve,
    )
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@dataclass
class ElasticResult:
    """Results from the Elastic Wave Equations solver.

    Attributes
    ----------
    vx : np.ndarray
        Final x-velocity field, shape (Nx, Nz)
    vz : np.ndarray
        Final z-velocity field, shape (Nx, Nz)
    tau_xx : np.ndarray
        Final normal stress in x, shape (Nx, Nz)
    tau_zz : np.ndarray
        Final normal stress in z, shape (Nx, Nz)
    tau_xz : np.ndarray
        Final shear stress, shape (Nx, Nz)
    x : np.ndarray
        x-coordinates, shape (Nx,)
    z : np.ndarray
        z-coordinates, shape (Nz,)
    t : float
        Final simulation time
    dt : float
        Time step used
    vx_snapshots : np.ndarray or None
        Saved snapshots of vx, shape (nsnaps, Nx, Nz)
    vz_snapshots : np.ndarray or None
        Saved snapshots of vz, shape (nsnaps, Nx, Nz)
    t_snapshots : np.ndarray or None
        Time values for snapshots
    """
    vx: np.ndarray
    vz: np.ndarray
    tau_xx: np.ndarray
    tau_zz: np.ndarray
    tau_xz: np.ndarray
    x: np.ndarray
    z: np.ndarray
    t: float
    dt: float
    vx_snapshots: np.ndarray | None = None
    vz_snapshots: np.ndarray | None = None
    t_snapshots: np.ndarray | None = None


def compute_lame_parameters(V_p: float, V_s: float, rho: float) -> tuple[float, float]:
    """Compute Lame parameters from wave velocities and density.

    Parameters
    ----------
    V_p : float
        P-wave velocity
    V_s : float
        S-wave velocity
    rho : float
        Density

    Returns
    -------
    lam : float
        First Lame parameter
    mu : float
        Shear modulus (second Lame parameter)

    Notes
    -----
    The relationships are:
        mu = rho * V_s^2
        lam = rho * V_p^2 - 2*mu
    """
    mu = rho * V_s**2
    lam = rho * V_p**2 - 2 * mu
    return lam, mu


def create_elastic_operator(
    v: "VectorTimeFunction",
    tau: "TensorTimeFunction",
    lam: "Function | float",
    mu: "Function | float",
    ro: "Function | float",
    grid: "Grid",
) -> "Operator":
    """Create the Devito operator for the Elastic Wave Equations.

    This function constructs the finite difference operator that solves
    the coupled velocity-stress system using a staggered grid approach.

    Parameters
    ----------
    v : VectorTimeFunction
        Velocity vector field (vx, vz)
    tau : TensorTimeFunction
        Stress tensor field (symmetric)
    lam : Function or float
        First Lame parameter [Pa]
    mu : Function or float
        Shear modulus [Pa]
    ro : Function or float
        Inverse density (buoyancy) [m^3/kg], i.e., 1/rho
    grid : Grid
        Devito computational grid

    Returns
    -------
    Operator
        Devito operator that advances the solution by one time step

    Notes
    -----
    The equations are:
        rho * dv/dt = div(tau)
        dtau/dt = lam * div(v) * I + mu * (grad(v) + grad(v)^T)

    Using ro = 1/rho for efficiency:
        dv/dt = ro * div(tau)
        dtau/dt = lam * diag(div(v)) + mu * (grad(v) + grad(v)^T)
    """
    # First order elastic wave equation
    # Momentum equation: dv/dt = (1/rho) * div(tau)
    pde_v = v.dt - ro * div(tau)

    # Stress equation: dtau/dt = lam * tr(grad(v)) * I + mu * (grad(v) + grad(v)^T)
    # Note: tr(grad(v)) = div(v), and we use diag() to create the diagonal tensor
    pde_tau = (
        tau.dt
        - lam * diag(div(v.forward))
        - mu * (grad(v.forward) + grad(v.forward).transpose(inner=False))
    )

    # Time update using solve() to isolate forward terms
    u_v = Eq(v.forward, solve(pde_v, v.forward))
    u_tau = Eq(tau.forward, solve(pde_tau, tau.forward))

    return Operator([u_v, u_tau])


def create_elastic_operator_with_source(
    v: "VectorTimeFunction",
    tau: "TensorTimeFunction",
    lam: "Function | float",
    mu: "Function | float",
    ro: "Function | float",
    grid: "Grid",
    src_term: list,
) -> "Operator":
    """Create elastic operator with source injection terms.

    Parameters
    ----------
    v : VectorTimeFunction
        Velocity vector field (vx, vz)
    tau : TensorTimeFunction
        Stress tensor field (symmetric)
    lam : Function or float
        First Lame parameter [Pa]
    mu : Function or float
        Shear modulus [Pa]
    ro : Function or float
        Inverse density (buoyancy) [m^3/kg]
    grid : Grid
        Devito computational grid
    src_term : list
        List of source injection equations

    Returns
    -------
    Operator
        Devito operator with source injection
    """
    # First order elastic wave equation
    pde_v = v.dt - ro * div(tau)
    pde_tau = (
        tau.dt
        - lam * diag(div(v.forward))
        - mu * (grad(v.forward) + grad(v.forward).transpose(inner=False))
    )

    # Time updates
    u_v = Eq(v.forward, solve(pde_v, v.forward))
    u_tau = Eq(tau.forward, solve(pde_tau, tau.forward))

    return Operator([u_v, u_tau] + src_term)


def solve_elastic_2d(
    Lx: float = 1500.0,
    Lz: float = 1500.0,
    Nx: int = 201,
    Nz: int = 201,
    T: float = 300.0,
    dt: float | None = None,
    V_p: float = 2.0,
    V_s: float = 1.0,
    rho: float = 1.8,
    space_order: int = 2,
    src_coords: tuple[float, float] | None = None,
    src_f0: float = 0.01,
    nsnaps: int = 0,
) -> ElasticResult:
    """Solve the 2D Elastic Wave Equations using Devito.

    Parameters
    ----------
    Lx : float
        Domain extent in x-direction [m]
    Lz : float
        Domain extent in z-direction [m]
    Nx : int
        Number of grid points in x-direction
    Nz : int
        Number of grid points in z-direction
    T : float
        Final simulation time (in same units as derived from V_p)
    dt : float, optional
        Time step. If None, computed from CFL condition.
    V_p : float
        P-wave velocity (default: 2.0)
    V_s : float
        S-wave velocity (default: 1.0)
    rho : float
        Density (default: 1.8)
    space_order : int
        Spatial discretization order (default: 2)
    src_coords : tuple, optional
        Source coordinates (x, z). Default: center of domain.
    src_f0 : float
        Source dominant frequency (default: 0.01)
    nsnaps : int
        Number of snapshots to save (0 = no snapshots)

    Returns
    -------
    ElasticResult
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

    # Create grid with explicit spacing
    dx = Lx / (Nx - 1)
    dz = Lz / (Nz - 1)

    x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=dx))
    z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=dz))
    grid = Grid(extent=(Lx, Lz), shape=(Nx, Nz), dimensions=(x, z))

    # Compute time step from CFL condition if not provided
    if dt is None:
        # CFL condition: dt <= dx / (sqrt(2) * V_p)
        dt = min(dx, dz) / (np.sqrt(2) * V_p) * 0.9  # 90% of CFL limit

    # Compute number of time steps
    Nt = int(T / dt)

    # Compute Lame parameters
    lam, mu = compute_lame_parameters(V_p, V_s, rho)

    # Inverse density (buoyancy)
    ro = 1.0 / rho

    # Create velocity and stress fields
    v = VectorTimeFunction(name='v', grid=grid, space_order=space_order, time_order=1)
    tau = TensorTimeFunction(name='t', grid=grid, space_order=space_order, time_order=1)

    # Initialize fields to zero
    v[0].data.fill(0.)
    v[1].data.fill(0.)
    tau[0, 0].data.fill(0.)
    tau[0, 1].data.fill(0.)
    tau[1, 1].data.fill(0.)

    # Set up source
    if src_coords is None:
        src_coords = (Lx / 2, Lz / 2)

    # Create source wavelet using Ricker wavelet
    t_vals = np.arange(0, T, dt)
    t0 = 1.0 / src_f0
    src_wavelet = ricker_wavelet(t_vals, src_f0, t0)

    # Find source grid indices
    src_ix = int(src_coords[0] / dx)
    src_iz = int(src_coords[1] / dz)

    # Create operator without external source (we'll inject manually)
    op = create_elastic_operator(v, tau, lam, mu, ro, grid)

    # Run simulation with manual source injection
    # For explosive source, inject into diagonal stress components
    for n in range(Nt):
        # Inject source at current time step
        if n < len(src_wavelet):
            tau[0, 0].data[0, src_ix, src_iz] += src_wavelet[n]
            tau[1, 1].data[0, src_ix, src_iz] += src_wavelet[n]

        # Apply operator for one time step
        op.apply(time_m=0, time_M=0, dt=dt)

    # Create coordinate arrays for output
    x_coords = np.linspace(0.0, Lx, Nx)
    z_coords = np.linspace(0.0, Lz, Nz)

    # Extract results
    vx_final = v[0].data[0, :, :].copy()
    vz_final = v[1].data[0, :, :].copy()
    tau_xx_final = tau[0, 0].data[0, :, :].copy()
    tau_zz_final = tau[1, 1].data[0, :, :].copy()
    tau_xz_final = tau[0, 1].data[0, :, :].copy()

    return ElasticResult(
        vx=vx_final,
        vz=vz_final,
        tau_xx=tau_xx_final,
        tau_zz=tau_zz_final,
        tau_xz=tau_xz_final,
        x=x_coords,
        z=z_coords,
        t=T,
        dt=dt,
        vx_snapshots=None,
        vz_snapshots=None,
        t_snapshots=None,
    )


def solve_elastic_2d_varying(
    Lx: float = 3000.0,
    Lz: float = 3000.0,
    Nx: int = 301,
    Nz: int = 301,
    T: float = 2000.0,
    dt: float | None = None,
    lam_field: np.ndarray | None = None,
    mu_field: np.ndarray | None = None,
    b_field: np.ndarray | None = None,
    space_order: int = 8,
    src_coords: tuple[float, float] | None = None,
    src_f0: float = 0.015,
    nsnaps: int = 0,
) -> ElasticResult:
    """Solve elastic wave equation with spatially varying parameters.

    Parameters
    ----------
    Lx : float
        Domain extent in x-direction [m]
    Lz : float
        Domain extent in z-direction [m]
    Nx : int
        Number of grid points in x-direction
    Nz : int
        Number of grid points in z-direction
    T : float
        Final simulation time
    dt : float, optional
        Time step. If None, computed from CFL condition.
    lam_field : ndarray, optional
        First Lame parameter field, shape (Nx, Nz)
    mu_field : ndarray, optional
        Shear modulus field, shape (Nx, Nz)
    b_field : ndarray, optional
        Buoyancy (1/rho) field, shape (Nx, Nz)
    space_order : int
        Spatial discretization order (default: 8)
    src_coords : tuple, optional
        Source coordinates (x, z). Default: near top center.
    src_f0 : float
        Source dominant frequency
    nsnaps : int
        Number of snapshots to save (0 = no snapshots)

    Returns
    -------
    ElasticResult
        Solution data including final fields and optional snapshots
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    # Create grid
    dx = Lx / (Nx - 1)
    dz = Lz / (Nz - 1)

    x_dim = SpaceDimension(name='x', spacing=Constant(name='h_x', value=dx))
    z_dim = SpaceDimension(name='z', spacing=Constant(name='h_z', value=dz))
    grid = Grid(extent=(Lx, Lz), shape=(Nx, Nz), dimensions=(x_dim, z_dim))

    # Default material properties (layered medium)
    if lam_field is None or mu_field is None or b_field is None:
        lam_field, mu_field, b_field = create_layered_model(Nx, Nz)

    # Compute maximum P-wave velocity for CFL
    rho_field = 1.0 / b_field
    V_p_max = np.sqrt((lam_field + 2 * mu_field) / rho_field).max()

    # Compute time step from CFL condition if not provided
    if dt is None:
        dt = min(dx, dz) / (np.sqrt(2) * V_p_max) * 0.9

    # Number of time steps
    Nt = int(T / dt)

    # Create Devito Functions for material parameters
    lam = Function(name='lam', grid=grid, space_order=space_order)
    mu = Function(name='mu', grid=grid, space_order=space_order)
    b = Function(name='b', grid=grid, space_order=space_order)

    lam.data[:] = lam_field
    mu.data[:] = mu_field
    b.data[:] = b_field

    # Create velocity and stress fields
    v = VectorTimeFunction(name='v', grid=grid, space_order=space_order, time_order=1)
    tau = TensorTimeFunction(name='t', grid=grid, space_order=space_order, time_order=1)

    # Initialize fields to zero
    v[0].data.fill(0.)
    v[1].data.fill(0.)
    tau[0, 0].data.fill(0.)
    tau[0, 1].data.fill(0.)
    tau[1, 1].data.fill(0.)

    # Set up source
    if src_coords is None:
        src_coords = (Lx / 2, 10.0)

    # Create source wavelet
    t_vals = np.arange(0, T, dt)
    t0 = 1.0 / src_f0
    src_wavelet = ricker_wavelet(t_vals, src_f0, t0)

    # Find source grid indices
    src_ix = int(src_coords[0] / dx)
    src_iz = int(src_coords[1] / dz)

    # First order elastic wave equation with varying parameters
    pde_v = v.dt - b * div(tau)
    pde_tau = (
        tau.dt
        - lam * diag(div(v.forward))
        - mu * (grad(v.forward) + grad(v.forward).transpose(inner=False))
    )

    # Time updates
    u_v = Eq(v.forward, solve(pde_v, v.forward))
    u_tau = Eq(tau.forward, solve(pde_tau, tau.forward))

    op = Operator([u_v, u_tau])

    # Run simulation
    for n in range(Nt):
        # Inject source
        if n < len(src_wavelet):
            tau[0, 0].data[0, src_ix, src_iz] += dt * src_wavelet[n]
            tau[1, 1].data[0, src_ix, src_iz] += dt * src_wavelet[n]

        op.apply(time_m=0, time_M=0, dt=dt)

    # Create coordinate arrays
    x_coords = np.linspace(0.0, Lx, Nx)
    z_coords = np.linspace(0.0, Lz, Nz)

    # Extract results
    vx_final = v[0].data[0, :, :].copy()
    vz_final = v[1].data[0, :, :].copy()
    tau_xx_final = tau[0, 0].data[0, :, :].copy()
    tau_zz_final = tau[1, 1].data[0, :, :].copy()
    tau_xz_final = tau[0, 1].data[0, :, :].copy()

    return ElasticResult(
        vx=vx_final,
        vz=vz_final,
        tau_xx=tau_xx_final,
        tau_zz=tau_zz_final,
        tau_xz=tau_xz_final,
        x=x_coords,
        z=z_coords,
        t=T,
        dt=dt,
        vx_snapshots=None,
        vz_snapshots=None,
        t_snapshots=None,
    )


def ricker_wavelet(t: np.ndarray, f0: float, t0: float = None) -> np.ndarray:
    """Generate a Ricker (Mexican hat) wavelet.

    Parameters
    ----------
    t : ndarray
        Time array
    f0 : float
        Dominant frequency
    t0 : float, optional
        Time shift (default: 1/f0)

    Returns
    -------
    ndarray
        Ricker wavelet values at times t
    """
    if t0 is None:
        t0 = 1.0 / f0

    pi_f0_t = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * pi_f0_t**2) * np.exp(-pi_f0_t**2)


def create_layered_model(
    Nx: int,
    Nz: int,
    nlayers: int = 5,
    V_p_range: tuple[float, float] = (1.5, 4.0),
    V_s_range: tuple[float, float] = (0.5, 2.3),
    rho_range: tuple[float, float] = (1.0, 3.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple layered velocity model.

    Parameters
    ----------
    Nx : int
        Number of grid points in x
    Nz : int
        Number of grid points in z
    nlayers : int
        Number of horizontal layers
    V_p_range : tuple
        Range of P-wave velocities (min, max)
    V_s_range : tuple
        Range of S-wave velocities (min, max)
    rho_range : tuple
        Range of densities (min, max)

    Returns
    -------
    lam : ndarray
        First Lame parameter, shape (Nx, Nz)
    mu : ndarray
        Shear modulus, shape (Nx, Nz)
    b : ndarray
        Buoyancy (1/rho), shape (Nx, Nz)
    """
    V_p = np.linspace(V_p_range[0], V_p_range[1], nlayers)
    V_s = np.linspace(V_s_range[0], V_s_range[1], nlayers)
    rho = np.linspace(rho_range[0], rho_range[1], nlayers)

    lam_layers = rho * (V_p**2 - 2 * V_s**2)
    mu_layers = rho * V_s**2
    b_layers = 1.0 / rho

    # Create 2D arrays
    lam = np.zeros((Nx, Nz))
    mu = np.zeros((Nx, Nz))
    b = np.zeros((Nx, Nz))

    layer_thickness = Nz // nlayers
    for i in range(nlayers):
        z_start = i * layer_thickness
        z_end = (i + 1) * layer_thickness if i < nlayers - 1 else Nz
        lam[:, z_start:z_end] = lam_layers[i]
        mu[:, z_start:z_end] = mu_layers[i]
        b[:, z_start:z_end] = b_layers[i]

    return lam, mu, b


def compute_wave_velocities(
    lam: np.ndarray,
    mu: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute P-wave and S-wave velocities from Lame parameters.

    Parameters
    ----------
    lam : ndarray
        First Lame parameter
    mu : ndarray
        Shear modulus
    rho : ndarray
        Density

    Returns
    -------
    V_p : ndarray
        P-wave velocity
    V_s : ndarray
        S-wave velocity
    """
    V_p = np.sqrt((lam + 2 * mu) / rho)
    V_s = np.sqrt(mu / rho)
    return V_p, V_s
