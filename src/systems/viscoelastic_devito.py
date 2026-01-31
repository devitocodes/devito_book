"""3D Viscoelastic Wave Equations Solver using Devito DSL.

Solves the viscoelastic wave equations using the velocity-stress formulation
with memory variables to model frequency-dependent attenuation for both
P-waves (Qp) and S-waves (Qs).

Physical background:
    - Viscoelastic media exhibit both elastic response and viscous dissipation
    - Different attenuation for P-waves (Qp) and S-waves (Qs)
    - Memory variables capture the history-dependent stress response
    - Essential for accurate modeling of seismic wave propagation in real rocks

The velocity-stress formulation with attenuation:
    rho * dv/dt = div(tau)
    dtau/dt = lambda*(tau_ep/tau_s)*div(v)*I + mu*(tau_es/tau_s)*(grad(v) + grad(v)^T) + r
    dr/dt + (1/tau_s)*(r + ...) = 0

where tau_ep, tau_es, tau_s are relaxation times for P and S waves.

Applications:
    - Full waveform inversion in attenuating media
    - Seismic imaging with Q compensation
    - Earthquake simulation in realistic earth models
    - Marine seismics (water/sediment interfaces)

Usage:
    from src.systems import solve_viscoelastic_3d

    result = solve_viscoelastic_3d(
        extent=(200., 100., 100.),  # Domain size [m]
        shape=(201, 101, 101),       # Grid points
        T=30.0,                      # Final time [ms]
        vp=2.2, vs=1.2,              # Wave velocities [km/s]
        Qp=100.0, Qs=70.0,           # Quality factors
    )

References:
    - Robertson et al. (1994): Viscoelastic finite-difference modeling, GEOPHYSICS
    - Thorbecke, FDELMODC implementation documentation
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


__all__ = [
    "ViscoelasticResult",
    "compute_viscoelastic_relaxation_parameters",
    "create_damping_field_3d",
    "create_layered_model_3d",
    "ricker_wavelet_3d",
    "solve_viscoelastic_3d",
]


@dataclass
class ViscoelasticResult:
    """Results from the 3D Viscoelastic Wave Equations solver.

    Attributes
    ----------
    vx : np.ndarray
        Final x-velocity field, shape (Nx, Ny, Nz)
    vy : np.ndarray
        Final y-velocity field, shape (Nx, Ny, Nz)
    vz : np.ndarray
        Final z-velocity field, shape (Nx, Ny, Nz)
    tau_xx : np.ndarray
        Final normal stress in x, shape (Nx, Ny, Nz)
    tau_yy : np.ndarray
        Final normal stress in y, shape (Nx, Ny, Nz)
    tau_zz : np.ndarray
        Final normal stress in z, shape (Nx, Ny, Nz)
    tau_xy : np.ndarray
        Final shear stress xy, shape (Nx, Ny, Nz)
    tau_xz : np.ndarray
        Final shear stress xz, shape (Nx, Ny, Nz)
    tau_yz : np.ndarray
        Final shear stress yz, shape (Nx, Ny, Nz)
    x : np.ndarray
        x-coordinates
    y : np.ndarray
        y-coordinates
    z : np.ndarray
        z-coordinates
    t : float
        Final simulation time
    dt : float
        Time step used
    """
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    tau_xx: np.ndarray
    tau_yy: np.ndarray
    tau_zz: np.ndarray
    tau_xy: np.ndarray
    tau_xz: np.ndarray
    tau_yz: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    t: float
    dt: float


def ricker_wavelet_3d(t: np.ndarray, f0: float, t0: float = None) -> np.ndarray:
    """Generate a Ricker (Mexican hat) wavelet.

    Parameters
    ----------
    t : ndarray
        Time array
    f0 : float
        Dominant frequency
    t0 : float, optional
        Time shift. Default: 1.5/f0

    Returns
    -------
    ndarray
        Ricker wavelet values at times t
    """
    if t0 is None:
        t0 = 1.5 / f0

    pi_f0_t = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * pi_f0_t**2) * np.exp(-pi_f0_t**2)


def compute_viscoelastic_relaxation_parameters(
    Qp: float | np.ndarray,
    Qs: float | np.ndarray,
    f0: float,
) -> tuple:
    """Compute relaxation parameters for viscoelastic modeling.

    Computes the stress relaxation time (t_s) and strain relaxation times
    for P-waves (t_ep) and S-waves (t_es) from quality factors Qp and Qs
    at reference frequency f0.

    Parameters
    ----------
    Qp : float or ndarray
        Quality factor for P-waves. Higher = less attenuation.
        Use large value (e.g., 10000) for no P-wave attenuation.
    Qs : float or ndarray
        Quality factor for S-waves. Higher = less attenuation.
        Use 0 or very small value for fluid (no shear waves).
    f0 : float
        Reference frequency

    Returns
    -------
    t_s : float or ndarray
        Stress relaxation time
    t_ep : float or ndarray
        Strain relaxation time for P-waves
    t_es : float or ndarray
        Strain relaxation time for S-waves

    Notes
    -----
    The relationships follow Robertson et al. (1994):
        t_s = (sqrt(1 + 1/Qp^2) - 1/Qp) / f0
        t_ep = 1 / (f0^2 * t_s)
        t_es = (1 + f0*Qs*t_s) / (f0*Qs - f0^2*t_s)

    For Qs = 0 (fluid), t_es is set to t_ep (no shear attenuation).
    """
    Qp = np.asarray(Qp)
    Qs = np.asarray(Qs)

    # Stress relaxation time (based on Qp)
    t_s = (np.sqrt(1.0 + 1.0 / Qp**2) - 1.0 / Qp) / f0

    # Strain relaxation time for P-waves
    t_ep = 1.0 / (f0**2 * t_s)

    # Strain relaxation time for S-waves
    # Handle Qs = 0 (fluid) case
    Qs_safe = np.where(Qs > 0, Qs, 1.0)  # Avoid division by zero

    denominator = f0 * Qs_safe - f0**2 * t_s
    # For numerical stability, ensure denominator doesn't go to zero
    denominator = np.where(np.abs(denominator) > 1e-10, denominator, 1e-10)

    t_es = (1.0 + f0 * Qs_safe * t_s) / denominator

    # Where Qs = 0, set t_es = t_ep (no shear attenuation)
    t_es = np.where(Qs > 0, t_es, t_ep)

    return t_s, t_ep, t_es


def create_damping_field_3d(
    grid: "Grid",
    nbl: int = 20,
    damping_coefficient: float = 0.05,
    space_order: int = 4,
) -> "Function":
    """Create a 3D absorbing boundary damping field.

    Parameters
    ----------
    grid : Grid
        Devito computational grid
    nbl : int
        Number of absorbing boundary layer points
    damping_coefficient : float
        Damping strength (higher = more absorption)
    space_order : int
        Spatial discretization order

    Returns
    -------
    Function
        Devito Function containing the damping field
    """
    damp = Function(name='damp', grid=grid, space_order=space_order)
    damp.data[:] = 1.0

    shape = grid.shape

    for dim in range(len(shape)):
        # Ensure nbl doesn't exceed half the grid dimension
        nbl_dim = min(nbl, shape[dim] // 2)
        if nbl_dim == 0:
            continue

        for i in range(nbl_dim):
            factor = 1.0 - damping_coefficient * ((nbl_dim - i) / nbl_dim)**2

            # Left boundary
            slices_left = [slice(None)] * len(shape)
            slices_left[dim] = i
            damp.data[tuple(slices_left)] *= factor

            # Right boundary
            slices_right = [slice(None)] * len(shape)
            slices_right[dim] = shape[dim] - 1 - i
            damp.data[tuple(slices_right)] *= factor

    return damp


def create_layered_model_3d(
    shape: tuple[int, int, int],
    vp_layers: list[float] = None,
    vs_layers: list[float] = None,
    Qp_layers: list[float] = None,
    Qs_layers: list[float] = None,
    rho_layers: list[float] = None,
    layer_depths: list[float] = None,
) -> tuple[np.ndarray, ...]:
    """Create a layered 3D model for viscoelastic simulation.

    Parameters
    ----------
    shape : tuple
        Grid shape (Nx, Ny, Nz)
    vp_layers : list, optional
        P-wave velocities for each layer
    vs_layers : list, optional
        S-wave velocities for each layer
    Qp_layers : list, optional
        P-wave quality factors for each layer
    Qs_layers : list, optional
        S-wave quality factors for each layer
    rho_layers : list, optional
        Densities for each layer
    layer_depths : list, optional
        Depth indices where each layer starts (in z-direction)

    Returns
    -------
    vp : ndarray
        P-wave velocity field
    vs : ndarray
        S-wave velocity field
    Qp : ndarray
        P-wave quality factor field
    Qs : ndarray
        S-wave quality factor field
    rho : ndarray
        Density field
    """
    Nx, Ny, Nz = shape

    # Default: 3-layer model (water, sediment, rock)
    if vp_layers is None:
        vp_layers = [1.52, 1.6, 2.2]  # km/s
    if vs_layers is None:
        vs_layers = [0.0, 0.4, 1.2]   # km/s (0 = fluid)
    if Qp_layers is None:
        Qp_layers = [10000., 40., 100.]
    if Qs_layers is None:
        Qs_layers = [0., 30., 70.]    # 0 = fluid (no shear)
    if rho_layers is None:
        rho_layers = [1.05, 1.3, 2.0]  # g/cm^3
    if layer_depths is None:
        # Default: layers at 0%, 50%, 54% depth
        layer_depths = [0, int(0.5 * Nz), int(0.5 * Nz) + 4]

    # Initialize arrays
    vp = np.zeros(shape, dtype=np.float32)
    vs = np.zeros(shape, dtype=np.float32)
    Qp = np.zeros(shape, dtype=np.float32)
    Qs = np.zeros(shape, dtype=np.float32)
    rho = np.zeros(shape, dtype=np.float32)

    # Use the minimum length across all layer arrays to avoid index errors
    nlayers = min(
        len(vp_layers), len(vs_layers), len(Qp_layers),
        len(Qs_layers), len(rho_layers), len(layer_depths)
    )

    # Fill layers
    for i in range(nlayers):
        z_start = layer_depths[i]
        z_end = layer_depths[i + 1] if i < nlayers - 1 else Nz

        vp[:, :, z_start:z_end] = vp_layers[i]
        vs[:, :, z_start:z_end] = vs_layers[i]
        Qp[:, :, z_start:z_end] = Qp_layers[i]
        Qs[:, :, z_start:z_end] = Qs_layers[i]
        rho[:, :, z_start:z_end] = rho_layers[i]

    return vp, vs, Qp, Qs, rho


def solve_viscoelastic_3d(
    extent: tuple[float, float, float] = (200., 100., 100.),
    shape: tuple[int, int, int] = (101, 51, 51),
    T: float = 30.0,
    dt: float | None = None,
    vp: float | np.ndarray = 2.0,
    vs: float | np.ndarray = 1.0,
    rho: float | np.ndarray = 2.0,
    Qp: float | np.ndarray = 100.0,
    Qs: float | np.ndarray = 50.0,
    f0: float = 0.12,
    space_order: int = 4,
    src_coords: tuple[float, float, float] | None = None,
    nbl: int = 20,
    use_damp: bool = True,
    dt_scale: float = 0.9,
) -> ViscoelasticResult:
    """Solve the 3D viscoelastic wave equations with attenuation.

    Implements the velocity-stress formulation with memory variables
    following Robertson et al. (1994). The system models both P-wave
    and S-wave attenuation through separate quality factors Qp and Qs.

    The equations are:
        dv/dt = (1/rho) * div(tau)
        dtau/dt = lambda*(t_ep/t_s)*div(v)*I + mu*(t_es/t_s)*strain - r
        dr/dt + (1/t_s)*(r + ...) = 0

    where:
        - tau is the stress tensor
        - r is the memory tensor
        - t_s, t_ep, t_es are relaxation times
        - strain = grad(v) + grad(v)^T

    Parameters
    ----------
    extent : tuple
        Domain size (Lx, Ly, Lz) [m]
    shape : tuple
        Number of grid points (Nx, Ny, Nz)
    T : float
        Final simulation time [ms]
    dt : float, optional
        Time step. If None, computed from CFL.
    vp : float or ndarray
        P-wave velocity [km/s]
    vs : float or ndarray
        S-wave velocity [km/s]. Use 0 for fluid layers.
    rho : float or ndarray
        Density [g/cm^3]
    Qp : float or ndarray
        P-wave quality factor. Use large value for no attenuation.
    Qs : float or ndarray
        S-wave quality factor. Use 0 for fluid (no shear).
    f0 : float
        Reference frequency [kHz]
    space_order : int
        Spatial discretization order
    src_coords : tuple, optional
        Source coordinates (x, y, z). Default: center-top.
    nbl : int
        Number of absorbing boundary layer points
    use_damp : bool
        Whether to apply absorbing boundary damping
    dt_scale : float
        Factor to reduce dt below CFL limit for stability

    Returns
    -------
    ViscoelasticResult
        Solution data including velocity and stress fields

    Raises
    ------
    ImportError
        If Devito is not installed

    Notes
    -----
    The viscoelastic wave equation can be unstable with the standard
    elastic CFL condition. A smaller dt (dt_scale < 1) is often needed.

    References
    ----------
    Robertson et al. (1994): Viscoelastic finite-difference modeling, GEOPHYSICS
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    Lx, Ly, Lz = extent
    Nx, Ny, Nz = shape

    # Create grid
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)
    h = min(dx, dy, dz)

    x_dim = SpaceDimension(name='x', spacing=Constant(name='h_x', value=dx))
    y_dim = SpaceDimension(name='y', spacing=Constant(name='h_y', value=dy))
    z_dim = SpaceDimension(name='z', spacing=Constant(name='h_z', value=dz))
    grid = Grid(extent=extent, shape=shape, dimensions=(x_dim, y_dim, z_dim),
                dtype=np.float32)

    # Handle scalar or array parameters
    vp_arr = np.asarray(vp, dtype=np.float32)
    vs_arr = np.asarray(vs, dtype=np.float32)
    rho_arr = np.asarray(rho, dtype=np.float32)
    Qp_arr = np.asarray(Qp, dtype=np.float32)
    Qs_arr = np.asarray(Qs, dtype=np.float32)

    if vp_arr.ndim == 0:
        vp_arr = np.full(shape, vp_arr, dtype=np.float32)
    if vs_arr.ndim == 0:
        vs_arr = np.full(shape, vs_arr, dtype=np.float32)
    if rho_arr.ndim == 0:
        rho_arr = np.full(shape, rho_arr, dtype=np.float32)
    if Qp_arr.ndim == 0:
        Qp_arr = np.full(shape, Qp_arr, dtype=np.float32)
    if Qs_arr.ndim == 0:
        Qs_arr = np.full(shape, Qs_arr, dtype=np.float32)

    vp_max = float(vp_arr.max())

    # CFL condition (with safety factor for viscoelastic stability)
    if dt is None:
        dt = h / (np.sqrt(3) * vp_max) * dt_scale

    Nt = int(T / dt)

    # Compute Lame parameters
    mu_arr = rho_arr * vs_arr**2
    lam_arr = rho_arr * vp_arr**2 - 2 * mu_arr
    b_arr = 1.0 / rho_arr  # Buoyancy

    # Compute relaxation parameters
    t_s_arr, t_ep_arr, t_es_arr = compute_viscoelastic_relaxation_parameters(
        Qp_arr, Qs_arr, f0
    )

    # Create Devito Functions for material parameters
    lam_fn = Function(name='l', grid=grid, space_order=space_order)
    mu_fn = Function(name='mu', grid=grid, space_order=space_order)
    b_fn = Function(name='b', grid=grid, space_order=space_order)

    lam_fn.data[:] = lam_arr
    mu_fn.data[:] = mu_arr
    b_fn.data[:] = b_arr

    # Relaxation time Functions
    t_s_fn = Function(name='t_s', grid=grid, space_order=space_order)
    t_ep_fn = Function(name='t_ep', grid=grid, space_order=space_order)
    t_es_fn = Function(name='t_es', grid=grid, space_order=space_order)

    t_s_fn.data[:] = t_s_arr
    t_ep_fn.data[:] = t_ep_arr
    t_es_fn.data[:] = t_es_arr

    # Damping for absorbing boundaries
    if use_damp:
        damp = create_damping_field_3d(grid, nbl, space_order=space_order)
    else:
        damp = Function(name='damp', grid=grid, space_order=space_order)
        damp.data[:] = 1.0

    # Create velocity (vector), stress (tensor), and memory (tensor) fields
    v = VectorTimeFunction(name='v', grid=grid, time_order=1,
                           space_order=space_order)
    tau = TensorTimeFunction(name='t', grid=grid, time_order=1,
                             space_order=space_order)
    r = TensorTimeFunction(name='r', grid=grid, time_order=1,
                           space_order=space_order)

    # Initialize fields to zero
    for i in range(3):
        v[i].data.fill(0.)
    for i in range(3):
        for j in range(3):
            tau[i, j].data.fill(0.)
            r[i, j].data.fill(0.)

    # Viscoelastic wave equations

    # Particle velocity: dv/dt = b * div(tau)
    pde_v = v.dt - b_fn * div(tau)
    u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

    # Strain tensor: e = grad(v) + grad(v)^T
    e = grad(v.forward) + grad(v.forward).transpose(inner=False)

    # Stress equation with relaxation:
    # dtau/dt = lam * (t_ep/t_s) * div(v) * I + mu * (t_es/t_s) * e - r
    pde_tau = (
        tau.dt
        - r.forward
        - lam_fn * (t_ep_fn / t_s_fn) * diag(div(v.forward))
        - mu_fn * (t_es_fn / t_s_fn) * e
    )
    u_tau = Eq(tau.forward, damp * solve(pde_tau, tau.forward))

    # Memory variable equation:
    # dr/dt + (1/t_s) * (r + lam*(t_ep/t_s - 1)*div(v)*I + mu*(t_es/t_s - 1)*e) = 0
    pde_r = (
        r.dt
        + (1.0 / t_s_fn) * (
            r
            + lam_fn * (t_ep_fn / t_s_fn - 1.0) * diag(div(v.forward))
            + mu_fn * (t_es_fn / t_s_fn - 1.0) * e
        )
    )
    u_r = Eq(r.forward, damp * solve(pde_r, r.forward))

    # Create operator
    op = Operator([u_v, u_r, u_tau])

    # Source setup
    if src_coords is None:
        src_coords = (Lx / 2, Ly / 2, 0.35 * Lz)  # Near top, center

    t_vals = np.arange(0, T, dt)
    src_wavelet = ricker_wavelet_3d(t_vals, f0)

    # Find source grid indices
    src_ix = max(0, min(int(src_coords[0] / dx), Nx - 1))
    src_it = max(0, min(int(src_coords[1] / dy), Ny - 1))
    src_iz = max(0, min(int(src_coords[2] / dz), Nz - 1))

    s = grid.stepping_dim.spacing  # Symbolic time step

    # Run simulation with explosive source (inject into diagonal stresses)
    for n in range(Nt):
        if n < len(src_wavelet):
            # Explosive source: inject into normal stress components
            src_val = dt * src_wavelet[n]
            tau[0, 0].data[0, src_ix, src_it, src_iz] += src_val
            tau[1, 1].data[0, src_ix, src_it, src_iz] += src_val
            tau[2, 2].data[0, src_ix, src_it, src_iz] += src_val

        op.apply(time_m=0, time_M=0, dt=dt)

    # Create coordinate arrays
    x_coords = np.linspace(0.0, Lx, Nx)
    y_coords = np.linspace(0.0, Ly, Ny)
    z_coords = np.linspace(0.0, Lz, Nz)

    # Extract results
    return ViscoelasticResult(
        vx=v[0].data[0, :, :, :].copy(),
        vy=v[1].data[0, :, :, :].copy(),
        vz=v[2].data[0, :, :, :].copy(),
        tau_xx=tau[0, 0].data[0, :, :, :].copy(),
        tau_yy=tau[1, 1].data[0, :, :, :].copy(),
        tau_zz=tau[2, 2].data[0, :, :, :].copy(),
        tau_xy=tau[0, 1].data[0, :, :, :].copy(),
        tau_xz=tau[0, 2].data[0, :, :, :].copy(),
        tau_yz=tau[1, 2].data[0, :, :, :].copy(),
        x=x_coords,
        y=y_coords,
        z=z_coords,
        t=T,
        dt=dt,
    )
