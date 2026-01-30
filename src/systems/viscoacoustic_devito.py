"""2D Viscoacoustic Wave Equations Solver using Devito DSL.

Solves the viscoacoustic wave equations with three different rheological models:

1. **SLS (Standard Linear Solid)** - Blanch & Symes (1995) / Dutta & Schuster (2014)
   Uses memory variables for accurate Q modeling across frequencies.

2. **Kelvin-Voigt** - Ren et al. (2014)
   Adds viscosity term to the standard acoustic equation.

3. **Maxwell** - Deng & McMechan (2007)
   Simple absorption coefficient approach.

The viscoacoustic equations model seismic wave propagation in attenuating media
where the quality factor Q describes energy loss per wavelength.

Physical background:
    - Real earth materials absorb seismic energy (convert to heat)
    - Q (quality factor) measures attenuation: low Q = high attenuation
    - Attenuation causes amplitude decay and phase dispersion
    - Important for seismic imaging and inversion in realistic media

Applications:
    - Seismic wave modeling with realistic attenuation
    - Full waveform inversion (FWI) in viscoacoustic media
    - Reverse time migration with Q compensation
    - Hydrocarbon detection (oil/gas causes attenuation)

Usage:
    from src.systems import solve_viscoacoustic_sls

    result = solve_viscoacoustic_sls(
        Lx=6000.0, Lz=6000.0,  # Domain size [m]
        Nx=301, Nz=301,         # Grid points
        T=2000.0,               # Final time [ms]
        vp=2.0,                 # P-wave velocity [km/s]
        Q=50.0,                 # Quality factor
        f0=0.005,               # Reference frequency [kHz]
    )

References:
    - Blanch & Symes (1995): SEG Technical Program Expanded Abstracts
    - Dutta & Schuster (2014): GEOPHYSICS, doi:10.1190/geo2013-0414.1
    - Ren et al. (2014): Geophysical Journal International
    - Deng & McMechan (2007): GEOPHYSICS
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
        VectorTimeFunction,
        div,
        grad,
        solve,
    )
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


__all__ = [
    "ViscoacousticResult",
    "compute_sls_relaxation_parameters",
    "create_damping_field",
    "ricker_wavelet",
    "solve_viscoacoustic_kv",
    "solve_viscoacoustic_maxwell",
    "solve_viscoacoustic_sls",
]


@dataclass
class ViscoacousticResult:
    """Results from the Viscoacoustic Wave Equations solver.

    Attributes
    ----------
    p : np.ndarray
        Final pressure field, shape (Nx, Nz)
    vx : np.ndarray
        Final x-velocity field, shape (Nx, Nz)
    vz : np.ndarray
        Final z-velocity field, shape (Nx, Nz)
    x : np.ndarray
        x-coordinates, shape (Nx,)
    z : np.ndarray
        z-coordinates, shape (Nz,)
    t : float
        Final simulation time
    dt : float
        Time step used
    p_snapshots : np.ndarray or None
        Saved snapshots of pressure, shape (nsnaps, Nx, Nz)
    t_snapshots : np.ndarray or None
        Time values for snapshots
    """
    p: np.ndarray
    vx: np.ndarray
    vz: np.ndarray
    x: np.ndarray
    z: np.ndarray
    t: float
    dt: float
    p_snapshots: np.ndarray | None = None
    t_snapshots: np.ndarray | None = None


def ricker_wavelet(t: np.ndarray, f0: float, t0: float = None) -> np.ndarray:
    """Generate a Ricker (Mexican hat) wavelet.

    The Ricker wavelet is the second derivative of a Gaussian and is
    commonly used as a seismic source signature.

    Parameters
    ----------
    t : ndarray
        Time array
    f0 : float
        Dominant (peak) frequency
    t0 : float, optional
        Time shift for the wavelet center. Default: 1.5/f0

    Returns
    -------
    ndarray
        Ricker wavelet values at times t

    Notes
    -----
    The Ricker wavelet is defined as:
        w(t) = (1 - 2*pi^2*f0^2*(t-t0)^2) * exp(-pi^2*f0^2*(t-t0)^2)

    The frequency content is centered around f0, with bandwidth
    approximately [0, 2.5*f0].
    """
    if t0 is None:
        t0 = 1.5 / f0

    pi_f0_t = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * pi_f0_t**2) * np.exp(-pi_f0_t**2)


def compute_sls_relaxation_parameters(
    Q: float | np.ndarray,
    f0: float,
) -> tuple:
    """Compute SLS relaxation parameters from Q and reference frequency.

    The Standard Linear Solid (SLS) model uses stress and strain relaxation
    times to model frequency-dependent attenuation. These parameters are
    derived from the quality factor Q at a reference frequency f0.

    Parameters
    ----------
    Q : float or ndarray
        Quality factor (dimensionless). Higher Q = less attenuation.
        Typical values: 20-200 for sedimentary rocks.
    f0 : float
        Reference frequency [same units as simulation]

    Returns
    -------
    t_s : float or ndarray
        Stress relaxation time
    t_ep : float or ndarray
        Strain relaxation time
    tau : float or ndarray
        Relaxation magnitude parameter (tau = t_ep/t_s - 1)

    Notes
    -----
    The relationships are:
        t_s = (sqrt(1 + 1/Q^2) - 1/Q) / f0
        t_ep = 1 / (f0^2 * t_s)
        tau = t_ep/t_s - 1

    For large Q (low attenuation):
        t_s -> 1/(2*f0*Q)
        tau -> 1/Q

    References
    ----------
    Blanch, J.O. and Symes, W.W., 1995. Efficient iterative viscoacoustic
    linearized inversion. SEG Technical Program Expanded Abstracts.
    """
    Q = np.asarray(Q)
    t_s = (np.sqrt(1.0 + 1.0 / Q**2) - 1.0 / Q) / f0
    t_ep = 1.0 / (f0**2 * t_s)
    tau = t_ep / t_s - 1.0
    return t_s, t_ep, tau


def create_damping_field(
    grid: "Grid",
    nbl: int = 40,
    damping_coefficient: float = 0.05,
    space_order: int = 8,
) -> "Function":
    """Create an absorbing boundary damping field.

    Creates a damping field that smoothly increases from 1.0 in the
    interior to a maximum value at the boundaries, used to implement
    absorbing boundary conditions.

    Parameters
    ----------
    grid : Grid
        Devito computational grid
    nbl : int
        Number of absorbing boundary layer points
    damping_coefficient : float
        Damping coefficient (higher = more absorption)
    space_order : int
        Spatial discretization order

    Returns
    -------
    Function
        Devito Function containing the damping field

    Notes
    -----
    The damping is applied multiplicatively to the solution at each
    time step: u_new = damp * u. Values close to 1.0 preserve the
    solution, while values < 1.0 attenuate it.
    """
    damp = Function(name='damp', grid=grid, space_order=space_order)

    # Initialize to 1.0 (no damping)
    damp.data[:] = 1.0

    # Get grid shape
    shape = grid.shape

    # Apply damping in absorbing boundary layers
    for dim in range(len(shape)):
        # Ensure nbl doesn't exceed half the grid dimension
        nbl_dim = min(nbl, shape[dim] // 2)
        if nbl_dim == 0:
            continue

        for i in range(nbl_dim):
            # Damping factor: 1 at interior edge, decreasing toward boundary
            factor = 1.0 - damping_coefficient * ((nbl_dim - i) / nbl_dim)**2

            # Create slice for this layer
            # Left boundary
            slices_left = [slice(None)] * len(shape)
            slices_left[dim] = i
            damp.data[tuple(slices_left)] *= factor

            # Right boundary
            slices_right = [slice(None)] * len(shape)
            slices_right[dim] = shape[dim] - 1 - i
            damp.data[tuple(slices_right)] *= factor

    return damp


def solve_viscoacoustic_sls(
    Lx: float = 6000.0,
    Lz: float = 6000.0,
    Nx: int = 301,
    Nz: int = 301,
    T: float = 2000.0,
    dt: float | None = None,
    vp: float | np.ndarray = 2.0,
    rho: float | np.ndarray = 1.0,
    Q: float | np.ndarray = 100.0,
    f0: float = 0.005,
    space_order: int = 8,
    src_coords: tuple[float, float] | None = None,
    nbl: int = 40,
    use_damp: bool = True,
) -> ViscoacousticResult:
    """Solve viscoacoustic wave equation using the SLS rheological model.

    The Standard Linear Solid (SLS) model, also known as the Zener model,
    uses a memory variable to accurately model frequency-dependent
    attenuation (Q) in viscoelastic/viscoacoustic media.

    The system of equations is:
        dP/dt + kappa*(tau + 1)*div(v) + r = S
        dv/dt + (1/rho)*grad(P) = 0
        dr/dt + (1/t_s)*(r + tau*kappa*div(v)) = 0

    where r is the memory variable, tau controls Q magnitude,
    and t_s is the stress relaxation time.

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
        Final simulation time [ms]
    dt : float, optional
        Time step. If None, computed from CFL condition.
    vp : float or ndarray
        P-wave velocity [km/s]. Scalar or array of shape (Nx, Nz).
    rho : float or ndarray
        Density [g/cm^3]. Scalar or array of shape (Nx, Nz).
    Q : float or ndarray
        Quality factor (dimensionless). Scalar or array of shape (Nx, Nz).
    f0 : float
        Reference frequency [kHz] for Q model
    space_order : int
        Spatial discretization order (default: 8)
    src_coords : tuple, optional
        Source coordinates (x, z). Default: center of domain.
    nbl : int
        Number of absorbing boundary layer points
    use_damp : bool
        Whether to apply absorbing boundary damping

    Returns
    -------
    ViscoacousticResult
        Solution data including final pressure, velocity fields

    Raises
    ------
    ImportError
        If Devito is not installed

    References
    ----------
    Blanch & Symes (1995), Dutta & Schuster (2014)
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    # Create grid with explicit spacing
    dx = Lx / (Nx - 1)
    dz = Lz / (Nz - 1)

    x_dim = SpaceDimension(name='x', spacing=Constant(name='h_x', value=dx))
    z_dim = SpaceDimension(name='z', spacing=Constant(name='h_z', value=dz))
    grid = Grid(extent=(Lx, Lz), shape=(Nx, Nz), dimensions=(x_dim, z_dim),
                dtype=np.float32)

    # Get time step symbol
    s = grid.stepping_dim.spacing

    # Handle scalar or array parameters
    vp_arr = np.asarray(vp, dtype=np.float32)
    rho_arr = np.asarray(rho, dtype=np.float32)
    Q_arr = np.asarray(Q, dtype=np.float32)

    if vp_arr.ndim == 0:
        vp_arr = np.full((Nx, Nz), vp_arr, dtype=np.float32)
    if rho_arr.ndim == 0:
        rho_arr = np.full((Nx, Nz), rho_arr, dtype=np.float32)
    if Q_arr.ndim == 0:
        Q_arr = np.full((Nx, Nz), Q_arr, dtype=np.float32)

    # Compute maximum velocity for CFL
    vp_max = float(vp_arr.max())

    # Compute time step from CFL condition if not provided
    if dt is None:
        dt = min(dx, dz) / (np.sqrt(2) * vp_max) * 0.9

    # Number of time steps
    Nt = int(T / dt)

    # Create Devito Functions for material parameters
    vp_fn = Function(name='vp', grid=grid, space_order=space_order)
    b_fn = Function(name='b', grid=grid, space_order=space_order)  # buoyancy = 1/rho
    qp_fn = Function(name='qp', grid=grid, space_order=space_order)

    vp_fn.data[:] = vp_arr
    b_fn.data[:] = 1.0 / rho_arr
    qp_fn.data[:] = Q_arr

    # Compute relaxation parameters (as Functions for spatially varying Q)
    t_s_fn = Function(name='t_s', grid=grid, space_order=space_order)
    tau_fn = Function(name='tau', grid=grid, space_order=space_order)

    t_s_arr, t_ep_arr, tau_arr = compute_sls_relaxation_parameters(Q_arr, f0)
    t_s_fn.data[:] = t_s_arr
    tau_fn.data[:] = tau_arr

    # Bulk modulus: kappa = rho * vp^2
    bm_fn = Function(name='bm', grid=grid, space_order=space_order)
    bm_fn.data[:] = rho_arr * vp_arr**2

    # Create damping field for absorbing boundaries
    if use_damp:
        damp = create_damping_field(grid, nbl, damping_coefficient=0.05,
                                    space_order=space_order)
    else:
        damp = Function(name='damp', grid=grid, space_order=space_order)
        damp.data[:] = 1.0

    # Create velocity, pressure, and memory variable fields
    v = VectorTimeFunction(name='v', grid=grid, time_order=1,
                           space_order=space_order)
    p = TimeFunction(name='p', grid=grid, time_order=1, space_order=space_order)
    r = TimeFunction(name='r', grid=grid, time_order=1, space_order=space_order)

    # Initialize fields to zero
    v[0].data.fill(0.)
    v[1].data.fill(0.)
    p.data.fill(0.)
    r.data.fill(0.)

    # SLS viscoacoustic equations
    # dv/dt + b * grad(p) = 0
    pde_v = v.dt + b_fn * grad(p)
    u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

    # dr/dt + (1/t_s) * (r + tau * bm * div(v.forward)) = 0
    pde_r = r.dt + (1.0 / t_s_fn) * (r + tau_fn * bm_fn * div(v.forward))
    u_r = Eq(r.forward, damp * solve(pde_r, r.forward))

    # dp/dt + bm * (tau + 1) * div(v.forward) + r.forward = 0
    pde_p = p.dt + bm_fn * (tau_fn + 1.0) * div(v.forward) + r.forward
    u_p = Eq(p.forward, damp * solve(pde_p, p.forward))

    # Create operator
    op = Operator([u_v, u_r, u_p])

    # Set up source
    if src_coords is None:
        src_coords = (Lx / 2, Lz / 2)

    # Create source wavelet
    t_vals = np.arange(0, T, dt)
    src_wavelet = ricker_wavelet(t_vals, f0)

    # Find source grid indices
    src_ix = int(src_coords[0] / dx)
    src_iz = int(src_coords[1] / dz)

    # Clip to valid range
    src_ix = max(0, min(src_ix, Nx - 1))
    src_iz = max(0, min(src_iz, Nz - 1))

    # Run simulation with source injection
    for n in range(Nt):
        # Inject source into pressure field
        if n < len(src_wavelet):
            p.data[0, src_ix, src_iz] += dt * src_wavelet[n]

        # Advance one time step
        op.apply(time_m=0, time_M=0, dt=dt)

    # Create coordinate arrays
    x_coords = np.linspace(0.0, Lx, Nx)
    z_coords = np.linspace(0.0, Lz, Nz)

    # Extract results
    p_final = p.data[0, :, :].copy()
    vx_final = v[0].data[0, :, :].copy()
    vz_final = v[1].data[0, :, :].copy()

    return ViscoacousticResult(
        p=p_final,
        vx=vx_final,
        vz=vz_final,
        x=x_coords,
        z=z_coords,
        t=T,
        dt=dt,
        p_snapshots=None,
        t_snapshots=None,
    )


def solve_viscoacoustic_kv(
    Lx: float = 6000.0,
    Lz: float = 6000.0,
    Nx: int = 301,
    Nz: int = 301,
    T: float = 2000.0,
    dt: float | None = None,
    vp: float | np.ndarray = 2.0,
    rho: float | np.ndarray = 1.0,
    Q: float | np.ndarray = 100.0,
    f0: float = 0.005,
    space_order: int = 8,
    src_coords: tuple[float, float] | None = None,
    nbl: int = 40,
    use_damp: bool = True,
) -> ViscoacousticResult:
    """Solve viscoacoustic wave equation using the Kelvin-Voigt model.

    The Kelvin-Voigt (KV) model adds a viscosity term to the standard
    acoustic wave equation. The viscosity coefficient is derived from
    the quality factor Q.

    The system of equations is:
        dP/dt + kappa*div(v) - eta*rho*div(b*grad(P)) = S
        dv/dt + (1/rho)*grad(P) = 0

    where eta = vp^2 / (omega_0 * Q) is the viscosity coefficient,
    and omega_0 = 2*pi*f0 is the angular reference frequency.

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
        Final simulation time [ms]
    dt : float, optional
        Time step. If None, computed from CFL condition.
    vp : float or ndarray
        P-wave velocity [km/s]
    rho : float or ndarray
        Density [g/cm^3]
    Q : float or ndarray
        Quality factor
    f0 : float
        Reference frequency [kHz]
    space_order : int
        Spatial discretization order
    src_coords : tuple, optional
        Source coordinates (x, z). Default: center.
    nbl : int
        Number of absorbing boundary layer points
    use_damp : bool
        Whether to apply absorbing boundary damping

    Returns
    -------
    ViscoacousticResult
        Solution data

    References
    ----------
    Ren et al. (2014), Geophysical Journal International
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
    grid = Grid(extent=(Lx, Lz), shape=(Nx, Nz), dimensions=(x_dim, z_dim),
                dtype=np.float32)

    # Handle parameters
    vp_arr = np.asarray(vp, dtype=np.float32)
    rho_arr = np.asarray(rho, dtype=np.float32)
    Q_arr = np.asarray(Q, dtype=np.float32)

    if vp_arr.ndim == 0:
        vp_arr = np.full((Nx, Nz), vp_arr, dtype=np.float32)
    if rho_arr.ndim == 0:
        rho_arr = np.full((Nx, Nz), rho_arr, dtype=np.float32)
    if Q_arr.ndim == 0:
        Q_arr = np.full((Nx, Nz), Q_arr, dtype=np.float32)

    vp_max = float(vp_arr.max())

    if dt is None:
        dt = min(dx, dz) / (np.sqrt(2) * vp_max) * 0.9

    Nt = int(T / dt)

    # Angular reference frequency
    omega = 2.0 * np.pi * f0

    # Create Devito Functions
    vp_fn = Function(name='vp', grid=grid, space_order=space_order)
    b_fn = Function(name='b', grid=grid, space_order=space_order)
    qp_fn = Function(name='qp', grid=grid, space_order=space_order)
    lam_fn = Function(name='lam', grid=grid, space_order=space_order)  # kappa = rho*vp^2

    vp_fn.data[:] = vp_arr
    b_fn.data[:] = 1.0 / rho_arr
    qp_fn.data[:] = Q_arr
    lam_fn.data[:] = rho_arr * vp_arr**2

    # Damping
    if use_damp:
        damp = create_damping_field(grid, nbl, space_order=space_order)
    else:
        damp = Function(name='damp', grid=grid, space_order=space_order)
        damp.data[:] = 1.0

    # Fields
    v = VectorTimeFunction(name='v', grid=grid, time_order=1,
                           space_order=space_order)
    p = TimeFunction(name='p', grid=grid, time_order=1, space_order=space_order)

    v[0].data.fill(0.)
    v[1].data.fill(0.)
    p.data.fill(0.)

    # Kelvin-Voigt equations
    # dv/dt + b * grad(p) = 0
    pde_v = v.dt + b_fn * grad(p)
    u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

    # dp/dt + lam * div(v.forward) - (lam / (omega * qp)) * laplacian(p) = 0
    # Using div(b * grad(p)) for the diffusion term
    pde_p = (
        p.dt
        + lam_fn * div(v.forward)
        - (lam_fn / (omega * qp_fn)) * div(b_fn * grad(p, shift=0.5), shift=-0.5)
    )
    u_p = Eq(p.forward, damp * solve(pde_p, p.forward))

    op = Operator([u_v, u_p])

    # Source setup
    if src_coords is None:
        src_coords = (Lx / 2, Lz / 2)

    t_vals = np.arange(0, T, dt)
    src_wavelet = ricker_wavelet(t_vals, f0)

    src_ix = max(0, min(int(src_coords[0] / dx), Nx - 1))
    src_iz = max(0, min(int(src_coords[1] / dz), Nz - 1))

    # Run simulation
    for n in range(Nt):
        if n < len(src_wavelet):
            p.data[0, src_ix, src_iz] += dt * src_wavelet[n]
        op.apply(time_m=0, time_M=0, dt=dt)

    x_coords = np.linspace(0.0, Lx, Nx)
    z_coords = np.linspace(0.0, Lz, Nz)

    return ViscoacousticResult(
        p=p.data[0, :, :].copy(),
        vx=v[0].data[0, :, :].copy(),
        vz=v[1].data[0, :, :].copy(),
        x=x_coords,
        z=z_coords,
        t=T,
        dt=dt,
    )


def solve_viscoacoustic_maxwell(
    Lx: float = 6000.0,
    Lz: float = 6000.0,
    Nx: int = 301,
    Nz: int = 301,
    T: float = 2000.0,
    dt: float | None = None,
    vp: float | np.ndarray = 2.0,
    rho: float | np.ndarray = 1.0,
    Q: float | np.ndarray = 100.0,
    f0: float = 0.005,
    space_order: int = 8,
    src_coords: tuple[float, float] | None = None,
    nbl: int = 40,
    use_damp: bool = True,
) -> ViscoacousticResult:
    """Solve viscoacoustic wave equation using the Maxwell model.

    The Maxwell model uses a simple absorption coefficient to model
    attenuation. This approach is computationally simpler than SLS
    but less accurate for broadband signals.

    The system of equations is:
        dP/dt + kappa*div(v) + (omega/Q)*P = S
        dv/dt + (1/rho)*grad(P) = 0

    where omega = 2*pi*f0 is the angular reference frequency, and
    the absorption coefficient is g = omega/Q = 2*pi*f0/Q.

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
        Final simulation time [ms]
    dt : float, optional
        Time step. If None, computed from CFL condition.
    vp : float or ndarray
        P-wave velocity [km/s]
    rho : float or ndarray
        Density [g/cm^3]
    Q : float or ndarray
        Quality factor
    f0 : float
        Reference frequency [kHz]
    space_order : int
        Spatial discretization order
    src_coords : tuple, optional
        Source coordinates (x, z). Default: center.
    nbl : int
        Number of absorbing boundary layer points
    use_damp : bool
        Whether to apply absorbing boundary damping

    Returns
    -------
    ViscoacousticResult
        Solution data

    References
    ----------
    Deng & McMechan (2007), GEOPHYSICS
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
    grid = Grid(extent=(Lx, Lz), shape=(Nx, Nz), dimensions=(x_dim, z_dim),
                dtype=np.float32)

    # Handle parameters
    vp_arr = np.asarray(vp, dtype=np.float32)
    rho_arr = np.asarray(rho, dtype=np.float32)
    Q_arr = np.asarray(Q, dtype=np.float32)

    if vp_arr.ndim == 0:
        vp_arr = np.full((Nx, Nz), vp_arr, dtype=np.float32)
    if rho_arr.ndim == 0:
        rho_arr = np.full((Nx, Nz), rho_arr, dtype=np.float32)
    if Q_arr.ndim == 0:
        Q_arr = np.full((Nx, Nz), Q_arr, dtype=np.float32)

    vp_max = float(vp_arr.max())

    if dt is None:
        dt = min(dx, dz) / (np.sqrt(2) * vp_max) * 0.9

    Nt = int(T / dt)

    # Angular reference frequency
    omega = 2.0 * np.pi * f0

    # Create Devito Functions
    b_fn = Function(name='b', grid=grid, space_order=space_order)
    qp_fn = Function(name='qp', grid=grid, space_order=space_order)
    lam_fn = Function(name='lam', grid=grid, space_order=space_order)

    b_fn.data[:] = 1.0 / rho_arr
    qp_fn.data[:] = Q_arr
    lam_fn.data[:] = rho_arr * vp_arr**2

    # Damping
    if use_damp:
        damp = create_damping_field(grid, nbl, space_order=space_order)
    else:
        damp = Function(name='damp', grid=grid, space_order=space_order)
        damp.data[:] = 1.0

    # Fields
    v = VectorTimeFunction(name='v', grid=grid, time_order=1,
                           space_order=space_order)
    p = TimeFunction(name='p', grid=grid, time_order=1, space_order=space_order)

    v[0].data.fill(0.)
    v[1].data.fill(0.)
    p.data.fill(0.)

    # Maxwell equations
    # dv/dt + b * grad(p) = 0
    pde_v = v.dt + b_fn * grad(p)
    u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

    # dp/dt + lam * div(v.forward) + (omega / qp) * p = 0
    pde_p = p.dt + lam_fn * div(v.forward) + (omega / qp_fn) * p
    u_p = Eq(p.forward, damp * solve(pde_p, p.forward))

    op = Operator([u_v, u_p])

    # Source setup
    if src_coords is None:
        src_coords = (Lx / 2, Lz / 2)

    t_vals = np.arange(0, T, dt)
    src_wavelet = ricker_wavelet(t_vals, f0)

    src_ix = max(0, min(int(src_coords[0] / dx), Nx - 1))
    src_iz = max(0, min(int(src_coords[1] / dz), Nz - 1))

    # Run simulation
    for n in range(Nt):
        if n < len(src_wavelet):
            p.data[0, src_ix, src_iz] += dt * src_wavelet[n]
        op.apply(time_m=0, time_M=0, dt=dt)

    x_coords = np.linspace(0.0, Lx, Nx)
    z_coords = np.linspace(0.0, Lz, Nz)

    return ViscoacousticResult(
        p=p.data[0, :, :].copy(),
        vx=v[0].data[0, :, :].copy(),
        vz=v[1].data[0, :, :].copy(),
        x=x_coords,
        z=z_coords,
        t=T,
        dt=dt,
    )
