"""Dispersion-Relation-Preserving (DRP) Wave Equation Solver using Devito.

This module implements DRP finite difference schemes for solving the acoustic
wave equation with minimized numerical dispersion. It provides:

- Pre-computed DRP coefficients for common stencil sizes
- Functions to compute custom DRP coefficients via optimization
- Devito-based wave equation solvers with DRP schemes

Usage:
    from src.highorder.drp_devito import (
        drp_coefficients,
        solve_wave_drp,
        WaveDRPResult,
    )

    # Use pre-computed DRP coefficients
    weights = drp_coefficients(M=4)

    # Solve 2D wave equation with DRP scheme
    result = solve_wave_drp(
        extent=(2000., 2000.),
        shape=(201, 201),
        velocity=1500.,
        f0=30.,
        t_end=0.5,
        use_drp=True
    )

References:
    [1] Tam, C.K.W., Webb, J.C. (1993). "Dispersion-Relation-Preserving
        Finite Difference Schemes for Computational Acoustics."
        J. Compute. Phys., 107(2), 262-281.
    [2] Liu, Y. (2013). "Globally optimal finite-difference schemes based
        on least squares." GEOPHYSICS, 78(4), 113-132.
"""

from dataclasses import dataclass

import numpy as np

try:
    from scipy import integrate, optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from devito import (
        Eq,
        Function,
        Grid,
        Operator,
        SparseTimeFunction,
        TimeFunction,
        solve,
    )
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False

from src.highorder.dispersion import fornberg_weights, ricker_wavelet

# Pre-computed DRP coefficients for the second derivative
# These are optimized using the Tam-Webb objective function
DRP_COEFFICIENTS = {
    # 5-point stencil (M=2)
    2: np.array([-2.65485432, 1.43656954, -0.10914239]),

    # 7-point stencil (M=3)
    3: np.array([-2.85678021, 1.60459224, -0.1962454, 0.02004326]),

    # 9-point stencil (M=4) - Tam-Webb optimized
    4: np.array([-2.96055679, 1.69342321, -0.25123233, 0.0425563, -0.00446879]),

    # 11-point stencil (M=5)
    5: np.array([-3.01383546e+00, 1.74043556e+00, -2.83135920e-01, 5.85762859e-02,
                 -9.87514765e-03, 9.16956477e-04]),
}

# Pre-computed Fornberg coefficients for comparison
FORNBERG_COEFFICIENTS = {
    2: np.array([-2.5, 4/3, -1/12]),
    3: np.array([-49/18, 3/2, -3/20, 1/90]),
    4: np.array([-205/72, 8/5, -1/5, 8/315, -1/560]),
    5: np.array([-5269/1800, 5/3, -5/21, 5/126, -5/1008, 1/3150]),
}


def drp_coefficients(M: int, use_fornberg: bool = False) -> np.ndarray:
    """Get finite difference coefficients for the second derivative.

    Parameters
    ----------
    M : int
        Stencil half-width (total 2M+1 points).
        Supported values: 2, 3, 4, 5.
    use_fornberg : bool, optional
        If True, return Fornberg (Taylor-optimal) coefficients instead of
        DRP-optimized coefficients. Default is False.

    Returns
    -------
    np.ndarray
        Symmetric weights [a_0, a_1, ..., a_M].

    Raises
    ------
    ValueError
        If M is not in the supported range.

    Examples
    --------
    >>> drp = drp_coefficients(M=4)
    >>> fornberg = drp_coefficients(M=4, use_fornberg=True)
    """
    coeffs = FORNBERG_COEFFICIENTS if use_fornberg else DRP_COEFFICIENTS

    if M not in coeffs:
        available = sorted(coeffs.keys())
        raise ValueError(
            f"M={M} not available. Supported values: {available}. "
            f"Use compute_drp_weights() for custom M."
        )

    return coeffs[M].copy()


def drp_objective_tamwebb(a: np.ndarray, M: int) -> float:
    """Tam-Webb DRP objective function for optimization.

    Minimizes the L2 error between the stencil's Fourier representation
    and the exact second derivative in Fourier space.

    Parameters
    ----------
    a : np.ndarray
        Coefficients [a_0, a_1, ..., a_M].
    M : int
        Stencil half-width.

    Returns
    -------
    float
        Objective function value.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "SciPy is required for DRP optimization. "
            "Install with: pip install scipy"
        )

    x = np.linspace(0, np.pi/2, 201)

    # Fourier representation of the stencil
    stencil_fourier = a[0] + 2 * np.sum(
        [a[i] * np.cos(i * x) for i in range(1, M + 1)],
        axis=0
    )

    # Error: should equal -x^2 for exact second derivative
    error = x**2 + stencil_fourier

    # Integrate squared error using trapezoidal rule
    return float(integrate.trapezoid(error**2, x=x))


def compute_drp_weights(
    M: int,
    method: str = 'tamwebb',
    verbose: bool = False
) -> np.ndarray:
    """Compute DRP-optimized finite difference weights via optimization.

    Parameters
    ----------
    M : int
        Stencil half-width (total 2M+1 points).
    method : str, optional
        Optimization method. Currently supported: 'tamwebb'.
        Default is 'tamwebb'.
    verbose : bool, optional
        If True, print optimization progress. Default is False.

    Returns
    -------
    np.ndarray
        Optimized symmetric weights [a_0, a_1, ..., a_M].

    Raises
    ------
    ImportError
        If SciPy is not available.
    ValueError
        If method is not recognized.

    Examples
    --------
    >>> weights = compute_drp_weights(M=4)
    >>> print(weights)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "SciPy is required for DRP optimization. "
            "Install with: pip install scipy"
        )

    # Initial guess: Fornberg weights
    initial = fornberg_weights(M)

    # Build constraints
    constraints = []

    # Constraint 1: a_0 + 2*sum(a_m) = 0 (consistency)
    constraints.append({
        'type': 'eq',
        'fun': lambda x: x[0] + 2 * np.sum(x[1:])
    })

    # Constraint 2: sum(a_m * m^2) = 1 (second-order accuracy)
    constraints.append({
        'type': 'eq',
        'fun': lambda x: np.sum([x[i] * i**2 for i in range(len(x))]) - 1
    })

    # Higher-order constraints (for n = 2 to M//2)
    for n in range(2, (M + 1) // 2):
        def constraint(x, n=n):
            return np.sum([x[i] * i**(2*n) for i in range(len(x))])
        constraints.append({'type': 'eq', 'fun': constraint})

    # Select objective function
    if method == 'tamwebb':
        objective = lambda a: drp_objective_tamwebb(a, M)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tamwebb'.")

    # Run optimization
    result = optimize.minimize(
        objective,
        initial,
        method='SLSQP',
        constraints=constraints,
        options={'ftol': 1e-15, 'maxiter': 500}
    )

    if verbose:
        print(f"Optimization {'succeeded' if result.success else 'failed'}")
        print(f"Message: {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Objective value: {result.fun:.6e}")

    if not result.success:
        import warnings
        warnings.warn(
            f"DRP optimization did not converge: {result.message}",
            stacklevel=2,
        )

    return result.x


def to_full_stencil(symmetric_weights: np.ndarray) -> np.ndarray:
    """Convert symmetric weights to full stencil format.

    Parameters
    ----------
    symmetric_weights : np.ndarray
        Symmetric weights [a_0, a_1, ..., a_M].

    Returns
    -------
    np.ndarray
        Full stencil [a_M, ..., a_1, a_0, a_1, ..., a_M].

    Examples
    --------
    >>> symmetric = np.array([-2.5, 1.33, -0.08])
    >>> full = to_full_stencil(symmetric)
    >>> print(full)
    [-0.08  1.33 -2.5   1.33 -0.08]
    """
    return np.concatenate([symmetric_weights[::-1], symmetric_weights[1:]])


@dataclass
class WaveDRPResult:
    """Results from the DRP wave equation solver.

    Attributes
    ----------
    u : np.ndarray
        Final wavefield, shape (Nx, Ny) for 2D or (Nx,) for 1D.
    x : np.ndarray
        x-coordinate array.
    y : np.ndarray or None
        y-coordinate array (None for 1D).
    t_final : float
        Final simulation time.
    dt : float
        Time step used.
    nt : int
        Number of time steps.
    weights : np.ndarray
        Stencil weights used.
    use_drp : bool
        Whether DRP coefficients were used.
    courant_number : float
        Actual Courant number.
    """
    u: np.ndarray
    x: np.ndarray
    y: np.ndarray | None
    t_final: float
    dt: float
    nt: int
    weights: np.ndarray
    use_drp: bool
    courant_number: float = 0.0


def solve_wave_drp_1d(
    L: float = 2000.0,
    Nx: int = 201,
    velocity: float | np.ndarray = 1500.0,
    f0: float = 30.0,
    t_end: float = 0.6,
    dt: float = 0.0008,
    source_location: float | None = None,
    use_drp: bool = True,
    space_order: int = 8,
) -> WaveDRPResult:
    """Solve 1D acoustic wave equation with optional DRP scheme.

    Parameters
    ----------
    L : float, optional
        Domain length in meters. Default is 2000 m.
    Nx : int, optional
        Number of grid points. Default is 201.
    velocity : float or np.ndarray, optional
        Wave velocity in m/s. Can be scalar (uniform) or array
        (heterogeneous). Default is 1500 m/s.
    f0 : float, optional
        Source peak frequency in Hz. Default is 30 Hz.
    t_end : float, optional
        Simulation end time in seconds. Default is 0.6 s.
    dt : float, optional
        Time step in seconds. Default is 0.0008 s.
    source_location : float, optional
        Source x-coordinate in meters. Default is center of domain.
    use_drp : bool, optional
        If True, use DRP coefficients; else use Fornberg. Default is True.
    space_order : int, optional
        Spatial order (must be even). Default is 8.

    Returns
    -------
    WaveDRPResult
        Solution data including final wavefield and metadata.

    Raises
    ------
    ImportError
        If Devito is not installed.
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    # Get stencil weights
    M = space_order // 2
    if M in DRP_COEFFICIENTS and use_drp:
        weights = drp_coefficients(M, use_fornberg=False)
    elif M in FORNBERG_COEFFICIENTS:
        weights = drp_coefficients(M, use_fornberg=True)
    else:
        weights = fornberg_weights(M)

    full_weights = to_full_stencil(weights)

    # Create grid
    grid = Grid(shape=(Nx,), extent=(L,))
    h = L / (Nx - 1)

    # Create wavefield
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order)

    # Velocity model
    vel = Function(name='vel', grid=grid, space_order=space_order)
    if np.isscalar(velocity):
        vel.data[:] = velocity
        v_max = velocity
    else:
        vel.data[:] = velocity
        v_max = np.max(velocity)

    # Compute Courant number
    courant = v_max * dt / h

    # Source setup
    nt = int(t_end / dt) + 1
    t_values = np.linspace(0, t_end, nt)

    if source_location is None:
        source_location = L / 2

    source = SparseTimeFunction(
        name='src',
        grid=grid,
        npoint=1,
        nt=nt,
        coordinates=[(source_location,)]
    )
    source.data[:, 0] = ricker_wavelet(t_values, f0=f0)

    # Wave equation with custom weights
    u_xx = u.dx2(weights=full_weights)
    pde = u.dt2 - vel**2 * u_xx
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Source injection
    src_term = source.inject(field=u.forward, expr=source * dt**2 * vel**2)

    # Build and run operator
    op = Operator([stencil] + src_term, subs=grid.spacing_map)
    op(time=nt-1, dt=dt)

    # Extract results
    x_coords = np.linspace(0, L, Nx)

    return WaveDRPResult(
        u=u.data[-1].copy(),
        x=x_coords,
        y=None,
        t_final=t_end,
        dt=dt,
        nt=nt,
        weights=weights,
        use_drp=use_drp,
        courant_number=courant,
    )


def solve_wave_drp(
    extent: tuple[float, float] = (2000.0, 2000.0),
    shape: tuple[int, int] = (201, 201),
    velocity: float | np.ndarray = 1500.0,
    f0: float = 30.0,
    t_end: float = 0.6,
    dt: float = 0.0008,
    source_location: tuple[float, float] | None = None,
    use_drp: bool = True,
    space_order: int = 8,
) -> WaveDRPResult:
    """Solve 2D acoustic wave equation with optional DRP scheme.

    Parameters
    ----------
    extent : tuple, optional
        Domain size (Lx, Ly) in meters. Default is (2000, 2000) m.
    shape : tuple, optional
        Grid shape (Nx, Ny). Default is (201, 201).
    velocity : float or np.ndarray, optional
        Wave velocity in m/s. Can be scalar (uniform) or 2D array
        (heterogeneous). Default is 1500 m/s.
    f0 : float, optional
        Source peak frequency in Hz. Default is 30 Hz.
    t_end : float, optional
        Simulation end time in seconds. Default is 0.6 s.
    dt : float, optional
        Time step in seconds. Default is 0.0008 s.
    source_location : tuple, optional
        Source (x, y) coordinates in meters. Default is center of domain.
    use_drp : bool, optional
        If True, use DRP coefficients; else use Fornberg. Default is True.
    space_order : int, optional
        Spatial order (must be even; stencil has space_order+1 points).
        Default is 8.

    Returns
    -------
    WaveDRPResult
        Solution data including final wavefield and metadata.

    Raises
    ------
    ImportError
        If Devito is not installed.

    Examples
    --------
    >>> result = solve_wave_drp(
    ...     extent=(2000., 2000.),
    ...     shape=(201, 201),
    ...     velocity=1500.,
    ...     f0=30.,
    ...     t_end=0.5,
    ...     use_drp=True
    ... )
    >>> print(f"Wavefield norm: {np.linalg.norm(result.u):.4f}")
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    # Get stencil weights
    M = space_order // 2
    if M in DRP_COEFFICIENTS and use_drp:
        weights = drp_coefficients(M, use_fornberg=False)
    elif M in FORNBERG_COEFFICIENTS:
        weights = drp_coefficients(M, use_fornberg=True)
    else:
        weights = fornberg_weights(M)

    full_weights = to_full_stencil(weights)

    # Create grid
    grid = Grid(shape=shape, extent=extent)
    x, y = grid.dimensions
    hx = extent[0] / (shape[0] - 1)
    hy = extent[1] / (shape[1] - 1)
    h = min(hx, hy)

    # Create wavefield
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order)

    # Velocity model
    vel = Function(name='vel', grid=grid, space_order=space_order)
    if np.isscalar(velocity):
        vel.data[:] = velocity
        v_max = velocity
    else:
        vel.data[:] = velocity
        v_max = np.max(velocity)

    # Compute Courant number
    courant = v_max * dt / h

    # Source setup
    nt = int(t_end / dt) + 1
    t_values = np.linspace(0, t_end, nt)

    if source_location is None:
        source_location = (extent[0] / 2, extent[1] / 2)

    source = SparseTimeFunction(
        name='src',
        grid=grid,
        npoint=1,
        nt=nt,
        coordinates=[source_location]
    )
    source.data[:, 0] = ricker_wavelet(t_values, f0=f0)

    # Wave equation with custom weights
    laplacian = u.dx2(weights=full_weights) + u.dy2(weights=full_weights)
    pde = u.dt2 - vel**2 * laplacian
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Source injection
    src_term = source.inject(field=u.forward, expr=source * dt**2 * vel**2)

    # Build and run operator
    op = Operator([stencil] + src_term, subs=grid.spacing_map)
    op(time=nt-1, dt=dt)

    # Extract results
    x_coords = np.linspace(0, extent[0], shape[0])
    y_coords = np.linspace(0, extent[1], shape[1])

    return WaveDRPResult(
        u=u.data[-1].copy(),
        x=x_coords,
        y=y_coords,
        t_final=t_end,
        dt=dt,
        nt=nt,
        weights=weights,
        use_drp=use_drp,
        courant_number=courant,
    )


def compare_dispersion_wavefields(
    extent: tuple[float, float] = (2000.0, 2000.0),
    shape: tuple[int, int] = (201, 201),
    velocity: float = 1500.0,
    f0: float = 30.0,
    t_end: float = 0.6,
    dt: float = 0.0008,
) -> tuple[WaveDRPResult, WaveDRPResult]:
    """Run simulations with both Fornberg and DRP schemes for comparison.

    Parameters
    ----------
    extent : tuple
        Domain size (Lx, Ly) in meters.
    shape : tuple
        Grid shape (Nx, Ny).
    velocity : float
        Wave velocity in m/s.
    f0 : float
        Source peak frequency in Hz.
    t_end : float
        Simulation end time in seconds.
    dt : float
        Time step in seconds.

    Returns
    -------
    tuple
        (fornberg_result, drp_result) as WaveDRPResult objects.
    """
    # Run with Fornberg (standard) scheme
    result_fornberg = solve_wave_drp(
        extent=extent,
        shape=shape,
        velocity=velocity,
        f0=f0,
        t_end=t_end,
        dt=dt,
        use_drp=False,
    )

    # Run with DRP scheme
    result_drp = solve_wave_drp(
        extent=extent,
        shape=shape,
        velocity=velocity,
        f0=f0,
        t_end=t_end,
        dt=dt,
        use_drp=True,
    )

    return result_fornberg, result_drp
