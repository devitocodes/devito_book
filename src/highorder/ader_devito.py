"""ADER (Arbitrary-order-accuracy via DERivatives) Wave Equation Solver using Devito.

This module implements ADER finite difference schemes for solving the first-order
acoustic wave equation with high-order time integration. ADER converts time
derivatives to spatial derivatives using the governing equations, enabling
temporal discretization accuracy to match spatial accuracy.

The key advantage of ADER is allowing larger CFL numbers than standard leapfrog
schemes while avoiding grid-grid decoupling artifacts.

Usage:
    from src.highorder.ader_devito import (
        solve_ader_2d,
        ADERResult,
        ricker_wavelet,
    )

    result = solve_ader_2d(
        extent=(1000., 1000.),
        shape=(101, 101),
        c_value=1.5,
        t_end=300.,
        courant=0.85,
    )

References:
    [1] Schwartzkopf, T., Munz, C.D., Toro, E.F. (2004). "Fast High Order ADER
        Schemes for Linear Hyperbolic Equations." J. Compute. Phys., 197(2).
"""

from dataclasses import dataclass

import numpy as np

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

try:
    from devito import (
        Eq,
        Function,
        Grid,
        Operator,
        SparseTimeFunction,
        TimeFunction,
        VectorTimeFunction,
        div,
        grad,
    )
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False

__all__ = [
    "ADERResult",
    "biharmonic",
    "graddiv",
    "gradlap",
    "gradlapdiv",
    "lapdiv",
    "ricker_wavelet",
    "solve_ader_2d",
]


def ricker_wavelet(t: np.ndarray, f0: float = 0.020, A: float = 1.0) -> np.ndarray:
    """Generate a Ricker wavelet (Mexican hat wavelet).

    Parameters
    ----------
    t : np.ndarray
        Time values.
    f0 : float, optional
        Peak frequency in kHz. Default is 0.020 kHz (20 Hz).
    A : float, optional
        Amplitude. Default is 1.0.

    Returns
    -------
    np.ndarray
        Wavelet values at times t.

    Notes
    -----
    The wavelet is centered at t = 1/f0.
    """
    tau = (np.pi * f0 * (t - 1.0 / f0)) ** 2
    return A * (1 - 2 * tau) * np.exp(-tau)


def graddiv(f):
    """Compute grad(div(f)) for a 2D vector field.

    This is NOT the same as applying a gradient stencil to a divergence stencil.
    Instead, we expand the continuous operator and then discretize:
        grad(div(f)) = [f_x.dx2 + f_y.dxdy, f_x.dxdy + f_y.dy2]

    Parameters
    ----------
    f : VectorTimeFunction or similar
        2D vector field with components f[0] and f[1].

    Returns
    -------
    sympy.Matrix
        2x1 matrix with gradient of divergence components.
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is required for graddiv")
    return sp.Matrix([
        [f[0].dx2 + f[1].dxdy],
        [f[0].dxdy + f[1].dy2]
    ])


def lapdiv(f):
    """Compute laplace(div(f)) for a 2D vector field.

    This is the Laplacian of the divergence of a vector field.

    Parameters
    ----------
    f : VectorTimeFunction or similar
        2D vector field with components f[0] and f[1].

    Returns
    -------
    sympy expression
        Scalar expression for laplace(div(f)).
    """
    return f[0].dx3 + f[0].dxdy2 + f[1].dx2dy + f[1].dy3


def gradlap(f):
    """Compute grad(laplace(f)) for a 2D scalar field.

    Parameters
    ----------
    f : TimeFunction or similar
        2D scalar field.

    Returns
    -------
    sympy.Matrix
        2x1 matrix with gradient of Laplacian components.
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is required for gradlap")
    return sp.Matrix([
        [f.dx3 + f.dxdy2],
        [f.dx2dy + f.dy3]
    ])


def gradlapdiv(f):
    """Compute grad(laplace(div(f))) for a 2D vector field.

    This is the gradient of the Laplacian of the divergence.

    Parameters
    ----------
    f : VectorTimeFunction or similar
        2D vector field with components f[0] and f[1].

    Returns
    -------
    sympy.Matrix
        2x1 matrix with gradient of Laplacian of divergence components.
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is required for gradlapdiv")
    return sp.Matrix([
        [f[0].dx4 + f[0].dx2dy2 + f[1].dx3dy + f[1].dxdy3],
        [f[0].dx3dy + f[0].dxdy3 + f[1].dx2dy2 + f[1].dy4]
    ])


def biharmonic(f):
    """Compute the biharmonic operator for a 2D scalar field.

    The biharmonic operator is: nabla^4 f = f_xxxx + 2*f_xxyy + f_yyyy

    Parameters
    ----------
    f : TimeFunction or similar
        2D scalar field.

    Returns
    -------
    sympy expression
        Scalar expression for nabla^4 f.
    """
    return f.dx4 + 2 * f.dx2dy2 + f.dy4


@dataclass
class ADERResult:
    """Results from the ADER wave equation solver.

    Attributes
    ----------
    p : np.ndarray
        Final pressure field, shape (Nx, Ny).
    vx : np.ndarray
        Final x-velocity field, shape (Nx, Ny).
    vy : np.ndarray
        Final y-velocity field, shape (Nx, Ny).
    x : np.ndarray
        x-coordinate array.
    y : np.ndarray
        y-coordinate array.
    t_final : float
        Final simulation time.
    dt : float
        Time step used.
    nt : int
        Number of time steps.
    courant : float
        Courant number used.
    """
    p: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t_final: float
    dt: float
    nt: int
    courant: float


def solve_ader_2d(
    extent: tuple[float, float] = (1000.0, 1000.0),
    shape: tuple[int, int] = (201, 201),
    c_value: float | np.ndarray = 1.5,
    rho_value: float | np.ndarray = 1.0,
    t_end: float = 450.0,
    courant: float = 0.85,
    f0: float = 0.020,
    source_location: tuple[float, float] | None = None,
    space_order: int = 16,
) -> ADERResult:
    """Solve 2D acoustic wave equation with 4th-order ADER time-stepping.

    This solver uses ADER (Arbitrary-order-accuracy via DERivatives) time
    integration, which converts time derivatives to spatial derivatives
    using the governing equations. This enables larger CFL numbers than
    standard leapfrog schemes.

    Parameters
    ----------
    extent : tuple, optional
        Domain size (Lx, Ly) in meters. Default is (1000, 1000) m.
    shape : tuple, optional
        Grid shape (Nx, Ny). Default is (201, 201).
    c_value : float or np.ndarray, optional
        Wave velocity in km/s. Can be scalar (uniform) or 2D array
        (heterogeneous). Default is 1.5 km/s.
    rho_value : float or np.ndarray, optional
        Density. Can be scalar (uniform) or 2D array. Default is 1.0.
    t_end : float, optional
        Simulation end time in ms. Default is 450 ms.
    courant : float, optional
        Courant number. ADER allows values up to ~0.85 (vs ~0.5 for leapfrog).
        Default is 0.85.
    f0 : float, optional
        Source peak frequency in kHz. Default is 0.020 kHz (20 Hz).
    source_location : tuple, optional
        Source (x, y) coordinates in meters. Default is center of domain.
    space_order : int, optional
        Spatial order for derivatives. Must be high enough for ADER accuracy.
        Default is 16.

    Returns
    -------
    ADERResult
        Solution data including pressure, velocity fields, and metadata.

    Raises
    ------
    ImportError
        If Devito or SymPy is not installed.

    Notes
    -----
    The ADER scheme uses a 4th-order Taylor expansion in time, converting
    time derivatives to spatial derivatives:

    - 1st time derivative: from governing equations
    - 2nd time derivative: c^2 * laplace(p), c^2 * grad(div(v))
    - 3rd time derivative: c^4 * laplace(div(v)), c^2/rho * grad(laplace(p))
    - 4th time derivative: c^4 * biharmonic(p), c^4 * grad(laplace(div(v)))

    This assumes constant material properties. For variable properties,
    the derivatives of material parameters must be included.

    Examples
    --------
    >>> result = solve_ader_2d(
    ...     extent=(1000., 1000.),
    ...     shape=(101, 101),
    ...     c_value=1.5,
    ...     t_end=300.,
    ...     courant=0.85
    ... )
    >>> print(f"Pressure field norm: {np.linalg.norm(result.p):.4f}")
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this solver. "
            "Install with: pip install devito"
        )

    if not SYMPY_AVAILABLE:
        raise ImportError(
            "SymPy is required for ADER schemes. "
            "Install with: pip install sympy"
        )

    # Create grid
    grid = Grid(shape=shape, extent=extent)

    # Create fields with no staggering (ADER uses collocated grid)
    p = TimeFunction(name='p', grid=grid, space_order=space_order)
    v = VectorTimeFunction(
        name='v', grid=grid, space_order=space_order,
        staggered=(None, None)  # No staggering
    )

    # Material parameters
    c = Function(name='c', grid=grid)
    rho = Function(name='rho', grid=grid)

    if np.isscalar(c_value):
        c.data[:] = c_value
        c_max = c_value
    else:
        c.data[:] = c_value
        c_max = np.amax(c_value)

    if np.isscalar(rho_value):
        rho.data[:] = rho_value
    else:
        rho.data[:] = rho_value

    # Derived quantities
    b = 1 / rho  # buoyancy
    c2 = c ** 2
    c4 = c ** 4

    # Time step from CFL condition
    h_min = np.amin(grid.spacing)
    dt = courant * h_min / c_max
    nt = int(t_end / dt) + 1

    # Time derivatives expressed as spatial derivatives
    # First time derivatives (from governing equations)
    pdt = rho * c2 * div(v)
    vdt = b * grad(p)

    # Second time derivatives
    pdt2 = c2 * p.laplace
    vdt2 = c2 * graddiv(v)

    # Third time derivatives
    pdt3 = rho * c4 * lapdiv(v)
    vdt3 = c2 * b * gradlap(p)

    # Fourth time derivatives
    pdt4 = c4 * biharmonic(p)
    vdt4 = c4 * gradlapdiv(v)

    # Time step symbol
    dt_sym = grid.stepping_dim.spacing

    # ADER update equations (4th order Taylor expansion)
    eq_p = Eq(
        p.forward,
        p + dt_sym * pdt
        + (dt_sym ** 2 / 2) * pdt2
        + (dt_sym ** 3 / 6) * pdt3
        + (dt_sym ** 4 / 24) * pdt4
    )

    eq_v = Eq(
        v.forward,
        v + dt_sym * vdt
        + (dt_sym ** 2 / 2) * vdt2
        + (dt_sym ** 3 / 6) * vdt3
        + (dt_sym ** 4 / 24) * vdt4
    )

    # Source setup
    t_values = np.linspace(0, t_end, nt)
    src_data = ricker_wavelet(t_values, f0=f0)

    if source_location is None:
        source_location = (extent[0] / 2, extent[1] / 2)

    source = SparseTimeFunction(
        name='src',
        grid=grid,
        npoint=1,
        nt=nt,
        coordinates=[source_location]
    )
    source.data[:, 0] = src_data

    # Source injection into pressure field
    src_term = source.inject(field=p.forward, expr=source)

    # Build and run operator
    op = Operator([eq_p, eq_v] + src_term)
    op.apply(dt=dt)

    # Extract results
    x_coords = np.linspace(0, extent[0], shape[0])
    y_coords = np.linspace(0, extent[1], shape[1])

    return ADERResult(
        p=p.data[-1].copy(),
        vx=v[0].data[-1].copy(),
        vy=v[1].data[-1].copy(),
        x=x_coords,
        y=y_coords,
        t_final=t_end,
        dt=dt,
        nt=nt,
        courant=courant,
    )


def compare_ader_vs_staggered(
    extent: tuple[float, float] = (1000.0, 1000.0),
    shape: tuple[int, int] = (201, 201),
    c_value: float = 1.5,
    t_end: float = 450.0,
) -> tuple[ADERResult, ADERResult]:
    """Compare ADER scheme with staggered leapfrog at same time step.

    This demonstrates the stability advantage of ADER, which can use larger
    CFL numbers than standard staggered leapfrog schemes.

    Parameters
    ----------
    extent : tuple
        Domain size (Lx, Ly).
    shape : tuple
        Grid shape (Nx, Ny).
    c_value : float
        Wave velocity.
    t_end : float
        Simulation end time.

    Returns
    -------
    tuple
        (ader_result, ader_result_low_cfl) - ADER results at CFL=0.85 and 0.5.

    Notes
    -----
    A standard staggered leapfrog scheme would be unstable at CFL=0.85.
    Both ADER runs should be stable, demonstrating ADER's advantage.
    """
    # Run ADER at high CFL (stable)
    result_high_cfl = solve_ader_2d(
        extent=extent,
        shape=shape,
        c_value=c_value,
        t_end=t_end,
        courant=0.85,
    )

    # Run ADER at standard CFL for comparison
    result_low_cfl = solve_ader_2d(
        extent=extent,
        shape=shape,
        c_value=c_value,
        t_end=t_end,
        courant=0.5,
    )

    return result_high_cfl, result_low_cfl
