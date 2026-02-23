"""1D Maxwell Equation Solver using Devito DSL (FDTD Method).

Solves the 1D Maxwell's equations (TM mode) using the Yee/FDTD scheme:

    dE_z/dt = (1/eps) * dH_y/dx
    dH_y/dt = (1/mu) * dE_z/dx

on domain [0, L] with:
    - Initial conditions: E_z(x, 0) = E_init(x), H_y(x, 0) = H_init(x)
    - Boundary conditions: PEC (E_z = 0) at both ends by default

The Yee scheme uses a staggered grid:
    - E_z defined at integer grid points: x_i = i * dx
    - H_y defined at half-integer points: x_{i+1/2} = (i + 0.5) * dx
    - Time stepping: E at integer times, H at half-integer times

Update formulas:
    E_z^{n+1}|_i = E_z^n|_i + (dt/eps/dx) * (H_y^{n+1/2}|_{i+1/2} - H_y^{n+1/2}|_{i-1/2})
    H_y^{n+3/2}|_{i+1/2} = H_y^{n+1/2}|_{i+1/2} + (dt/mu/dx) * (E_z^{n+1}|_{i+1} - E_z^{n+1}|_i)

References
----------
.. [1] K.S. Yee, "Numerical solution of initial boundary value problems
       involving Maxwell's equations in isotropic media," IEEE Trans.
       Antennas Propag., vol. 14, no. 3, pp. 302-307, 1966.

Usage
-----
>>> from src.em import solve_maxwell_1d
>>> result = solve_maxwell_1d(
...     L=1.0, Nx=200, T=5e-9,
...     E_init=lambda x: np.exp(-((x - 0.5)**2) / 0.01**2),
... )
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.em.units import EMConstants, compute_cfl_dt, courant_number_1d

try:
    from devito import (
        Eq,
        Function,
        Grid,
        Operator,
        SparseTimeFunction,
        SubDomain,
        TimeFunction,
    )

    DEVITO_AVAILABLE = True
except Exception:
    # Devito import can fail in restricted environments (e.g. hardened sandboxes)
    # due to platform introspection during import-time initialization.
    DEVITO_AVAILABLE = False


@dataclass
class MaxwellResult1D:
    """Results from the 1D Maxwell FDTD solver.

    Attributes
    ----------
    E_z : np.ndarray
        Electric field (z-component) at final time, shape (Nx+1,)
    H_y : np.ndarray
        Magnetic field (y-component) at final time, shape (Nx,)
    x_E : np.ndarray
        Spatial coordinates for E_z (integer grid points)
    x_H : np.ndarray
        Spatial coordinates for H_y (half-integer grid points)
    t : float
        Final simulation time [s]
    dt : float
        Time step used [s]
    dx : float
        Grid spacing [m]
    c : float
        Wave speed [m/s]
    C : float
        Courant number used
    E_history : np.ndarray, optional
        Full E_z history, shape (Nt+1, Nx+1)
    H_history : np.ndarray, optional
        Full H_y history, shape (Nt+1, Nx)
    t_history : np.ndarray, optional
        Time points for history
    """
    E_z: np.ndarray
    H_y: np.ndarray
    x_E: np.ndarray
    x_H: np.ndarray
    t: float
    dt: float
    dx: float
    c: float
    C: float
    E_history: np.ndarray | None = None
    H_history: np.ndarray | None = None
    t_history: np.ndarray | None = None


if DEVITO_AVAILABLE:

    class _HUpdateDomain(SubDomain):
        """Update domain for the staggered H field.

        With an E grid of size `Nx+1`, the staggered H field is logically defined on
        `Nx` half-cells. We therefore update indices `[0, Nx-1]` (i.e., exclude the
        final x-index), leaving the last staggered point unused.
        """

        name = "h_update"

        def define(self, dimensions):
            (x,) = dimensions
            return {x: ("middle", 0, 1)}


def _solve_maxwell_1d_numpy(
    L: float,
    Nx: int,
    T: float,
    CFL: float,
    eps_r: float | np.ndarray,
    mu_r: float | np.ndarray,
    sigma: float | np.ndarray,
    E_init: Callable[[np.ndarray], np.ndarray] | None,
    H_init: Callable[[np.ndarray], np.ndarray] | None,
    source_func: Callable[[float], float] | None,
    source_position: float | None,
    bc_left: str,
    bc_right: str,
    save_history: bool,
    dtype: np.dtype,
) -> MaxwellResult1D:
    """Pure-NumPy fallback for restricted environments where Devito is unavailable."""
    const = EMConstants()
    dx = L / Nx

    x_E = np.linspace(0.0, L, Nx + 1, dtype=dtype)
    x_H = np.linspace(dx / 2, L - dx / 2, Nx, dtype=dtype)

    if isinstance(eps_r, np.ndarray):
        eps_r_E = eps_r if len(eps_r) == Nx + 1 else np.interp(x_E, np.linspace(0, L, len(eps_r)), eps_r)
    else:
        eps_r_E = np.full(Nx + 1, eps_r, dtype=dtype)

    if isinstance(mu_r, np.ndarray):
        mu_r_H = mu_r if len(mu_r) == Nx else np.interp(x_H, np.linspace(0, L, len(mu_r)), mu_r)
    else:
        mu_r_H = np.full(Nx, mu_r, dtype=dtype)

    if isinstance(sigma, np.ndarray):
        sigma_E = sigma if len(sigma) == Nx + 1 else np.interp(x_E, np.linspace(0, L, len(sigma)), sigma)
    else:
        sigma_E = np.full(Nx + 1, sigma, dtype=dtype)

    c_min = float(const.c0 / np.sqrt(np.max(eps_r_E) * (np.max(mu_r_H) if mu_r_H.size else 1.0)))
    dt = float(compute_cfl_dt(dx=dx, c=c_min, CFL=CFL))

    if T == 0.0:
        E0 = E_init(x_E) if E_init else np.zeros(Nx + 1, dtype=dtype)
        H0 = H_init(x_H) if H_init else np.zeros(Nx, dtype=dtype)
        return MaxwellResult1D(
            E_z=E0.copy(),
            H_y=H0.copy(),
            x_E=x_E,
            x_H=x_H,
            t=0.0,
            dt=0.0,
            dx=dx,
            c=c_min,
            C=0.0,
        )

    Nt = int(np.ceil(T / dt))
    dt = float(T / Nt)
    C_actual = float(courant_number_1d(c=c_min, dt=dt, dx=dx))

    eps_E = eps_r_E * const.eps0
    denom = 1.0 + sigma_E * dt / (2.0 * eps_E)
    Ca = (1.0 - sigma_E * dt / (2.0 * eps_E)) / denom
    Cb = (dt / eps_E / dx) / denom

    mu_H = mu_r_H * const.mu0
    Ch = dt / (mu_H * dx)

    E_z = E_init(x_E) if E_init else np.zeros(Nx + 1, dtype=dtype)
    H_y = H_init(x_H) if H_init else np.zeros(Nx, dtype=dtype)

    if source_func is not None:
        if source_position is None:
            raise ValueError("source_position required when source_func is provided")
        source_idx = int(round(source_position / dx))
        source_idx = max(1, min(source_idx, Nx - 1))
    else:
        source_idx = None

    if save_history:
        E_history = np.zeros((Nt + 1, Nx + 1), dtype=dtype)
        H_history = np.zeros((Nt + 1, Nx), dtype=dtype)
        t_history = np.linspace(0.0, T, Nt + 1, dtype=dtype)
        E_history[0, :] = E_z
        H_history[0, :] = H_y
    else:
        E_history = None
        H_history = None
        t_history = None

    mur = (C_actual - 1.0) / (C_actual + 1.0) if C_actual > 0 else 0.0

    for n in range(Nt):
        E_old = E_z.copy()

        H_y[:] = H_y[:] + Ch * (E_old[1:] - E_old[:-1])
        E_z[1:-1] = Ca[1:-1] * E_old[1:-1] + Cb[1:-1] * (H_y[1:] - H_y[:-1])

        if source_idx is not None:
            E_z[source_idx] += float(source_func((n + 1) * dt))

        if bc_left == "pec":
            E_z[0] = 0.0
        elif bc_left == "abc":
            E_z[0] = E_old[1] + mur * (E_z[1] - E_old[0])
        elif bc_left == "pmc":
            E_z[0] = E_z[1]
        else:
            raise ValueError(f"Unknown bc_left={bc_left!r}")

        if bc_right == "pec":
            E_z[-1] = 0.0
        elif bc_right == "abc":
            E_z[-1] = E_old[-2] + mur * (E_z[-2] - E_old[-1])
        elif bc_right == "pmc":
            E_z[-1] = E_z[-2]
        else:
            raise ValueError(f"Unknown bc_right={bc_right!r}")

        if save_history:
            E_history[n + 1, :] = E_z
            H_history[n + 1, :] = H_y

    return MaxwellResult1D(
        E_z=E_z.copy(),
        H_y=H_y.copy(),
        x_E=x_E,
        x_H=x_H,
        t=T,
        dt=dt,
        dx=dx,
        c=c_min,
        C=C_actual,
        E_history=E_history,
        H_history=H_history,
        t_history=t_history,
    )


def solve_maxwell_1d(
    L: float = 1.0,
    Nx: int = 200,
    T: float = 5e-9,
    CFL: float = 0.9,
    eps_r: float | np.ndarray = 1.0,
    mu_r: float | np.ndarray = 1.0,
    sigma: float | np.ndarray = 0.0,
    E_init: Callable[[np.ndarray], np.ndarray] | None = None,
    H_init: Callable[[np.ndarray], np.ndarray] | None = None,
    source_func: Callable[[float], float] | None = None,
    source_position: float | None = None,
    bc_left: str = "pec",
    bc_right: str = "pec",
    save_history: bool = False,
    dtype: np.dtype = np.float64,
) -> MaxwellResult1D:
    """Solve 1D Maxwell's equations (TM mode) using Devito FDTD.

    Parameters
    ----------
    L : float
        Domain length [m]
    Nx : int
        Number of spatial grid intervals (E_z has Nx+1 points, H_y has Nx points)
    T : float
        Final simulation time [s]
    CFL : float
        Courant number (c*dt/dx). Must be <= 1 for stability. Default: 0.9
    eps_r : float or np.ndarray
        Relative permittivity. Can be spatially varying.
    mu_r : float or np.ndarray
        Relative permeability. Can be spatially varying.
    sigma : float or np.ndarray
        Conductivity [S/m] for lossy media. Default: 0 (lossless).
    E_init : callable, optional
        Initial E_z field: E_init(x) -> E_z(x, 0)
        Default: zero everywhere
    H_init : callable, optional
        Initial H_y field: H_init(x) -> H_y(x, 0)
        Default: zero everywhere
    source_func : callable, optional
        Time-dependent source: source_func(t) -> amplitude
        Injected as a soft source at source_position
    source_position : float, optional
        x-coordinate for source injection [m]. Required if source_func given.
    bc_left : str
        Left boundary condition: "pec" (default), "pmc", or "abc"
    bc_right : str
        Right boundary condition: "pec" (default), "pmc", or "abc"
    save_history : bool
        If True, save full solution history
    dtype : np.dtype
        Floating-point precision. Default: np.float64

    Returns
    -------
    MaxwellResult1D
        Solution data including final fields and optionally history

    Raises
    ------
    ImportError
        If Devito is not installed
    ValueError
        If CFL > 1 (unstable) or invalid parameters
    """
    if Nx < 2:
        raise ValueError("Nx must be >= 2")
    if L <= 0.0:
        raise ValueError("L must be > 0")
    if T < 0.0:
        raise ValueError("T must be >= 0")
    if CFL <= 0.0:
        raise ValueError("CFL must be > 0")
    if CFL > 1.0:
        raise ValueError(f"CFL={CFL} > 1 violates the 1D Yee stability condition")

    if not DEVITO_AVAILABLE:
        return _solve_maxwell_1d_numpy(
            L=L,
            Nx=Nx,
            T=T,
            CFL=CFL,
            eps_r=eps_r,
            mu_r=mu_r,
            sigma=sigma,
            E_init=E_init,
            H_init=H_init,
            source_func=source_func,
            source_position=source_position,
            bc_left=bc_left,
            bc_right=bc_right,
            save_history=save_history,
            dtype=dtype,
        )

    const = EMConstants()
    dx = L / Nx

    # Coordinates (for user-facing outputs and initial conditions).
    x_E = np.linspace(0.0, L, Nx + 1, dtype=dtype)
    x_H = np.linspace(dx / 2, L - dx / 2, Nx, dtype=dtype)

    # Material arrays.
    if isinstance(eps_r, np.ndarray):
        eps_r_E = eps_r if len(eps_r) == Nx + 1 else np.interp(x_E, np.linspace(0, L, len(eps_r)), eps_r)
    else:
        eps_r_E = np.full(Nx + 1, eps_r, dtype=dtype)

    if isinstance(mu_r, np.ndarray):
        mu_r_H = mu_r if len(mu_r) == Nx else np.interp(x_H, np.linspace(0, L, len(mu_r)), mu_r)
    else:
        mu_r_H = np.full(Nx, mu_r, dtype=dtype)

    if isinstance(sigma, np.ndarray):
        sigma_E = sigma if len(sigma) == Nx + 1 else np.interp(x_E, np.linspace(0, L, len(sigma)), sigma)
    else:
        sigma_E = np.full(Nx + 1, sigma, dtype=dtype)

    # Conservative wave speed for stability (use maximum material slowness).
    c_local = const.c0 / np.sqrt(eps_r_E * (np.max(mu_r_H) if mu_r_H.size else 1.0))
    c_min = float(np.min(c_local))

    # Stable dt from CFL.
    dt = float(compute_cfl_dt(dx=dx, c=c_min, CFL=CFL))

    if T == 0.0:
        E0 = E_init(x_E) if E_init else np.zeros(Nx + 1, dtype=dtype)
        H0 = H_init(x_H) if H_init else np.zeros(Nx, dtype=dtype)
        return MaxwellResult1D(
            E_z=E0.copy(),
            H_y=H0.copy(),
            x_E=x_E,
            x_H=x_H,
            t=0.0,
            dt=0.0,
            dx=dx,
            c=c_min,
            C=0.0,
        )

    Nt = int(np.ceil(T / dt))
    dt = float(T / Nt)  # Hit T exactly.
    C_actual = float(courant_number_1d(c=c_min, dt=dt, dx=dx))
    if C_actual > 1.0 + 1e-12:
        raise ValueError(f"Adjusted CFL={C_actual:.6f} > 1; reduce CFL or increase Nx")

    # Lossy-medium coefficients on E grid:
    # E^{n+1} = Ca * E^n + Cb * (dH/dx)
    eps_E = eps_r_E * const.eps0
    denom = 1.0 + sigma_E * dt / (2.0 * eps_E)
    Ca = (1.0 - sigma_E * dt / (2.0 * eps_E)) / denom
    Cb = (dt / eps_E) / denom

    # H coefficient on H grid: dt/mu. Store on a length-(Nx+1) array for Devito and ignore the last entry.
    mu_H = mu_r_H * const.mu0
    Ch = np.zeros(Nx + 1, dtype=dtype)
    Ch[:Nx] = dt / mu_H

    grid = Grid(shape=(Nx + 1,), extent=(L,), dtype=dtype, subdomains=(_HUpdateDomain(),))
    t = grid.stepping_dim
    x = grid.dimensions[0]

    if save_history:
        E = TimeFunction(name="E_z", grid=grid, time_order=1, space_order=2, save=Nt + 1, dtype=dtype)
        H = TimeFunction(
            name="H_y", grid=grid, time_order=1, space_order=2, save=Nt + 1, staggered=x, dtype=dtype
        )
    else:
        E = TimeFunction(name="E_z", grid=grid, time_order=1, space_order=2, dtype=dtype)
        H = TimeFunction(name="H_y", grid=grid, time_order=1, space_order=2, staggered=x, dtype=dtype)

    Ca_f = Function(name="Ca", grid=grid, space_order=0, dtype=dtype)
    Cb_f = Function(name="Cb", grid=grid, space_order=0, dtype=dtype)
    Ch_f = Function(name="Ch", grid=grid, space_order=0, staggered=x, dtype=dtype)
    Ca_f.data[:] = Ca
    Cb_f.data[:] = Cb
    Ch_f.data[:] = Ch

    # Initial conditions: store H at a half time-step behind E (leapfrog).
    E0 = E_init(x_E) if E_init else np.zeros(Nx + 1, dtype=dtype)
    H0 = H_init(x_H) if H_init else np.zeros(Nx, dtype=dtype)
    if bc_left == "pec":
        E0[0] = 0.0
    if bc_right == "pec":
        E0[-1] = 0.0

    # Convert H(t=0) to H(t=-dt/2) for leapfrog consistency:
    # H^{-1/2}_{i+1/2} = H^0_{i+1/2} - (dt/(2*mu*dx)) * (E^0_{i+1} - E^0_i)
    H_minus_half = H0 - 0.5 * (Ch[:Nx] / dx) * (E0[1:] - E0[:-1])

    if save_history:
        E.data[0, :] = E0
        H.data[0, :Nx] = H_minus_half
        H.data[0, Nx] = 0.0
    else:
        E.data[0, :] = E0
        H.data[0, :Nx] = H_minus_half
        H.data[0, Nx] = 0.0

    # Yee updates using symbolic derivatives on the staggered grid.
    # E.dx at staggered H locations gives (E[i+1] - E[i]) / h_x (forward diff).
    # H.forward.dx at node E locations gives (H[i+½] - H[i-½]) / h_x (centered diff).
    update_H = Eq(
        H.forward,
        H + Ch_f * E.dx,
        subdomain=grid.subdomains["h_update"],
    )
    update_E = Eq(
        E.forward,
        Ca_f * E + Cb_f * H.forward.dx,
        subdomain=grid.interior,
    )

    eqs = [update_H, update_E]

    # Optional soft source on E via SparseTimeFunction injection.
    if source_func is not None:
        if source_position is None:
            raise ValueError("source_position required when source_func is provided")
        src = SparseTimeFunction(name="src", grid=grid, npoint=1, nt=Nt + 1, dtype=dtype)
        src.coordinates.data[0, 0] = float(source_position)
        src.data[:, 0] = np.array([source_func(n * dt) for n in range(Nt + 1)], dtype=dtype)
        eqs += src.inject(field=E.forward, expr=src)

    # Boundary conditions on E.
    if bc_left == "pec":
        eqs.append(Eq(E[t + 1, 0], 0.0))
    elif bc_left == "abc":
        mur = (C_actual - 1.0) / (C_actual + 1.0)
        eqs.append(Eq(E[t + 1, 0], E[t, 1] + mur * (E[t + 1, 1] - E[t, 0])))
    elif bc_left == "pmc":
        eqs.append(Eq(E[t + 1, 0], E[t + 1, 1]))
    else:
        raise ValueError(f"Unknown bc_left={bc_left!r}")

    if bc_right == "pec":
        eqs.append(Eq(E[t + 1, Nx], 0.0))
    elif bc_right == "abc":
        mur = (C_actual - 1.0) / (C_actual + 1.0)
        eqs.append(Eq(E[t + 1, Nx], E[t, Nx - 1] + mur * (E[t + 1, Nx - 1] - E[t, Nx])))
    elif bc_right == "pmc":
        eqs.append(Eq(E[t + 1, Nx], E[t + 1, Nx - 1]))
    else:
        raise ValueError(f"Unknown bc_right={bc_right!r}")

    op = Operator(eqs)
    if save_history:
        # With save=Nt+1, valid indices are 0..Nt. The loop writes at t+1,
        # so time_M must be Nt-1 to keep the last write at index Nt.
        op(time_M=Nt - 1, dt=dt)
    else:
        op(time=Nt, dt=dt)

    if save_history:
        E_history = E.data[: Nt + 1, :].copy()
        H_history = H.data[: Nt + 1, :Nx].copy()
        t_history = np.linspace(0.0, T, Nt + 1, dtype=dtype)
        E_final = E_history[-1]
        H_final = H_history[-1]
    else:
        E_history = None
        H_history = None
        t_history = None
        final_tidx = Nt % 2  # time_order=1 → 2 buffers
        E_final = E.data[final_tidx, :].copy()
        H_final = H.data[final_tidx, :Nx].copy()

    return MaxwellResult1D(
        E_z=E_final,
        H_y=H_final,
        x_E=x_E,
        x_H=x_H,
        t=T,
        dt=dt,
        dx=dx,
        c=c_min,
        C=C_actual,
        E_history=E_history,
        H_history=H_history,
        t_history=t_history,
    )


def exact_plane_wave_1d(
    x: np.ndarray,
    t: float,
    amplitude: float = 1.0,
    k: float = None,
    omega: float = None,
    wavelength: float = None,
    frequency: float = None,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
    direction: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact solution for a plane wave in 1D.

    E_z = A * cos(k*x - omega*t)  for wave traveling in +x direction
    H_y = A/eta * cos(k*x - omega*t)

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates [m]
    t : float
        Time [s]
    amplitude : float
        Electric field amplitude [V/m]
    k : float, optional
        Wavenumber [rad/m]. One of k, omega, wavelength, frequency required.
    omega : float, optional
        Angular frequency [rad/s]
    wavelength : float, optional
        Wavelength [m]
    frequency : float, optional
        Frequency [Hz]
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability
    direction : int
        +1 for +x propagation, -1 for -x propagation

    Returns
    -------
    tuple
        (E_z, H_y) field arrays
    """
    const = EMConstants()
    c = const.c0 / np.sqrt(eps_r * mu_r)
    eta = const.eta0 * np.sqrt(mu_r / eps_r)

    # Determine k and omega from given parameters
    if k is not None:
        omega_val = k * c
    elif omega is not None:
        omega_val = omega
        k = omega / c
    elif wavelength is not None:
        k = 2 * np.pi / wavelength
        omega_val = k * c
    elif frequency is not None:
        omega_val = 2 * np.pi * frequency
        k = omega_val / c
    else:
        raise ValueError("One of k, omega, wavelength, or frequency must be provided")

    # Plane wave solution
    phase = k * x - direction * omega_val * t
    E_z = amplitude * np.cos(phase)
    H_y = direction * amplitude / eta * np.cos(phase)

    return E_z, H_y


def gaussian_pulse_1d(
    x: np.ndarray,
    x0: float,
    sigma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Gaussian pulse initial condition for E_z.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates [m]
    x0 : float
        Pulse center [m]
    sigma : float
        Pulse width (standard deviation) [m]
    amplitude : float
        Peak amplitude [V/m]

    Returns
    -------
    np.ndarray
        E_z initial condition
    """
    return amplitude * np.exp(-((x - x0)**2) / (2 * sigma**2))


def ricker_wavelet(t: np.ndarray, f0: float, t0: float = None) -> np.ndarray:
    """Ricker wavelet (Mexican hat) for source injection.

    r(t) = (1 - 2*(pi*f0*(t-t0))^2) * exp(-(pi*f0*(t-t0))^2)

    Parameters
    ----------
    t : np.ndarray
        Time array [s]
    f0 : float
        Peak frequency [Hz]
    t0 : float, optional
        Time shift [s]. Default: 1/f0 (one period delay)

    Returns
    -------
    np.ndarray
        Wavelet amplitude at each time
    """
    if t0 is None:
        t0 = 1.0 / f0

    tau = np.pi * f0 * (t - t0)
    return (1 - 2 * tau**2) * np.exp(-tau**2)


def convergence_test_maxwell_1d(
    grid_sizes: list = None,
    T: float = 1e-9,
    CFL: float = 0.5,
    wavelength: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run convergence test for 1D Maxwell solver.

    Uses a plane wave exact solution for error computation.
    Tests second-order convergence of the Yee scheme.

    Parameters
    ----------
    grid_sizes : list, optional
        List of Nx values to test. Default: [50, 100, 200, 400]
    T : float
        Final time [s]. Should be short to avoid boundary effects.
    CFL : float
        Courant number. Using 0.5 for cleaner convergence.
    wavelength : float
        Wavelength of test wave [m]

    Returns
    -------
    tuple
        (grid_sizes, errors, observed_order)
    """
    if grid_sizes is None:
        grid_sizes = [50, 100, 200, 400]

    const = EMConstants()
    L = 1.0  # Domain length
    k = 2 * np.pi / wavelength
    omega = k * const.c0

    errors = []

    for Nx in grid_sizes:
        # Initial condition: plane wave
        def E_init(x):
            return np.cos(k * x)

        def H_init(x):
            return np.cos(k * x) / const.eta0

        result = solve_maxwell_1d(
            L=L, Nx=Nx, T=T, CFL=CFL,
            E_init=E_init, H_init=H_init,
            bc_left="abc", bc_right="abc",
        )

        # Exact solution at final time
        E_exact, _ = exact_plane_wave_1d(
            result.x_E, result.t,
            amplitude=1.0, k=k,
        )

        # L2 error
        error = np.sqrt(np.mean((result.E_z - E_exact)**2))
        errors.append(error)

    errors = np.array(errors)
    grid_sizes = np.array(grid_sizes)

    # Compute observed order via linear regression
    dx_vals = L / grid_sizes
    log_dx = np.log(dx_vals)
    log_err = np.log(errors)

    # Only use last 3 points to avoid pre-asymptotic regime
    coeffs = np.polyfit(log_dx[-3:], log_err[-3:], 1)
    observed_order = coeffs[0]

    return grid_sizes, errors, observed_order


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MaxwellResult1D",
    "convergence_test_maxwell_1d",
    "exact_plane_wave_1d",
    "gaussian_pulse_1d",
    "ricker_wavelet",
    "solve_maxwell_1d",
]
