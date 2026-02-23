"""Verification utilities for Maxwell FDTD solvers.

Provides tools for Method of Manufactured Solutions (MMS), convergence
testing, and validation against exact solutions for electromagnetic
simulations.

The key verification tests are:
1. Plane wave propagation at correct speed
2. Reflection at PEC boundaries
3. Second-order convergence in space and time
4. Energy conservation in lossless media
5. Comparison with published results (Monk-Süli, Taflove)

References
----------
.. [1] P.J. Roache, "Fundamentals of Verification and Validation,"
       Hermosa Publishers, 2009.

.. [2] P. Monk and E. Süli, "Error estimates for Yee's method on
       non-uniform grids," IEEE Trans. Magnetics, vol. 30, pp. 3200-3203, 1994.
"""

from collections.abc import Callable

import numpy as np

from src.em.units import EMConstants


def manufactured_solution_1d(
    x: np.ndarray,
    t: float,
    omega: float = 2 * np.pi * 1e9,
    k: float = None,
    alpha: float = 1e8,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate manufactured solution for 1D Maxwell verification.

    Creates a smooth, exponentially decaying standing wave that satisfies
    Maxwell's equations with a known source term.

    u(x, t) = sin(k*x) * cos(omega*t) * exp(-alpha*t)

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates [m]
    t : float
        Time [s]
    omega : float
        Angular frequency [rad/s]
    k : float, optional
        Wavenumber [rad/m]. If None, computed from omega.
    alpha : float
        Decay rate [1/s]
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability

    Returns
    -------
    tuple
        (E_z, H_y, source_term)
        E_z: manufactured E-field solution
        H_y: manufactured H-field solution
        source_term: required source to satisfy equations
    """
    const = EMConstants()
    c = const.c0 / np.sqrt(eps_r * mu_r)
    eta = const.eta0 * np.sqrt(mu_r / eps_r)

    if k is None:
        k = omega / c

    # Manufactured E_z field
    E_z = np.sin(k * x) * np.cos(omega * t) * np.exp(-alpha * t)

    # Consistent H_y from Faraday's law
    # dH_y/dt = (1/mu) * dE_z/dx
    # H_y = (1/mu) * integral(k*cos(k*x)*cos(omega*t)*exp(-alpha*t)) dt
    # This is approximate; for exact MMS we compute source term
    H_y_approx = (k / (const.mu0 * mu_r * omega)) * np.cos(k * x) * np.sin(omega * t) * np.exp(-alpha * t)

    # Compute required source term
    # From Maxwell: dE_z/dt = (1/eps) * dH_y/dx + source
    # source = dE_z/dt - (1/eps) * dH_y/dx
    eps = const.eps0 * eps_r
    mu = const.mu0 * mu_r

    # Time derivative of E_z
    dEz_dt = np.sin(k * x) * (-omega * np.sin(omega * t) - alpha * np.cos(omega * t)) * np.exp(-alpha * t)

    # Space derivative of H_y (approximate)
    dHy_dx = -(k**2 / (mu * omega)) * np.sin(k * x) * np.sin(omega * t) * np.exp(-alpha * t)

    source_term = dEz_dt - (1/eps) * dHy_dx

    return E_z, H_y_approx, source_term


def compute_mms_error(
    solver_result,
    manufactured_func: Callable,
    norm: str = 'L2',
) -> float:
    """Compute error between numerical and manufactured solution.

    Parameters
    ----------
    solver_result : MaxwellResult1D or MaxwellResult2D
        Result from Maxwell solver
    manufactured_func : callable
        Function returning manufactured solution: manufactured_func(x, t) -> E
    norm : str
        Error norm: 'L2', 'Linf', or 'L1'

    Returns
    -------
    float
        Error in specified norm
    """
    E_mms = manufactured_func(solver_result.x_E, solver_result.t)
    diff = solver_result.E_z - E_mms

    if norm == 'L2':
        return np.sqrt(np.mean(diff**2))
    elif norm == 'Linf':
        return np.max(np.abs(diff))
    elif norm == 'L1':
        return np.mean(np.abs(diff))
    else:
        raise ValueError(f"Unknown norm: {norm}")


def verify_wave_speed(
    E_history: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    expected_c: float,
    tolerance: float = 0.05,
) -> tuple[bool, float]:
    """Verify that wave travels at expected speed.

    Tracks the peak of a propagating pulse and measures velocity.

    Parameters
    ----------
    E_history : np.ndarray
        Solution history, shape (Nt, Nx)
    x : np.ndarray
        Spatial coordinates
    t : np.ndarray
        Time coordinates
    expected_c : float
        Expected wave speed [m/s]
    tolerance : float
        Relative tolerance for verification

    Returns
    -------
    tuple
        (passed: bool, measured_c: float)
    """
    # Find peak position at each time
    peak_positions = []
    peak_times = []

    for i, E in enumerate(E_history):
        peak_idx = np.argmax(np.abs(E))
        # Only track if peak is well-defined (not at boundary)
        if 0.1 * len(x) < peak_idx < 0.9 * len(x):
            peak_positions.append(x[peak_idx])
            peak_times.append(t[i])

    if len(peak_positions) < 2:
        return False, 0.0

    peak_positions = np.array(peak_positions)
    peak_times = np.array(peak_times)

    # Linear fit to get velocity
    if len(peak_times) > 2:
        coeffs = np.polyfit(peak_times, peak_positions, 1)
        measured_c = abs(coeffs[0])
    else:
        measured_c = abs(peak_positions[-1] - peak_positions[0]) / (peak_times[-1] - peak_times[0])

    relative_error = abs(measured_c - expected_c) / expected_c
    passed = relative_error < tolerance

    return passed, measured_c


def verify_pec_reflection(
    E_history: np.ndarray,
    x: np.ndarray,
    boundary_idx: int,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """Verify that E_z = 0 is maintained at PEC boundary.

    Parameters
    ----------
    E_history : np.ndarray
        Solution history, shape (Nt, Nx)
    x : np.ndarray
        Spatial coordinates
    boundary_idx : int
        Index of PEC boundary (0 for left, -1 for right)
    tolerance : float
        Maximum allowed E-field at boundary

    Returns
    -------
    tuple
        (passed: bool, max_error: float)
    """
    boundary_values = E_history[:, boundary_idx]
    max_error = np.max(np.abs(boundary_values))
    passed = max_error < tolerance

    return passed, max_error


def verify_energy_conservation(
    E_history: np.ndarray,
    H_history: np.ndarray,
    dx: float,
    eps: float,
    mu: float,
    tolerance: float = 0.01,
) -> tuple[bool, float, np.ndarray]:
    """Verify energy conservation in lossless medium.

    Total electromagnetic energy:
    U = (1/2) * integral(eps*E^2 + mu*H^2) dx

    Parameters
    ----------
    E_history : np.ndarray
        E-field history, shape (Nt, Nx_E)
    H_history : np.ndarray
        H-field history, shape (Nt, Nx_H)
    dx : float
        Grid spacing [m]
    eps : float
        Permittivity [F/m]
    mu : float
        Permeability [H/m]
    tolerance : float
        Maximum allowed relative energy change

    Returns
    -------
    tuple
        (passed: bool, max_relative_change: float, energy_vs_time: np.ndarray)
    """
    energy = []

    for E, H in zip(E_history, H_history):
        # Electric energy (at E grid points)
        U_E = 0.5 * eps * np.sum(E**2) * dx

        # Magnetic energy (at H grid points)
        U_H = 0.5 * mu * np.sum(H**2) * dx

        energy.append(U_E + U_H)

    energy = np.array(energy)
    initial_energy = energy[0]

    if initial_energy > 0:
        relative_change = np.abs(energy - initial_energy) / initial_energy
        max_change = np.max(relative_change)
    else:
        max_change = 0.0

    passed = max_change < tolerance

    return passed, max_change, energy


def convergence_rate(
    dx_values: np.ndarray,
    errors: np.ndarray,
) -> float:
    """Compute convergence rate from grid refinement study.

    Fits error = C * dx^p to find order p.

    Parameters
    ----------
    dx_values : np.ndarray
        Grid spacing values
    errors : np.ndarray
        Corresponding error values

    Returns
    -------
    float
        Observed convergence order
    """
    log_dx = np.log(dx_values)
    log_err = np.log(errors)

    # Linear regression
    coeffs = np.polyfit(log_dx, log_err, 1)
    return coeffs[0]


def verify_monk_suli_convergence(
    solver_func: Callable,
    grid_sizes: list = None,
    expected_order: float = 2.0,
    tolerance: float = 0.2,
) -> tuple[bool, float, list]:
    """Reproduce convergence results from Monk & Süli (1994).

    Tests that the Yee scheme achieves second-order convergence
    in the L2 norm for smooth solutions.

    Parameters
    ----------
    solver_func : callable
        Function that takes grid size N and returns (error, dx)
    grid_sizes : list
        Grid sizes to test
    expected_order : float
        Expected convergence order (2.0 for Yee scheme)
    tolerance : float
        Tolerance for order verification

    Returns
    -------
    tuple
        (passed: bool, observed_order: float, errors: list)

    References
    ----------
    P. Monk and E. Süli, "Error estimates for Yee's method on
    non-uniform grids," IEEE Trans. Magnetics, vol. 30, 1994.
    """
    if grid_sizes is None:
        grid_sizes = [25, 50, 100, 200]

    errors = []
    dx_vals = []

    for N in grid_sizes:
        err, dx = solver_func(N)
        errors.append(err)
        dx_vals.append(dx)

    errors = np.array(errors)
    dx_vals = np.array(dx_vals)

    observed_order = convergence_rate(dx_vals, errors)
    passed = abs(observed_order - expected_order) < tolerance

    return passed, observed_order, errors.tolist()


def taflove_dispersion_formula(
    omega: float,
    c: float,
    dx: float,
    dt: float,
    theta: float = 0.0,
) -> float:
    """Compute numerical phase velocity from Taflove's dispersion formula.

    For 1D FDTD, the dispersion relation is:
    sin^2(omega_num*dt/2) = C^2 * sin^2(k*dx/2)

    where C = c*dt/dx is the Courant number.

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s]
    c : float
        Physical wave speed [m/s]
    dx : float
        Grid spacing [m]
    dt : float
        Time step [s]
    theta : float
        Propagation angle (for 2D/3D) [rad]

    Returns
    -------
    float
        Numerical phase velocity / physical phase velocity ratio

    References
    ----------
    A. Taflove, "Application of the finite-difference time-domain
    method to sinusoidal steady-state electromagnetic-penetration
    problems," IEEE TEMC, vol. 22, 1980.
    """
    k = omega / c  # Physical wavenumber
    C = c * dt / dx  # Courant number

    # From dispersion relation
    sin_omega_dt_2 = C * np.sin(k * dx / 2)

    # Clamp to valid range
    sin_omega_dt_2 = np.clip(sin_omega_dt_2, -1, 1)

    # Numerical angular frequency
    omega_num = 2 * np.arcsin(sin_omega_dt_2) / dt

    # Numerical phase velocity ratio
    if omega > 0:
        return omega_num / omega
    else:
        return 1.0


def verify_cfl_stability_boundary(
    solver_func: Callable,
    C_values: list = None,
    stable_threshold: float = 10.0,
) -> tuple[float, list]:
    """Verify instability occurs at the CFL boundary.

    Tests that:
    - C < 1: stable (max field bounded)
    - C > 1: unstable (field grows exponentially)

    Parameters
    ----------
    solver_func : callable
        Function that takes Courant number and returns max field value
    C_values : list
        Courant numbers to test
    stable_threshold : float
        Maximum field value considered stable

    Returns
    -------
    tuple
        (critical_C: float, results: list of (C, max_field, is_stable))
    """
    if C_values is None:
        C_values = [0.5, 0.7, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1]

    results = []
    for C in C_values:
        try:
            max_field = solver_func(C)
            is_stable = max_field < stable_threshold
        except (ValueError, RuntimeError):
            max_field = np.inf
            is_stable = False

        results.append((C, max_field, is_stable))

    # Find critical C (transition point)
    critical_C = 1.0
    for C, _, is_stable in results:
        if not is_stable:
            critical_C = C
            break

    return critical_C, results


def compute_reflection_coefficient_numerical(
    E_incident: np.ndarray,
    E_reflected: np.ndarray,
) -> float:
    """Compute numerical reflection coefficient from field data.

    Parameters
    ----------
    E_incident : np.ndarray
        Incident wave field (peak amplitude or time series)
    E_reflected : np.ndarray
        Reflected wave field

    Returns
    -------
    float
        Reflection coefficient |R| = |E_reflected| / |E_incident|
    """
    A_inc = np.max(np.abs(E_incident))
    A_ref = np.max(np.abs(E_reflected))

    if A_inc > 0:
        return A_ref / A_inc
    else:
        return 0.0


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "compute_mms_error",
    "compute_reflection_coefficient_numerical",
    "convergence_rate",
    "manufactured_solution_1d",
    "taflove_dispersion_formula",
    "verify_cfl_stability_boundary",
    "verify_energy_conservation",
    "verify_monk_suli_convergence",
    "verify_pec_reflection",
    "verify_wave_speed",
]
