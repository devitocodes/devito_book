"""Dispersion analysis for Maxwell FDTD solvers.

Analyzes the numerical dispersion properties of the Yee/FDTD scheme
for Maxwell's equations. The scheme introduces numerical dispersion
where the phase velocity depends on:
- Wavelength (points per wavelength)
- Courant number
- Propagation direction (grid anisotropy in 2D/3D)

The numerical dispersion relation for 1D FDTD is:
    sin^2(omega_num * dt/2) / (dt/2)^2 = c^2 * sin^2(k * dx/2) / (dx/2)^2

This simplifies to:
    sin(omega_num * dt/2) = C * sin(k * dx/2)

where C = c*dt/dx is the Courant number.

The "magic time step" C = 1 eliminates dispersion in 1D (waves travel
at exactly the correct speed).

References
----------
.. [1] A. Taflove, "Application of the finite-difference time-domain
       method to sinusoidal steady-state electromagnetic-penetration
       problems," IEEE TEMC, vol. 22, pp. 191-202, 1980.

.. [2] L.N. Trefethen, "Group velocity in finite difference schemes,"
       SIAM Review, vol. 24, pp. 113-136, 1982.
"""

import numpy as np


def numerical_dispersion_relation_1d(
    k: float | np.ndarray,
    c: float,
    dx: float,
    dt: float,
) -> float | np.ndarray:
    """Compute numerical angular frequency from dispersion relation.

    Parameters
    ----------
    k : float or np.ndarray
        Physical wavenumber(s) [rad/m]
    c : float
        Wave speed [m/s]
    dx : float
        Grid spacing [m]
    dt : float
        Time step [s]

    Returns
    -------
    float or np.ndarray
        Numerical angular frequency [rad/s]
    """
    C = c * dt / dx  # Courant number

    # sin(omega_num * dt/2) = C * sin(k * dx/2)
    sin_arg = C * np.sin(k * dx / 2)

    # Clamp to valid range for arcsin
    sin_arg = np.clip(sin_arg, -1, 1)

    omega_num = 2 * np.arcsin(sin_arg) / dt
    return omega_num


def phase_velocity_error_1d(
    k: float | np.ndarray,
    c: float,
    dx: float,
    dt: float,
) -> float | np.ndarray:
    """Compute relative phase velocity error.

    Returns (v_num - c) / c where v_num = omega_num / k.

    Parameters
    ----------
    k : float or np.ndarray
        Physical wavenumber(s) [rad/m]
    c : float
        Wave speed [m/s]
    dx : float
        Grid spacing [m]
    dt : float
        Time step [s]

    Returns
    -------
    float or np.ndarray
        Relative phase velocity error
    """
    omega_num = numerical_dispersion_relation_1d(k, c, dx, dt)

    # Handle k=0 case
    k_arr = np.atleast_1d(k)
    error = np.zeros_like(k_arr, dtype=float)
    nonzero = k_arr != 0
    error[nonzero] = (omega_num[nonzero] / k_arr[nonzero] - c) / c

    if np.isscalar(k):
        return error[0]
    return error


def numerical_dispersion_relation_2d(
    kx: float | np.ndarray,
    ky: float | np.ndarray,
    c: float,
    dx: float,
    dy: float,
    dt: float,
) -> float | np.ndarray:
    """Compute numerical angular frequency for 2D FDTD.

    The 2D dispersion relation is:
    sin^2(omega*dt/2) = Sx^2 * sin^2(kx*dx/2) + Sy^2 * sin^2(ky*dy/2)

    where Sx = c*dt/dx, Sy = c*dt/dy.

    Parameters
    ----------
    kx, ky : float or np.ndarray
        Wavenumber components [rad/m]
    c : float
        Wave speed [m/s]
    dx, dy : float
        Grid spacing [m]
    dt : float
        Time step [s]

    Returns
    -------
    float or np.ndarray
        Numerical angular frequency [rad/s]
    """
    Sx = c * dt / dx
    Sy = c * dt / dy

    sin2_omega = (Sx**2 * np.sin(kx * dx / 2)**2 +
                  Sy**2 * np.sin(ky * dy / 2)**2)

    # Clamp for stability
    sin2_omega = np.clip(sin2_omega, 0, 1)

    omega_num = 2 * np.arcsin(np.sqrt(sin2_omega)) / dt
    return omega_num


def phase_velocity_error_2d(
    k_mag: float,
    theta: float,
    c: float,
    dx: float,
    dy: float,
    dt: float,
) -> float:
    """Compute relative phase velocity error in 2D at given angle.

    Parameters
    ----------
    k_mag : float
        Wavenumber magnitude [rad/m]
    theta : float
        Propagation angle from x-axis [rad]
    c : float
        Wave speed [m/s]
    dx, dy : float
        Grid spacing [m]
    dt : float
        Time step [s]

    Returns
    -------
    float
        Relative phase velocity error
    """
    kx = k_mag * np.cos(theta)
    ky = k_mag * np.sin(theta)

    omega_num = numerical_dispersion_relation_2d(kx, ky, c, dx, dy, dt)

    if k_mag > 0:
        v_num = omega_num / k_mag
        return (v_num - c) / c
    return 0.0


def compute_dispersion_error(
    points_per_wavelength: float | np.ndarray,
    courant_number: float,
    dim: int = 1,
    theta: float = 0.0,
) -> float | np.ndarray:
    """Compute dispersion error as function of resolution and Courant number.

    Parameters
    ----------
    points_per_wavelength : float or np.ndarray
        Number of grid points per wavelength (N_lambda = lambda/dx)
    courant_number : float
        Courant number C = c*dt/dx (1D) or c*dt*sqrt(1/dx^2+1/dy^2) (2D)
    dim : int
        Dimension (1 or 2)
    theta : float
        Propagation angle for 2D [rad]

    Returns
    -------
    float or np.ndarray
        Relative phase velocity error
    """
    N_lambda = np.atleast_1d(points_per_wavelength)
    C = courant_number

    # Wavenumber: k = 2*pi/lambda, and N_lambda = lambda/dx, so k*dx = 2*pi/N_lambda
    k_dx = 2 * np.pi / N_lambda

    if dim == 1:
        # sin(omega*dt/2) = C * sin(k*dx/2)
        # omega = k*c (exact), so omega*dt = k*c*dt = k*dx*C
        # Numerical: omega_num = (2/dt) * arcsin(C * sin(k*dx/2))
        # Phase velocity ratio: v_num/c = omega_num/(k*c) = omega_num*dx/(k*dx*c)

        sin_arg = C * np.sin(k_dx / 2)
        sin_arg = np.clip(sin_arg, -1, 1)

        # omega_num * dt = 2 * arcsin(...)
        # v_num / c = omega_num / (k*c) = omega_num * dt / (k*dx*C)
        #           = 2*arcsin(C*sin(k*dx/2)) / (k*dx*C)

        omega_num_dt = 2 * np.arcsin(sin_arg)
        v_ratio = omega_num_dt / (k_dx * C)
        error = v_ratio - 1.0

    else:  # 2D
        # Assume dx = dy
        kx_dx = k_dx * np.cos(theta)
        ky_dy = k_dx * np.sin(theta)

        # For 2D with dx=dy, the stability limit is C <= 1/sqrt(2)
        # Use Sx = Sy = C/sqrt(2) to get C_2d = C
        Sx = Sy = C / np.sqrt(2)

        sin2_omega = (Sx**2 * np.sin(kx_dx / 2)**2 +
                      Sy**2 * np.sin(ky_dy / 2)**2)
        sin2_omega = np.clip(sin2_omega, 0, 1)

        omega_num_dt = 2 * np.arcsin(np.sqrt(sin2_omega))

        # Exact: omega * dt = k * c * dt = k * dx * C
        # For 2D with angle: k = sqrt(kx^2 + ky^2), and k*dx = k_dx
        v_ratio = omega_num_dt / (k_dx * C / np.sqrt(2))
        error = v_ratio - 1.0

    if np.isscalar(points_per_wavelength):
        return error[0]
    return error


def magic_time_step_error(
    points_per_wavelength: float | np.ndarray,
) -> float | np.ndarray:
    """Compute error at the "magic" time step C=1.

    At C=1 in 1D, the numerical dispersion vanishes exactly for all
    wavenumbers. This function verifies this property.

    Parameters
    ----------
    points_per_wavelength : float or np.ndarray
        Number of grid points per wavelength

    Returns
    -------
    float or np.ndarray
        Dispersion error (should be zero to machine precision)
    """
    return compute_dispersion_error(points_per_wavelength, courant_number=1.0, dim=1)


def group_velocity_error_1d(
    k: float | np.ndarray,
    c: float,
    dx: float,
    dt: float,
) -> float | np.ndarray:
    """Compute relative group velocity error.

    Group velocity: v_g = d(omega)/dk

    Parameters
    ----------
    k : float or np.ndarray
        Physical wavenumber(s) [rad/m]
    c : float
        Wave speed [m/s]
    dx : float
        Grid spacing [m]
    dt : float
        Time step [s]

    Returns
    -------
    float or np.ndarray
        Relative group velocity error
    """
    C = c * dt / dx

    # d(omega_num)/dk = (c * cos(k*dx/2)) / sqrt(1 - C^2 * sin^2(k*dx/2))
    sin_term = np.sin(k * dx / 2)
    cos_term = np.cos(k * dx / 2)

    denom = np.sqrt(1 - C**2 * sin_term**2)
    denom = np.maximum(denom, 1e-10)  # Avoid division by zero

    v_g_num = c * cos_term / denom

    return (v_g_num - c) / c


def stability_limit_1d() -> float:
    """Return 1D CFL stability limit."""
    return 1.0


def stability_limit_2d() -> float:
    """Return 2D CFL stability limit."""
    return 1.0 / np.sqrt(2)


def stability_limit_3d() -> float:
    """Return 3D CFL stability limit."""
    return 1.0 / np.sqrt(3)


def optimal_courant_1d(
    min_wavelength: float,
    dx: float,
    accuracy_target: float = 0.01,
) -> float:
    """Find optimal Courant number for given accuracy target.

    In 1D, C=1 gives zero dispersion, but this may not be achievable
    with other constraints. This finds the best C for a given
    target accuracy.

    Parameters
    ----------
    min_wavelength : float
        Minimum wavelength in simulation [m]
    dx : float
        Grid spacing [m]
    accuracy_target : float
        Maximum acceptable phase velocity error

    Returns
    -------
    float
        Recommended Courant number
    """
    # At C=1, dispersion is zero
    # For other C, error increases. Find C that gives target error.
    N_lambda = min_wavelength / dx

    # For small N_lambda, even C=1 may not be achievable
    # Binary search for optimal C
    C_low, C_high = 0.5, 1.0

    for _ in range(50):
        C_mid = (C_low + C_high) / 2
        error = abs(compute_dispersion_error(N_lambda, C_mid, dim=1))

        if error < accuracy_target:
            C_low = C_mid
        else:
            C_high = C_mid

    return C_low


def plot_dispersion_polar(
    k_magnitude: float,
    c: float,
    dx: float,
    dy: float,
    dt: float,
    n_angles: int = 360,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data for polar plot of dispersion error vs angle.

    Parameters
    ----------
    k_magnitude : float
        Wavenumber magnitude [rad/m]
    c : float
        Wave speed [m/s]
    dx, dy : float
        Grid spacing [m]
    dt : float
        Time step [s]
    n_angles : int
        Number of angles to compute

    Returns
    -------
    tuple
        (angles [rad], phase velocity ratios)
    """
    angles = np.linspace(0, 2 * np.pi, n_angles)
    ratios = np.zeros(n_angles)

    for i, theta in enumerate(angles):
        error = phase_velocity_error_2d(k_magnitude, theta, c, dx, dy, dt)
        ratios[i] = 1 + error  # Convert error to ratio

    return angles, ratios


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "compute_dispersion_error",
    "group_velocity_error_1d",
    "magic_time_step_error",
    "numerical_dispersion_relation_1d",
    "numerical_dispersion_relation_2d",
    "optimal_courant_1d",
    "phase_velocity_error_1d",
    "phase_velocity_error_2d",
    "plot_dispersion_polar",
    "stability_limit_1d",
    "stability_limit_2d",
    "stability_limit_3d",
]
