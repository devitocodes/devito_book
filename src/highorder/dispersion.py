"""Dispersion Analysis Utilities for Finite Difference Schemes.

This module provides tools for analyzing the dispersion properties of
finite difference schemes used in wave equation solvers. It includes
functions for computing:

- Numerical and analytical dispersion relations
- Fornberg finite difference weights
- Dispersion error metrics

Usage:
    from src.highorder.dispersion import (
        numerical_dispersion_relation,
        analytical_dispersion_relation,
        fornberg_weights,
        dispersion_error,
        dispersion_ratio,
    )

    # Compute dispersion ratio for a 9-point stencil
    weights = fornberg_weights(M=4)
    ratio = dispersion_ratio(weights, h=10.0, dt=0.001, v=1500.0, k=0.1)

References:
    [1] Fornberg, B. (1988). "Generation of Finite Difference Formulas on
        Arbitrarily Spaced Grids." Mathematics of Computation, 51(184).
    [2] Tam, C.K.W., Webb, J.C. (1993). "Dispersion-Relation-Preserving
        Finite Difference Schemes for Computational Acoustics."
        J. Compute. Phys., 107(2), 262-281.
    [3] Chen, G., Peng, Z., Li, Y. (2022). "A framework for automatically
        choosing the optimal parameters of finite-difference scheme in
        the acoustic wave modeling." Computers & Geosciences, 159.
"""


import numpy as np

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


def fornberg_weights(M: int, derivative: int = 2) -> np.ndarray:
    """Compute Fornberg finite difference weights for a symmetric stencil.

    Uses the Fornberg algorithm to compute optimal (in Taylor series sense)
    finite difference coefficients for approximating derivatives on a
    symmetric stencil with 2M+1 points.

    Parameters
    ----------
    M : int
        Number of points on each side of center (total 2M+1 points).
        For example, M=4 gives a 9-point stencil.
    derivative : int, optional
        Order of derivative to approximate. Default is 2 (second derivative).

    Returns
    -------
    np.ndarray
        Symmetric weights [a_0, a_1, ..., a_M] where a_m = a_{-m}.
        The full stencil is [a_M, ..., a_1, a_0, a_1, ..., a_M].

    Raises
    ------
    ImportError
        If SymPy is not available.
    ValueError
        If M < 1 or derivative > 2*M.

    Examples
    --------
    >>> weights = fornberg_weights(M=2)  # 5-point stencil
    >>> print(weights)
    [-2.5         1.33333333 -0.08333333]

    >>> weights = fornberg_weights(M=4)  # 9-point stencil
    >>> print(weights)
    [-2.84722222  1.6        -0.2         0.02539683 -0.00178571]

    Notes
    -----
    The weights approximate the second derivative as:
        d^2f/dx^2 = (1/h^2) * sum_{m=-M}^{M} a_m * f(x + m*h) + O(h^{2M})

    where h is the grid spacing.
    """
    if not SYMPY_AVAILABLE:
        raise ImportError(
            "SymPy is required for fornberg_weights. "
            "Install with: pip install sympy"
        )

    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")

    if derivative > 2 * M:
        raise ValueError(
            f"Derivative order {derivative} too high for M={M}. "
            f"Maximum derivative order is {2*M}."
        )

    # Generate points ordered by distance from center: 0, 1, -1, 2, -2, ...
    x = [(1 - (-1)**n * (2*n + 1)) // 4 for n in range(2*M + 1)]

    # Compute weights using Fornberg's algorithm via SymPy
    weights = sp.finite_diff_weights(derivative, x, 0)

    # Extract weights for the requested derivative
    # weights[derivative][-1] gives the full stencil weights
    full_weights = weights[derivative][-1]

    # Convert to symmetric form [a_0, a_1, ..., a_M]
    # Points are ordered: 0, 1, -1, 2, -2, ...
    # We take every other weight starting from index 0
    symmetric = np.array([float(full_weights[i]) for i in range(0, 2*M+1, 2)])

    return symmetric


def analytical_dispersion_relation(k: float | np.ndarray, c: float) -> float | np.ndarray:
    """Compute the analytical dispersion relation for the wave equation.

    For the continuous wave equation u_tt = c^2 * u_xx, the dispersion
    relation is omega = c * k (non-dispersive).

    Parameters
    ----------
    k : float or np.ndarray
        Wavenumber(s).
    c : float
        Wave velocity.

    Returns
    -------
    float or np.ndarray
        Angular frequency omega = c * k.

    Examples
    --------
    >>> omega = analytical_dispersion_relation(k=0.1, c=1500.0)
    >>> print(omega)
    150.0
    """
    return c * k


def numerical_dispersion_relation(
    weights: np.ndarray,
    h: float,
    dt: float,
    k: float | np.ndarray,
    c: float
) -> float | np.ndarray:
    """Compute the numerical dispersion relation for a finite difference scheme.

    For a discretized wave equation with spatial stencil weights a_m and
    time-stepping, this computes the numerical angular frequency.

    Parameters
    ----------
    weights : np.ndarray
        Symmetric stencil weights [a_0, a_1, ..., a_M] for the second
        spatial derivative (before division by h^2).
    h : float
        Grid spacing.
    dt : float
        Time step.
    k : float or np.ndarray
        Wavenumber(s).
    c : float
        Wave velocity.

    Returns
    -------
    float or np.ndarray
        Numerical angular frequency omega_numerical.

    Notes
    -----
    The numerical dispersion relation is derived by substituting a plane
    wave solution u_i^n = exp(i*(k*i*h - omega*n*dt)) into the discretized
    equation and solving for omega.
    """
    # Compute the spatial operator in Fourier space
    # sum_{m=-M}^{M} a_m * exp(i*m*k*h) = a_0 + 2*sum_{m=1}^{M} a_m * cos(m*k*h)
    M = len(weights) - 1
    spatial_term = weights[0] + 2 * np.sum(
        [weights[m] * np.cos(m * k * h) for m in range(1, M + 1)],
        axis=0
    )

    # From the discretized equation:
    # (2 - 2*cos(omega*dt)) / dt^2 = c^2 * spatial_term / h^2
    # cos(omega*dt) = 1 + (c^2 * dt^2 / h^2) * spatial_term / 2

    cos_omega_dt = 1 + 0.5 * (c**2 * dt**2 / h**2) * spatial_term

    # Clamp to valid range for arccos
    cos_omega_dt = np.clip(cos_omega_dt, -1, 1)

    omega = np.arccos(cos_omega_dt) / dt

    return omega


def dispersion_ratio(
    weights: np.ndarray,
    h: float,
    dt: float,
    v: float,
    k: float,
    alpha: float = 0.0
) -> float:
    """Compute the velocity error ratio for a finite difference scheme.

    The velocity error ratio delta = v_FD / v measures how accurately the
    numerical scheme preserves wave velocity. A value of 1.0 indicates
    perfect preservation; deviations indicate numerical dispersion.

    Parameters
    ----------
    weights : np.ndarray
        Symmetric stencil weights [a_0, a_1, ..., a_M].
    h : float
        Grid spacing.
    dt : float
        Time step.
    v : float
        True wave velocity.
    k : float
        Wavenumber.
    alpha : float, optional
        Propagation angle in radians (for 2D/3D). Default is 0 (1D case or
        propagation aligned with x-axis).

    Returns
    -------
    float
        Velocity error ratio v_FD / v.

    Examples
    --------
    >>> weights = fornberg_weights(M=4)
    >>> ratio = dispersion_ratio(weights, h=10.0, dt=0.001, v=1500.0, k=0.1)
    >>> print(f"Velocity ratio: {ratio:.4f}")
    """
    if k == 0:
        return 1.0

    M = len(weights) - 1

    # Compute the cosine sum for 2D propagation
    # In 2D, the stencil applies to both x and y directions
    cosines = np.array([
        np.cos(m * k * h * np.cos(alpha)) +
        np.cos(m * k * h * np.sin(alpha)) - 2
        for m in range(1, M + 1)
    ])

    total = np.sum(weights[1:] * cosines)

    # Argument of arccos
    arg = 1 + (v**2 * dt**2 / h**2) * total

    # Clamp to valid range for arccos (numerical safety)
    arg = np.clip(arg, -1, 1)

    # Compute velocity ratio
    ratio = np.arccos(arg) / (v * k * dt)

    return float(ratio)


def dispersion_difference(
    weights: np.ndarray,
    h: float,
    dt: float,
    v: float,
    k: float,
    alpha: float = 0.0
) -> float:
    """Compute the absolute velocity error for a finite difference scheme.

    Parameters
    ----------
    weights : np.ndarray
        Symmetric stencil weights [a_0, a_1, ..., a_M].
    h : float
        Grid spacing.
    dt : float
        Time step.
    v : float
        True wave velocity.
    k : float
        Wavenumber.
    alpha : float, optional
        Propagation angle in radians.

    Returns
    -------
    float
        Absolute velocity error |v_FD - v|.
    """
    if k == 0:
        return 0.0

    M = len(weights) - 1

    cosines = np.array([
        np.cos(m * k * h * np.cos(alpha)) +
        np.cos(m * k * h * np.sin(alpha)) - 2
        for m in range(1, M + 1)
    ])

    total = np.sum(weights[1:] * cosines)
    theta = 1 + (v**2 * dt**2 / h**2) * total

    # Clamp to valid range
    theta = np.clip(theta, -1, 1)

    v_fd = np.arccos(theta) / (k * dt)
    return abs(v_fd - v)


def dispersion_error(
    weights: np.ndarray,
    h: float,
    dt: float,
    v: float,
    k_max: float,
    n_samples: int = 100
) -> float:
    """Compute the maximum dispersion error over a wavenumber range.

    Parameters
    ----------
    weights : np.ndarray
        Symmetric stencil weights [a_0, a_1, ..., a_M].
    h : float
        Grid spacing.
    dt : float
        Time step.
    v : float
        Wave velocity.
    k_max : float
        Maximum wavenumber to consider.
    n_samples : int, optional
        Number of wavenumber samples. Default is 100.

    Returns
    -------
    float
        Maximum absolute velocity error ratio |delta - 1| over [0, k_max].

    Examples
    --------
    >>> weights = fornberg_weights(M=4)
    >>> max_err = dispersion_error(weights, h=10.0, dt=0.001, v=1500.0, k_max=0.2)
    >>> print(f"Maximum dispersion error: {max_err:.4f}")
    """
    k_range = np.linspace(1e-10, k_max, n_samples)  # Avoid k=0
    errors = []

    for k in k_range:
        ratio = dispersion_ratio(weights, h, dt, v, k)
        errors.append(abs(ratio - 1))

    return max(errors)


def critical_dt(
    weights: np.ndarray,
    h: float,
    v_max: float,
    ndim: int = 2
) -> float:
    """Compute the critical time step for stability (CFL condition).

    For explicit time integration of the wave equation, the time step
    must satisfy the CFL condition to ensure stability.

    Parameters
    ----------
    weights : np.ndarray
        Symmetric stencil weights [a_0, a_1, ..., a_M].
    h : float
        Grid spacing (assumed uniform in all dimensions).
    v_max : float
        Maximum wave velocity in the model.
    ndim : int, optional
        Number of spatial dimensions. Default is 2.

    Returns
    -------
    float
        Critical time step. Use dt < dt_critical for stability.

    Notes
    -----
    The formula is:
        dt_critical = h / v_max * sqrt(sum|a_time| / (ndim * sum|a_space|))

    For second-order time discretization, sum|a_time| = 4.

    Examples
    --------
    >>> weights = fornberg_weights(M=4)
    >>> dt_crit = critical_dt(weights, h=10.0, v_max=4500.0)
    >>> print(f"Critical dt: {dt_crit:.6f} s")
    """
    sum_abs_space = np.sum(np.abs(weights))
    sum_abs_time = 4.0  # For second-order time: |1| + |-2| + |1|

    dt_critical = h * np.sqrt(sum_abs_time / (ndim * sum_abs_space)) / v_max
    return float(dt_critical)


def cfl_number(weights: np.ndarray, ndim: int = 2) -> float:
    """Compute the CFL factor for a given stencil.

    The critical time step is: dt_critical = h / v_max * cfl_factor

    Parameters
    ----------
    weights : np.ndarray
        Symmetric stencil weights [a_0, a_1, ..., a_M].
    ndim : int, optional
        Number of spatial dimensions. Default is 2.

    Returns
    -------
    float
        CFL factor.

    Examples
    --------
    >>> weights = fornberg_weights(M=4)
    >>> cfl = cfl_number(weights)
    >>> print(f"CFL factor: {cfl:.4f}")
    """
    sum_abs_space = np.sum(np.abs(weights))
    sum_abs_time = 4.0

    return float(np.sqrt(sum_abs_time / (ndim * sum_abs_space)))


def ricker_wavelet(
    t: np.ndarray,
    f0: float = 30.0,
    A: float = 1.0
) -> np.ndarray:
    """Generate a Ricker wavelet (Mexican hat wavelet).

    The Ricker wavelet is commonly used as a seismic source signature.

    Parameters
    ----------
    t : np.ndarray
        Time values.
    f0 : float, optional
        Peak frequency in Hz. Default is 30 Hz.
    A : float, optional
        Amplitude. Default is 1.

    Returns
    -------
    np.ndarray
        Wavelet values at times t.

    Notes
    -----
    The Ricker wavelet is defined as:
        w(t) = A * (1 - 2*pi^2*f0^2*(t - 1/f0)^2) * exp(-pi^2*f0^2*(t - 1/f0)^2)

    The wavelet is centered at t = 1/f0 and has maximum frequency content
    around 2.5 * f0.

    Examples
    --------
    >>> t = np.linspace(0, 0.1, 1000)
    >>> wavelet = ricker_wavelet(t, f0=30.0)
    """
    tau = (np.pi * f0 * (t - 1.0 / f0)) ** 2
    return A * (1 - 2 * tau) * np.exp(-tau)


def max_frequency_ricker(f0: float, threshold: float = 0.01) -> float:
    """Estimate the maximum significant frequency of a Ricker wavelet.

    Parameters
    ----------
    f0 : float
        Peak frequency of the Ricker wavelet in Hz.
    threshold : float, optional
        Amplitude threshold for "significant" frequency content.
        Default is 0.01 (1% of peak).

    Returns
    -------
    float
        Approximate maximum frequency where amplitude exceeds threshold.

    Notes
    -----
    As a rule of thumb, the maximum frequency is approximately 2.5 * f0
    to 3.0 * f0 for typical thresholds.
    """
    # The Ricker wavelet spectrum peaks at f0 and decays.
    # Empirically, f_max ~ 2.5 * f0 for 1% amplitude threshold
    return 2.5 * f0


def nyquist_spacing(f_max: float, v_min: float) -> float:
    """Compute the Nyquist-limited grid spacing.

    The grid must resolve the shortest wavelength: h <= v_min / (2 * f_max)

    Parameters
    ----------
    f_max : float
        Maximum frequency in the simulation (Hz).
    v_min : float
        Minimum velocity in the model (m/s).

    Returns
    -------
    float
        Maximum allowable grid spacing (m).

    Examples
    --------
    >>> h_max = nyquist_spacing(f_max=100.0, v_min=1500.0)
    >>> print(f"Maximum grid spacing: {h_max:.2f} m")
    """
    return v_min / (2.0 * f_max)
