"""Analytical solutions for electromagnetic problems.

This module provides exact solutions for verification of FDTD simulations:
    - 1D and 2D plane wave propagation
    - Rectangular cavity resonant modes
    - Waveguide cutoff frequencies

These solutions enable rigorous convergence testing and validation
of the numerical implementation.

Physical constants used:
    - c0 = 299792458 m/s (speed of light in vacuum)
    - μ0 = 4π × 10⁻⁷ H/m (permeability of free space)
    - ε0 = 8.854187817 × 10⁻¹² F/m (permittivity of free space)
"""

import numpy as np

# Physical constants
C0 = 299792458.0  # Speed of light in vacuum [m/s]
MU0 = 4.0 * np.pi * 1e-7  # Permeability of free space [H/m]
EPS0 = 8.854187817e-12  # Permittivity of free space [F/m]
ETA0 = np.sqrt(MU0 / EPS0)  # Impedance of free space ≈ 377 Ω


def exact_plane_wave_1d(
    x: np.ndarray,
    t: float,
    f0: float,
    E0: float = 1.0,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
    direction: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact solution for 1D plane wave propagation.

    For a forward-traveling plane wave in a lossless medium:
        E_y(x, t) = E0 * sin(ω*t - k*x)
        H_z(x, t) = E0/η * sin(ω*t - k*x)

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates [m]
    t : float
        Time [s]
    f0 : float
        Frequency [Hz]
    E0 : float
        Electric field amplitude [V/m]
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability
    direction : int
        Propagation direction: +1 for +x, -1 for -x

    Returns
    -------
    Ey : np.ndarray
        Electric field at positions x and time t
    Hz : np.ndarray
        Magnetic field at positions x and time t

    Notes
    -----
    The wave speed is c = c0 / sqrt(eps_r * mu_r)
    The wave impedance is η = η0 * sqrt(mu_r / eps_r)
    The wave number is k = ω * sqrt(eps_r * mu_r) / c0
    """
    omega = 2.0 * np.pi * f0
    c = C0 / np.sqrt(eps_r * mu_r)
    k = omega / c
    eta = ETA0 * np.sqrt(mu_r / eps_r)

    phase = omega * t - direction * k * x
    Ey = E0 * np.sin(phase)
    Hz = direction * (E0 / eta) * np.sin(phase)

    return Ey, Hz


def exact_plane_wave_2d(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    f0: float,
    theta: float = 0.0,
    E0: float = 1.0,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
    polarization: str = "TMz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact solution for 2D plane wave at arbitrary angle.

    For TMz polarization (Ez, Hx, Hy):
        Ez(x, y, t) = E0 * sin(ω*t - kx*x - ky*y)
        Hx(x, y, t) = -(ky/ωμ) * E0 * sin(ω*t - kx*x - ky*y)
        Hy(x, y, t) = (kx/ωμ) * E0 * sin(ω*t - kx*x - ky*y)

    For TEz polarization (Hz, Ex, Ey):
        Hz(x, y, t) = H0 * sin(ω*t - kx*x - ky*y)
        Ex(x, y, t) = (ky/ωε) * H0 * sin(ω*t - kx*x - ky*y)
        Ey(x, y, t) = -(kx/ωε) * H0 * sin(ω*t - kx*x - ky*y)

    Parameters
    ----------
    x : np.ndarray
        x-coordinates [m], shape (Nx,) or (Nx, Ny)
    y : np.ndarray
        y-coordinates [m], shape (Ny,) or (Nx, Ny)
    t : float
        Time [s]
    f0 : float
        Frequency [Hz]
    theta : float
        Angle of propagation from +x axis [radians]
    E0 : float
        Field amplitude [V/m or A/m]
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability
    polarization : str
        "TMz" (Ez polarization) or "TEz" (Hz polarization)

    Returns
    -------
    For TMz: (Ez, Hx, Hy)
    For TEz: (Hz, Ex, Ey)

    Notes
    -----
    The wave vector components are:
        kx = k * cos(θ)
        ky = k * sin(θ)
    """
    # Create meshgrid if needed
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y, indexing='ij')
    else:
        X, Y = x, y

    omega = 2.0 * np.pi * f0
    c = C0 / np.sqrt(eps_r * mu_r)
    k = omega / c
    kx = k * np.cos(theta)
    ky = k * np.sin(theta)

    eps = EPS0 * eps_r
    mu = MU0 * mu_r
    eta = np.sqrt(mu / eps)

    phase = omega * t - kx * X - ky * Y

    if polarization.upper() == "TMZ":
        Ez = E0 * np.sin(phase)
        Hx = -(E0 / eta) * np.sin(theta) * np.sin(phase)
        Hy = (E0 / eta) * np.cos(theta) * np.sin(phase)
        return Ez, Hx, Hy
    elif polarization.upper() == "TEZ":
        H0 = E0 / eta  # Convert to H-field amplitude
        Hz = H0 * np.sin(phase)
        Ex = eta * H0 * np.sin(theta) * np.sin(phase)
        Ey = -eta * H0 * np.cos(theta) * np.sin(phase)
        return Hz, Ex, Ey
    else:
        raise ValueError(f"Unknown polarization: {polarization}")


def cavity_resonant_frequencies(
    a: float,
    b: float,
    c: float = None,
    m_max: int = 3,
    n_max: int = 3,
    p_max: int = 0,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
) -> list[dict]:
    """Compute resonant frequencies of a rectangular cavity.

    For a rectangular cavity with dimensions a × b × c, the resonant
    frequencies are given by:

    2D (a × b):
        f_mn = (c0 / (2*sqrt(eps_r*mu_r))) * sqrt((m/a)² + (n/b)²)

    3D (a × b × c):
        f_mnp = (c0 / (2*sqrt(eps_r*mu_r))) * sqrt((m/a)² + (n/b)² + (p/c)²)

    Parameters
    ----------
    a : float
        Cavity dimension in x [m]
    b : float
        Cavity dimension in y [m]
    c : float, optional
        Cavity dimension in z [m]. If None, 2D cavity.
    m_max : int
        Maximum mode number in x
    n_max : int
        Maximum mode number in y
    p_max : int
        Maximum mode number in z (for 3D)
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability

    Returns
    -------
    list of dict
        List of mode info dictionaries, sorted by frequency:
        - 'f': resonant frequency [Hz]
        - 'm', 'n', 'p': mode numbers
        - 'mode': string like "TM_11" or "TE_110"

    Notes
    -----
    For TMz modes in 2D, m ≥ 1 and n ≥ 1.
    For TEz modes in 2D, m ≥ 1 or n ≥ 1 (not both zero).
    """
    v = C0 / np.sqrt(eps_r * mu_r)  # Wave velocity

    modes = []

    if c is None:
        # 2D cavity
        for m in range(m_max + 1):
            for n in range(n_max + 1):
                if m == 0 and n == 0:
                    continue  # No (0,0) mode

                f = (v / 2.0) * np.sqrt((m / a) ** 2 + (n / b) ** 2)

                # Determine mode type
                if m > 0 and n > 0:
                    mode_type = f"TM_{m}{n}"
                else:
                    mode_type = f"TE_{m}{n}"

                modes.append({
                    "f": f,
                    "m": m,
                    "n": n,
                    "p": 0,
                    "mode": mode_type,
                })
    else:
        # 3D cavity
        for m in range(m_max + 1):
            for n in range(n_max + 1):
                for p in range(p_max + 1):
                    # At least two indices must be non-zero
                    nonzero = (m > 0) + (n > 0) + (p > 0)
                    if nonzero < 2:
                        continue

                    f = (v / 2.0) * np.sqrt(
                        (m / a) ** 2 + (n / b) ** 2 + (p / c) ** 2
                    )

                    mode_type = f"TE/TM_{m}{n}{p}"

                    modes.append({
                        "f": f,
                        "m": m,
                        "n": n,
                        "p": p,
                        "mode": mode_type,
                    })

    # Sort by frequency
    modes.sort(key=lambda x: x["f"])

    return modes


def cavity_field_2d_tmz(
    x: np.ndarray,
    y: np.ndarray,
    a: float,
    b: float,
    m: int,
    n: int,
    t: float,
    E0: float = 1.0,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact field distribution for TMz mode in 2D rectangular cavity.

    The TMz modes have:
        Ez = E0 * sin(m*π*x/a) * sin(n*π*y/b) * cos(ω*t)
        Hx = (n*π/(ωμ*b)) * E0 * sin(m*π*x/a) * cos(n*π*y/b) * sin(ω*t)
        Hy = -(m*π/(ωμ*a)) * E0 * cos(m*π*x/a) * sin(n*π*y/b) * sin(ω*t)

    Parameters
    ----------
    x : np.ndarray
        x-coordinates [m], shape (Nx,)
    y : np.ndarray
        y-coordinates [m], shape (Ny,)
    a : float
        Cavity width in x [m]
    b : float
        Cavity width in y [m]
    m : int
        Mode number in x (m ≥ 1)
    n : int
        Mode number in y (n ≥ 1)
    t : float
        Time [s]
    E0 : float
        Electric field amplitude [V/m]
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability

    Returns
    -------
    Ez : np.ndarray
        Electric field, shape (Nx, Ny)
    Hx : np.ndarray
        Magnetic field x-component, shape (Nx, Ny)
    Hy : np.ndarray
        Magnetic field y-component, shape (Nx, Ny)
    """
    if m < 1 or n < 1:
        raise ValueError(f"TMz modes require m ≥ 1 and n ≥ 1, got m={m}, n={n}")

    X, Y = np.meshgrid(x, y, indexing='ij')

    # Resonant frequency
    v = C0 / np.sqrt(eps_r * mu_r)
    f = (v / 2.0) * np.sqrt((m / a) ** 2 + (n / b) ** 2)
    omega = 2.0 * np.pi * f
    mu = MU0 * mu_r

    # Spatial patterns
    sin_mx = np.sin(m * np.pi * X / a)
    sin_ny = np.sin(n * np.pi * Y / b)
    cos_mx = np.cos(m * np.pi * X / a)
    cos_ny = np.cos(n * np.pi * Y / b)

    # Field components
    Ez = E0 * sin_mx * sin_ny * np.cos(omega * t)
    Hx = (n * np.pi / (omega * mu * b)) * E0 * sin_mx * cos_ny * np.sin(omega * t)
    Hy = -(m * np.pi / (omega * mu * a)) * E0 * cos_mx * sin_ny * np.sin(omega * t)

    return Ez, Hx, Hy


def waveguide_cutoff_frequency(
    a: float,
    b: float = None,
    m: int = 1,
    n: int = 0,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
) -> float:
    """Compute cutoff frequency for rectangular waveguide mode.

    The cutoff frequency for the TE_mn or TM_mn mode is:
        f_c = (c0 / (2*sqrt(eps_r*mu_r))) * sqrt((m/a)² + (n/b)²)

    Parameters
    ----------
    a : float
        Waveguide width (larger dimension) [m]
    b : float, optional
        Waveguide height [m]. Default: a/2
    m : int
        Mode number in x (broad dimension)
    n : int
        Mode number in y (narrow dimension)
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability

    Returns
    -------
    float
        Cutoff frequency [Hz]

    Notes
    -----
    The dominant mode in a rectangular waveguide is TE_10
    (m=1, n=0), which has the lowest cutoff frequency.

    For propagation, the operating frequency must be above
    the cutoff: f > f_c.

    The waveguide wavelength is:
        λ_g = λ_0 / sqrt(1 - (f_c/f)²)
    """
    if b is None:
        b = a / 2

    v = C0 / np.sqrt(eps_r * mu_r)
    f_c = (v / 2.0) * np.sqrt((m / a) ** 2 + (n / b) ** 2)

    return f_c


def standing_wave_electric_field(
    x: np.ndarray,
    t: float,
    L: float,
    n: int,
    E0: float = 1.0,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
) -> np.ndarray:
    """Electric field for nth standing wave mode between PEC boundaries.

    For a region 0 ≤ x ≤ L with PEC (E = 0) at both ends:
        E(x, t) = E0 * sin(n*π*x/L) * cos(ω_n*t)

    where ω_n = n*π*c/L.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates [m]
    t : float
        Time [s]
    L : float
        Cavity length [m]
    n : int
        Mode number (n ≥ 1)
    E0 : float
        Amplitude [V/m]
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability

    Returns
    -------
    np.ndarray
        Electric field at positions x and time t
    """
    c = C0 / np.sqrt(eps_r * mu_r)
    omega_n = n * np.pi * c / L

    return E0 * np.sin(n * np.pi * x / L) * np.cos(omega_n * t)


def gaussian_pulse_analytical(
    x: np.ndarray,
    t: float,
    x0: float,
    sigma: float,
    c: float,
    direction: int = 1,
) -> np.ndarray:
    """Analytical Gaussian pulse propagation in 1D.

    A Gaussian pulse traveling in a lossless medium:
        E(x, t) = exp(-((x - x0 - c*t) / sigma)²)  for +x direction
        E(x, t) = exp(-((x - x0 + c*t) / sigma)²)  for -x direction

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates [m]
    t : float
        Time [s]
    x0 : float
        Initial pulse center [m]
    sigma : float
        Pulse width [m]
    c : float
        Wave speed [m/s]
    direction : int
        +1 for +x propagation, -1 for -x propagation

    Returns
    -------
    np.ndarray
        Gaussian pulse profile
    """
    return np.exp(-((x - x0 - direction * c * t) / sigma) ** 2)
