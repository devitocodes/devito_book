"""Perfectly Matched Layer (PML) for FDTD simulations.

This module implements the Convolutional Perfectly Matched Layer (CPML),
which provides excellent absorption of outgoing waves with minimal
reflection across a wide frequency range and angles of incidence.

The CPML formulation uses auxiliary differential equations (ADE) with
complex frequency-shifted (CFS) coordinate stretching:

    s_i(ω) = κ_i + σ_i / (α_i + jω)

where:
    - κ_i: stretching factor (≥ 1)
    - σ_i: conductivity profile [S/m]
    - α_i: complex frequency shift [S/m]

References:
    - Berenger, J.P. (1994). "A Perfectly Matched Layer for the
      Absorption of Electromagnetic Waves." J. Compute. Phys., 114, 185-200.
    - Roden & Gedney (2000). "Convolutional PML (CPML): An efficient
      FDTD implementation of the CFS-PML." Microw. Opt. Tech. Lett., 27, 334-339.
    - Kuzuoglu & Mittra (1996). "Frequency dependence of the constitutive
      parameters of causal perfectly matched anisotropic absorbers."
      IEEE Microw. Guided Wave Lett., 6, 447-449.
"""

import numpy as np


def create_pml_sigma(
    n_pml: int,
    dx: float,
    sigma_max: float | None = None,
    m: float = 3.0,
    profile: str = "polynomial",
) -> np.ndarray:
    """Create conductivity profile for PML region.

    Parameters
    ----------
    n_pml : int
        Number of PML cells
    dx : float
        Grid spacing [m]
    sigma_max : float, optional
        Maximum conductivity [S/m].
        Default: calculated for optimal reflection coefficient.
    m : float
        Polynomial grading order (typically 3-4)
    profile : str
        Profile type: "polynomial" or "geometric"

    Returns
    -------
    np.ndarray
        Conductivity profile, shape (n_pml,)

    Notes
    -----
    The polynomial profile is:
        σ(x) = σ_max * (x / d)^m

    where d is the PML thickness and x is distance from the inner edge.

    For reflection coefficient R, the optimal σ_max is:
        σ_max = -(m + 1) * ln(R) / (2 * η * d)

    where η = sqrt(μ/ε) is the impedance.
    """
    d = n_pml * dx  # PML thickness

    if sigma_max is None:
        # Optimal value for reflection coefficient R ~ 1e-6
        # Using free-space impedance η0 ≈ 377 Ω
        R = 1e-6
        eta0 = 377.0
        sigma_max = -(m + 1) * np.log(R) / (2.0 * eta0 * d)

    # Distance from inner PML boundary (0 at inner edge, d at outer)
    x = np.linspace(0.5 * dx, d - 0.5 * dx, n_pml)

    if profile == "polynomial":
        sigma = sigma_max * (x / d) ** m
    elif profile == "geometric":
        # Geometric grading: σ(x) = σ_0 * g^(x/dx)
        g = (sigma_max / 1e-6) ** (dx / d)
        sigma = 1e-6 * g ** (x / dx)
    else:
        raise ValueError(f"Unknown profile type: {profile}")

    return sigma


def create_cpml_coefficients(
    n_pml: int,
    dx: float,
    dt: float,
    sigma_max: float | None = None,
    alpha_max: float = 0.0,
    kappa_max: float = 1.0,
    m_sigma: float = 3.0,
    m_alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    """Create CPML update coefficients.

    The CPML uses auxiliary variables (ψ) that convolve with field
    derivatives. The update equations are:

        ψ_n+1 = b * ψ_n + a * (∂E/∂x or ∂H/∂x)
        ∂/∂x̃ = ∂/∂x + ψ

    where:
        b = exp(-(σ/κ + α) * dt / ε₀)
        a = (σ / (σ*κ + α*κ²)) * (b - 1)   if σ > 0

    Parameters
    ----------
    n_pml : int
        Number of PML cells
    dx : float
        Grid spacing [m]
    dt : float
        Time step [s]
    sigma_max : float, optional
        Maximum PML conductivity [S/m]
    alpha_max : float
        Maximum CFS alpha value [S/m] (helps low-frequency absorption)
    kappa_max : float
        Maximum stretching factor (typically 1-15)
    m_sigma : float
        Polynomial order for sigma profile
    m_alpha : float
        Polynomial order for alpha profile

    Returns
    -------
    dict
        Dictionary containing:
        - 'b': decay coefficient array, shape (n_pml,)
        - 'a': update coefficient array, shape (n_pml,)
        - 'kappa': stretching factor array, shape (n_pml,)
        - 'sigma': conductivity array, shape (n_pml,)

    Notes
    -----
    The coefficients are designed for explicit FDTD update equations.
    Apply b and a to auxiliary field arrays at each time step.
    """
    # Constants
    eps0 = 8.854187817e-12  # Free-space permittivity [F/m]

    d = n_pml * dx  # PML thickness

    # Default sigma_max for ~1e-6 reflection
    if sigma_max is None:
        R = 1e-6
        eta0 = 377.0  # Free-space impedance
        sigma_max = -(m_sigma + 1) * np.log(R) / (2.0 * eta0 * d)

    # Normalized distance from inner boundary (0 at inner, 1 at outer)
    # Use half-cell offset for proper Yee grid alignment
    rho = (np.arange(n_pml) + 0.5) / n_pml

    # Graded profiles
    sigma = sigma_max * rho**m_sigma
    alpha = alpha_max * (1.0 - rho) ** m_alpha  # Decreases outward
    kappa = 1.0 + (kappa_max - 1.0) * rho**m_sigma

    # CPML coefficients
    # b = exp(-(sigma/kappa + alpha) * dt / eps0)
    b = np.exp(-(sigma / kappa + alpha) * dt / eps0)

    # a = (sigma / (sigma*kappa + alpha*kappa^2)) * (b - 1)
    # Handle sigma = 0 case (a = 0 when sigma = 0)
    denom = sigma * kappa + alpha * kappa**2
    a = np.where(
        sigma > 1e-20,
        (sigma / denom) * (b - 1.0),
        0.0,
    )

    return {
        "b": b,
        "a": a,
        "kappa": kappa,
        "sigma": sigma,
        "alpha": alpha,
    }


def create_pml_region_2d(
    Nx: int,
    Ny: int,
    n_pml: int,
    dx: float,
    dy: float,
    dt: float,
    sigma_max: float | None = None,
) -> dict[str, np.ndarray]:
    """Create 2D CPML coefficient arrays.

    Creates coefficient arrays for all four boundaries (left, right,
    bottom, top) in a 2D domain.

    Parameters
    ----------
    Nx : int
        Number of grid points in x-direction
    Ny : int
        Number of grid points in y-direction
    n_pml : int
        PML thickness in grid cells
    dx : float
        Grid spacing in x [m]
    dy : float
        Grid spacing in y [m]
    dt : float
        Time step [s]
    sigma_max : float, optional
        Maximum PML conductivity

    Returns
    -------
    dict
        Dictionary containing 2D coefficient arrays:
        - 'bx', 'ax': x-direction PML coefficients
        - 'by', 'ay': y-direction PML coefficients
        - 'kappa_x', 'kappa_y': stretching factors

    Notes
    -----
    The returned arrays have full grid size (Nx, Ny) with non-unity
    values only in the PML regions. The interior has b=1, a=0, kappa=1.
    """
    # Get 1D coefficients
    cpml_x = create_cpml_coefficients(n_pml, dx, dt, sigma_max)
    cpml_y = create_cpml_coefficients(n_pml, dy, dt, sigma_max)

    # Initialize full arrays with interior values
    bx = np.ones((Nx, Ny))
    ax = np.zeros((Nx, Ny))
    kappa_x = np.ones((Nx, Ny))

    by = np.ones((Nx, Ny))
    ay = np.zeros((Nx, Ny))
    kappa_y = np.ones((Nx, Ny))

    # Fill PML regions (coefficients go from boundary toward interior)

    # Left boundary (x = 0)
    for i in range(n_pml):
        idx = n_pml - 1 - i  # Reverse: outer (i=0) uses last coeff
        bx[i, :] = cpml_x["b"][idx]
        ax[i, :] = cpml_x["a"][idx]
        kappa_x[i, :] = cpml_x["kappa"][idx]

    # Right boundary (x = Nx-1)
    for i in range(n_pml):
        idx = i  # Forward: inner (i=0) uses first coeff
        bx[Nx - n_pml + i, :] = cpml_x["b"][idx]
        ax[Nx - n_pml + i, :] = cpml_x["a"][idx]
        kappa_x[Nx - n_pml + i, :] = cpml_x["kappa"][idx]

    # Bottom boundary (y = 0)
    for j in range(n_pml):
        idx = n_pml - 1 - j
        by[:, j] = cpml_y["b"][idx]
        ay[:, j] = cpml_y["a"][idx]
        kappa_y[:, j] = cpml_y["kappa"][idx]

    # Top boundary (y = Ny-1)
    for j in range(n_pml):
        idx = j
        by[:, Ny - n_pml + j] = cpml_y["b"][idx]
        ay[:, Ny - n_pml + j] = cpml_y["a"][idx]
        kappa_y[:, Ny - n_pml + j] = cpml_y["kappa"][idx]

    return {
        "bx": bx,
        "ax": ax,
        "kappa_x": kappa_x,
        "by": by,
        "ay": ay,
        "kappa_y": kappa_y,
    }


def pml_reflection_coefficient(
    n_pml: int,
    dx: float,
    sigma_max: float,
    m: float = 3.0,
) -> float:
    """Compute theoretical reflection coefficient for PML.

    Parameters
    ----------
    n_pml : int
        Number of PML cells
    dx : float
        Grid spacing [m]
    sigma_max : float
        Maximum conductivity [S/m]
    m : float
        Polynomial grading order

    Returns
    -------
    float
        Theoretical reflection coefficient R

    Notes
    -----
    The theoretical reflection for normal incidence is:
        R = exp(-2 * ∫₀ᵈ σ(x)/η dx)
          = exp(-2 * σ_max * d / ((m+1) * η))

    This is a best-case estimate; actual reflection may be higher
    at oblique incidence or with discrete sampling effects.
    """
    eta0 = 377.0  # Free-space impedance
    d = n_pml * dx
    R = np.exp(-2.0 * sigma_max * d / ((m + 1) * eta0))
    return R
