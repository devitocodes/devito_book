"""
Von Neumann stability analysis tools.

This module provides functions for:
- Computing amplification factors for common FD schemes
- Checking stability conditions (CFL)
- Computing maximum stable time steps

Theory
------
Von Neumann stability analysis examines the growth of Fourier modes
in a finite difference scheme. For a mode u_j^n = g^n * exp(i*theta*j),
the scheme is stable if |g| <= 1 for all wave numbers theta in [0, 2*pi].
"""


import numpy as np


def amplification_factor_diffusion(
    r: float,
    theta: float | np.ndarray
) -> float | np.ndarray:
    """
    Compute amplification factor for FTCS diffusion scheme.

    The Forward-Time Central-Space scheme for u_t = alpha * u_xx:
        u_j^{n+1} = u_j^n + r*(u_{j+1}^n - 2*u_j^n + u_{j-1}^n)

    where r = alpha * dt / dx^2 (mesh ratio / Fourier number).

    Parameters
    ----------
    r : float
        Mesh ratio (Fourier number): alpha * dt / dx^2
    theta : float or ndarray
        Dimensionless wave number(s): xi * dx, in [0, 2*pi]

    Returns
    -------
    g : float or ndarray
        Amplification factor(s)

    Notes
    -----
    Stability requires |g| <= 1, which gives 0 <= r <= 0.5.
    """
    return 1 - 4 * r * np.sin(theta / 2) ** 2


def amplification_factor_advection_upwind(
    nu: float,
    theta: float | np.ndarray
) -> complex | np.ndarray:
    """
    Compute amplification factor for first-order upwind advection.

    The upwind scheme for u_t + c*u_x = 0 (c > 0):
        u_j^{n+1} = (1-nu)*u_j^n + nu*u_{j-1}^n

    where nu = c * dt / dx (Courant number).

    Parameters
    ----------
    nu : float
        Courant number: c * dt / dx
    theta : float or ndarray
        Dimensionless wave number(s): xi * dx, in [0, 2*pi]

    Returns
    -------
    g : complex or ndarray
        Complex amplification factor(s)

    Notes
    -----
    Stability requires |g| <= 1, which gives 0 <= nu <= 1 (CFL condition).
    """
    return 1 - nu * (1 - np.cos(theta)) - 1j * nu * np.sin(theta)


def amplification_factor_wave(
    nu: float,
    theta: float | np.ndarray
) -> complex | np.ndarray:
    """
    Compute amplification factor for leapfrog wave equation scheme.

    The leapfrog scheme for u_tt = c^2 * u_xx:
        u_j^{n+1} = 2*u_j^n - u_j^{n-1} + nu^2*(u_{j+1}^n - 2*u_j^n + u_{j-1}^n)

    where nu = c * dt / dx (Courant number).

    Parameters
    ----------
    nu : float
        Courant number: c * dt / dx
    theta : float or ndarray
        Dimensionless wave number(s): xi * dx, in [0, 2*pi]

    Returns
    -------
    g : complex or ndarray
        Complex amplification factor(s). Returns one root of the
        quadratic; both roots have |g| = 1 when stable.

    Notes
    -----
    Stability requires |g| = 1 (no growth or decay), which gives
    nu <= 1 (CFL condition).
    """
    sin2 = np.sin(theta / 2) ** 2
    a = 1 - 2 * nu**2 * sin2

    # Discriminant of quadratic g^2 - 2*a*g + 1 = 0
    discriminant = a**2 - 1

    if np.isscalar(discriminant):
        if discriminant < 0:
            # Stable: |g| = 1
            return a + 1j * np.sqrt(-discriminant)
        else:
            # Unstable: |g| != 1
            return a + np.sqrt(discriminant)
    else:
        # Array case
        result = np.zeros_like(theta, dtype=complex)
        stable = discriminant < 0
        result[stable] = a[stable] + 1j * np.sqrt(-discriminant[stable])
        result[~stable] = a[~stable] + np.sqrt(discriminant[~stable])
        return result


def compute_cfl(
    c: float,
    dt: float,
    dx: float,
    ndim: int = 1
) -> float:
    """
    Compute CFL number for wave equation.

    The CFL number is a dimensionless ratio that characterizes the
    relationship between the physical wave speed and the numerical
    propagation speed.

    Parameters
    ----------
    c : float
        Wave speed (velocity)
    dt : float
        Time step
    dx : float
        Grid spacing (minimum if non-uniform)
    ndim : int, optional
        Number of spatial dimensions (default: 1)

    Returns
    -------
    cfl : float
        CFL number. For stability, cfl <= 1/sqrt(ndim).

    Notes
    -----
    For d dimensions with equal spacing, stability requires:
        CFL <= 1/sqrt(d)

    That is:
        1D: CFL <= 1
        2D: CFL <= 1/sqrt(2) ≈ 0.707
        3D: CFL <= 1/sqrt(3) ≈ 0.577
    """
    return c * dt / dx


def stable_timestep_diffusion(
    alpha: float,
    dx: float,
    cfl_max: float = 0.4,
    ndim: int = 1
) -> float:
    """
    Compute maximum stable time step for explicit diffusion.

    Parameters
    ----------
    alpha : float
        Diffusion coefficient
    dx : float
        Grid spacing (minimum if non-uniform)
    cfl_max : float, optional
        Safety factor (default: 0.4, max stable is 0.5 in 1D)
    ndim : int, optional
        Number of spatial dimensions (default: 1)

    Returns
    -------
    dt : float
        Maximum stable time step

    Notes
    -----
    For FTCS scheme in d dimensions:
        dt <= dx^2 / (2 * d * alpha)

    The cfl_max parameter should be < 0.5/d for safety margin.
    """
    return cfl_max * dx**2 / (ndim * alpha)


def stable_timestep_wave(
    c: float,
    dx: float,
    cfl_max: float = 0.9,
    ndim: int = 1
) -> float:
    """
    Compute maximum stable time step for explicit wave equation.

    Parameters
    ----------
    c : float
        Wave speed (maximum velocity in heterogeneous media)
    dx : float
        Grid spacing (minimum if non-uniform)
    cfl_max : float, optional
        Target CFL number (default: 0.9)
    ndim : int, optional
        Number of spatial dimensions (default: 1)

    Returns
    -------
    dt : float
        Maximum stable time step

    Notes
    -----
    For leapfrog scheme in d dimensions:
        dt <= dx / (c * sqrt(d))

    The cfl_max parameter should account for this factor of sqrt(d).
    """
    return cfl_max * dx / (c * np.sqrt(ndim))


def check_stability_diffusion(
    alpha: float,
    dt: float,
    dx: float,
    ndim: int = 1
) -> tuple[bool, float, float]:
    """
    Check stability of FTCS diffusion scheme.

    Parameters
    ----------
    alpha : float
        Diffusion coefficient
    dt : float
        Time step
    dx : float
        Grid spacing
    ndim : int, optional
        Number of spatial dimensions (default: 1)

    Returns
    -------
    is_stable : bool
        True if scheme is stable
    r : float
        Mesh ratio (Fourier number): alpha * dt / dx^2
    r_max : float
        Maximum stable mesh ratio: 1 / (2 * ndim)

    Examples
    --------
    >>> stable, r, r_max = check_stability_diffusion(1.0, 0.001, 0.1)
    >>> print(f"r = {r:.3f}, r_max = {r_max:.3f}, stable = {stable}")
    r = 0.100, r_max = 0.500, stable = True
    """
    r = alpha * dt / dx**2
    r_max = 0.5 / ndim
    return r <= r_max, r, r_max


def check_stability_wave(
    c: float,
    dt: float,
    dx: float,
    ndim: int = 1
) -> tuple[bool, float, float]:
    """
    Check stability of leapfrog wave equation scheme.

    Parameters
    ----------
    c : float
        Wave speed
    dt : float
        Time step
    dx : float
        Grid spacing
    ndim : int, optional
        Number of spatial dimensions (default: 1)

    Returns
    -------
    is_stable : bool
        True if scheme is stable
    cfl : float
        CFL number: c * dt / dx
    cfl_max : float
        Maximum stable CFL: 1 / sqrt(ndim)

    Examples
    --------
    >>> stable, cfl, cfl_max = check_stability_wave(1500., 0.0001, 10.)
    >>> print(f"CFL = {cfl:.3f}, CFL_max = {cfl_max:.3f}, stable = {stable}")
    CFL = 0.015, CFL_max = 1.000, stable = True
    """
    cfl = c * dt / dx
    cfl_max = 1.0 / np.sqrt(ndim)
    return cfl <= cfl_max, cfl, cfl_max


def plot_amplification_factors(save_path: str = None):
    """
    Plot amplification factors for various schemes.

    Creates a figure showing |g(theta)| for:
    - Diffusion (FTCS) with different r values
    - Advection (upwind) with different nu values
    - Wave equation (leapfrog) with different nu values

    Parameters
    ----------
    save_path : str, optional
        Path to save figure (if None, displays interactively)
    """
    import matplotlib.pyplot as plt

    theta = np.linspace(0, 2*np.pi, 200)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Diffusion
    ax = axes[0]
    for r in [0.1, 0.25, 0.4, 0.5, 0.6]:
        g = amplification_factor_diffusion(r, theta)
        label = f'r = {r}'
        linestyle = '-' if r <= 0.5 else '--'
        ax.plot(theta, np.abs(g), label=label, linestyle=linestyle)
    ax.axhline(y=1, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$|g|$')
    ax.set_title('Diffusion (FTCS)')
    ax.legend()
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 1.5)

    # Advection
    ax = axes[1]
    for nu in [0.25, 0.5, 0.75, 1.0, 1.25]:
        g = amplification_factor_advection_upwind(nu, theta)
        label = f'$\\nu$ = {nu}'
        linestyle = '-' if nu <= 1.0 else '--'
        ax.plot(theta, np.abs(g), label=label, linestyle=linestyle)
    ax.axhline(y=1, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$|g|$')
    ax.set_title('Advection (Upwind)')
    ax.legend()
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 1.5)

    # Wave equation
    ax = axes[2]
    for nu in [0.5, 0.75, 0.9, 1.0, 1.1]:
        g = amplification_factor_wave(nu, theta)
        label = f'$\\nu$ = {nu}'
        linestyle = '-' if nu <= 1.0 else '--'
        ax.plot(theta, np.abs(g), label=label, linestyle=linestyle)
    ax.axhline(y=1, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$|g|$')
    ax.set_title('Wave (Leapfrog)')
    ax.legend()
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    # Example usage and verification
    print("Von Neumann Stability Analysis Examples")
    print("=" * 50)

    # Diffusion stability
    print("\n1. Diffusion (FTCS) Stability")
    print("-" * 30)
    alpha = 0.1  # diffusion coefficient
    dx = 0.01    # grid spacing

    for r in [0.3, 0.5, 0.6]:
        dt = r * dx**2 / alpha
        stable, r_actual, r_max = check_stability_diffusion(alpha, dt, dx)
        status = "STABLE" if stable else "UNSTABLE"
        print(f"r = {r:.2f}: dt = {dt:.6f}, {status}")

    # Advection stability
    print("\n2. Advection (Upwind) Stability")
    print("-" * 30)
    c = 1.0  # wave speed
    dx = 0.01

    for nu in [0.5, 1.0, 1.5]:
        dt = nu * dx / c
        cfl = compute_cfl(c, dt, dx)
        status = "STABLE" if cfl <= 1.0 else "UNSTABLE"
        print(f"nu = {nu:.2f}: dt = {dt:.6f}, CFL = {cfl:.2f}, {status}")

    # Wave equation stability
    print("\n3. Wave Equation (Leapfrog) Stability")
    print("-" * 30)
    c = 1500.0  # velocity m/s
    dx = 10.0   # grid spacing m

    for ndim in [1, 2, 3]:
        dt = stable_timestep_wave(c, dx, cfl_max=0.9, ndim=ndim)
        stable, cfl, cfl_max = check_stability_wave(c, dt, dx, ndim)
        print(f"{ndim}D: dt = {dt:.6f}, CFL = {cfl:.3f}, CFL_max = {cfl_max:.3f}")

    # Generate plots
    print("\nGenerating amplification factor plots...")
    plot_amplification_factors("amplification_factors.png")
    print("Saved to amplification_factors.png")
