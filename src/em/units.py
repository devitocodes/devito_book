"""Electromagnetic unit handling and physical constants.

Provides unit-aware physical constants and utilities for verifying
dimensional consistency in electromagnetic simulations using Pint.

The module ensures all electromagnetic quantities maintain proper units:
- Electric field E [V/m]
- Magnetic field H [A/m]
- Permittivity epsilon [F/m]
- Permeability mu [H/m]
- Conductivity sigma [S/m]

Example
-------
>>> from src.em.units import EMConstants, compute_cfl_dt
>>> c = EMConstants()
>>> print(f"Speed of light: {c.c0:.6e}")
>>> dt = compute_cfl_dt(dx=0.01, c=c.c0, CFL=0.9)
>>> print(f"Time step: {dt:.2e} s")
"""

from dataclasses import dataclass

import numpy as np

# Try to import pint for unit handling
try:
    import pint
    PINT_AVAILABLE = True
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity
except ImportError:
    PINT_AVAILABLE = False
    ureg = None
    Q_ = None


@dataclass
class EMConstants:
    """Electromagnetic physical constants.

    All values are in SI units. The class provides both raw float values
    and Pint Quantity objects (if Pint is available) for unit verification.

    Attributes
    ----------
    c0 : float
        Speed of light in vacuum [m/s]
    eps0 : float
        Permittivity of free space [F/m]
    mu0 : float
        Permeability of free space [H/m]
    eta0 : float
        Impedance of free space [Ohm]
    """

    c0: float = 299792458.0  # Speed of light [m/s]
    eps0: float = 8.8541878128e-12  # Permittivity of free space [F/m]
    mu0: float = 1.25663706212e-6  # Permeability of free space [H/m]

    def __post_init__(self):
        """Compute derived quantities and verify consistency."""
        # Impedance of free space
        self.eta0 = np.sqrt(self.mu0 / self.eps0)

        # Verify fundamental relation: c = 1/sqrt(eps0 * mu0)
        c_computed = 1.0 / np.sqrt(self.eps0 * self.mu0)
        if not np.isclose(c_computed, self.c0, rtol=1e-6):
            raise ValueError(
                f"Inconsistent constants: c0={self.c0}, but 1/sqrt(eps0*mu0)={c_computed}"
            )

    @property
    def c0_pint(self):
        """Speed of light with units (requires Pint)."""
        if not PINT_AVAILABLE:
            raise ImportError("Pint is required for unit-aware quantities")
        return self.c0 * ureg.meter / ureg.second

    @property
    def eps0_pint(self):
        """Permittivity of free space with units (requires Pint)."""
        if not PINT_AVAILABLE:
            raise ImportError("Pint is required for unit-aware quantities")
        return self.eps0 * ureg.farad / ureg.meter

    @property
    def mu0_pint(self):
        """Permeability of free space with units (requires Pint)."""
        if not PINT_AVAILABLE:
            raise ImportError("Pint is required for unit-aware quantities")
        return self.mu0 * ureg.henry / ureg.meter

    @property
    def eta0_pint(self):
        """Impedance of free space with units (requires Pint)."""
        if not PINT_AVAILABLE:
            raise ImportError("Pint is required for unit-aware quantities")
        return self.eta0 * ureg.ohm


def compute_wave_speed(eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Compute electromagnetic wave speed in a medium.

    Parameters
    ----------
    eps_r : float
        Relative permittivity (dielectric constant)
    mu_r : float
        Relative permeability

    Returns
    -------
    float
        Wave speed [m/s]
    """
    c = EMConstants()
    return c.c0 / np.sqrt(eps_r * mu_r)


def compute_wavelength(frequency: float, eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Compute wavelength in a medium.

    Parameters
    ----------
    frequency : float
        Frequency [Hz]
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability

    Returns
    -------
    float
        Wavelength [m]
    """
    c = compute_wave_speed(eps_r, mu_r)
    return c / frequency


def compute_cfl_dt(
    dx: float,
    c: float = None,
    CFL: float = 0.9,
    dy: float = None,
    dz: float = None,
) -> float:
    """Compute stable time step from CFL condition.

    For FDTD, the CFL condition in d dimensions is:
        c * dt <= 1/sqrt(1/dx^2 + 1/dy^2 + ...)

    Parameters
    ----------
    dx : float
        Grid spacing in x [m]
    c : float, optional
        Wave speed [m/s]. Default: speed of light.
    CFL : float
        CFL number (0 < CFL <= 1). Default: 0.9
    dy : float, optional
        Grid spacing in y [m] (for 2D/3D)
    dz : float, optional
        Grid spacing in z [m] (for 3D)

    Returns
    -------
    float
        Stable time step [s]
    """
    if c is None:
        c = EMConstants().c0

    # Compute stability limit
    inv_dx_sq = 1.0 / dx**2
    if dy is not None:
        inv_dx_sq += 1.0 / dy**2
    if dz is not None:
        inv_dx_sq += 1.0 / dz**2

    dt_max = 1.0 / (c * np.sqrt(inv_dx_sq))
    return CFL * dt_max


def compute_impedance(eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Compute wave impedance in a medium.

    Parameters
    ----------
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability

    Returns
    -------
    float
        Wave impedance [Ohm]
    """
    c = EMConstants()
    return c.eta0 * np.sqrt(mu_r / eps_r)


def reflection_coefficient(
    eps_r1: float, eps_r2: float, mu_r1: float = 1.0, mu_r2: float = 1.0
) -> float:
    """Compute reflection coefficient at normal incidence.

    Parameters
    ----------
    eps_r1 : float
        Relative permittivity of medium 1 (incident)
    eps_r2 : float
        Relative permittivity of medium 2 (transmitted)
    mu_r1 : float
        Relative permeability of medium 1
    mu_r2 : float
        Relative permeability of medium 2

    Returns
    -------
    float
        Reflection coefficient (can be negative for phase reversal)
    """
    eta1 = compute_impedance(eps_r1, mu_r1)
    eta2 = compute_impedance(eps_r2, mu_r2)
    return (eta2 - eta1) / (eta2 + eta1)


def transmission_coefficient(
    eps_r1: float, eps_r2: float, mu_r1: float = 1.0, mu_r2: float = 1.0
) -> float:
    """Compute transmission coefficient at normal incidence.

    Parameters
    ----------
    eps_r1 : float
        Relative permittivity of medium 1 (incident)
    eps_r2 : float
        Relative permittivity of medium 2 (transmitted)
    mu_r1 : float
        Relative permeability of medium 1
    mu_r2 : float
        Relative permeability of medium 2

    Returns
    -------
    float
        Transmission coefficient
    """
    eta1 = compute_impedance(eps_r1, mu_r1)
    eta2 = compute_impedance(eps_r2, mu_r2)
    return 2 * eta2 / (eta2 + eta1)


def skin_depth(frequency: float, sigma: float, eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Compute skin depth in a lossy medium.

    For a good conductor (sigma >> omega*eps), this simplifies to:
        delta = sqrt(2 / (omega * mu * sigma))

    Parameters
    ----------
    frequency : float
        Frequency [Hz]
    sigma : float
        Conductivity [S/m]
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability

    Returns
    -------
    float
        Skin depth [m]
    """
    c = EMConstants()
    omega = 2 * np.pi * frequency
    mu = mu_r * c.mu0
    eps = eps_r * c.eps0

    # General formula
    alpha = omega * np.sqrt(
        mu * eps / 2 * (np.sqrt(1 + (sigma / (omega * eps))**2) - 1)
    )
    if alpha > 0:
        return 1.0 / alpha
    else:
        return np.inf


def verify_units(
    E_magnitude: float,
    H_magnitude: float,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
    tolerance: float = 0.01,
) -> tuple[bool, float]:
    """Verify that E and H field magnitudes are consistent.

    In a plane wave, |E| = eta * |H| where eta is the wave impedance.

    Parameters
    ----------
    E_magnitude : float
        Electric field magnitude [V/m]
    H_magnitude : float
        Magnetic field magnitude [A/m]
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability
    tolerance : float
        Relative tolerance for verification

    Returns
    -------
    tuple
        (is_consistent: bool, ratio_error: float)
    """
    eta = compute_impedance(eps_r, mu_r)
    expected_ratio = eta
    actual_ratio = E_magnitude / H_magnitude if H_magnitude > 0 else np.inf

    error = abs(actual_ratio - expected_ratio) / expected_ratio
    is_consistent = error < tolerance

    return is_consistent, error


def points_per_wavelength(dx: float, frequency: float, eps_r: float = 1.0) -> float:
    """Compute the number of grid points per wavelength.

    A rule of thumb for FDTD is to use at least 10-20 points per wavelength
    for acceptable accuracy.

    Parameters
    ----------
    dx : float
        Grid spacing [m]
    frequency : float
        Frequency [Hz]
    eps_r : float
        Relative permittivity

    Returns
    -------
    float
        Points per wavelength
    """
    wavelength = compute_wavelength(frequency, eps_r)
    return wavelength / dx


def courant_number_1d(c: float, dt: float, dx: float) -> float:
    """Compute the 1D Courant number.

    Parameters
    ----------
    c : float
        Wave speed [m/s]
    dt : float
        Time step [s]
    dx : float
        Grid spacing [m]

    Returns
    -------
    float
        Courant number C = c*dt/dx
    """
    return c * dt / dx


def courant_number_2d(c: float, dt: float, dx: float, dy: float) -> float:
    """Compute the 2D Courant number.

    Parameters
    ----------
    c : float
        Wave speed [m/s]
    dt : float
        Time step [s]
    dx : float
        Grid spacing in x [m]
    dy : float
        Grid spacing in y [m]

    Returns
    -------
    float
        Courant number C = c*dt*sqrt(1/dx^2 + 1/dy^2)
    """
    return c * dt * np.sqrt(1/dx**2 + 1/dy**2)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PINT_AVAILABLE",
    "EMConstants",
    "compute_cfl_dt",
    "compute_impedance",
    "compute_wave_speed",
    "compute_wavelength",
    "courant_number_1d",
    "courant_number_2d",
    "points_per_wavelength",
    "reflection_coefficient",
    "skin_depth",
    "transmission_coefficient",
    "verify_units",
]
