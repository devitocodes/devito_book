"""Dielectric slab waveguide utilities.

Provides analytical solutions and utilities for dielectric slab waveguides,
including mode calculation, effective index computation, and mode profiles.

The dielectric slab waveguide consists of:
- Core: refractive index n_core (higher)
- Cladding: refractive index n_clad (lower)
- Guided modes exist when n_clad < n_eff < n_core

For TE modes, the eigenvalue equation is:
    tan(k_x * d/2) = gamma / k_x  (symmetric modes)
    cot(k_x * d/2) = -gamma / k_x (antisymmetric modes)

where:
    k_x = k_0 * sqrt(n_core^2 - n_eff^2)  (transverse wavenumber in core)
    gamma = k_0 * sqrt(n_eff^2 - n_clad^2)  (decay constant in cladding)
    k_0 = 2*pi/lambda_0  (free-space wavenumber)

References
----------
.. [1] B.E.A. Saleh and M.C. Teich, "Fundamentals of Photonics,"
       2nd ed., Wiley, 2007, Chapter 8.

.. [2] A. Yariv and P. Yeh, "Photonics: Optical Electronics in Modern
       Communications," 6th ed., Oxford University Press, 2007.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass
class WaveguideMode:
    """Represents a guided mode of a dielectric waveguide.

    Attributes
    ----------
    mode_number : int
        Mode index (0 = fundamental, 1 = first higher order, etc.)
    n_eff : float
        Effective refractive index
    k_x : float
        Transverse wavenumber in core [rad/m]
    gamma : float
        Decay constant in cladding [rad/m]
    symmetry : str
        'symmetric' or 'antisymmetric'
    beta : float
        Propagation constant [rad/m]
    """
    mode_number: int
    n_eff: float
    k_x: float
    gamma: float
    symmetry: str
    beta: float


@dataclass
class SlabWaveguide:
    """Dielectric slab waveguide parameters and mode solver.

    Attributes
    ----------
    n_core : float
        Core refractive index
    n_clad : float
        Cladding refractive index
    thickness : float
        Core thickness [m]
    wavelength : float
        Free-space wavelength [m]
    """
    n_core: float
    n_clad: float
    thickness: float
    wavelength: float

    def __post_init__(self):
        """Compute derived quantities."""
        if self.n_core <= self.n_clad:
            raise ValueError("n_core must be greater than n_clad for guiding")

        self.k0 = 2 * np.pi / self.wavelength
        self.d = self.thickness

        # V-number (normalized frequency)
        self.V = self.k0 * (self.d / 2) * np.sqrt(self.n_core**2 - self.n_clad**2)

        # Maximum number of guided modes (approximate)
        self.max_modes = int(np.floor(self.V / (np.pi/2))) + 1

    def _eigenvalue_equation_symmetric(self, n_eff: float) -> float:
        """Eigenvalue equation for symmetric TE modes.

        Returns: tan(k_x * d/2) - gamma/k_x
        """
        if n_eff >= self.n_core or n_eff <= self.n_clad:
            return np.inf

        k_x = self.k0 * np.sqrt(self.n_core**2 - n_eff**2)
        gamma = self.k0 * np.sqrt(n_eff**2 - self.n_clad**2)

        return np.tan(k_x * self.d / 2) - gamma / k_x

    def _eigenvalue_equation_antisymmetric(self, n_eff: float) -> float:
        """Eigenvalue equation for antisymmetric TE modes.

        Returns: cot(k_x * d/2) + gamma/k_x
        """
        if n_eff >= self.n_core or n_eff <= self.n_clad:
            return np.inf

        k_x = self.k0 * np.sqrt(self.n_core**2 - n_eff**2)
        gamma = self.k0 * np.sqrt(n_eff**2 - self.n_clad**2)

        tan_val = np.tan(k_x * self.d / 2)
        if abs(tan_val) < 1e-10:
            return np.inf

        return 1/tan_val + gamma / k_x

    def find_modes(self, num_points: int = 1000) -> list[WaveguideMode]:
        """Find all guided modes of the waveguide.

        Uses root-finding on the eigenvalue equation to locate modes.

        Parameters
        ----------
        num_points : int
            Number of search points for initial bracketing

        Returns
        -------
        list
            List of WaveguideMode objects, sorted by effective index
        """
        modes = []

        # Search range for n_eff
        n_eff_min = self.n_clad + 1e-10
        n_eff_max = self.n_core - 1e-10

        n_eff_vals = np.linspace(n_eff_min, n_eff_max, num_points)

        # Find symmetric modes
        for i in range(len(n_eff_vals) - 1):
            try:
                f1 = self._eigenvalue_equation_symmetric(n_eff_vals[i])
                f2 = self._eigenvalue_equation_symmetric(n_eff_vals[i+1])

                if np.isfinite(f1) and np.isfinite(f2) and f1 * f2 < 0:
                    n_eff = brentq(
                        self._eigenvalue_equation_symmetric,
                        n_eff_vals[i], n_eff_vals[i+1],
                        xtol=1e-12
                    )
                    modes.append(self._create_mode(n_eff, 'symmetric'))
            except (ValueError, RuntimeError):
                continue

        # Find antisymmetric modes
        for i in range(len(n_eff_vals) - 1):
            try:
                f1 = self._eigenvalue_equation_antisymmetric(n_eff_vals[i])
                f2 = self._eigenvalue_equation_antisymmetric(n_eff_vals[i+1])

                if np.isfinite(f1) and np.isfinite(f2) and f1 * f2 < 0:
                    n_eff = brentq(
                        self._eigenvalue_equation_antisymmetric,
                        n_eff_vals[i], n_eff_vals[i+1],
                        xtol=1e-12
                    )
                    modes.append(self._create_mode(n_eff, 'antisymmetric'))
            except (ValueError, RuntimeError):
                continue

        # Sort by effective index (highest first = fundamental)
        modes.sort(key=lambda m: -m.n_eff)

        # Assign mode numbers
        for i, mode in enumerate(modes):
            mode.mode_number = i

        return modes

    def _create_mode(self, n_eff: float, symmetry: str) -> WaveguideMode:
        """Create WaveguideMode object from effective index."""
        k_x = self.k0 * np.sqrt(self.n_core**2 - n_eff**2)
        gamma = self.k0 * np.sqrt(n_eff**2 - self.n_clad**2)
        beta = self.k0 * n_eff

        return WaveguideMode(
            mode_number=-1,  # Will be set later
            n_eff=n_eff,
            k_x=k_x,
            gamma=gamma,
            symmetry=symmetry,
            beta=beta,
        )

    def mode_profile(
        self,
        mode: WaveguideMode,
        x: np.ndarray,
    ) -> np.ndarray:
        """Compute the transverse electric field profile of a mode.

        For TE modes, the field profile is:
        - Core: E_y = A * cos(k_x * x)  (symmetric)
               E_y = A * sin(k_x * x)  (antisymmetric)
        - Cladding: E_y = B * exp(-gamma * |x|)

        Parameters
        ----------
        mode : WaveguideMode
            Mode to compute profile for
        x : np.ndarray
            Transverse coordinate array (centered at x=0)

        Returns
        -------
        np.ndarray
            Electric field amplitude at each x position
        """
        E = np.zeros_like(x, dtype=float)

        # Core region: |x| < d/2
        core_mask = np.abs(x) <= self.d / 2

        if mode.symmetry == 'symmetric':
            # Symmetric: cos profile in core
            E[core_mask] = np.cos(mode.k_x * x[core_mask])

            # Match amplitude at core/cladding interface
            E_boundary = np.cos(mode.k_x * self.d / 2)
        else:
            # Antisymmetric: sin profile in core
            E[core_mask] = np.sin(mode.k_x * x[core_mask])
            E_boundary = np.sin(mode.k_x * self.d / 2)

        # Cladding region: |x| > d/2
        # Exponential decay
        clad_left = x < -self.d / 2
        clad_right = x > self.d / 2

        if mode.symmetry == 'symmetric':
            E[clad_left] = E_boundary * np.exp(mode.gamma * (x[clad_left] + self.d/2))
            E[clad_right] = E_boundary * np.exp(-mode.gamma * (x[clad_right] - self.d/2))
        else:
            E[clad_left] = -E_boundary * np.exp(mode.gamma * (x[clad_left] + self.d/2))
            E[clad_right] = E_boundary * np.exp(-mode.gamma * (x[clad_right] - self.d/2))

        # Normalize to unit power (approximately)
        power = np.trapezoid(E**2, x)
        if power > 0:
            E = E / np.sqrt(power)

        return E

    def confinement_factor(self, mode: WaveguideMode) -> float:
        """Compute power confinement factor in core.

        Gamma = (power in core) / (total power)

        Parameters
        ----------
        mode : WaveguideMode
            Mode to analyze

        Returns
        -------
        float
            Confinement factor (0 to 1)
        """
        # Analytical formula for TE modes
        k_x = mode.k_x
        gamma = mode.gamma
        d = self.d

        if mode.symmetry == 'symmetric':
            # Gamma = (d/2 + sin(k_x*d)/(2*k_x)) / (d/2 + sin(k_x*d)/(2*k_x) + 1/gamma)
            core_integral = d/2 + np.sin(k_x * d) / (2 * k_x)
        else:
            core_integral = d/2 - np.sin(k_x * d) / (2 * k_x)

        clad_integral = 1 / gamma

        total = core_integral + clad_integral
        if total > 0:
            return core_integral / total
        return 0.0

    def group_index(self, mode: WaveguideMode, delta_lambda: float = 1e-9) -> float:
        """Compute group index using numerical differentiation.

        n_g = n_eff - lambda * d(n_eff)/d(lambda)

        Parameters
        ----------
        mode : WaveguideMode
            Mode to analyze
        delta_lambda : float
            Wavelength step for numerical derivative [m]

        Returns
        -------
        float
            Group refractive index
        """
        # Create waveguide at slightly different wavelengths
        wg_plus = SlabWaveguide(
            self.n_core, self.n_clad, self.thickness,
            self.wavelength + delta_lambda
        )
        wg_minus = SlabWaveguide(
            self.n_core, self.n_clad, self.thickness,
            self.wavelength - delta_lambda
        )

        # Find corresponding modes
        modes_plus = wg_plus.find_modes()
        modes_minus = wg_minus.find_modes()

        if mode.mode_number < len(modes_plus) and mode.mode_number < len(modes_minus):
            n_eff_plus = modes_plus[mode.mode_number].n_eff
            n_eff_minus = modes_minus[mode.mode_number].n_eff

            dn_eff_dlambda = (n_eff_plus - n_eff_minus) / (2 * delta_lambda)
            n_g = mode.n_eff - self.wavelength * dn_eff_dlambda
            return n_g

        return mode.n_eff


def cutoff_wavelength(
    n_core: float,
    n_clad: float,
    thickness: float,
    mode_number: int = 1,
) -> float:
    """Compute cutoff wavelength for a given mode.

    At cutoff, n_eff = n_clad and the mode becomes radiating.

    Parameters
    ----------
    n_core : float
        Core refractive index
    n_clad : float
        Cladding refractive index
    thickness : float
        Core thickness [m]
    mode_number : int
        Mode number (1 = first higher-order mode)

    Returns
    -------
    float
        Cutoff wavelength [m]
    """
    # V-number at cutoff for mode m is V_c = m * pi/2
    V_c = mode_number * np.pi / 2

    # V = k0 * (d/2) * sqrt(n_core^2 - n_clad^2)
    # V_c = (2*pi/lambda_c) * (d/2) * sqrt(n_core^2 - n_clad^2)
    # lambda_c = pi * d * sqrt(n_core^2 - n_clad^2) / V_c

    NA = np.sqrt(n_core**2 - n_clad**2)  # Numerical aperture
    lambda_c = np.pi * thickness * NA / V_c

    return lambda_c


def single_mode_condition(
    n_core: float,
    n_clad: float,
    wavelength: float,
) -> float:
    """Compute maximum thickness for single-mode operation.

    Parameters
    ----------
    n_core : float
        Core refractive index
    n_clad : float
        Cladding refractive index
    wavelength : float
        Operating wavelength [m]

    Returns
    -------
    float
        Maximum core thickness [m] for single-mode operation
    """
    NA = np.sqrt(n_core**2 - n_clad**2)

    # V < pi/2 for single mode
    # (2*pi/lambda) * (d/2) * NA < pi/2
    # d < lambda / (2 * NA)

    d_max = wavelength / (2 * NA)
    return d_max


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SlabWaveguide",
    "WaveguideMode",
    "cutoff_wavelength",
    "single_mode_condition",
]
