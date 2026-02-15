"""Material models for electromagnetic simulations.

Provides models for various electromagnetic media including:
- Lossy dielectrics (conductivity)
- Debye relaxation (frequency-dependent permittivity)
- Cole-Cole model (broad frequency dispersion)
- Soil models for GPR applications

The material properties are frequency-dependent in general:
    eps*(omega) = eps_inf + (eps_s - eps_inf) / (1 + j*omega*tau)

For FDTD, these frequency-dependent materials require auxiliary
differential equations (ADE) or recursive convolution methods.

References
----------
.. [1] A. Taflove and S.C. Hagness, "Computational Electrodynamics,"
       3rd ed., Chapter 9: Dispersive Materials.

.. [2] C. Warren et al., "gprMax: Open source software to simulate
       electromagnetic wave propagation for Ground Penetrating Radar,"
       Computer Physics Communications, vol. 209, pp. 163-170, 2016.
"""

from dataclasses import dataclass

import numpy as np

from src.em.units import EMConstants


@dataclass
class DielectricMaterial:
    """Simple dielectric material model.

    Attributes
    ----------
    name : str
        Material name
    eps_r : float
        Relative permittivity
    mu_r : float
        Relative permeability
    sigma : float
        Conductivity [S/m]
    """
    name: str
    eps_r: float = 1.0
    mu_r: float = 1.0
    sigma: float = 0.0

    @property
    def is_lossy(self) -> bool:
        """Check if material has losses."""
        return self.sigma > 0

    def wave_speed(self) -> float:
        """Compute wave speed in material [m/s]."""
        const = EMConstants()
        return const.c0 / np.sqrt(self.eps_r * self.mu_r)

    def wavelength(self, frequency: float) -> float:
        """Compute wavelength in material [m]."""
        return self.wave_speed() / frequency

    def attenuation_coefficient(self, frequency: float) -> float:
        """Compute attenuation coefficient [Np/m].

        For a good conductor or lossy dielectric:
        alpha = omega * sqrt(mu*eps/2) * sqrt(sqrt(1 + (sigma/(omega*eps))^2) - 1)
        """
        const = EMConstants()
        omega = 2 * np.pi * frequency
        eps = self.eps_r * const.eps0
        mu = self.mu_r * const.mu0

        if self.sigma == 0:
            return 0.0

        ratio = self.sigma / (omega * eps)
        alpha = omega * np.sqrt(mu * eps / 2) * np.sqrt(np.sqrt(1 + ratio**2) - 1)
        return alpha

    def skin_depth(self, frequency: float) -> float:
        """Compute skin depth [m]."""
        alpha = self.attenuation_coefficient(frequency)
        if alpha > 0:
            return 1.0 / alpha
        return np.inf


@dataclass
class DebyeMaterial:
    """Debye relaxation model for frequency-dependent materials.

    The complex permittivity is:
    eps*(omega) = eps_inf + (eps_s - eps_inf) / (1 + j*omega*tau)

    Attributes
    ----------
    name : str
        Material name
    eps_s : float
        Static (DC) relative permittivity
    eps_inf : float
        High-frequency (optical) relative permittivity
    tau : float
        Relaxation time [s]
    sigma_dc : float
        DC conductivity [S/m] (added loss term)
    mu_r : float
        Relative permeability
    """
    name: str
    eps_s: float
    eps_inf: float
    tau: float
    sigma_dc: float = 0.0
    mu_r: float = 1.0

    def complex_permittivity(self, frequency: float) -> complex:
        """Compute complex relative permittivity at given frequency.

        Parameters
        ----------
        frequency : float
            Frequency [Hz]

        Returns
        -------
        complex
            Complex relative permittivity
        """
        const = EMConstants()
        omega = 2 * np.pi * frequency

        # Debye term
        eps_debye = self.eps_inf + (self.eps_s - self.eps_inf) / (1 + 1j * omega * self.tau)

        # Add DC conductivity loss
        if self.sigma_dc > 0 and omega > 0:
            eps_debye = eps_debye - 1j * self.sigma_dc / (omega * const.eps0)

        return eps_debye

    def real_permittivity(self, frequency: float) -> float:
        """Real part of relative permittivity."""
        return self.complex_permittivity(frequency).real

    def loss_tangent(self, frequency: float) -> float:
        """Loss tangent tan(delta) = eps''/eps'."""
        eps = self.complex_permittivity(frequency)
        if eps.real > 0:
            return -eps.imag / eps.real
        return 0.0

    def effective_conductivity(self, frequency: float) -> float:
        """Effective conductivity [S/m] at given frequency."""
        const = EMConstants()
        omega = 2 * np.pi * frequency
        eps = self.complex_permittivity(frequency)
        return -omega * const.eps0 * eps.imag


@dataclass
class ColeCole:
    """Cole-Cole model for broad frequency dispersion.

    The complex permittivity is:
    eps*(omega) = eps_inf + (eps_s - eps_inf) / (1 + (j*omega*tau)^alpha)

    where alpha (0 < alpha <= 1) controls the breadth of dispersion.
    alpha = 1 reduces to the Debye model.

    Attributes
    ----------
    name : str
        Material name
    eps_s : float
        Static relative permittivity
    eps_inf : float
        High-frequency relative permittivity
    tau : float
        Characteristic relaxation time [s]
    alpha : float
        Cole-Cole exponent (0 < alpha <= 1)
    sigma_dc : float
        DC conductivity [S/m]
    """
    name: str
    eps_s: float
    eps_inf: float
    tau: float
    alpha: float = 1.0
    sigma_dc: float = 0.0

    def complex_permittivity(self, frequency: float) -> complex:
        """Compute complex relative permittivity at given frequency."""
        const = EMConstants()
        omega = 2 * np.pi * frequency

        # Cole-Cole term
        eps_cc = self.eps_inf + (self.eps_s - self.eps_inf) / (
            1 + (1j * omega * self.tau)**self.alpha
        )

        # Add DC conductivity loss
        if self.sigma_dc > 0 and omega > 0:
            eps_cc = eps_cc - 1j * self.sigma_dc / (omega * const.eps0)

        return eps_cc


# =============================================================================
# Predefined Materials
# =============================================================================

# Common dielectric materials
VACUUM = DielectricMaterial(name="Vacuum", eps_r=1.0, mu_r=1.0, sigma=0.0)
AIR = DielectricMaterial(name="Air", eps_r=1.0006, mu_r=1.0, sigma=0.0)
WATER = DielectricMaterial(name="Water (pure)", eps_r=80.0, mu_r=1.0, sigma=5e-4)
GLASS = DielectricMaterial(name="Glass", eps_r=4.0, mu_r=1.0, sigma=1e-12)
TEFLON = DielectricMaterial(name="Teflon", eps_r=2.1, mu_r=1.0, sigma=1e-16)
FR4 = DielectricMaterial(name="FR-4 (PCB)", eps_r=4.5, mu_r=1.0, sigma=0.0)

# Metals (approximated as lossy dielectrics for FDTD)
COPPER = DielectricMaterial(name="Copper", eps_r=1.0, mu_r=1.0, sigma=5.8e7)
ALUMINUM = DielectricMaterial(name="Aluminum", eps_r=1.0, mu_r=1.0, sigma=3.8e7)
IRON = DielectricMaterial(name="Iron", eps_r=1.0, mu_r=4000.0, sigma=1.0e7)


# =============================================================================
# Soil Models for GPR
# =============================================================================

@dataclass
class SoilModel:
    """Soil material model for GPR applications.

    Common soil types with typical electromagnetic properties.
    Based on empirical data from GPR literature.

    Attributes
    ----------
    name : str
        Soil type name
    eps_r : float
        Relative permittivity (typical range 3-40)
    sigma : float
        Conductivity [S/m] (typical range 0.001-0.1)
    water_content : float
        Volumetric water content (0-1)
    """
    name: str
    eps_r: float
    sigma: float
    water_content: float = 0.0

    def to_dielectric(self) -> DielectricMaterial:
        """Convert to DielectricMaterial for FDTD."""
        return DielectricMaterial(
            name=self.name,
            eps_r=self.eps_r,
            mu_r=1.0,
            sigma=self.sigma,
        )


def topp_equation(water_content: float) -> float:
    """Topp's equation for soil permittivity from water content.

    Empirical relation between volumetric water content and
    relative permittivity of soil.

    Parameters
    ----------
    water_content : float
        Volumetric water content (0 to ~0.5)

    Returns
    -------
    float
        Estimated relative permittivity

    References
    ----------
    G.C. Topp et al., "Electromagnetic determination of soil water
    content," Water Resources Research, vol. 16, pp. 574-582, 1980.
    """
    theta = water_content
    # Topp's polynomial
    eps_r = 3.03 + 9.3*theta + 146.0*theta**2 - 76.7*theta**3
    return max(eps_r, 1.0)


def soil_conductivity_from_water(
    water_content: float,
    clay_content: float = 0.1,
    temperature: float = 20.0,
) -> float:
    """Estimate soil conductivity from water and clay content.

    Parameters
    ----------
    water_content : float
        Volumetric water content (0 to ~0.5)
    clay_content : float
        Clay fraction (0 to 1)
    temperature : float
        Temperature [C]

    Returns
    -------
    float
        Estimated conductivity [S/m]
    """
    # Simplified empirical model
    # Higher clay and water content increases conductivity
    sigma_base = 0.01  # Base conductivity for dry soil [S/m]
    sigma = sigma_base * (1 + 10*water_content + 5*clay_content)
    # Temperature correction (conductivity increases ~2% per degree)
    sigma *= 1 + 0.02 * (temperature - 20)
    return sigma


# Predefined soil types
DRY_SAND = SoilModel(name="Dry Sand", eps_r=3.0, sigma=0.001, water_content=0.02)
WET_SAND = SoilModel(name="Wet Sand", eps_r=25.0, sigma=0.01, water_content=0.25)
DRY_CLAY = SoilModel(name="Dry Clay", eps_r=3.5, sigma=0.05, water_content=0.05)
WET_CLAY = SoilModel(name="Wet Clay", eps_r=35.0, sigma=0.1, water_content=0.35)
LOAM = SoilModel(name="Loam", eps_r=12.0, sigma=0.02, water_content=0.15)
CONCRETE = SoilModel(name="Concrete", eps_r=6.0, sigma=0.01, water_content=0.05)
ASPHALT = SoilModel(name="Asphalt", eps_r=4.0, sigma=0.005, water_content=0.02)


def create_layered_model(
    layers: list[tuple[float, DielectricMaterial]],
    Nx: int,
    L: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create 1D layered material model for FDTD.

    Parameters
    ----------
    layers : list of (thickness, material) tuples
        Each layer specified by thickness [m] and material
    Nx : int
        Number of grid points
    L : float
        Total domain length [m]

    Returns
    -------
    tuple
        (eps_r_array, sigma_array) of shape (Nx+1,)
    """
    eps_r = np.ones(Nx + 1)
    sigma = np.zeros(Nx + 1)

    x = np.linspace(0, L, Nx + 1)
    z_current = 0.0

    for thickness, material in layers:
        z_next = z_current + thickness
        mask = (x >= z_current) & (x < z_next)
        eps_r[mask] = material.eps_r
        sigma[mask] = material.sigma
        z_current = z_next

    return eps_r, sigma


def create_halfspace_model(
    material: DielectricMaterial,
    interface_depth: float,
    Nx: int,
    L: float,
    background: DielectricMaterial = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create 1D halfspace model (e.g., air/soil interface).

    Parameters
    ----------
    material : DielectricMaterial
        Material for lower halfspace
    interface_depth : float
        Depth of interface from top [m]
    Nx : int
        Number of grid points
    L : float
        Total domain length [m]
    background : DielectricMaterial, optional
        Material for upper halfspace. Default: air.

    Returns
    -------
    tuple
        (eps_r_array, sigma_array) of shape (Nx+1,)
    """
    if background is None:
        background = AIR

    eps_r = np.ones(Nx + 1) * background.eps_r
    sigma = np.ones(Nx + 1) * background.sigma

    x = np.linspace(0, L, Nx + 1)
    mask = x >= interface_depth

    eps_r[mask] = material.eps_r
    sigma[mask] = material.sigma

    return eps_r, sigma


def create_cylinder_model_2d(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    center: tuple[float, float],
    radius: float,
    cylinder_material: DielectricMaterial,
    background: DielectricMaterial = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create 2D model with cylindrical scatterer.

    Parameters
    ----------
    Nx, Ny : int
        Number of grid points in x and y
    Lx, Ly : float
        Domain dimensions [m]
    center : tuple
        (x, y) center of cylinder [m]
    radius : float
        Cylinder radius [m]
    cylinder_material : DielectricMaterial
        Material inside cylinder
    background : DielectricMaterial, optional
        Background material. Default: vacuum.

    Returns
    -------
    tuple
        (eps_r_array, sigma_array) of shape (Nx+1, Ny+1)
    """
    if background is None:
        background = VACUUM

    eps_r = np.ones((Nx + 1, Ny + 1)) * background.eps_r
    sigma = np.ones((Nx + 1, Ny + 1)) * background.sigma

    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    r = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = r <= radius

    eps_r[mask] = cylinder_material.eps_r
    sigma[mask] = cylinder_material.sigma

    return eps_r, sigma


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Predefined materials
    "AIR",
    "ALUMINUM",
    "ASPHALT",
    "CONCRETE",
    "COPPER",
    "DRY_CLAY",
    "DRY_SAND",
    "FR4",
    "GLASS",
    "IRON",
    "LOAM",
    "TEFLON",
    "VACUUM",
    "WATER",
    "WET_CLAY",
    "WET_SAND",
    # Material classes
    "ColeCole",
    "DebyeMaterial",
    "DielectricMaterial",
    "SoilModel",
    # Model creation functions
    "create_cylinder_model_2d",
    "create_halfspace_model",
    "create_layered_model",
    "soil_conductivity_from_water",
    "topp_equation",
]
