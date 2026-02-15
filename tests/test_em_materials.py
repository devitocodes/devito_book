"""Tests for src.em.materials — material models for EM simulations."""

import numpy as np
import pytest

from src.em.materials import (
    AIR,
    ALUMINUM,
    ASPHALT,
    CONCRETE,
    COPPER,
    DRY_CLAY,
    DRY_SAND,
    GLASS,
    IRON,
    LOAM,
    VACUUM,
    WATER,
    WET_CLAY,
    WET_SAND,
    ColeCole,
    DebyeMaterial,
    DielectricMaterial,
    create_cylinder_model_2d,
    create_halfspace_model,
    create_layered_model,
    soil_conductivity_from_water,
    topp_equation,
)
from src.em.units import EMConstants

# ---------------------------------------------------------------------------
# DielectricMaterial
# ---------------------------------------------------------------------------


class TestDielectricMaterial:

    def test_vacuum_wave_speed(self):
        c = VACUUM.wave_speed()
        assert c == pytest.approx(EMConstants().c0, rel=1e-6)

    def test_dielectric_slows_wave(self):
        mat = DielectricMaterial(name="glass", eps_r=4.0)
        c0 = EMConstants().c0
        assert mat.wave_speed() == pytest.approx(c0 / 2, rel=1e-6)

    def test_wavelength(self):
        mat = DielectricMaterial(name="test", eps_r=4.0)
        lam = mat.wavelength(1e9)
        expected = mat.wave_speed() / 1e9
        assert lam == pytest.approx(expected, rel=1e-10)

    def test_is_lossy(self):
        assert not VACUUM.is_lossy
        assert COPPER.is_lossy
        assert WATER.is_lossy

    def test_attenuation_lossless(self):
        assert VACUUM.attenuation_coefficient(1e9) == 0.0

    def test_attenuation_lossy(self):
        alpha = COPPER.attenuation_coefficient(1e9)
        assert alpha > 0

    def test_skin_depth_lossless_infinite(self):
        assert VACUUM.skin_depth(1e9) == np.inf

    def test_skin_depth_copper_small(self):
        delta = COPPER.skin_depth(1e9)
        assert 0 < delta < 1e-4  # micrometers at 1 GHz


# ---------------------------------------------------------------------------
# DebyeMaterial
# ---------------------------------------------------------------------------


class TestDebyeMaterial:

    @pytest.fixture
    def water_debye(self):
        return DebyeMaterial(
            name="Water (Debye)",
            eps_s=80.0, eps_inf=4.9,
            tau=9.4e-12,  # ~9.4 ps relaxation
        )

    def test_dc_permittivity(self, water_debye):
        """At f→0, eps should approach eps_s."""
        eps = water_debye.complex_permittivity(1.0)  # 1 Hz ≈ DC
        assert eps.real == pytest.approx(80.0, rel=1e-3)

    def test_high_freq_permittivity(self, water_debye):
        """At f→∞, eps should approach eps_inf."""
        eps = water_debye.complex_permittivity(1e15)  # ~THz
        assert eps.real == pytest.approx(4.9, rel=0.1)

    def test_imaginary_part_negative(self, water_debye):
        """Imaginary part should be negative (lossy)."""
        eps = water_debye.complex_permittivity(1e9)
        assert eps.imag < 0

    def test_real_permittivity_method(self, water_debye):
        eps_r = water_debye.real_permittivity(1e9)
        eps_c = water_debye.complex_permittivity(1e9)
        assert eps_r == pytest.approx(eps_c.real)

    def test_loss_tangent_positive(self, water_debye):
        assert water_debye.loss_tangent(1e9) > 0

    def test_effective_conductivity(self, water_debye):
        sigma = water_debye.effective_conductivity(1e9)
        assert sigma > 0

    def test_dc_conductivity_adds_loss(self):
        mat = DebyeMaterial(
            name="lossy", eps_s=10.0, eps_inf=5.0,
            tau=1e-11, sigma_dc=0.1,
        )
        eps_no_sigma = DebyeMaterial(
            name="lossless", eps_s=10.0, eps_inf=5.0,
            tau=1e-11, sigma_dc=0.0,
        )
        # With sigma_dc, imaginary part should be larger (more negative)
        assert mat.complex_permittivity(1e9).imag < eps_no_sigma.complex_permittivity(1e9).imag


# ---------------------------------------------------------------------------
# ColeCole
# ---------------------------------------------------------------------------


class TestColeCole:

    def test_reduces_to_debye(self):
        """alpha=1 should give same result as Debye model."""
        cc = ColeCole(name="cc", eps_s=80.0, eps_inf=4.9, tau=9.4e-12, alpha=1.0)
        db = DebyeMaterial(name="db", eps_s=80.0, eps_inf=4.9, tau=9.4e-12)

        eps_cc = cc.complex_permittivity(1e9)
        eps_db = db.complex_permittivity(1e9)
        assert eps_cc.real == pytest.approx(eps_db.real, rel=1e-10)
        assert eps_cc.imag == pytest.approx(eps_db.imag, rel=1e-10)

    def test_dc_limit(self):
        cc = ColeCole(name="cc", eps_s=20.0, eps_inf=5.0, tau=1e-10, alpha=0.8)
        eps = cc.complex_permittivity(1.0)
        assert eps.real == pytest.approx(20.0, rel=1e-3)

    def test_dc_conductivity(self):
        cc = ColeCole(name="cc", eps_s=20.0, eps_inf=5.0, tau=1e-10,
                      alpha=0.8, sigma_dc=0.05)
        eps = cc.complex_permittivity(1e9)
        assert eps.imag < 0  # lossy


# ---------------------------------------------------------------------------
# SoilModel
# ---------------------------------------------------------------------------


class TestSoilModel:

    def test_to_dielectric(self):
        soil = DRY_SAND
        mat = soil.to_dielectric()
        assert isinstance(mat, DielectricMaterial)
        assert mat.eps_r == soil.eps_r
        assert mat.sigma == soil.sigma
        assert mat.mu_r == 1.0

    def test_predefined_soils_valid(self):
        for soil in [DRY_SAND, WET_SAND, DRY_CLAY, WET_CLAY, LOAM, CONCRETE, ASPHALT]:
            assert soil.eps_r > 1.0
            assert soil.sigma >= 0


# ---------------------------------------------------------------------------
# Predefined materials
# ---------------------------------------------------------------------------


class TestPredefinedMaterials:

    def test_vacuum_eps_r_one(self):
        assert VACUUM.eps_r == 1.0
        assert VACUUM.sigma == 0.0

    def test_air_near_vacuum(self):
        assert AIR.eps_r == pytest.approx(1.0, abs=0.01)

    def test_water_high_permittivity(self):
        assert WATER.eps_r == 80.0

    def test_metals_high_conductivity(self):
        assert COPPER.sigma > 1e6
        assert ALUMINUM.sigma > 1e6

    def test_iron_magnetic(self):
        assert IRON.mu_r > 1.0

    def test_glass_dielectric(self):
        assert GLASS.eps_r > 1.0
        assert GLASS.sigma < 1e-6


# ---------------------------------------------------------------------------
# Topp equation and soil conductivity
# ---------------------------------------------------------------------------


class TestSoilFunctions:

    def test_topp_dry_soil(self):
        eps_r = topp_equation(0.0)
        assert eps_r == pytest.approx(3.03, rel=0.01)

    def test_topp_increases_with_water(self):
        assert topp_equation(0.3) > topp_equation(0.1)

    def test_topp_floor_at_one(self):
        assert topp_equation(-0.5) >= 1.0

    def test_conductivity_increases_with_water(self):
        s1 = soil_conductivity_from_water(0.05)
        s2 = soil_conductivity_from_water(0.30)
        assert s2 > s1

    def test_conductivity_increases_with_clay(self):
        s1 = soil_conductivity_from_water(0.1, clay_content=0.0)
        s2 = soil_conductivity_from_water(0.1, clay_content=0.5)
        assert s2 > s1

    def test_conductivity_temperature_effect(self):
        s1 = soil_conductivity_from_water(0.1, temperature=10.0)
        s2 = soil_conductivity_from_water(0.1, temperature=30.0)
        assert s2 > s1


# ---------------------------------------------------------------------------
# Model creation functions
# ---------------------------------------------------------------------------


class TestCreateLayeredModel:

    def test_shape(self):
        eps_r, sigma = create_layered_model(
            layers=[(0.5, VACUUM), (0.5, GLASS)],
            Nx=100, L=1.0,
        )
        assert eps_r.shape == (101,)
        assert sigma.shape == (101,)

    def test_layer_values(self):
        eps_r, sigma = create_layered_model(
            layers=[(0.5, VACUUM), (0.5, GLASS)],
            Nx=100, L=1.0,
        )
        # First half: vacuum
        assert eps_r[0] == pytest.approx(1.0)
        # Second half: glass
        assert eps_r[75] == pytest.approx(GLASS.eps_r)

    def test_three_layers(self):
        eps_r, _ = create_layered_model(
            layers=[
                (0.2, AIR),
                (0.3, DRY_SAND.to_dielectric()),
                (0.5, WET_CLAY.to_dielectric()),
            ],
            Nx=200, L=1.0,
        )
        assert eps_r[10] == pytest.approx(AIR.eps_r)   # x=0.05
        assert eps_r[70] == pytest.approx(DRY_SAND.eps_r)  # x=0.35
        assert eps_r[150] == pytest.approx(WET_CLAY.eps_r)  # x=0.75


class TestCreateHalfspaceModel:

    def test_shape(self):
        eps_r, sigma = create_halfspace_model(
            material=DRY_SAND.to_dielectric(),
            interface_depth=0.3,
            Nx=100, L=1.0,
        )
        assert eps_r.shape == (101,)

    def test_interface_values(self):
        mat = DielectricMaterial(name="soil", eps_r=9.0, sigma=0.01)
        eps_r, sigma = create_halfspace_model(
            material=mat, interface_depth=0.5,
            Nx=100, L=1.0,
        )
        # Above interface: air
        assert eps_r[10] == pytest.approx(AIR.eps_r)
        # Below interface: soil
        assert eps_r[75] == pytest.approx(9.0)
        assert sigma[75] == pytest.approx(0.01)

    def test_custom_background(self):
        eps_r, _ = create_halfspace_model(
            material=GLASS,
            interface_depth=0.5,
            Nx=100, L=1.0,
            background=WATER,
        )
        assert eps_r[10] == pytest.approx(WATER.eps_r)
        assert eps_r[75] == pytest.approx(GLASS.eps_r)


class TestCreateCylinderModel2d:

    def test_shape(self):
        mat = DielectricMaterial(name="pipe", eps_r=10.0, sigma=0.0)
        eps_r, sigma = create_cylinder_model_2d(
            Nx=50, Ny=50, Lx=1.0, Ly=1.0,
            center=(0.5, 0.5), radius=0.1,
            cylinder_material=mat,
        )
        assert eps_r.shape == (51, 51)
        assert sigma.shape == (51, 51)

    def test_center_has_cylinder_material(self):
        mat = DielectricMaterial(name="pipe", eps_r=10.0, sigma=0.5)
        eps_r, sigma = create_cylinder_model_2d(
            Nx=100, Ny=100, Lx=1.0, Ly=1.0,
            center=(0.5, 0.5), radius=0.2,
            cylinder_material=mat,
        )
        assert eps_r[50, 50] == pytest.approx(10.0)
        assert sigma[50, 50] == pytest.approx(0.5)

    def test_corner_has_background(self):
        mat = DielectricMaterial(name="pipe", eps_r=10.0, sigma=0.0)
        eps_r, _ = create_cylinder_model_2d(
            Nx=100, Ny=100, Lx=1.0, Ly=1.0,
            center=(0.5, 0.5), radius=0.1,
            cylinder_material=mat,
        )
        # Corner (0,0) should be vacuum (default background)
        assert eps_r[0, 0] == pytest.approx(VACUUM.eps_r)

    def test_custom_background(self):
        mat = DielectricMaterial(name="pipe", eps_r=10.0, sigma=0.0)
        bg = DielectricMaterial(name="soil", eps_r=5.0, sigma=0.01)
        eps_r, sigma = create_cylinder_model_2d(
            Nx=50, Ny=50, Lx=1.0, Ly=1.0,
            center=(0.5, 0.5), radius=0.1,
            cylinder_material=mat, background=bg,
        )
        assert eps_r[0, 0] == pytest.approx(5.0)
        assert sigma[0, 0] == pytest.approx(0.01)
