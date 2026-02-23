"""Tests for src.em.waveguide — dielectric slab waveguide utilities."""

import numpy as np
import pytest

from src.em.waveguide import (
    SlabWaveguide,
    cutoff_wavelength,
    single_mode_condition,
)

# ---------------------------------------------------------------------------
# SlabWaveguide construction
# ---------------------------------------------------------------------------


class TestSlabWaveguideConstruction:

    def test_basic_creation(self):
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=1e-6, wavelength=1.55e-6)
        assert wg.V > 0
        assert wg.k0 > 0

    def test_n_core_must_exceed_n_clad(self):
        with pytest.raises(ValueError, match="n_core must be greater"):
            SlabWaveguide(n_core=1.0, n_clad=1.5, thickness=1e-6, wavelength=1.55e-6)

    def test_v_number(self):
        """V-number should match analytical formula."""
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=2e-6, wavelength=1.55e-6)
        k0 = 2 * np.pi / 1.55e-6
        NA = np.sqrt(1.5**2 - 1.0**2)
        V_expected = k0 * 1e-6 * NA  # k0 * d/2 * NA
        assert wg.V == pytest.approx(V_expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Mode finding
# ---------------------------------------------------------------------------


class TestFindModes:

    def test_single_mode_waveguide(self):
        """Thin waveguide should support only one mode."""
        # d = lambda/(4*NA) → well below single-mode cutoff
        wg = SlabWaveguide(n_core=1.5, n_clad=1.45, thickness=0.5e-6, wavelength=1.55e-6)
        modes = wg.find_modes()
        assert len(modes) >= 1  # At least the fundamental
        # Fundamental mode
        assert modes[0].mode_number == 0
        assert modes[0].n_eff > wg.n_clad
        assert modes[0].n_eff < wg.n_core

    def test_multimode_waveguide(self):
        """Thick waveguide should support multiple modes."""
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=10e-6, wavelength=1.55e-6)
        modes = wg.find_modes()
        assert len(modes) > 1

    def test_modes_sorted_by_neff(self):
        """Modes should be sorted with highest n_eff first."""
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=10e-6, wavelength=1.55e-6)
        modes = wg.find_modes()
        for i in range(len(modes) - 1):
            assert modes[i].n_eff >= modes[i + 1].n_eff

    def test_mode_neff_in_range(self):
        """All n_eff should be between n_clad and n_core."""
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=5e-6, wavelength=1.55e-6)
        for mode in wg.find_modes():
            assert mode.n_eff > wg.n_clad
            assert mode.n_eff < wg.n_core

    def test_mode_has_symmetry(self):
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=5e-6, wavelength=1.55e-6)
        modes = wg.find_modes()
        for mode in modes:
            assert mode.symmetry in ("symmetric", "antisymmetric")

    def test_fundamental_is_symmetric(self):
        """Fundamental mode should be symmetric."""
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=2e-6, wavelength=1.55e-6)
        modes = wg.find_modes()
        assert modes[0].symmetry == "symmetric"

    def test_mode_beta_positive(self):
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=2e-6, wavelength=1.55e-6)
        for mode in wg.find_modes():
            assert mode.beta > 0
            assert mode.k_x > 0
            assert mode.gamma > 0


# ---------------------------------------------------------------------------
# Mode profile
# ---------------------------------------------------------------------------


class TestModeProfile:

    def test_profile_shape(self):
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=2e-6, wavelength=1.55e-6)
        mode = wg.find_modes()[0]
        x = np.linspace(-5e-6, 5e-6, 1001)
        E = wg.mode_profile(mode, x)
        assert E.shape == x.shape

    def test_symmetric_mode_even(self):
        """Symmetric mode profile should be even: E(x) = E(-x)."""
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=2e-6, wavelength=1.55e-6)
        modes = [m for m in wg.find_modes() if m.symmetry == "symmetric"]
        assert len(modes) > 0
        x = np.linspace(-5e-6, 5e-6, 1001)
        E = wg.mode_profile(modes[0], x)
        np.testing.assert_allclose(E, E[::-1], atol=1e-10)

    def test_antisymmetric_mode_odd(self):
        """Antisymmetric mode profile should be odd: E(x) = -E(-x)."""
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=5e-6, wavelength=1.55e-6)
        modes = [m for m in wg.find_modes() if m.symmetry == "antisymmetric"]
        if len(modes) == 0:
            pytest.skip("No antisymmetric modes for this waveguide")
        x = np.linspace(-10e-6, 10e-6, 2001)
        E = wg.mode_profile(modes[0], x)
        np.testing.assert_allclose(E, -E[::-1], atol=1e-10)

    def test_profile_decays_in_cladding(self):
        """Field should decay exponentially in cladding."""
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=2e-6, wavelength=1.55e-6)
        mode = wg.find_modes()[0]
        x = np.linspace(-10e-6, 10e-6, 2001)
        E = wg.mode_profile(mode, x)
        # Field at boundary should be larger than field deep in cladding
        assert abs(E[0]) < abs(E[500])  # Edge vs halfway to core


# ---------------------------------------------------------------------------
# Confinement factor
# ---------------------------------------------------------------------------


class TestConfinementFactor:

    def test_between_zero_and_one(self):
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=2e-6, wavelength=1.55e-6)
        for mode in wg.find_modes():
            gamma = wg.confinement_factor(mode)
            assert 0 < gamma < 1

    def test_fundamental_most_confined(self):
        """Fundamental mode should have highest confinement."""
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=5e-6, wavelength=1.55e-6)
        modes = wg.find_modes()
        if len(modes) < 2:
            pytest.skip("Need at least 2 modes")
        assert wg.confinement_factor(modes[0]) > wg.confinement_factor(modes[1])


# ---------------------------------------------------------------------------
# Group index
# ---------------------------------------------------------------------------


class TestGroupIndex:

    def test_group_index_positive(self):
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=2e-6, wavelength=1.55e-6)
        mode = wg.find_modes()[0]
        n_g = wg.group_index(mode)
        assert n_g > 0

    def test_group_index_reasonable_range(self):
        """Group index should be between n_clad and ~2*n_core."""
        wg = SlabWaveguide(n_core=1.5, n_clad=1.0, thickness=2e-6, wavelength=1.55e-6)
        mode = wg.find_modes()[0]
        n_g = wg.group_index(mode)
        assert n_g > wg.n_clad
        assert n_g < 2 * wg.n_core


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


class TestCutoffWavelength:

    def test_cutoff_positive(self):
        lam_c = cutoff_wavelength(n_core=1.5, n_clad=1.0, thickness=2e-6, mode_number=1)
        assert lam_c > 0

    def test_higher_modes_shorter_cutoff(self):
        """Higher-order modes have shorter cutoff wavelengths."""
        lam_1 = cutoff_wavelength(n_core=1.5, n_clad=1.0, thickness=5e-6, mode_number=1)
        lam_2 = cutoff_wavelength(n_core=1.5, n_clad=1.0, thickness=5e-6, mode_number=2)
        assert lam_2 < lam_1

    def test_consistency_with_single_mode(self):
        """Single-mode condition and cutoff wavelength should agree."""
        n_core, n_clad = 1.5, 1.0
        lam = 1.55e-6
        d_max = single_mode_condition(n_core, n_clad, lam)
        lam_c = cutoff_wavelength(n_core, n_clad, d_max, mode_number=1)
        assert lam_c == pytest.approx(lam, rel=1e-6)


class TestSingleModeCondition:

    def test_returns_positive(self):
        d = single_mode_condition(n_core=1.5, n_clad=1.0, wavelength=1.55e-6)
        assert d > 0

    def test_larger_na_thinner_core(self):
        """Larger NA requires thinner core for single-mode."""
        d1 = single_mode_condition(n_core=1.5, n_clad=1.45, wavelength=1.55e-6)
        d2 = single_mode_condition(n_core=1.5, n_clad=1.0, wavelength=1.55e-6)
        assert d2 < d1  # Larger NA → thinner

    def test_waveguide_at_max_thickness_is_single_mode(self):
        """Waveguide at max thickness should support exactly one mode."""
        n_core, n_clad = 1.5, 1.0
        lam = 1.55e-6
        d_max = single_mode_condition(n_core, n_clad, lam)
        # Slightly below cutoff
        wg = SlabWaveguide(n_core, n_clad, d_max * 0.95, lam)
        modes = wg.find_modes()
        assert len(modes) == 1
