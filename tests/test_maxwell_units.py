"""Tests for electromagnetic units and constants."""

import numpy as np

from src.em.units import (
    EMConstants,
    compute_cfl_dt,
    compute_impedance,
    compute_wave_speed,
    compute_wavelength,
    courant_number_1d,
    courant_number_2d,
    points_per_wavelength,
    reflection_coefficient,
    skin_depth,
    transmission_coefficient,
    verify_units,
)


class TestEMConstants:
    """Tests for electromagnetic constants."""

    def test_speed_of_light_value(self):
        """Speed of light should be approximately 3e8 m/s."""
        const = EMConstants()
        assert abs(const.c0 - 299792458) < 1

    def test_fundamental_relation(self):
        """c = 1/sqrt(eps0 * mu0) should hold."""
        const = EMConstants()
        c_computed = 1.0 / np.sqrt(const.eps0 * const.mu0)
        assert np.isclose(c_computed, const.c0, rtol=1e-6)

    def test_impedance_of_free_space(self):
        """eta0 should be approximately 377 Ohm."""
        const = EMConstants()
        assert abs(const.eta0 - 377) < 1


class TestWaveSpeed:
    """Tests for wave speed calculations."""

    def test_free_space_speed(self):
        """Wave speed in vacuum should be c0."""
        const = EMConstants()
        c = compute_wave_speed(eps_r=1.0, mu_r=1.0)
        assert np.isclose(c, const.c0, rtol=1e-10)

    def test_dielectric_slows_wave(self):
        """Wave should slow down in dielectric."""
        const = EMConstants()
        c_glass = compute_wave_speed(eps_r=4.0, mu_r=1.0)
        assert c_glass < const.c0
        assert np.isclose(c_glass, const.c0 / 2, rtol=1e-10)


class TestWavelength:
    """Tests for wavelength calculations."""

    def test_wavelength_at_1ghz(self):
        """Wavelength at 1 GHz in vacuum should be 0.3 m."""
        wavelength = compute_wavelength(frequency=1e9)
        assert np.isclose(wavelength, 0.3, rtol=0.01)

    def test_wavelength_in_dielectric(self):
        """Wavelength should be shorter in dielectric."""
        lambda_vacuum = compute_wavelength(frequency=1e9, eps_r=1.0)
        lambda_glass = compute_wavelength(frequency=1e9, eps_r=4.0)
        assert np.isclose(lambda_glass, lambda_vacuum / 2, rtol=1e-10)


class TestCFLCondition:
    """Tests for CFL time step computation."""

    def test_1d_cfl(self):
        """1D CFL should give dt = CFL * dx / c."""
        const = EMConstants()
        dx = 0.01
        CFL = 0.9
        dt = compute_cfl_dt(dx=dx, CFL=CFL)

        expected = CFL * dx / const.c0
        assert np.isclose(dt, expected, rtol=1e-10)

    def test_2d_cfl_more_restrictive(self):
        """2D CFL should be more restrictive than 1D."""
        dt_1d = compute_cfl_dt(dx=0.01, CFL=0.9)
        dt_2d = compute_cfl_dt(dx=0.01, dy=0.01, CFL=0.9)
        assert dt_2d < dt_1d

    def test_2d_cfl_uniform_grid(self):
        """2D CFL with dx=dy should give sqrt(2) factor."""
        dx = 0.01
        CFL = 1.0
        dt_2d = compute_cfl_dt(dx=dx, dy=dx, CFL=CFL)
        dt_1d_equiv = compute_cfl_dt(dx=dx, CFL=CFL)

        assert np.isclose(dt_2d, dt_1d_equiv / np.sqrt(2), rtol=1e-10)


class TestImpedance:
    """Tests for wave impedance calculations."""

    def test_vacuum_impedance(self):
        """Vacuum impedance should be eta0."""
        const = EMConstants()
        eta = compute_impedance(eps_r=1.0, mu_r=1.0)
        assert np.isclose(eta, const.eta0, rtol=1e-10)

    def test_dielectric_impedance(self):
        """Dielectric should have lower impedance than vacuum."""
        const = EMConstants()
        eta_glass = compute_impedance(eps_r=4.0, mu_r=1.0)
        assert eta_glass < const.eta0
        assert np.isclose(eta_glass, const.eta0 / 2, rtol=1e-10)


class TestReflectionCoefficient:
    """Tests for reflection coefficient calculations."""

    def test_same_media_no_reflection(self):
        """Same media should have zero reflection."""
        R = reflection_coefficient(eps_r1=4.0, eps_r2=4.0)
        assert np.isclose(R, 0.0, atol=1e-10)

    def test_vacuum_to_glass(self):
        """Vacuum to glass reflection coefficient."""
        # R = (eta2 - eta1) / (eta2 + eta1)
        # eta1 = eta0, eta2 = eta0/2
        # R = (eta0/2 - eta0) / (eta0/2 + eta0) = -1/3
        R = reflection_coefficient(eps_r1=1.0, eps_r2=4.0)
        assert np.isclose(R, -1/3, rtol=1e-10)

    def test_glass_to_vacuum(self):
        """Glass to vacuum should have positive reflection."""
        R = reflection_coefficient(eps_r1=4.0, eps_r2=1.0)
        assert np.isclose(R, 1/3, rtol=1e-10)


class TestTransmissionCoefficient:
    """Tests for transmission coefficient calculations."""

    def test_same_media_full_transmission(self):
        """Same media should have T=1."""
        T = transmission_coefficient(eps_r1=4.0, eps_r2=4.0)
        assert np.isclose(T, 1.0, rtol=1e-10)

    def test_vacuum_to_glass(self):
        """Vacuum to glass transmission coefficient."""
        T = transmission_coefficient(eps_r1=1.0, eps_r2=4.0)
        # T = 2*eta2 / (eta2 + eta1) = 2*(eta0/2) / (eta0/2 + eta0) = 2/3
        assert np.isclose(T, 2/3, rtol=1e-10)


class TestVerifyUnits:
    """Tests for field unit verification."""

    def test_consistent_plane_wave(self):
        """Plane wave fields should be consistent."""
        const = EMConstants()
        E_mag = 100.0  # V/m
        H_mag = E_mag / const.eta0  # A/m

        consistent, error = verify_units(E_mag, H_mag)
        assert consistent
        assert error < 0.01

    def test_inconsistent_fields(self):
        """Inconsistent fields should be detected."""
        const = EMConstants()
        E_mag = 100.0
        H_mag = E_mag / (2 * const.eta0)  # Wrong by factor of 2

        consistent, error = verify_units(E_mag, H_mag)
        assert not consistent
        assert error > 0.1


class TestCourantNumbers:
    """Tests for Courant number calculations."""

    def test_1d_courant(self):
        """1D Courant number calculation."""
        const = EMConstants()
        C = courant_number_1d(c=const.c0, dt=1e-10, dx=0.03)
        # C = c*dt/dx = 3e8 * 1e-10 / 0.03 = 1.0
        assert np.isclose(C, 1.0, rtol=0.01)

    def test_2d_courant(self):
        """2D Courant number calculation."""
        const = EMConstants()
        dx = dy = 0.03
        dt = 1e-10
        C = courant_number_2d(c=const.c0, dt=dt, dx=dx, dy=dy)
        # C_2d = c*dt*sqrt(1/dx^2 + 1/dy^2) = c*dt*sqrt(2)/dx
        expected = const.c0 * dt * np.sqrt(2) / dx
        assert np.isclose(C, expected, rtol=1e-10)


class TestPointsPerWavelength:
    """Tests for resolution calculation."""

    def test_typical_resolution(self):
        """10 points per wavelength at 1 GHz."""
        dx = 0.03  # 3 cm
        freq = 1e9  # 1 GHz, lambda = 0.3 m
        ppw = points_per_wavelength(dx=dx, frequency=freq)
        assert np.isclose(ppw, 10.0, rtol=0.01)

    def test_in_dielectric(self):
        """Fewer points per wavelength in dielectric."""
        dx = 0.03
        freq = 1e9
        ppw_vacuum = points_per_wavelength(dx=dx, frequency=freq, eps_r=1.0)
        ppw_glass = points_per_wavelength(dx=dx, frequency=freq, eps_r=4.0)
        assert ppw_glass == ppw_vacuum / 2


class TestSkinDepth:
    """Tests for skin depth calculation."""

    def test_good_conductor(self):
        """Skin depth in copper at 1 GHz."""
        delta = skin_depth(frequency=1e9, sigma=5.8e7)  # Copper
        # Should be very small (micrometers)
        assert delta < 10e-6

    def test_poor_conductor(self):
        """Skin depth in slightly lossy dielectric."""
        delta = skin_depth(frequency=1e9, sigma=0.01, eps_r=4.0)
        # Should be larger than copper
        assert delta > 0.01  # More than 1 cm

    def test_lossless_infinite(self):
        """Lossless medium should have infinite skin depth."""
        delta = skin_depth(frequency=1e9, sigma=0.0)
        assert delta == np.inf
