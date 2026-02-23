"""Tests for Maxwell solver verification utilities."""

import numpy as np

from src.em.units import EMConstants
from src.em.verification import (
    convergence_rate,
    manufactured_solution_1d,
    taflove_dispersion_formula,
    verify_energy_conservation,
    verify_pec_reflection,
    verify_wave_speed,
)


class TestVerifyWaveSpeed:
    """Tests for wave speed verification."""

    def test_correct_speed_detected(self):
        """Should pass when wave travels at expected speed."""
        const = EMConstants()

        # Simulate wave traveling at c
        Nx = 100
        Nt = 50
        dx = 0.01
        dt = 0.9 * dx / const.c0

        x = np.linspace(0, Nx * dx, Nx)
        t = np.linspace(0, Nt * dt, Nt)

        # Gaussian pulse moving at c
        E_history = []
        x0 = 0.2
        sigma = 0.05
        for ti in t:
            E = np.exp(-((x - x0 - const.c0 * ti) ** 2) / (2 * sigma ** 2))
            E_history.append(E)
        E_history = np.array(E_history)

        passed, measured_c = verify_wave_speed(
            E_history, x, t, expected_c=const.c0, tolerance=0.1
        )

        assert passed
        assert abs(measured_c - const.c0) / const.c0 < 0.1


class TestVerifyPECReflection:
    """Tests for PEC boundary verification."""

    def test_zero_field_at_boundary(self):
        """Should pass when E=0 at boundary."""
        E_history = np.random.rand(10, 100)
        E_history[:, 0] = 0.0  # Zero at left boundary

        x = np.linspace(0, 1, 100)
        passed, max_error = verify_pec_reflection(E_history, x, 0)

        assert passed
        assert max_error == 0.0

    def test_nonzero_detected(self):
        """Should fail when E != 0 at boundary."""
        E_history = np.random.rand(10, 100)
        E_history[:, 0] = 0.001  # Small nonzero value

        x = np.linspace(0, 1, 100)
        passed, max_error = verify_pec_reflection(E_history, x, 0, tolerance=1e-6)

        assert not passed
        assert max_error > 0


class TestVerifyEnergyConservation:
    """Tests for energy conservation verification."""

    def test_constant_energy_passes(self):
        """Should pass when energy is constant."""
        const = EMConstants()
        Nx = 100
        Nt = 20
        dx = 0.01

        # Create constant-energy field configuration
        E_history = np.ones((Nt, Nx + 1)) * 0.5
        H_history = np.ones((Nt, Nx)) * 0.5 / const.eta0

        passed, max_change, energy = verify_energy_conservation(
            E_history, H_history, dx, const.eps0, const.mu0
        )

        assert passed
        assert max_change < 0.01

    def test_varying_energy_detected(self):
        """Should fail when energy changes significantly."""
        const = EMConstants()
        Nx = 100
        Nt = 20
        dx = 0.01

        # Create field with increasing energy
        E_history = np.array([np.ones(Nx + 1) * (1 + 0.1 * i) for i in range(Nt)])
        H_history = np.array([np.ones(Nx) * (1 + 0.1 * i) / const.eta0 for i in range(Nt)])

        passed, max_change, energy = verify_energy_conservation(
            E_history, H_history, dx, const.eps0, const.mu0, tolerance=0.01
        )

        assert not passed


class TestManufacturedSolution:
    """Tests for manufactured solutions."""

    def test_solution_has_correct_shape(self):
        """Manufactured solution should return arrays of correct shape."""
        x = np.linspace(0, 1, 101)
        t = 0.5e-9

        E_mms, H_mms, source = manufactured_solution_1d(x, t)

        assert E_mms.shape == x.shape
        assert H_mms.shape == x.shape
        assert source.shape == x.shape

    def test_smooth_solution(self):
        """Manufactured solution should be smooth."""
        x = np.linspace(0, 1, 101)
        t = 0.5e-9

        E_mms, _, _ = manufactured_solution_1d(x, t)

        # Check no NaN or Inf
        assert np.all(np.isfinite(E_mms))

        # Check smoothness (finite differences should be bounded)
        dE = np.diff(E_mms)
        assert np.max(np.abs(dE)) < 1.0  # Reasonable gradient


class TestTafloveDispersion:
    """Tests for Taflove dispersion formula."""

    def test_magic_timestep_no_dispersion(self):
        """At C=1, should have no dispersion in 1D."""
        const = EMConstants()
        c = const.c0
        dx = 0.01
        dt = dx / c  # C = 1

        # Test at various frequencies
        for omega in [1e9, 5e9, 10e9]:
            ratio = taflove_dispersion_formula(omega, c, dx, dt)
            # At magic timestep, ratio should be 1
            assert abs(ratio - 1.0) < 0.01

    def test_dispersion_increases_with_k(self):
        """Dispersion should increase for higher wavenumbers."""
        const = EMConstants()
        c = const.c0
        dx = 0.01
        dt = 0.5 * dx / c  # C = 0.5

        ratio_low = taflove_dispersion_formula(1e9, c, dx, dt)
        ratio_high = taflove_dispersion_formula(10e9, c, dx, dt)

        # Higher frequency should have more dispersion (lower ratio)
        assert ratio_high < ratio_low


class TestConvergenceRate:
    """Tests for convergence rate calculation."""

    def test_known_order(self):
        """Should recover known convergence order."""
        # Generate synthetic data with second-order convergence
        dx_values = np.array([0.1, 0.05, 0.025, 0.0125])
        C = 0.1  # Constant
        order = 2.0
        errors = C * dx_values ** order

        computed_order = convergence_rate(dx_values, errors)
        assert abs(computed_order - order) < 0.1

    def test_first_order(self):
        """Should detect first-order convergence."""
        dx_values = np.array([0.1, 0.05, 0.025, 0.0125])
        errors = 0.1 * dx_values ** 1.0

        computed_order = convergence_rate(dx_values, errors)
        assert abs(computed_order - 1.0) < 0.1
