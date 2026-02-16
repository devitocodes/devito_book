"""Smoke tests for src/em/analysis/dispersion_maxwell.py."""

import numpy as np
import pytest


def test_dispersion_1d_positive_frequency():
    """Numerical frequency should be positive for positive wavenumber."""
    from src.em.analysis.dispersion_maxwell import numerical_dispersion_relation_1d

    omega = numerical_dispersion_relation_1d(k=1.0, c=3e8, dx=0.001, dt=1e-12)
    assert omega > 0


def test_phase_velocity_error_1d_small_for_resolved():
    """Phase velocity error should be small for well-resolved waves."""
    from src.em.analysis.dispersion_maxwell import phase_velocity_error_1d

    c = 1.0
    dx = 0.01
    dt = 0.005  # C = 0.5
    k = np.array([2 * np.pi / (20 * dx)])  # 20 points per wavelength
    error = phase_velocity_error_1d(k, c, dx, dt)
    assert np.all(np.abs(error) < 0.01)  # Less than 1% error


def test_magic_time_step_zero_dispersion():
    """At C=1 in 1D, numerical dispersion vanishes."""
    from src.em.analysis.dispersion_maxwell import magic_time_step_error

    error = magic_time_step_error(points_per_wavelength=10.0)
    assert abs(error) < 1e-10


def test_stability_limits():
    """CFL stability limits for 1D, 2D, 3D."""
    from src.em.analysis.dispersion_maxwell import (
        stability_limit_1d,
        stability_limit_2d,
        stability_limit_3d,
    )

    assert stability_limit_1d() == 1.0
    assert stability_limit_2d() == pytest.approx(1.0 / np.sqrt(2))
    assert stability_limit_3d() == pytest.approx(1.0 / np.sqrt(3))


def test_compute_dispersion_error_1d():
    """compute_dispersion_error returns a finite value for 1D."""
    from src.em.analysis.dispersion_maxwell import compute_dispersion_error

    err = compute_dispersion_error(
        points_per_wavelength=10.0, courant_number=0.5, dim=1
    )
    assert np.isfinite(err)
    assert abs(err) < 0.1  # Should be small for 10 ppw


def test_compute_dispersion_error_2d():
    """compute_dispersion_error returns a finite value for 2D."""
    from src.em.analysis.dispersion_maxwell import compute_dispersion_error

    err = compute_dispersion_error(
        points_per_wavelength=10.0,
        courant_number=0.5,
        dim=2,
        theta=np.pi / 4,
    )
    assert np.isfinite(err)


def test_group_velocity_error_1d():
    """Group velocity error should be finite and bounded."""
    from src.em.analysis.dispersion_maxwell import group_velocity_error_1d

    c = 1.0
    dx = 0.01
    dt = 0.005
    k = 2 * np.pi / (20 * dx)
    error = group_velocity_error_1d(k, c, dx, dt)
    assert np.isfinite(error)
    assert abs(error) < 0.1


def test_plot_dispersion_polar_shape():
    """plot_dispersion_polar returns arrays of correct shape."""
    from src.em.analysis.dispersion_maxwell import plot_dispersion_polar

    angles, ratios = plot_dispersion_polar(
        k_magnitude=1.0, c=1.0, dx=0.01, dy=0.01, dt=0.005, n_angles=36
    )
    assert angles.shape == (36,)
    assert ratios.shape == (36,)
    assert np.all(np.isfinite(ratios))
