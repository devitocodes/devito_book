"""Smoke tests for src/advec/twopt_BVP.py."""

import numpy as np
import pytest


def test_solver_centered_basic():
    """Centered solver returns correct shape and boundary values."""
    from src.advec.twopt_BVP import solver

    u, x = solver(eps=0.1, Nx=20, method="centered")
    assert len(u) == 21
    assert len(x) == 21
    assert u[0] == pytest.approx(0.0)
    assert u[-1] == pytest.approx(1.0)


def test_solver_upwind_basic():
    """Upwind solver returns correct shape and boundary values."""
    from src.advec.twopt_BVP import solver

    u, x = solver(eps=0.1, Nx=20, method="upwind")
    assert len(u) == 21
    assert u[0] == pytest.approx(0.0)
    assert u[-1] == pytest.approx(1.0)


def test_solver_monotone():
    """Solution should be monotonically increasing for moderate eps."""
    from src.advec.twopt_BVP import solver

    u, x = solver(eps=0.5, Nx=40, method="centered")
    assert np.all(np.diff(u) >= -1e-10), "Solution should be approximately monotone"


def test_exact_solution():
    """Exact solution matches boundary conditions."""
    from src.advec.twopt_BVP import u_exact

    x = np.array([0.0, 1.0])
    u = u_exact(x, eps=0.1)
    assert u[0] == pytest.approx(0.0, abs=1e-10)
    assert u[-1] == pytest.approx(1.0, abs=1e-10)
