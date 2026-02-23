"""Smoke tests for src/verification.py."""

import numpy as np
import sympy as sp


def test_verify_identity_exact_match():
    """verify_identity returns True for identical expressions."""
    from src.verification import verify_identity

    x, h = sp.symbols("x h")
    f = sp.sin(x)
    assert verify_identity(f, f, h) is True


def test_verify_identity_fd_approximation():
    """Forward difference approximates first derivative to O(h)."""
    from src.verification import verify_identity

    x, h = sp.symbols("x h")
    f = sp.exp(x)
    fd = (f.subs(x, x + h) - f) / h
    deriv = f.diff(x)
    # Should match to at least order 1
    assert verify_identity(fd, deriv, h, order=1) is True


def test_check_stencil_order_central():
    """Central difference of exp(x) should be order 2."""
    from src.verification import check_stencil_order

    x, h = sp.symbols("x h")
    f = sp.Function("f")
    central = (f(x + h) - 2 * f(x) + f(x - h)) / h**2
    exact = f(x).diff(x, 2)
    order = check_stencil_order(central, exact, h)
    assert order == 2


def test_verify_stability_wave():
    """Stability check for wave equation with CFL <= 1."""
    from src.verification import verify_stability_condition

    stable, msg = verify_stability_condition(
        "wave_1d", {"c": 1.0, "dt": 0.01, "dx": 0.02}
    )
    assert stable is True
    assert "Courant" in msg

    unstable, msg = verify_stability_condition(
        "wave_1d", {"c": 1.0, "dt": 0.05, "dx": 0.02}
    )
    assert unstable is False


def test_verify_stability_diffusion():
    """Stability check for explicit diffusion with Fourier number <= 0.5."""
    from src.verification import verify_stability_condition

    stable, _ = verify_stability_condition(
        "explicit_diffusion", {"alpha": 1.0, "dt": 0.001, "dx": 0.1}
    )
    assert stable is True

    unstable, _ = verify_stability_condition(
        "explicit_diffusion", {"alpha": 1.0, "dt": 0.1, "dx": 0.1}
    )
    assert unstable is False


def test_convergence_test_second_order():
    """convergence_test detects second-order convergence."""
    from src.verification import convergence_test

    def solver(n):
        x = np.linspace(0, 1, n + 1)
        dx = x[1] - x[0]
        # Fake a second-order solution: exact + O(dx^2) error
        u_exact = np.sin(np.pi * x)
        u_num = u_exact + 0.1 * dx**2 * np.cos(np.pi * x)
        return x, u_num

    def exact(x):
        return np.sin(np.pi * x)

    passed, order, errors = convergence_test(
        solver, exact, [20, 40, 80, 160], expected_order=2.0
    )
    assert passed
    assert abs(order - 2.0) < 0.5
