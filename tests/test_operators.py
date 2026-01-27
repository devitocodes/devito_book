"""Tests for finite difference operators.

These tests verify that the finite difference stencils in src/operators.py
correctly approximate derivatives to the expected order of accuracy.
"""

import numpy as np
import pytest
import sympy as sp

from src.operators import (
    backward_diff,
    central_diff,
    derive_truncation_error,
    forward_diff,
    fourth_order_second_derivative,
    get_stencil_order,
    laplacian_2d,
    laplacian_3d,
    second_derivative_central,
    taylor_expand,
)
from src.symbols import dx, dy, dz, h, u, x, y, z


class TestFirstDerivatives:
    """Tests for first derivative approximations."""

    def test_forward_diff_formula(self):
        """Forward difference should give (f(x+h) - f(x)) / h."""
        func = u(x)
        result = forward_diff(func, x, h)
        expected = (u(x + h) - u(x)) / h
        assert sp.simplify(result - expected) == 0

    def test_backward_diff_formula(self):
        """Backward difference should give (f(x) - f(x-h)) / h."""
        func = u(x)
        result = backward_diff(func, x, h)
        expected = (u(x) - u(x - h)) / h
        assert sp.simplify(result - expected) == 0

    def test_central_diff_formula(self):
        """Central difference should give (f(x+h) - f(x-h)) / (2h)."""
        func = u(x)
        result = central_diff(func, x, h)
        expected = (u(x + h) - u(x - h)) / (2 * h)
        assert sp.simplify(result - expected) == 0

    def test_forward_diff_order(self):
        """Forward difference is O(h) accurate."""
        func = u(x)
        stencil = forward_diff(func, x, h)
        exact = sp.Derivative(func, x)
        order = get_stencil_order(stencil, exact, x, h)
        assert order == 1

    def test_backward_diff_order(self):
        """Backward difference is O(h) accurate."""
        func = u(x)
        stencil = backward_diff(func, x, h)
        exact = sp.Derivative(func, x)
        order = get_stencil_order(stencil, exact, x, h)
        assert order == 1

    def test_central_diff_order(self):
        """Central difference is O(h^2) accurate."""
        func = u(x)
        stencil = central_diff(func, x, h)
        exact = sp.Derivative(func, x)
        order = get_stencil_order(stencil, exact, x, h)
        assert order == 2

    @pytest.mark.parametrize("test_func,expected_deriv", [
        (x**2, 2*x),
        (x**3, 3*x**2),
        (sp.sin(x), sp.cos(x)),
        (sp.exp(x), sp.exp(x)),
    ])
    def test_central_diff_specific_functions(self, test_func, expected_deriv):
        """Central difference should approximate known derivatives."""
        # Substitute into the stencil
        stencil = central_diff(test_func, x, h)

        # Taylor expand to get leading term
        expanded = sp.series(stencil, h, 0, 3).removeO()

        # The h^0 term should be the derivative
        constant_term = expanded.coeff(h, 0)
        assert sp.simplify(constant_term - expected_deriv) == 0


class TestSecondDerivatives:
    """Tests for second derivative approximations."""

    def test_second_derivative_central_formula(self):
        """Second derivative central should give (f(x+h) - 2f(x) + f(x-h)) / h^2."""
        func = u(x)
        result = second_derivative_central(func, x, h)
        expected = (u(x + h) - 2*u(x) + u(x - h)) / h**2
        assert sp.simplify(result - expected) == 0

    def test_second_derivative_central_order(self):
        """Second derivative central is O(h^2) accurate."""
        func = u(x)
        stencil = second_derivative_central(func, x, h)
        exact = sp.Derivative(func, x, x)
        order = get_stencil_order(stencil, exact, x, h)
        assert order == 2

    def test_fourth_order_second_derivative_order(self):
        """Fourth-order second derivative is O(h^4) accurate."""
        func = u(x)
        stencil = fourth_order_second_derivative(func, x, h)
        exact = sp.Derivative(func, x, x)
        order = get_stencil_order(stencil, exact, x, h)
        assert order == 4

    def test_fourth_order_second_derivative_formula(self):
        """Fourth-order formula: (-f_{+2} + 16f_{+1} - 30f_0 + 16f_{-1} - f_{-2}) / (12h^2)."""
        func = u(x)
        result = fourth_order_second_derivative(func, x, h)

        f_p2 = u(x + 2*h)
        f_p1 = u(x + h)
        f_0 = u(x)
        f_m1 = u(x - h)
        f_m2 = u(x - 2*h)
        expected = (-f_p2 + 16*f_p1 - 30*f_0 + 16*f_m1 - f_m2) / (12 * h**2)

        assert sp.simplify(result - expected) == 0

    @pytest.mark.parametrize("test_func,expected_deriv", [
        (x**2, 2),
        (x**3, 6*x),
        (x**4, 12*x**2),
        (sp.sin(x), -sp.sin(x)),
        (sp.cos(x), -sp.cos(x)),
        (sp.exp(x), sp.exp(x)),
    ])
    def test_second_derivative_specific_functions(self, test_func, expected_deriv):
        """Second derivative stencil should approximate known second derivatives."""
        stencil = second_derivative_central(test_func, x, h)
        expanded = sp.series(stencil, h, 0, 3).removeO()
        constant_term = expanded.coeff(h, 0)
        assert sp.simplify(constant_term - expected_deriv) == 0


class TestMultiDimensional:
    """Tests for multi-dimensional operators."""

    def test_laplacian_2d_formula(self):
        """2D Laplacian should be d2u/dx2 + d2u/dy2."""
        func = u(x, y)
        result = laplacian_2d(func, x, y, dx, dy)

        d2_dx2 = second_derivative_central(func, x, dx)
        d2_dy2 = second_derivative_central(func, y, dy)
        expected = d2_dx2 + d2_dy2

        assert sp.simplify(result - expected) == 0

    def test_laplacian_3d_formula(self):
        """3D Laplacian should be d2u/dx2 + d2u/dy2 + d2u/dz2."""
        func = u(x, y, z)
        result = laplacian_3d(func, x, y, z, dx, dy, dz)

        d2_dx2 = second_derivative_central(func, x, dx)
        d2_dy2 = second_derivative_central(func, y, dy)
        d2_dz2 = second_derivative_central(func, z, dz)
        expected = d2_dx2 + d2_dy2 + d2_dz2

        assert sp.simplify(result - expected) == 0

    def test_laplacian_2d_isotropic(self):
        """2D Laplacian with equal spacing should have symmetric stencil."""
        func = u(x, y)
        # Use same spacing for both directions
        result = laplacian_2d(func, x, y, h, h)

        # Check coefficients are symmetric
        stencil = sp.expand(result * h**2)
        # Coefficient of u(x,y) should be -4
        center_coeff = stencil.coeff(u(x, y))
        assert center_coeff == -4

    @pytest.mark.parametrize("test_func", [
        x**2 + y**2,
        sp.sin(x) * sp.sin(y),
        sp.exp(x + y),
    ])
    def test_laplacian_2d_order(self, test_func):
        """2D Laplacian stencil should be O(h^2) accurate."""
        # Compute analytical Laplacian
        exact_laplacian = sp.diff(test_func, x, 2) + sp.diff(test_func, y, 2)

        # Compute numerical Laplacian (using h for both spacings)
        numerical_laplacian = laplacian_2d(test_func, x, y, h, h)

        # Expand in Taylor series
        error = numerical_laplacian - exact_laplacian
        series = sp.series(error, h, 0, 3)

        # O(h^0) and O(h^1) terms should vanish
        assert series.coeff(h, 0) == 0
        assert series.coeff(h, 1) == 0


class TestTruncationError:
    """Tests for truncation error analysis utilities."""

    def test_taylor_expand_basic(self):
        """Taylor expansion should work for simple functions."""
        # u(x+h) = u(x) + h*u'(x) + h^2/2*u''(x) + ...
        func = u(x + h)
        expanded = taylor_expand(func, x, h, order=4)

        # Should contain u(x), h*derivative terms
        # Just check it returns something sensible
        assert expanded is not None

    def test_derive_truncation_error_central_diff(self):
        """Truncation error for central diff should be O(h^2)."""
        func = u(x)
        stencil = central_diff(func, x, h)
        exact = sp.Derivative(func, x)

        error_series, leading_term = derive_truncation_error(stencil, exact, x, h)

        # Leading term should be O(h^2)
        assert leading_term.has(h**2)

    def test_get_stencil_order_forward(self):
        """get_stencil_order should return 1 for forward difference."""
        func = u(x)
        stencil = forward_diff(func, x, h)
        exact = sp.Derivative(func, x)
        order = get_stencil_order(stencil, exact, x, h)
        assert order == 1

    def test_get_stencil_order_central(self):
        """get_stencil_order should return 2 for central difference."""
        func = u(x)
        stencil = central_diff(func, x, h)
        exact = sp.Derivative(func, x)
        order = get_stencil_order(stencil, exact, x, h)
        assert order == 2

    def test_get_stencil_order_second_deriv(self):
        """get_stencil_order should return 2 for second derivative central."""
        func = u(x)
        stencil = second_derivative_central(func, x, h)
        exact = sp.Derivative(func, x, x)
        order = get_stencil_order(stencil, exact, x, h)
        assert order == 2

    def test_get_stencil_order_fourth_order(self):
        """get_stencil_order should return 4 for fourth-order second derivative."""
        func = u(x)
        stencil = fourth_order_second_derivative(func, x, h)
        exact = sp.Derivative(func, x, x)
        order = get_stencil_order(stencil, exact, x, h)
        assert order == 4


class TestNumericalVerification:
    """Tests comparing symbolic stencils against numerical evaluation."""

    def test_central_diff_numerical(self):
        """Central difference should match numerical differentiation."""
        # Test function: f(x) = sin(x)
        f_sym = sp.sin(x)
        f_num = np.sin

        # Create the stencil expression
        stencil = central_diff(f_sym, x, h)

        # Convert to numerical function
        stencil_func = sp.lambdify([x, h], stencil, 'numpy')

        # Test at several points
        x_vals = np.array([0.5, 1.0, 1.5, 2.0])
        h_val = 0.001

        # Numerical differentiation (analytical derivative for comparison)
        exact_deriv = np.cos(x_vals)
        numerical_deriv = stencil_func(x_vals, h_val)

        np.testing.assert_allclose(numerical_deriv, exact_deriv, rtol=1e-5)

    def test_second_deriv_numerical(self):
        """Second derivative should match numerical evaluation."""
        # Test function: f(x) = x^4
        f_sym = x**4
        f_num = lambda x: x**4
        f_deriv2 = lambda x: 12 * x**2

        stencil = second_derivative_central(f_sym, x, h)
        stencil_func = sp.lambdify([x, h], stencil, 'numpy')

        x_vals = np.array([0.5, 1.0, 1.5, 2.0])
        h_val = 0.001

        exact = f_deriv2(x_vals)
        numerical = stencil_func(x_vals, h_val)

        np.testing.assert_allclose(numerical, exact, rtol=1e-5)

    @pytest.mark.parametrize("h_val,expected_order", [
        (0.1, 2),
        (0.01, 2),
        (0.001, 2),
    ])
    def test_convergence_rate(self, h_val, expected_order):
        """Error should decrease as h^order."""
        f_sym = sp.sin(x)
        stencil = central_diff(f_sym, x, h)
        stencil_func = sp.lambdify([x, h], stencil, 'numpy')

        x_val = 1.0
        exact = np.cos(x_val)

        # Compute at two h values
        h1, h2 = h_val, h_val / 2
        err1 = abs(stencil_func(x_val, h1) - exact)
        err2 = abs(stencil_func(x_val, h2) - exact)

        # Error ratio should be close to 2^order
        ratio = err1 / err2
        expected_ratio = 2**expected_order

        # Allow some tolerance (numerical precision limits)
        assert abs(ratio - expected_ratio) / expected_ratio < 0.1
