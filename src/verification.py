"""Verification utilities for finite difference derivations.

Provides tools to verify mathematical identities, check stencil accuracy,
and validate PDE solutions using the Method of Manufactured Solutions (MMS).

Usage:
    from src.verification import verify_identity, check_stencil_order

    # Verify a Taylor series derivation
    assert verify_identity(central_diff(u(x), x, h), u(x).diff(x), h, order=2)

    # Check stencil order of accuracy
    order = check_stencil_order(stencil_expr, exact_deriv, h)
"""

from collections.abc import Callable

import numpy as np
import sympy as sp


def verify_identity(
    lhs,
    rhs,
    step_symbol,
    order: int = 6,
    tolerance: float = 1e-12,
) -> bool:
    """Verify that two expressions are equal up to truncation error.

    Expands both sides in Taylor series and checks if they match
    to the specified order.

    Parameters
    ----------
    lhs : sympy expression
        Left-hand side expression (e.g., finite difference approximation)
    rhs : sympy expression
        Right-hand side expression (e.g., exact derivative)
    step_symbol : sympy Symbol
        Grid spacing symbol (e.g., dx, dt, h)
    order : int
        Order of Taylor expansion for comparison
    tolerance : float
        Numerical tolerance for coefficient comparison

    Returns
    -------
    bool
        True if expressions match to specified order

    Examples
    --------
    >>> from src.operators import central_diff
    >>> from src.symbols import x, dx, u
    >>> verify_identity(central_diff(u(x), x, dx), u(x).diff(x), dx, order=2)
    True
    """
    # Expand both sides
    lhs_expanded = sp.series(lhs, step_symbol, 0, order).removeO()
    rhs_expanded = sp.series(rhs, step_symbol, 0, order).removeO()

    # Compare
    diff = sp.simplify(lhs_expanded - rhs_expanded)

    # Check if difference vanishes (or is purely higher order)
    diff_series = sp.series(diff, step_symbol, 0, order)

    # Extract coefficients and check they're all zero
    for power in range(order):
        coeff = diff_series.coeff(step_symbol, power)
        if coeff != 0:
            # Try numerical evaluation if symbolic simplification fails
            try:
                # Substitute dummy values for any remaining symbols
                test_val = complex(coeff.evalf())
                if abs(test_val) > tolerance:
                    return False
            except (TypeError, ValueError):
                # If can't evaluate numerically, it's likely non-zero
                return False

    return True


def check_stencil_order(
    stencil_expr,
    exact_derivative,
    step_symbol,
    max_order: int = 8,
) -> int:
    """Determine the order of accuracy of a finite difference stencil.

    Parameters
    ----------
    stencil_expr : sympy expression
        The finite difference approximation
    exact_derivative : sympy expression
        The exact derivative being approximated
    step_symbol : sympy Symbol
        Grid spacing symbol
    max_order : int
        Maximum order to check

    Returns
    -------
    int
        Order of accuracy (e.g., 2 for O(h^2))

    Examples
    --------
    >>> from src.operators import second_derivative_central
    >>> order = check_stencil_order(
    ...     second_derivative_central(u(x), x, dx),
    ...     u(x).diff(x, 2),
    ...     dx
    ... )
    >>> order
    2
    """
    error = sp.simplify(stencil_expr - exact_derivative)
    series = sp.series(error, step_symbol, 0, max_order + 1)

    for power in range(max_order + 1):
        coeff = series.coeff(step_symbol, power)
        if coeff != 0:
            # Try to simplify/evaluate the coefficient
            coeff_simplified = sp.simplify(coeff)
            if coeff_simplified != 0:
                return power

    return max_order


def get_truncation_error(
    stencil_expr,
    exact_derivative,
    step_symbol,
    order: int = 4,
) -> tuple[sp.Expr, sp.Expr]:
    """Compute the truncation error of a stencil.

    Parameters
    ----------
    stencil_expr : sympy expression
        The finite difference approximation
    exact_derivative : sympy expression
        The exact derivative being approximated
    step_symbol : sympy Symbol
        Grid spacing symbol
    order : int
        Order of Taylor expansion

    Returns
    -------
    tuple
        (full_error_series, leading_term)
    """
    error = sp.simplify(stencil_expr - exact_derivative)
    series = sp.series(error, step_symbol, 0, order + 1)

    # Find leading term
    for power in range(order + 1):
        coeff = series.coeff(step_symbol, power)
        if coeff != 0:
            leading_term = coeff * step_symbol**power
            break
    else:
        leading_term = sp.Integer(0)

    return series.removeO(), leading_term


def verify_pde_solution(
    pde_lhs,
    pde_rhs,
    solution,
    variables: dict,
    tolerance: float = 1e-10,
) -> bool:
    """Verify that a function satisfies a PDE.

    Useful for Method of Manufactured Solutions (MMS) verification.

    Parameters
    ----------
    pde_lhs : sympy expression
        Left-hand side of PDE (e.g., u.diff(t) - alpha*u.diff(x,2))
    pde_rhs : sympy expression
        Right-hand side of PDE (usually 0 or source term)
    solution : sympy expression
        Proposed solution to substitute
    variables : dict
        Mapping of function symbols to solution (e.g., {u(x,t): exp(-t)*sin(x)})
    tolerance : float
        Numerical tolerance for verification

    Returns
    -------
    bool
        True if solution satisfies PDE

    Examples
    --------
    >>> from src.symbols import x, t, alpha, u
    >>> # Heat equation: u_t = alpha * u_xx
    >>> pde = u(x,t).diff(t) - alpha * u(x,t).diff(x,2)
    >>> sol = sp.exp(-alpha * sp.pi**2 * t) * sp.sin(sp.pi * x)
    >>> verify_pde_solution(pde, 0, sol, {u(x,t): sol})
    True
    """
    # Substitute solution into PDE
    residual = pde_lhs - pde_rhs
    for func, expr in variables.items():
        residual = residual.subs(func, expr)

    # Simplify
    residual = sp.simplify(residual)

    # Check if zero
    if residual == 0:
        return True

    # Try numerical evaluation at random points
    try:
        free_syms = residual.free_symbols
        test_values = {s: np.random.uniform(0.1, 1.0) for s in free_syms}
        numerical_residual = complex(residual.subs(test_values).evalf())
        return abs(numerical_residual) < tolerance
    except (TypeError, ValueError):
        return False


def numerical_verify(
    symbolic_expr,
    numerical_func: Callable,
    test_points: np.ndarray,
    tolerance: float = 1e-8,
    **param_values,
) -> tuple[bool, float]:
    """Compare symbolic expression against numerical implementation.

    Parameters
    ----------
    symbolic_expr : sympy expression
        SymPy expression to evaluate
    numerical_func : callable
        Python/NumPy function to compare against
    test_points : np.ndarray
        Points at which to evaluate
    tolerance : float
        Maximum allowed difference
    **param_values : dict
        Values for symbolic parameters

    Returns
    -------
    tuple
        (passed: bool, max_error: float)

    Examples
    --------
    >>> expr = x**2 + 2*x + 1
    >>> def f(x): return x**2 + 2*x + 1
    >>> points = np.linspace(0, 10, 100)
    >>> passed, error = numerical_verify(expr, f, points)
    >>> passed
    True
    """
    # Create numerical function from symbolic expression
    free_syms = sorted(symbolic_expr.free_symbols, key=lambda s: s.name)

    # Substitute parameter values
    expr_substituted = symbolic_expr
    for name, value in param_values.items():
        sym = sp.Symbol(name)
        if sym in expr_substituted.free_symbols:
            expr_substituted = expr_substituted.subs(sym, value)

    # Lambdify the expression
    remaining_syms = sorted(expr_substituted.free_symbols, key=lambda s: s.name)
    if len(remaining_syms) == 1:
        sym_func = sp.lambdify(remaining_syms[0], expr_substituted, 'numpy')
    else:
        sym_func = sp.lambdify(remaining_syms, expr_substituted, 'numpy')

    # Evaluate both
    try:
        sym_values = sym_func(test_points)
        num_values = numerical_func(test_points)

        max_error = np.max(np.abs(sym_values - num_values))
        passed = max_error < tolerance

        return passed, max_error
    except Exception as e:
        return False, float('inf')


def convergence_test(
    solver_func: Callable,
    exact_solution: Callable,
    grid_sizes: list,
    expected_order: float,
    norm: str = 'L2',
    tolerance: float = 0.5,
) -> tuple[bool, float, list]:
    """Verify convergence order of a numerical solver.

    Parameters
    ----------
    solver_func : callable
        Function that takes grid size and returns numerical solution
        Signature: solver_func(n) -> (x_grid, u_numerical)
    exact_solution : callable
        Exact solution function: exact_solution(x) -> u_exact
    grid_sizes : list
        List of grid sizes to test (e.g., [10, 20, 40, 80])
    expected_order : float
        Expected convergence order
    norm : str
        Error norm: 'L2', 'Linf', or 'L1'
    tolerance : float
        Tolerance for order verification (e.g., 0.5 means order must be
        within expected_order +/- 0.5)

    Returns
    -------
    tuple
        (passed: bool, observed_order: float, errors: list)
    """
    errors = []

    for n in grid_sizes:
        x_grid, u_num = solver_func(n)
        u_exact = exact_solution(x_grid)

        diff = u_num - u_exact

        if norm == 'L2':
            err = np.sqrt(np.mean(diff**2))
        elif norm == 'Linf':
            err = np.max(np.abs(diff))
        elif norm == 'L1':
            err = np.mean(np.abs(diff))
        else:
            raise ValueError(f"Unknown norm: {norm}")

        errors.append(err)

    # Compute convergence rates
    errors = np.array(errors)
    grid_sizes = np.array(grid_sizes)

    # Linear regression in log-log space
    log_h = np.log(1.0 / grid_sizes)
    log_err = np.log(errors)

    # Fit line
    coeffs = np.polyfit(log_h, log_err, 1)
    observed_order = coeffs[0]

    # Check if within tolerance
    passed = abs(observed_order - expected_order) < tolerance

    return passed, observed_order, errors.tolist()


def verify_stability_condition(
    scheme_name: str,
    params: dict,
) -> tuple[bool, str]:
    """Check if numerical parameters satisfy stability conditions.

    Parameters
    ----------
    scheme_name : str
        Name of scheme: 'explicit_diffusion', 'wave_1d', 'advection_upwind'
    params : dict
        Dictionary containing relevant parameters (dt, dx, c, alpha, etc.)

    Returns
    -------
    tuple
        (is_stable: bool, message: str)
    """
    if scheme_name == 'explicit_diffusion':
        # Fourier number F = alpha * dt / dx^2 <= 0.5
        alpha = params.get('alpha', params.get('kappa', 1.0))
        dt = params['dt']
        dx = params['dx']
        F = alpha * dt / dx**2
        is_stable = F <= 0.5
        msg = f"Fourier number F = {F:.4f} {'<=' if is_stable else '>'} 0.5"
        return is_stable, msg

    elif scheme_name == 'wave_1d':
        # Courant number C = c * dt / dx <= 1
        c = params['c']
        dt = params['dt']
        dx = params['dx']
        C = c * dt / dx
        is_stable = C <= 1.0
        msg = f"Courant number C = {C:.4f} {'<=' if is_stable else '>'} 1.0"
        return is_stable, msg

    elif scheme_name == 'advection_upwind':
        # CFL: c * dt / dx <= 1
        c = params['c']
        dt = params['dt']
        dx = params['dx']
        CFL = abs(c) * dt / dx
        is_stable = CFL <= 1.0
        msg = f"CFL = {CFL:.4f} {'<=' if is_stable else '>'} 1.0"
        return is_stable, msg

    else:
        return True, f"Unknown scheme: {scheme_name} (no stability check)"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'check_stencil_order',
    'convergence_test',
    'get_truncation_error',
    'numerical_verify',
    'verify_identity',
    'verify_pde_solution',
    'verify_stability_condition',
]
