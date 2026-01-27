"""Finite difference operators and stencil generation.

All operators return SymPy expressions that can be:
1. Displayed as LaTeX via sp.latex()
2. Verified via Taylor series expansion
3. Related to Devito stencils

Usage:
    from src.operators import central_diff, second_derivative_central
    from src.symbols import u, x, dx

    # First derivative approximation
    du_dx = central_diff(u(x), x, dx)

    # Second derivative approximation
    d2u_dx2 = second_derivative_central(u(x), x, dx)
"""

import sympy as sp

# =============================================================================
# First Derivative Operators
# =============================================================================

def forward_diff(func, var, step):
    """First-order forward difference approximation.

    Approximates: df/dx at x
    Formula: (f(x+h) - f(x)) / h
    Order: O(h)

    Parameters
    ----------
    func : sympy expression
        Function to differentiate (e.g., u(x, t))
    var : sympy Symbol
        Variable to differentiate with respect to
    step : sympy Symbol
        Grid spacing (e.g., dx)

    Returns
    -------
    sympy expression
        Finite difference approximation
    """
    shifted = func.subs(var, var + step)
    return (shifted - func) / step


def backward_diff(func, var, step):
    """First-order backward difference approximation.

    Approximates: df/dx at x
    Formula: (f(x) - f(x-h)) / h
    Order: O(h)

    Parameters
    ----------
    func : sympy expression
        Function to differentiate
    var : sympy Symbol
        Variable to differentiate with respect to
    step : sympy Symbol
        Grid spacing

    Returns
    -------
    sympy expression
        Finite difference approximation
    """
    shifted = func.subs(var, var - step)
    return (func - shifted) / step


def central_diff(func, var, step):
    """Second-order central difference approximation for first derivative.

    Approximates: df/dx at x
    Formula: (f(x+h) - f(x-h)) / (2h)
    Order: O(h^2)

    Parameters
    ----------
    func : sympy expression
        Function to differentiate
    var : sympy Symbol
        Variable to differentiate with respect to
    step : sympy Symbol
        Grid spacing

    Returns
    -------
    sympy expression
        Finite difference approximation
    """
    forward = func.subs(var, var + step)
    backward = func.subs(var, var - step)
    return (forward - backward) / (2 * step)


# =============================================================================
# Second Derivative Operators
# =============================================================================

def second_derivative_central(func, var, step):
    """Second-order central difference for second derivative.

    Approximates: d²f/dx² at x
    Formula: (f(x+h) - 2f(x) + f(x-h)) / h²
    Order: O(h^2)

    Parameters
    ----------
    func : sympy expression
        Function to differentiate
    var : sympy Symbol
        Variable to differentiate with respect to
    step : sympy Symbol
        Grid spacing

    Returns
    -------
    sympy expression
        Finite difference approximation
    """
    forward = func.subs(var, var + step)
    backward = func.subs(var, var - step)
    return (forward - 2*func + backward) / step**2


def fourth_order_second_derivative(func, var, step):
    """Fourth-order accurate central difference for second derivative.

    Approximates: d²f/dx² at x
    Formula: (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h²)
    Order: O(h^4)

    Parameters
    ----------
    func : sympy expression
        Function to differentiate
    var : sympy Symbol
        Variable to differentiate with respect to
    step : sympy Symbol
        Grid spacing

    Returns
    -------
    sympy expression
        Finite difference approximation
    """
    f_p2 = func.subs(var, var + 2*step)
    f_p1 = func.subs(var, var + step)
    f_m1 = func.subs(var, var - step)
    f_m2 = func.subs(var, var - 2*step)

    return (-f_p2 + 16*f_p1 - 30*func + 16*f_m1 - f_m2) / (12 * step**2)


# =============================================================================
# Multi-Dimensional Operators
# =============================================================================

def laplacian_2d(func, x_var, y_var, hx, hy):
    """2D Laplacian using central differences.

    Approximates: ∇²f = ∂²f/∂x² + ∂²f/∂y²
    Order: O(hx², hy²)

    Parameters
    ----------
    func : sympy expression
        Function (e.g., u(x, y, t))
    x_var, y_var : sympy Symbols
        Spatial variables
    hx, hy : sympy Symbols
        Grid spacings in x and y directions

    Returns
    -------
    sympy expression
        Finite difference approximation of Laplacian
    """
    d2_dx2 = second_derivative_central(func, x_var, hx)
    d2_dy2 = second_derivative_central(func, y_var, hy)
    return d2_dx2 + d2_dy2


def laplacian_3d(func, x_var, y_var, z_var, hx, hy, hz):
    """3D Laplacian using central differences.

    Approximates: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
    Order: O(hx², hy², hz²)
    """
    d2_dx2 = second_derivative_central(func, x_var, hx)
    d2_dy2 = second_derivative_central(func, y_var, hy)
    d2_dz2 = second_derivative_central(func, z_var, hz)
    return d2_dx2 + d2_dy2 + d2_dz2


# =============================================================================
# Time Derivative Operators
# =============================================================================

def forward_euler_dt(func, t_var, dt_step):
    """Forward Euler time derivative (explicit).

    Approximates: ∂f/∂t at time n
    Formula: (f^{n+1} - f^n) / dt
    This is rearranged to: f^{n+1} = f^n + dt * (rhs)

    Parameters
    ----------
    func : sympy expression
        Function at time level n
    t_var : sympy Symbol
        Time variable
    dt_step : sympy Symbol
        Time step

    Returns
    -------
    sympy expression
        Forward difference in time
    """
    return forward_diff(func, t_var, dt_step)


def backward_euler_dt(func, t_var, dt_step):
    """Backward Euler time derivative (implicit).

    Approximates: ∂f/∂t at time n+1
    Formula: (f^{n+1} - f^n) / dt evaluated at n+1
    """
    return backward_diff(func, t_var, dt_step)


def central_time_second_derivative(func, t_var, dt_step):
    """Central difference for second time derivative.

    Approximates: ∂²f/∂t² at time n
    Formula: (f^{n+1} - 2f^n + f^{n-1}) / dt²
    Order: O(dt²)
    """
    return second_derivative_central(func, t_var, dt_step)


# =============================================================================
# Truncation Error Analysis
# =============================================================================

def taylor_expand(func, var, step, point=None, order: int = 6):
    """Expand function in Taylor series about a point.

    Parameters
    ----------
    func : sympy expression
        Function to expand (e.g., u(x+h, t))
    var : sympy Symbol
        Variable of expansion
    step : sympy Symbol
        Expansion parameter (appears in series)
    point : optional
        Point about which to expand (default: current value of var)
    order : int
        Number of terms to keep

    Returns
    -------
    sympy expression
        Taylor series expansion (without O() term)
    """
    return sp.series(func, step, 0, order).removeO()


def _expand_stencil_taylor(stencil_expr, func, var, step, order: int = 8):
    """Expand a stencil expression using Taylor series.

    This substitutes a polynomial test function and expands all shifted
    function evaluations u(x+h), u(x-h), etc. in Taylor series.

    Parameters
    ----------
    stencil_expr : sympy expression
        Stencil with function calls like u(x+h)
    func : sympy Function
        The function symbol (e.g., u)
    var : sympy Symbol
        The variable (e.g., x)
    step : sympy Symbol
        Grid spacing (e.g., h)
    order : int
        Order of Taylor expansion

    Returns
    -------
    sympy expression
        Stencil expanded as series in step
    """
    # Find all function applications in the expression
    result = stencil_expr

    # Get the function class
    if hasattr(func, 'func'):
        func_class = func.func
    else:
        func_class = func

    # Find all atoms that are function applications
    for atom in stencil_expr.atoms(sp.Function):
        if atom.func == func_class:
            # Get the argument
            args = atom.args
            if len(args) == 1:
                arg = args[0]
                # If argument contains step, expand in Taylor series
                if arg.has(step):
                    # Extract the shift: arg = var + shift
                    shift = sp.simplify(arg - var)
                    if shift != 0:
                        # Taylor expand f(var + shift) around var
                        expanded = sp.Integer(0)
                        f_at_var = func_class(var)
                        for n in range(order):
                            deriv = sp.diff(f_at_var, var, n)
                            expanded += deriv * shift**n / sp.factorial(n)
                        result = result.subs(atom, expanded)

    return sp.series(result, step, 0, order).removeO()


def derive_truncation_error(stencil_expr, exact_derivative, var, step, order: int = 6):
    """Compute truncation error of a finite difference stencil.

    Parameters
    ----------
    stencil_expr : sympy expression
        The finite difference approximation
    exact_derivative : sympy expression
        The exact derivative being approximated
    var : sympy Symbol
        Spatial or temporal variable
    step : sympy Symbol
        Grid spacing
    order : int
        Order of Taylor expansion for analysis

    Returns
    -------
    tuple
        (truncation_error_series, leading_order_term)
    """
    # Find the function in the stencil
    func = None
    for atom in stencil_expr.atoms(sp.Function):
        func = atom.func
        break

    if func is None:
        # No function found, try direct series
        error = sp.simplify(stencil_expr - exact_derivative)
        series = sp.series(error, step, 0, order)
    else:
        # Expand stencil using Taylor series
        expanded_stencil = _expand_stencil_taylor(stencil_expr, func, var, step, order)

        # The exact derivative should match the leading term
        # Get the derivative form for comparison
        if isinstance(exact_derivative, sp.Derivative):
            # It's a symbolic derivative - evaluate it
            deriv_order = exact_derivative.derivative_count
            exact_val = sp.diff(func(var), var, deriv_order)
        else:
            exact_val = exact_derivative

        error = sp.simplify(expanded_stencil - exact_val)
        series = sp.series(error, step, 0, order)

    # Find leading order term
    leading_term = sp.Integer(0)
    for power in range(order):
        coeff = series.coeff(step, power)
        if coeff != 0:
            coeff_simplified = sp.simplify(coeff)
            if coeff_simplified != 0:
                leading_term = coeff_simplified * step**power
                break

    return series.removeO(), leading_term


def get_stencil_order(stencil_expr, exact_derivative, var, step, max_order: int = 8):
    """Determine the order of accuracy of a stencil.

    Parameters
    ----------
    stencil_expr : sympy expression
        The finite difference approximation
    exact_derivative : sympy expression
        The exact derivative being approximated
    var : sympy Symbol
        Variable
    step : sympy Symbol
        Grid spacing
    max_order : int
        Maximum order to check

    Returns
    -------
    int
        Order of accuracy (e.g., 2 for O(h²))
    """
    # Find the function in the stencil
    func = None
    for atom in stencil_expr.atoms(sp.Function):
        func = atom.func
        break

    if func is None:
        # No function applications - try direct computation
        error = sp.simplify(stencil_expr - exact_derivative)
        series = sp.series(error, step, 0, max_order + 1)
    else:
        # Expand stencil using Taylor series
        expanded_stencil = _expand_stencil_taylor(stencil_expr, func, var, step, max_order + 1)

        # Get the derivative value for comparison
        if isinstance(exact_derivative, sp.Derivative):
            deriv_order = exact_derivative.derivative_count
            exact_val = sp.diff(func(var), var, deriv_order)
        else:
            exact_val = exact_derivative

        error = sp.simplify(expanded_stencil - exact_val)
        series = sp.series(error, step, 0, max_order + 1)

    for power in range(max_order + 1):
        coeff = series.coeff(step, power)
        if coeff != 0:
            coeff_simplified = sp.simplify(coeff)
            if coeff_simplified != 0:
                return power

    return max_order  # If all terms vanish, it's at least max_order accurate


# =============================================================================
# Devito Connection Helpers
# =============================================================================

def stencil_to_devito_hint(derivative_type: str, space_order: int) -> str:
    """Generate hint showing equivalent Devito syntax.

    Parameters
    ----------
    derivative_type : str
        Type of derivative: 'dx', 'dx2', 'dt', 'dt2', 'laplace'
    space_order : int
        Devito space_order parameter

    Returns
    -------
    str
        Comment showing Devito equivalent
    """
    hints = {
        'dx': f'u.dx  # with space_order={space_order}',
        'dx2': f'u.dx2  # with space_order={space_order}',
        'dy': f'u.dy  # with space_order={space_order}',
        'dy2': f'u.dy2  # with space_order={space_order}',
        'dt': 'u.dt  # with time_order=1',
        'dt2': 'u.dt2  # with time_order=2',
        'laplace': f'u.laplace  # with space_order={space_order}',
    }
    return hints.get(derivative_type, f'# Unknown derivative type: {derivative_type}')


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'backward_diff',
    'backward_euler_dt',
    'central_diff',
    'central_time_second_derivative',
    'derive_truncation_error',
    'forward_diff',
    'forward_euler_dt',
    'fourth_order_second_derivative',
    'get_stencil_order',
    'laplacian_2d',
    'laplacian_3d',
    'second_derivative_central',
    'stencil_to_devito_hint',
    'taylor_expand',
]
