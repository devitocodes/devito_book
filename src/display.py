"""Display utilities for SymPy expressions in Quarto documents.

Provides consistent LaTeX rendering of equations with optional labels
for cross-referencing in Quarto documents.

Usage:
    from src.display import show_eq, show_eq_aligned, show_derivation

    # Single equation with label
    show_eq(u.dt - alpha * u.dx2, label='eq-heat')

    # Aligned equations
    show_eq_aligned([
        (u.dt, alpha * u.dx2),
        ('u^{n+1}', 'u^n + \\Delta t \\cdot \\alpha \\cdot u_{xx}'),
    ], label='eq-heat-steps')
"""

import sympy as sp
from IPython.display import Math, display


def latex_expr(expr, **kwargs) -> str:
    """Convert SymPy expression to LaTeX string.

    Parameters
    ----------
    expr : sympy expression or str
        Expression to convert
    **kwargs : dict
        Additional arguments passed to sp.latex()

    Returns
    -------
    str
        LaTeX string representation
    """
    if isinstance(expr, str):
        return expr
    return sp.latex(expr, **kwargs)


def show_eq(
    lhs,
    rhs=None,
    label: str | None = None,
    inline: bool = False,
) -> str:
    """Display a single equation with optional Quarto label.

    Parameters
    ----------
    lhs : sympy expression or str
        Left-hand side of equation (or entire equation if rhs is None)
    rhs : sympy expression or str, optional
        Right-hand side of equation
    label : str, optional
        Quarto cross-reference label (e.g., 'eq-heat-pde')
    inline : bool
        If True, return inline math ($...$), otherwise display math ($$...$$)

    Returns
    -------
    str
        LaTeX string suitable for Quarto markdown

    Examples
    --------
    >>> show_eq(u.dt, alpha * u.dx2, label='eq-heat')
    '$$\\frac{\\partial u}{\\partial t} = \\alpha \\frac{\\partial^2 u}{\\partial x^2}$$ {#eq-heat}'
    """
    lhs_latex = latex_expr(lhs)

    if rhs is not None:
        rhs_latex = latex_expr(rhs)
        eq_latex = f"{lhs_latex} = {rhs_latex}"
    else:
        eq_latex = lhs_latex

    if inline:
        return f"${eq_latex}$"

    if label:
        return f"$$\n{eq_latex}\n$$ {{#{label}}}"
    else:
        return f"$$\n{eq_latex}\n$$"


def show_eq_aligned(
    equations: list[tuple],
    label: str | None = None,
    env: str = 'aligned',
) -> str:
    """Display multiple aligned equations.

    Parameters
    ----------
    equations : list of tuples
        Each tuple is (lhs, rhs) for one line of the alignment
    label : str, optional
        Quarto cross-reference label for the entire block
    env : str
        LaTeX environment: 'aligned' (single number), 'align' (each line numbered)

    Returns
    -------
    str
        LaTeX string with aligned equations

    Examples
    --------
    >>> show_eq_aligned([
    ...     (u.dt, alpha * u.dx2),
    ...     ('u^{n+1}', 'u^n + dt * alpha * u_xx'),
    ... ], label='eq-heat-discretized')
    """
    lines = []
    for lhs, rhs in equations:
        lhs_latex = latex_expr(lhs)
        rhs_latex = latex_expr(rhs)
        lines.append(f"{lhs_latex} &= {rhs_latex}")

    content = " \\\\\n".join(lines)

    if env == 'aligned':
        # Single equation number for block
        latex_block = f"\\begin{{aligned}}\n{content}\n\\end{{aligned}}"
    else:
        # Each line gets a number (use with \label{} for individual refs)
        latex_block = f"\\begin{{{env}}}\n{content}\n\\end{{{env}}}"

    if label:
        return f"$$\n{latex_block}\n$$ {{#{label}}}"
    else:
        return f"$$\n{latex_block}\n$$"


def show_derivation(
    steps: list[tuple[str, sp.Expr | str]],
    label: str | None = None,
) -> str:
    """Display a mathematical derivation with descriptions.

    Parameters
    ----------
    steps : list of tuples
        Each tuple is (description, expression)
    label : str, optional
        Quarto label for the derivation block

    Returns
    -------
    str
        Formatted derivation suitable for Quarto

    Examples
    --------
    >>> show_derivation([
    ...     ('Start with the PDE', pde),
    ...     ('Apply forward difference in time', fd_time),
    ...     ('Rearrange for u^{n+1}', update_eq),
    ... ], label='eq-heat-derivation')
    """
    output_lines = []

    for description, expr in steps:
        expr_latex = latex_expr(expr)
        output_lines.append(f"**{description}:**")
        output_lines.append(f"$$\n{expr_latex}\n$$")
        output_lines.append("")  # Blank line between steps

    result = "\n".join(output_lines)

    # Note: Quarto doesn't support labeling multi-block derivations directly
    # The label would need to be applied to a specific equation
    return result


def inline_latex(expr, wrap: bool = True) -> str:
    """Convert expression to inline LaTeX.

    Parameters
    ----------
    expr : sympy expression or str
        Expression to convert
    wrap : bool
        If True, wrap in $...$ delimiters

    Returns
    -------
    str
        Inline LaTeX string
    """
    latex_str = latex_expr(expr)
    if wrap:
        return f"${latex_str}$"
    return latex_str


def display_sympy(expr):
    """Display SymPy expression in Jupyter/Quarto with LaTeX rendering.

    Parameters
    ----------
    expr : sympy expression
        Expression to display
    """
    display(Math(sp.latex(expr)))


def print_latex(expr, label: str | None = None):
    """Print LaTeX suitable for copy-paste into Quarto.

    Parameters
    ----------
    expr : sympy expression
        Expression to convert
    label : str, optional
        Quarto equation label
    """
    output = show_eq(expr, label=label)
    print(output)


# =============================================================================
# Macro Replacement Helpers
# =============================================================================

def macro_to_sympy(macro_name: str):
    """Map common LaTeX macros to SymPy equivalents.

    This helps migrate from custom LaTeX macros to SymPy-generated LaTeX.

    Parameters
    ----------
    macro_name : str
        Name of the LaTeX macro (without backslash)

    Returns
    -------
    sympy expression or str
        Equivalent SymPy expression or LaTeX string
    """
    from .symbols import dt, dx, dy, dz, t, u, x

    # Mapping of common macros to SymPy expressions
    macro_map = {
        'Ddt': sp.Derivative(u(x, t), t),
        'Ddx': sp.Derivative(u(x, t), x),
        'uex': sp.Function('u_e'),
        'half': sp.Rational(1, 2),
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'dt': dt,
        'tp': sp.Symbol('t^+'),
        'tm': sp.Symbol('t^-'),
        'xp': sp.Symbol('x^+'),
        'xm': sp.Symbol('x^-'),
        'Oof': lambda n: sp.O(sp.Symbol('h')**n),
    }

    return macro_map.get(macro_name, f'\\{macro_name}')


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'display_sympy',
    'inline_latex',
    'latex_expr',
    'macro_to_sympy',
    'print_latex',
    'show_derivation',
    'show_eq',
    'show_eq_aligned',
]
