"""Canonical SymPy symbols used throughout the book.

Centralising symbols ensures:
1. Consistent notation across chapters
2. No accidental shadowing (e.g., redefining x differently)
3. Assumptions are uniform (real, positive, etc.)
4. LaTeX rendering is consistent

Usage:
    from src.symbols import x, t, dx, dt, u, c
    # or
    from src import x, t, dx, dt, u, c
"""

import sympy as sp

# =============================================================================
# Spatial Variables
# =============================================================================
x = sp.Symbol('x', real=True)
y = sp.Symbol('y', real=True)
z = sp.Symbol('z', real=True)

# =============================================================================
# Temporal Variable
# =============================================================================
t = sp.Symbol('t', real=True, nonnegative=True)

# =============================================================================
# Grid Spacing (always positive)
# =============================================================================
dx = sp.Symbol(r'\Delta x', positive=True, real=True)
dy = sp.Symbol(r'\Delta y', positive=True, real=True)
dz = sp.Symbol(r'\Delta z', positive=True, real=True)
dt = sp.Symbol(r'\Delta t', positive=True, real=True)

# Alternative notation aliases (common in numerical analysis texts)
h = sp.Symbol('h', positive=True, real=True)  # Spatial step alias
k = sp.Symbol('k', positive=True, real=True)  # Temporal step alias

# =============================================================================
# Generic Function Symbols
# =============================================================================
u = sp.Function('u')
v = sp.Function('v')
w = sp.Function('w')
f = sp.Function('f')
g = sp.Function('g')

# =============================================================================
# Physical Parameters (typically positive)
# =============================================================================
alpha = sp.Symbol(r'\alpha', positive=True)      # Diffusivity / thermal diffusivity
c = sp.Symbol('c', positive=True)                 # Wave speed
nu = sp.Symbol(r'\nu', positive=True)            # Kinematic viscosity
kappa = sp.Symbol(r'\kappa', positive=True)      # Thermal conductivity
rho = sp.Symbol(r'\rho', positive=True)          # Density
omega = sp.Symbol(r'\omega', positive=True)      # Angular frequency
L = sp.Symbol('L', positive=True)                 # Domain length
T_final = sp.Symbol('T', positive=True)          # Final time

# =============================================================================
# Index Variables for Stencils
# =============================================================================
i = sp.Symbol('i', integer=True)
j = sp.Symbol('j', integer=True)
n = sp.Symbol('n', integer=True, nonnegative=True)
m = sp.Symbol('m', integer=True, nonnegative=True)

# =============================================================================
# Dimensionless Numbers
# =============================================================================
C = sp.Symbol('C', positive=True)    # Courant number: c*dt/dx
F = sp.Symbol('F', positive=True)    # Fourier/diffusion number: alpha*dt/dx^2
Re = sp.Symbol('Re', positive=True)  # Reynolds number
Pe = sp.Symbol('Pe', positive=True)  # Peclet number

# =============================================================================
# Error Analysis
# =============================================================================
xi = sp.Symbol(r'\xi', real=True)    # Intermediate point in Taylor expansion
eta = sp.Symbol(r'\eta', real=True)  # Another intermediate point
theta = sp.Symbol(r'\theta', real=True)  # Fourier mode / angle


# =============================================================================
# Helper Functions
# =============================================================================
def exact(func_name: str = 'u') -> sp.Function:
    """Create exact solution symbol with subscript 'e'.

    Example:
        u_e = exact('u')
        u_e(x, t)  # Represents u_e(x, t)
    """
    return sp.Function(f'{func_name}_e')


def numerical(func_name: str = 'u') -> sp.Function:
    """Create numerical solution symbol (no subscript).

    This is typically the same as the base function but
    semantically represents the numerical approximation.
    """
    return sp.Function(func_name)


def grid_func(func_name: str = 'u'):
    """Create a symbol for grid function notation u_i^n.

    Returns a function that can be called as grid_func('u')(i, n)
    to represent u at grid point i and time level n.
    """
    return sp.Function(func_name)


# =============================================================================
# Commonly Used Expressions
# =============================================================================
def half():
    """Return SymPy Rational 1/2 for exact arithmetic."""
    return sp.Rational(1, 2)


def third():
    """Return SymPy Rational 1/3 for exact arithmetic."""
    return sp.Rational(1, 3)


def quarter():
    """Return SymPy Rational 1/4 for exact arithmetic."""
    return sp.Rational(1, 4)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    'C',
    'F',
    'L',
    'Pe',
    'Re',
    'T_final',
    'alpha',
    'c',
    'dt',
    'dx',
    'dy',
    'dz',
    'eta',
    'exact',
    'f',
    'g',
    'grid_func',
    'h',
    'half',
    'i',
    'j',
    'k',
    'kappa',
    'm',
    'n',
    'nu',
    'numerical',
    'omega',
    'quarter',
    'rho',
    'sp',
    't',
    'theta',
    'third',
    'u',
    'v',
    'w',
    'x',
    'xi',
    'y',
    'z',
]
