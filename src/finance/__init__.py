"""Computational Finance solvers using Devito DSL.

This module provides solvers for financial PDEs using Devito's
symbolic finite difference framework, including the Black-Scholes
equation for option pricing.

The Black-Scholes equation:
    dV/dt + 0.5 * sigma^2 * S^2 * d2V/dS2 + r * S * dV/dS - r * V = 0

where:
    V(S, t) = option value as a function of stock price S and time t
    sigma = volatility of the underlying asset
    r = risk-free interest rate
    S = underlying asset price

The module implements:
1. European call and put options
2. Analytical Black-Scholes formulas for verification
3. Greeks computation (Delta, Gamma, Theta)

Key features:
- Custom SpaceDimension for asset price grid
- Time-stepping from expiration backward to present
- Boundary conditions for far-field behavior
- Second-order accurate finite differences

Examples
--------
Price a European call option:

    >>> from src.finance import solve_bs_european_call
    >>> result = solve_bs_european_call(
    ...     S_max=200.0,      # Maximum asset price
    ...     K=100.0,          # Strike price
    ...     T=1.0,            # Time to expiration
    ...     r=0.05,           # Risk-free rate
    ...     sigma=0.2,        # Volatility
    ...     nS=100,           # Asset price grid points
    ...     nt=1000,          # Time steps
    ... )
    >>> print(f"Option value at S=100: {result.V_at_S(100.0):.4f}")

Compute Greeks:

    >>> from src.finance import compute_greeks
    >>> greeks = compute_greeks(
    ...     V=result.V,
    ...     S=result.S,
    ...     dt=result.dt,
    ...     r=0.05,
    ... )
    >>> print(f"Delta at S=100: {greeks.delta_at_S(100.0):.4f}")

Compare with analytical solution:

    >>> from src.finance import black_scholes_analytical
    >>> V_exact = black_scholes_analytical(
    ...     S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2,
    ...     option_type='call'
    ... )
    >>> print(f"Analytical value: {V_exact:.4f}")
"""

from src.finance.black_scholes_devito import (
    BlackScholesResult,
    GreeksResult,
    analytical_greeks,
    black_scholes_analytical,
    compute_greeks,
    solve_bs_european_call,
    solve_bs_european_put,
)

__all__ = [
    "BlackScholesResult",
    "GreeksResult",
    "analytical_greeks",
    "black_scholes_analytical",
    "compute_greeks",
    "solve_bs_european_call",
    "solve_bs_european_put",
]
