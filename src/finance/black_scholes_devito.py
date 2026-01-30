"""Black-Scholes option pricing solver using Devito DSL.

Solves the Black-Scholes PDE for European call and put options:
    V_t + 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V = 0

Time is measured backward from expiration T to present (t=0).
Using tau = T - t, the forward PDE becomes:
    V_tau = 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V

Boundary conditions for call option:
    V(0, tau) = 0                       (worthless if S=0)
    V(S_max, tau) ~ S - K*exp(-r*tau)   (deep in-the-money)

Boundary conditions for put option:
    V(0, tau) = K*exp(-r*tau)           (worth K at S=0)
    V(S_max, tau) = 0                   (worthless if S >> K)

Terminal condition (payoff at expiration):
    Call: V(S, 0) = max(S - K, 0)
    Put:  V(S, 0) = max(K - S, 0)
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

try:
    from devito import (
        Constant,
        Eq,
        Function,
        Grid,
        Operator,
        TimeFunction,
    )

    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@dataclass
class BlackScholesResult:
    """Results from Black-Scholes PDE solver.

    Attributes
    ----------
    V : np.ndarray
        Option values at t=0 (present), shape (nS+1,)
    S : np.ndarray
        Asset price grid points
    t : float
        Time (0 for present, T for expiration)
    dt : float
        Time step used
    K : float
        Strike price
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    T : float
        Time to expiration
    V_history : np.ndarray, optional
        Full solution history from expiration to present
    """

    V: np.ndarray
    S: np.ndarray
    t: float
    dt: float
    K: float
    r: float
    sigma: float
    T: float
    V_history: np.ndarray | None = None

    def V_at_S(self, S_target: float) -> float:
        """Interpolate option value at a specific asset price."""
        return np.interp(S_target, self.S, self.V)


@dataclass
class GreeksResult:
    """Greeks (sensitivities) for an option.

    Attributes
    ----------
    delta : np.ndarray
        dV/dS - sensitivity to underlying price
    gamma : np.ndarray
        d2V/dS2 - sensitivity of delta to underlying price
    theta : np.ndarray
        dV/dt - sensitivity to time (time decay)
    S : np.ndarray
        Asset price grid
    """

    delta: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    S: np.ndarray

    def delta_at_S(self, S_target: float) -> float:
        """Interpolate delta at a specific asset price."""
        return np.interp(S_target, self.S, self.delta)

    def gamma_at_S(self, S_target: float) -> float:
        """Interpolate gamma at a specific asset price."""
        return np.interp(S_target, self.S, self.gamma)

    def theta_at_S(self, S_target: float) -> float:
        """Interpolate theta at a specific asset price."""
        return np.interp(S_target, self.S, self.theta)


def black_scholes_analytical(
    S: float | np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float | np.ndarray:
    """Analytical Black-Scholes formula for European options.

    Parameters
    ----------
    S : float or np.ndarray
        Current stock price(s)
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized)
    option_type : str
        'call' or 'put'

    Returns
    -------
    float or np.ndarray
        Option value(s)
    """
    if T <= 0:
        # At expiration
        if option_type.lower() == "call":
            return np.maximum(S - K, 0.0)
        else:
            return np.maximum(K - S, 0.0)

    S = np.asarray(S)
    # Handle S=0 case
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        value = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        # Handle S=0: call is worthless
        value = np.where(S <= 0, 0.0, value)
    else:  # put
        value = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        # Handle S=0: put is worth K*exp(-rT)
        value = np.where(S <= 0, K * np.exp(-r * T), value)

    return float(value) if value.ndim == 0 else value


def solve_bs_european_call(
    S_max: float = 200.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    nS: int = 100,
    nt: int = 1000,
    save_history: bool = False,
) -> BlackScholesResult:
    """Solve Black-Scholes PDE for European call option.

    Uses explicit finite difference scheme with time stepping from
    expiration (t=T) backward to present (t=0).

    Parameters
    ----------
    S_max : float
        Maximum asset price in grid (should be several times K)
    K : float
        Strike price
    T : float
        Time to expiration (years)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized)
    nS : int
        Number of asset price grid intervals
    nt : int
        Number of time steps
    save_history : bool
        If True, save full solution history

    Returns
    -------
    BlackScholesResult
        Solution including option values at present time
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required. Install with: pip install devito")

    dS = S_max / nS
    dt = T / nt

    # Stability check for explicit scheme
    # For BS equation: dt < dS^2 / (sigma^2 * S_max^2)
    stability_dt = dS**2 / (sigma**2 * S_max**2 + abs(r) * S_max * dS)
    if dt > stability_dt:
        raise ValueError(
            f"Time step dt={dt:.6f} may be unstable. "
            f"Use nt >= {int(T / stability_dt) + 1} for stability."
        )

    # Create standard 1D grid
    grid = Grid(shape=(nS + 1,), extent=(S_max,))

    # Create TimeFunction for option value
    V = TimeFunction(name="V", grid=grid, time_order=1, space_order=2)

    # Asset price array
    S = np.linspace(0, S_max, nS + 1)

    # Terminal condition: payoff at expiration
    V.data[0, :] = np.maximum(S - K, 0)
    V.data[1, :] = V.data[0, :]

    # Coefficients as functions of S
    # V_tau = 0.5*sigma^2*S^2*V_SS + r*S*V_S - r*V
    sigma_const = Constant(name="sigma", value=sigma)
    r_const = Constant(name="r", value=r)
    dt_const = Constant(name="dt", value=dt)

    # S as a Function for the coefficients
    S_func = Function(name="S_arr", grid=grid)
    S_func.data[:] = S

    # Build the PDE update: explicit forward in tau
    # V^{n+1} = V^n + dt * (0.5*sigma^2*S^2*V_SS + r*S*V_S - r*V)
    # In 1D: laplace = d2V/dx2, dx = dV/dx
    diffusion = 0.5 * sigma_const**2 * S_func**2 * V.dx2
    convection = r_const * S_func * V.dx
    reaction = -r_const * V

    pde_rhs = diffusion + convection + reaction
    update_eq = Eq(V.forward, V + dt_const * pde_rhs, subdomain=grid.interior)

    # Boundary conditions
    t = grid.stepping_dim

    # At S=0: V = 0 (call is worthless)
    bc_S0 = Eq(V[t + 1, 0], 0.0)

    # At S=S_max: V = S - K*exp(-r*tau), approximate as S - K for large S
    # For stability, use linear extrapolation or fixed boundary
    bc_Smax = Eq(V[t + 1, nS], V[t + 1, nS - 1] + dS)

    # Create operator
    op = Operator([update_eq, bc_S0, bc_Smax])

    # Storage for history
    if save_history:
        V_history = np.zeros((nt + 1, nS + 1))
        V_history[0, :] = V.data[0, :]

    # Time stepping (backward in real time = forward in tau)
    for n in range(nt):
        op.apply(time_m=0, time_M=0, dt=dt)
        V.data[0, :] = V.data[1, :]

        if save_history:
            V_history[n + 1, :] = V.data[0, :]

    return BlackScholesResult(
        V=V.data[0, :].copy(),
        S=S,
        t=0.0,
        dt=dt,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        V_history=V_history if save_history else None,
    )


def solve_bs_european_put(
    S_max: float = 200.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    nS: int = 100,
    nt: int = 1000,
    save_history: bool = False,
) -> BlackScholesResult:
    """Solve Black-Scholes PDE for European put option.

    Parameters are the same as solve_bs_european_call.
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required. Install with: pip install devito")

    dS = S_max / nS
    dt = T / nt

    # Stability check
    stability_dt = dS**2 / (sigma**2 * S_max**2 + abs(r) * S_max * dS)
    if dt > stability_dt:
        raise ValueError(
            f"Time step dt={dt:.6f} may be unstable. "
            f"Use nt >= {int(T / stability_dt) + 1} for stability."
        )

    # Create standard 1D grid
    grid = Grid(shape=(nS + 1,), extent=(S_max,))

    V = TimeFunction(name="V", grid=grid, time_order=1, space_order=2)
    S = np.linspace(0, S_max, nS + 1)

    # Terminal condition: put payoff
    V.data[0, :] = np.maximum(K - S, 0)
    V.data[1, :] = V.data[0, :]

    sigma_const = Constant(name="sigma", value=sigma)
    r_const = Constant(name="r", value=r)
    dt_const = Constant(name="dt", value=dt)

    S_func = Function(name="S_arr", grid=grid)
    S_func.data[:] = S

    diffusion = 0.5 * sigma_const**2 * S_func**2 * V.dx2
    convection = r_const * S_func * V.dx
    reaction = -r_const * V

    pde_rhs = diffusion + convection + reaction
    update_eq = Eq(V.forward, V + dt_const * pde_rhs, subdomain=grid.interior)

    t = grid.stepping_dim

    # At S=0: V = K*exp(-r*tau), start with K and decay
    bc_S0 = Eq(V[t + 1, 0], K)  # Approximation; exact is K*exp(-r*tau)

    # At S=S_max: V = 0 (put is worthless)
    bc_Smax = Eq(V[t + 1, nS], 0.0)

    op = Operator([update_eq, bc_S0, bc_Smax])

    if save_history:
        V_history = np.zeros((nt + 1, nS + 1))
        V_history[0, :] = V.data[0, :]

    for n in range(nt):
        op.apply(time_m=0, time_M=0, dt=dt)
        V.data[0, :] = V.data[1, :]

        # Update S=0 boundary with time-dependent value
        tau = (n + 1) * dt
        V.data[0, 0] = K * np.exp(-r * tau)

        if save_history:
            V_history[n + 1, :] = V.data[0, :]

    return BlackScholesResult(
        V=V.data[0, :].copy(),
        S=S,
        t=0.0,
        dt=dt,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        V_history=V_history if save_history else None,
    )


def compute_greeks(
    V: np.ndarray,
    S: np.ndarray,
    dt: float,
    r: float,
    sigma: float,
    V_prev: np.ndarray | None = None,
) -> GreeksResult:
    """Compute option Greeks from numerical solution.

    Parameters
    ----------
    V : np.ndarray
        Option values at current time
    S : np.ndarray
        Asset price grid
    dt : float
        Time step for theta calculation
    r : float
        Risk-free rate
    sigma : float
        Volatility
    V_prev : np.ndarray, optional
        Option values at previous time step (for theta)

    Returns
    -------
    GreeksResult
        Greeks (delta, gamma, theta)
    """
    dS = S[1] - S[0]

    # Delta = dV/dS (central difference)
    delta = np.zeros_like(V)
    delta[1:-1] = (V[2:] - V[:-2]) / (2 * dS)
    delta[0] = (V[1] - V[0]) / dS
    delta[-1] = (V[-1] - V[-2]) / dS

    # Gamma = d2V/dS2 (central difference)
    gamma = np.zeros_like(V)
    gamma[1:-1] = (V[2:] - 2 * V[1:-1] + V[:-2]) / dS**2

    # Theta = dV/dt (from time stepping if available)
    if V_prev is not None:
        theta = (V - V_prev) / dt
    else:
        # Estimate from PDE: theta = -0.5*sigma^2*S^2*gamma - r*S*delta + r*V
        theta = -0.5 * sigma**2 * S**2 * gamma - r * S * delta + r * V

    return GreeksResult(delta=delta, gamma=gamma, theta=theta, S=S)


def analytical_greeks(
    S: float | np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """Compute analytical Greeks from Black-Scholes formulas.

    Parameters
    ----------
    S : float or np.ndarray
        Current stock price(s)
    K : float
        Strike price
    T : float
        Time to expiration (years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'

    Returns
    -------
    dict
        Dictionary with keys 'delta', 'gamma', 'theta', 'vega', 'rho'
    """
    if T <= 0:
        # At expiration
        S = np.asarray(S)
        if option_type.lower() == "call":
            delta = np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5))
        else:
            delta = np.where(S > K, 0.0, np.where(S < K, -1.0, -0.5))
        zeros = np.zeros_like(S, dtype=float)
        return {
            "delta": float(delta) if delta.ndim == 0 else delta,
            "gamma": float(zeros) if zeros.ndim == 0 else zeros,
            "theta": float(zeros) if zeros.ndim == 0 else zeros,
            "vega": float(zeros) if zeros.ndim == 0 else zeros,
            "rho": float(zeros) if zeros.ndim == 0 else zeros,
        }

    S = np.asarray(S)
    sqrt_T = np.sqrt(T)

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

    # PDF and CDF of standard normal
    phi_d1 = stats.norm.pdf(d1)
    N_d1 = stats.norm.cdf(d1)
    N_d2 = stats.norm.cdf(d2)
    N_neg_d1 = stats.norm.cdf(-d1)
    N_neg_d2 = stats.norm.cdf(-d2)

    # Gamma (same for call and put)
    gamma = np.where(S > 0, phi_d1 / (S * sigma * sqrt_T), 0.0)

    # Vega (same for call and put)
    vega = np.where(S > 0, S * phi_d1 * sqrt_T, 0.0)

    if option_type.lower() == "call":
        delta = np.where(S > 0, N_d1, 0.0)
        theta = np.where(
            S > 0,
            -S * phi_d1 * sigma / (2 * sqrt_T) - r * K * np.exp(-r * T) * N_d2,
            -r * K * np.exp(-r * T),
        )
        rho = np.where(S > 0, K * T * np.exp(-r * T) * N_d2, 0.0)
    else:
        delta = np.where(S > 0, N_d1 - 1, -1.0)
        theta = np.where(
            S > 0,
            -S * phi_d1 * sigma / (2 * sqrt_T) + r * K * np.exp(-r * T) * N_neg_d2,
            r * K * np.exp(-r * T),
        )
        rho = np.where(S > 0, -K * T * np.exp(-r * T) * N_neg_d2, -K * T * np.exp(-r * T))

    def _to_scalar_if_needed(arr):
        return float(arr) if arr.ndim == 0 else arr

    return {
        "delta": _to_scalar_if_needed(delta),
        "gamma": _to_scalar_if_needed(gamma),
        "theta": _to_scalar_if_needed(theta),
        "vega": _to_scalar_if_needed(vega),
        "rho": _to_scalar_if_needed(rho),
    }
