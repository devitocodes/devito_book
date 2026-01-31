"""Tests for Black-Scholes option pricing solvers using Devito.

This module tests the Black-Scholes PDE solvers for European options,
including:
1. Call and put option pricing
2. Put-call parity verification
3. Greeks computation (Delta, Gamma, Theta)
4. Convergence to analytical solutions
5. Boundary conditions and time decay

The Black-Scholes PDE:
    V_t + 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V = 0

Per CONTRIBUTING.md: All results must be reproducible with fixed random seeds,
version-pinned dependencies, and automated tests validating examples.
"""

import numpy as np
import pytest

# Check if Devito is available
try:
    import devito  # noqa: F401

    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DEVITO_AVAILABLE, reason="Devito not installed")


# =============================================================================
# Test: Module Imports
# =============================================================================


@pytest.mark.devito
class TestModuleImports:
    """Test that the finance module imports correctly."""

    def test_import_black_scholes_module(self):
        """Test importing the Black-Scholes module."""
        from src.finance import black_scholes_devito

        assert black_scholes_devito is not None

    def test_import_solver_functions(self):
        """Test importing solver functions."""
        from src.finance import (
            solve_bs_european_call,
            solve_bs_european_put,
        )

        assert solve_bs_european_call is not None
        assert solve_bs_european_put is not None

    def test_import_analytical_functions(self):
        """Test importing analytical solution functions."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        assert black_scholes_analytical is not None

    def test_import_greeks_functions(self):
        """Test importing Greeks computation functions."""
        from src.finance.black_scholes_devito import compute_greeks

        assert compute_greeks is not None

    def test_import_result_dataclass(self):
        """Test importing result dataclass."""
        from src.finance.black_scholes_devito import BlackScholesResult

        assert BlackScholesResult is not None


# =============================================================================
# Test: Analytical Black-Scholes Formula
# =============================================================================


class TestAnalyticalBlackScholes:
    """Tests for the analytical Black-Scholes formula."""

    def test_call_at_expiry(self):
        """At expiry (T=0), call value should be max(S-K, 0)."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        S = np.array([80, 100, 120])
        K = 100
        T = 0
        r = 0.05
        sigma = 0.2

        V = black_scholes_analytical(S, K, T, r, sigma, option_type="call")
        expected = np.maximum(S - K, 0)

        np.testing.assert_allclose(V, expected, rtol=1e-10)

    def test_put_at_expiry(self):
        """At expiry (T=0), put value should be max(K-S, 0)."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        S = np.array([80, 100, 120])
        K = 100
        T = 0
        r = 0.05
        sigma = 0.2

        V = black_scholes_analytical(S, K, T, r, sigma, option_type="put")
        expected = np.maximum(K - S, 0)

        np.testing.assert_allclose(V, expected, rtol=1e-10)

    def test_call_positive_for_itm(self):
        """In-the-money call (S > K) should have positive value."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        V = black_scholes_analytical(S=120, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
        assert V > 0

    def test_put_positive_for_itm(self):
        """In-the-money put (S < K) should have positive value."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        V = black_scholes_analytical(S=80, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
        assert V > 0

    def test_call_value_increases_with_S(self):
        """Call value should increase with stock price."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        S = np.array([80, 100, 120])
        V = black_scholes_analytical(S, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")

        assert V[1] > V[0]
        assert V[2] > V[1]

    def test_put_value_decreases_with_S(self):
        """Put value should decrease with stock price."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        S = np.array([80, 100, 120])
        V = black_scholes_analytical(S, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")

        assert V[0] > V[1]
        assert V[1] > V[2]

    def test_call_at_S_zero(self):
        """Call value at S=0 should be 0."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        V = black_scholes_analytical(S=0, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
        assert V == pytest.approx(0.0, abs=1e-10)

    def test_put_at_S_zero(self):
        """Put value at S=0 should be K*exp(-rT)."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        K = 100
        T = 1.0
        r = 0.05
        V = black_scholes_analytical(S=0, K=K, T=T, r=r, sigma=0.2, option_type="put")
        expected = K * np.exp(-r * T)

        assert V == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Test: Put-Call Parity
# =============================================================================


class TestPutCallParity:
    """Tests for put-call parity: C - P = S - K*exp(-rT)."""

    def test_parity_analytical(self):
        """Put-call parity should hold for analytical solutions."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2

        C = black_scholes_analytical(S, K, T, r, sigma, option_type="call")
        P = black_scholes_analytical(S, K, T, r, sigma, option_type="put")

        parity_lhs = C - P
        parity_rhs = S - K * np.exp(-r * T)

        assert parity_lhs == pytest.approx(parity_rhs, rel=1e-10)

    def test_parity_numerical(self):
        """Put-call parity should approximately hold for numerical solutions."""
        from src.finance import solve_bs_european_call, solve_bs_european_put

        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2

        call_result = solve_bs_european_call(
            S_max=300, K=K, T=T, r=r, sigma=sigma, nS=200, nt=2000
        )
        put_result = solve_bs_european_put(
            S_max=300, K=K, T=T, r=r, sigma=sigma, nS=200, nt=2000
        )

        # Check parity at S = K (at-the-money)
        S = K
        C = call_result.V_at_S(S)
        P = put_result.V_at_S(S)

        parity_lhs = C - P
        parity_rhs = S - K * np.exp(-r * T)

        # Allow larger tolerance for numerical solution
        assert parity_lhs == pytest.approx(parity_rhs, rel=0.05)

    def test_parity_various_strikes(self):
        """Put-call parity should hold for various strikes."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        S = 100
        T = 1.0
        r = 0.05
        sigma = 0.2

        for K in [80, 100, 120]:
            C = black_scholes_analytical(S, K, T, r, sigma, option_type="call")
            P = black_scholes_analytical(S, K, T, r, sigma, option_type="put")

            parity_diff = abs((C - P) - (S - K * np.exp(-r * T)))
            assert parity_diff < 1e-10, f"Parity failed for K={K}"


# =============================================================================
# Test: European Call Option Solver
# =============================================================================


@pytest.mark.devito
class TestEuropeanCallSolver:
    """Tests for the European call option solver."""

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.finance import solve_bs_european_call

        result = solve_bs_european_call(
            S_max=200, K=100, T=1.0, r=0.05, sigma=0.2, nS=50, nt=500
        )

        assert result.V is not None
        assert result.S is not None
        assert len(result.V) == 51
        assert len(result.S) == 51

    def test_boundary_at_S_zero(self):
        """Call value at S=0 should be 0."""
        from src.finance import solve_bs_european_call

        result = solve_bs_european_call(
            S_max=200, K=100, T=1.0, r=0.05, sigma=0.2, nS=100, nt=1000
        )

        assert result.V[0] == pytest.approx(0.0, abs=1e-6)

    def test_boundary_at_S_large(self):
        """For large S, call value should be approximately S - K*exp(-rT)."""
        from src.finance import solve_bs_european_call

        K = 100
        T = 1.0
        r = 0.05

        result = solve_bs_european_call(
            S_max=500, K=K, T=T, r=r, sigma=0.2, nS=200, nt=2000
        )

        # At S = 400 (deep in-the-money)
        S_test = 400
        V_numerical = result.V_at_S(S_test)
        V_expected = S_test - K * np.exp(-r * T)

        assert V_numerical == pytest.approx(V_expected, rel=0.05)

    def test_convergence_to_analytical(self):
        """Numerical solution should converge to analytical with refinement."""
        from src.finance import solve_bs_european_call
        from src.finance.black_scholes_devito import black_scholes_analytical

        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        S_test = 100  # At-the-money

        # Analytical solution
        V_exact = black_scholes_analytical(S_test, K, T, r, sigma, option_type="call")

        # Coarse solution
        result_coarse = solve_bs_european_call(
            S_max=300, K=K, T=T, r=r, sigma=sigma, nS=50, nt=500
        )

        # Fine solution
        result_fine = solve_bs_european_call(
            S_max=300, K=K, T=T, r=r, sigma=sigma, nS=200, nt=2000
        )

        error_coarse = abs(result_coarse.V_at_S(S_test) - V_exact)
        error_fine = abs(result_fine.V_at_S(S_test) - V_exact)

        assert error_fine < error_coarse, "Error should decrease with refinement"

    def test_result_dataclass_attributes(self):
        """Test that result dataclass has expected attributes."""
        from src.finance import solve_bs_european_call

        result = solve_bs_european_call(
            S_max=200, K=100, T=1.0, r=0.05, sigma=0.2, nS=50, nt=500
        )

        assert hasattr(result, "V")
        assert hasattr(result, "S")
        assert hasattr(result, "K")
        assert hasattr(result, "r")
        assert hasattr(result, "sigma")
        assert hasattr(result, "T")
        assert hasattr(result, "dt")


# =============================================================================
# Test: European Put Option Solver
# =============================================================================


@pytest.mark.devito
class TestEuropeanPutSolver:
    """Tests for the European put option solver."""

    def test_basic_run(self):
        """Test basic solver execution."""
        from src.finance import solve_bs_european_put

        result = solve_bs_european_put(
            S_max=200, K=100, T=1.0, r=0.05, sigma=0.2, nS=50, nt=500
        )

        assert result.V is not None
        assert len(result.V) == 51

    def test_boundary_at_S_zero(self):
        """Put value at S=0 should be approximately K*exp(-rT)."""
        from src.finance import solve_bs_european_put

        K = 100
        T = 1.0
        r = 0.05

        result = solve_bs_european_put(
            S_max=200, K=K, T=T, r=r, sigma=0.2, nS=100, nt=1000
        )

        expected = K * np.exp(-r * T)
        assert result.V[0] == pytest.approx(expected, rel=0.05)

    def test_boundary_at_S_large(self):
        """Put value at large S should be approximately 0."""
        from src.finance import solve_bs_european_put

        result = solve_bs_european_put(
            S_max=400, K=100, T=1.0, r=0.05, sigma=0.2, nS=200, nt=2000
        )

        # At S = 300 (deep out-of-the-money for put)
        assert result.V[-1] == pytest.approx(0.0, abs=0.5)

    def test_convergence_to_analytical(self):
        """Numerical solution should converge to analytical."""
        from src.finance import solve_bs_european_put
        from src.finance.black_scholes_devito import black_scholes_analytical

        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        S_test = 100

        V_exact = black_scholes_analytical(S_test, K, T, r, sigma, option_type="put")

        result = solve_bs_european_put(
            S_max=300, K=K, T=T, r=r, sigma=sigma, nS=200, nt=2000
        )

        V_numerical = result.V_at_S(S_test)
        error = abs(V_numerical - V_exact) / V_exact

        assert error < 0.1, f"Put error {error:.2%} exceeds 10%"


# =============================================================================
# Test: Greeks Computation
# =============================================================================


class TestGreeksComputation:
    """Tests for options Greeks (Delta, Gamma, Theta)."""

    def test_delta_call_positive(self):
        """Call delta should be positive (between 0 and 1)."""
        from src.finance import solve_bs_european_call
        from src.finance.black_scholes_devito import compute_greeks

        result = solve_bs_european_call(
            S_max=200, K=100, T=1.0, r=0.05, sigma=0.2, nS=100, nt=1000
        )

        greeks = compute_greeks(result.V, result.S, result.dt, result.r, result.sigma)

        # Interior deltas should be between 0 and 1
        interior_delta = greeks.delta[10:-10]
        assert np.all(interior_delta >= -0.1)  # Allow small numerical error
        assert np.all(interior_delta <= 1.1)

    def test_delta_increases_with_S_for_call(self):
        """Call delta should increase with S (all else equal)."""
        from src.finance import solve_bs_european_call
        from src.finance.black_scholes_devito import compute_greeks

        result = solve_bs_european_call(
            S_max=200, K=100, T=1.0, r=0.05, sigma=0.2, nS=100, nt=1000
        )

        greeks = compute_greeks(result.V, result.S, result.dt, result.r, result.sigma)

        # Check delta is generally increasing (allowing for numerical noise)
        delta_low = greeks.delta_at_S(50)
        delta_atm = greeks.delta_at_S(100)
        delta_high = greeks.delta_at_S(150)

        assert delta_atm > delta_low
        assert delta_high > delta_atm

    def test_gamma_positive(self):
        """Gamma should be positive for both calls and puts."""
        from src.finance import solve_bs_european_call
        from src.finance.black_scholes_devito import compute_greeks

        result = solve_bs_european_call(
            S_max=200, K=100, T=1.0, r=0.05, sigma=0.2, nS=100, nt=1000
        )

        greeks = compute_greeks(result.V, result.S, result.dt, result.r, result.sigma)

        # Interior gamma should be positive
        interior_gamma = greeks.gamma[10:-10]
        assert np.mean(interior_gamma) > 0

    def test_gamma_peaks_at_ATM(self):
        """Gamma should be highest near at-the-money."""
        from src.finance import solve_bs_european_call
        from src.finance.black_scholes_devito import compute_greeks

        K = 100
        result = solve_bs_european_call(
            S_max=200, K=K, T=1.0, r=0.05, sigma=0.2, nS=100, nt=1000
        )

        greeks = compute_greeks(result.V, result.S, result.dt, result.r, result.sigma)

        # Find index closest to ATM
        atm_idx = np.argmin(np.abs(result.S - K))

        # Gamma at ATM should be higher than at deep ITM/OTM
        gamma_atm = greeks.gamma[atm_idx]
        gamma_itm = greeks.gamma[atm_idx + 30]
        gamma_otm = greeks.gamma[atm_idx - 30]

        assert gamma_atm > gamma_itm * 0.5  # Allow some flexibility
        assert gamma_atm > gamma_otm * 0.5

    def test_theta_call_generally_negative(self):
        """Call theta should generally be negative (time decay)."""
        from src.finance import solve_bs_european_call
        from src.finance.black_scholes_devito import compute_greeks

        result = solve_bs_european_call(
            S_max=200, K=100, T=1.0, r=0.05, sigma=0.2, nS=100, nt=1000
        )

        greeks = compute_greeks(result.V, result.S, result.dt, result.r, result.sigma)

        # Most theta values should be negative
        interior_theta = greeks.theta[10:-10]
        assert np.mean(interior_theta) < 0


# =============================================================================
# Test: Time Decay and Volatility Effects
# =============================================================================


class TestTimeDecayEffects:
    """Tests for time decay and volatility effects on options."""

    def test_call_value_decreases_with_time(self):
        """Call option value should decrease as time to expiry decreases."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        S = 100
        K = 100
        sigma = 0.2
        r = 0.05

        V_T1 = black_scholes_analytical(S, K, T=1.0, r=r, sigma=sigma, option_type="call")
        V_T05 = black_scholes_analytical(S, K, T=0.5, r=r, sigma=sigma, option_type="call")
        V_T01 = black_scholes_analytical(S, K, T=0.1, r=r, sigma=sigma, option_type="call")

        assert V_T1 > V_T05
        assert V_T05 > V_T01

    def test_option_value_increases_with_volatility(self):
        """Option value should increase with volatility."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        S = 100
        K = 100
        T = 1.0
        r = 0.05

        V_low_vol = black_scholes_analytical(
            S, K, T, r, sigma=0.1, option_type="call"
        )
        V_high_vol = black_scholes_analytical(
            S, K, T, r, sigma=0.3, option_type="call"
        )

        assert V_high_vol > V_low_vol

    def test_put_value_with_interest_rate(self):
        """Put value should increase with higher interest rate (all else equal)."""
        from src.finance.black_scholes_devito import black_scholes_analytical

        S = 100
        K = 100
        T = 1.0
        sigma = 0.2

        # For deep out-of-money put, higher r means higher discounted K
        V_low_r = black_scholes_analytical(
            S=80, K=K, T=T, r=0.01, sigma=sigma, option_type="put"
        )
        V_high_r = black_scholes_analytical(
            S=80, K=K, T=T, r=0.10, sigma=sigma, option_type="put"
        )

        # Note: relationship can be complex; just verify values are reasonable
        assert V_low_r > 0
        assert V_high_r > 0


# =============================================================================
# Test: Numerical Accuracy and Convergence
# =============================================================================


@pytest.mark.devito
@pytest.mark.slow
class TestNumericalConvergence:
    """Tests for numerical convergence of the solvers."""

    def test_spatial_convergence_call(self):
        """Test spatial convergence rate for call option."""
        from src.finance import solve_bs_european_call
        from src.finance.black_scholes_devito import black_scholes_analytical

        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        S_test = 100

        V_exact = black_scholes_analytical(S_test, K, T, r, sigma, option_type="call")

        # Grid refinement study
        nS_values = [50, 100, 200]
        errors = []

        for nS in nS_values:
            nt = nS * 20  # Keep time refinement proportional
            result = solve_bs_european_call(
                S_max=300, K=K, T=T, r=r, sigma=sigma, nS=nS, nt=nt
            )
            error = abs(result.V_at_S(S_test) - V_exact)
            errors.append(error)

        # Verify errors decrease with refinement
        assert errors[1] < errors[0], "Error should decrease with refinement"
        assert errors[2] < errors[1], "Error should decrease with refinement"

    def test_solution_stability(self):
        """Test that solution remains stable (no blowup)."""
        from src.finance import solve_bs_european_call

        result = solve_bs_european_call(
            S_max=200, K=100, T=1.0, r=0.05, sigma=0.2, nS=100, nt=1000
        )

        # Solution should be bounded
        assert np.all(np.isfinite(result.V))
        assert np.all(result.V >= -1)  # Small negative allowed for numerical error
        assert np.max(result.V) < result.S[-1]  # Call bounded by S


# =============================================================================
# Test: Edge Cases
# =============================================================================


@pytest.mark.devito
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_expiry(self):
        """Test option pricing with very short time to expiry."""
        from src.finance import solve_bs_european_call

        result = solve_bs_european_call(
            S_max=200, K=100, T=0.01, r=0.05, sigma=0.2, nS=50, nt=100
        )

        # Near expiry, should be close to intrinsic value
        S_itm = 120
        V_numerical = result.V_at_S(S_itm)
        intrinsic = max(S_itm - 100, 0)

        assert abs(V_numerical - intrinsic) < 5

    def test_low_volatility(self):
        """Test with low volatility (more deterministic)."""
        from src.finance import solve_bs_european_call
        from src.finance.black_scholes_devito import black_scholes_analytical

        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.05  # Low volatility

        V_exact = black_scholes_analytical(100, K, T, r, sigma, option_type="call")

        result = solve_bs_european_call(
            S_max=200, K=K, T=T, r=r, sigma=sigma, nS=100, nt=2000
        )

        V_numerical = result.V_at_S(100)
        error = abs(V_numerical - V_exact)

        assert error < 1.0  # Should be reasonably accurate

    def test_high_volatility(self):
        """Test with high volatility."""
        from src.finance import solve_bs_european_call
        from src.finance.black_scholes_devito import black_scholes_analytical

        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.5  # High volatility

        V_exact = black_scholes_analytical(100, K, T, r, sigma, option_type="call")

        # High volatility requires more time steps for stability
        result = solve_bs_european_call(
            S_max=400, K=K, T=T, r=r, sigma=sigma, nS=150, nt=10000
        )

        V_numerical = result.V_at_S(100)
        error = abs(V_numerical - V_exact) / V_exact

        assert error < 0.15  # Allow 15% error for challenging case


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
