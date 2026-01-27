"""Pytest configuration and fixtures for the test suite.

This module provides shared fixtures and configuration for testing
finite difference derivations and Devito solvers.
"""

# Import project modules
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.plotting import RANDOM_SEED, set_seed
from src.symbols import (
    alpha,
    c,
    dt,
    dx,
    dy,
    dz,
    f,
    h,
    i,
    j,
    k,
    n,
    nu,
    t,
    u,
    v,
    x,
    y,
    z,
)

# =============================================================================
# Session Setup
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "devito: marks tests that require Devito installation"
    )
    config.addinivalue_line(
        "markers", "derivation: marks tests that verify mathematical derivations"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_random_seed():
    """Ensure reproducibility across all tests."""
    set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    yield


# =============================================================================
# Symbol Fixtures
# =============================================================================

@pytest.fixture
def spatial_symbols():
    """Provide standard spatial symbols."""
    return {'x': x, 'y': y, 'z': z, 'dx': dx, 'dy': dy, 'dz': dz, 'h': h}


@pytest.fixture
def temporal_symbols():
    """Provide standard temporal symbols."""
    return {'t': t, 'dt': dt, 'k': k}


@pytest.fixture
def function_symbols():
    """Provide standard function symbols."""
    return {'u': u, 'v': v, 'f': f}


@pytest.fixture
def index_symbols():
    """Provide standard index symbols."""
    return {'i': i, 'j': j, 'n': n}


@pytest.fixture
def physical_params():
    """Provide standard physical parameter symbols."""
    return {'alpha': alpha, 'c': c, 'nu': nu}


# =============================================================================
# Test Function Fixtures
# =============================================================================

@pytest.fixture
def test_polynomial():
    """Simple polynomial for testing derivatives: f(x) = x^3 + 2x^2 + x + 1."""
    return x**3 + 2*x**2 + x + 1


@pytest.fixture
def test_trig():
    """Trigonometric function: f(x) = sin(x)."""
    return sp.sin(x)


@pytest.fixture
def test_exp():
    """Exponential function: f(x) = exp(-x^2)."""
    return sp.exp(-x**2)


@pytest.fixture
def test_function_u():
    """Generic function u(x) for derivative testing."""
    return u(x)


@pytest.fixture
def test_function_u_xt():
    """Generic function u(x, t) for PDE testing."""
    return u(x, t)


# =============================================================================
# Analytical Solutions
# =============================================================================

@pytest.fixture
def heat_equation_solution():
    """Analytical solution to heat equation u_t = alpha * u_xx.

    Solution: u(x, t) = exp(-alpha * pi^2 * t) * sin(pi * x)
    Domain: x in [0, 1], t >= 0
    BCs: u(0, t) = u(1, t) = 0
    IC: u(x, 0) = sin(pi * x)
    """
    return sp.exp(-alpha * sp.pi**2 * t) * sp.sin(sp.pi * x)


@pytest.fixture
def wave_equation_solution():
    """Analytical solution to wave equation u_tt = c^2 * u_xx.

    Solution: u(x, t) = sin(pi * x) * cos(pi * c * t)
    Domain: x in [0, 1], t >= 0
    BCs: u(0, t) = u(1, t) = 0
    IC: u(x, 0) = sin(pi * x), u_t(x, 0) = 0
    """
    return sp.sin(sp.pi * x) * sp.cos(sp.pi * c * t)


@pytest.fixture
def advection_equation_solution():
    """Analytical solution to advection equation u_t + c * u_x = 0.

    Solution: u(x, t) = f(x - c*t) for any smooth f
    Example: u(x, t) = exp(-(x - c*t)^2)
    """
    return sp.exp(-(x - c*t)**2)


# =============================================================================
# Grid Fixtures
# =============================================================================

@pytest.fixture
def grid_1d():
    """Standard 1D grid for testing."""
    Nx = 101
    Lx = 1.0
    x_grid = np.linspace(0, Lx, Nx)
    dx_val = x_grid[1] - x_grid[0]
    return {'x': x_grid, 'dx': dx_val, 'Nx': Nx, 'Lx': Lx}


@pytest.fixture
def grid_2d():
    """Standard 2D grid for testing."""
    Nx, Ny = 51, 51
    Lx, Ly = 1.0, 1.0
    x_grid = np.linspace(0, Lx, Nx)
    y_grid = np.linspace(0, Ly, Ny)
    dx_val = x_grid[1] - x_grid[0]
    dy_val = y_grid[1] - y_grid[0]
    X, Y = np.meshgrid(x_grid, y_grid)
    return {
        'x': x_grid, 'y': y_grid,
        'X': X, 'Y': Y,
        'dx': dx_val, 'dy': dy_val,
        'Nx': Nx, 'Ny': Ny,
        'Lx': Lx, 'Ly': Ly,
    }


@pytest.fixture
def time_grid():
    """Standard time discretization for testing."""
    Nt = 100
    T_final = 0.1
    t_grid = np.linspace(0, T_final, Nt + 1)
    dt_val = t_grid[1] - t_grid[0]
    return {'t': t_grid, 'dt': dt_val, 'Nt': Nt, 'T_final': T_final}


# =============================================================================
# Devito Fixtures (conditional on Devito availability)
# =============================================================================

@pytest.fixture
def devito_available():
    """Check if Devito is available."""
    import importlib.util
    return importlib.util.find_spec("devito") is not None


@pytest.fixture
def devito_grid_1d(devito_available):
    """Devito 1D grid fixture."""
    if not devito_available:
        pytest.skip("Devito not available")

    from devito import Grid
    return Grid(shape=(101,), extent=(1.0,))


@pytest.fixture
def devito_grid_2d(devito_available):
    """Devito 2D grid fixture."""
    if not devito_available:
        pytest.skip("Devito not available")

    from devito import Grid
    return Grid(shape=(101, 101), extent=(1.0, 1.0))


# =============================================================================
# Helper Functions
# =============================================================================

@pytest.fixture
def assert_sympy_equal():
    """Fixture providing a function to compare SymPy expressions."""
    def _compare(expr1, expr2, expand=True, simplify=True):
        """Compare two SymPy expressions for equality.

        Parameters
        ----------
        expr1, expr2 : sympy expressions
            Expressions to compare
        expand : bool
            Whether to expand expressions before comparing
        simplify : bool
            Whether to simplify the difference

        Returns
        -------
        bool
            True if expressions are equal
        """
        if expand:
            expr1 = sp.expand(expr1)
            expr2 = sp.expand(expr2)

        diff = expr1 - expr2

        if simplify:
            diff = sp.simplify(diff)

        return diff == 0

    return _compare


@pytest.fixture
def numerical_tolerance():
    """Standard numerical tolerance for floating-point comparisons."""
    return 1e-10
