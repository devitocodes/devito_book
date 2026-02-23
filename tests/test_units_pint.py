import pytest

pint = pytest.importorskip("pint")


@pytest.fixture(scope="module")
def ureg():
    ureg = pint.UnitRegistry()
    # A generic "field" unit for scalar PDE unknowns (e.g., u(x,t)).
    ureg.define("field = [field]")
    ureg.define("velocity_field = meter / second")
    return ureg


def _is_dimensionless(q) -> bool:
    return q.dimensionality == q._REGISTRY.dimensionless.dimensionality


def _assert_dimensionless(q):
    assert q.dimensionality == q._REGISTRY.dimensionless.dimensionality


def test_diffusion_fourier_numbers_dimensionless(ureg):
    # Used in multiple diffusion snippets (Forward Euler in 1D).
    L = 1.0 * ureg.meter
    alpha = 1.0 * (ureg.meter**2 / ureg.second)

    for Nx, F in [(100, 0.5), (100, 0.4), (80, 0.4), (50, 0.4)]:
        dx = L / Nx
        dt = F * dx**2 / alpha
        _assert_dimensionless(alpha * dt / dx**2)
        assert (alpha * dt / dx**2).to_base_units().magnitude == pytest.approx(F)


def test_wave_cfl_numbers_dimensionless(ureg):
    L = 1.0 * ureg.meter
    c = 1.0 * (ureg.meter / ureg.second)

    # 1D wave snippets use dx = L/Nx.
    for Nx, C in [(100, 0.5), (100, 0.9), (200, 0.9), (80, 0.9)]:
        dx = L / Nx
        dt = C * dx / c
        _assert_dimensionless(c * dt / dx)
        assert (c * dt / dx).to_base_units().magnitude == pytest.approx(C)

    # 2D wave snippet (`bc_2d_dirichlet_wave.py`) uses dx = L/(Nx-1).
    Nx = 51
    C = 0.5
    dx = L / (Nx - 1)
    dt = C * dx / c
    _assert_dimensionless(c * dt / dx)
    assert (c * dt / dx).to_base_units().magnitude == pytest.approx(C)


def test_wave_update_term_units_match_field(ureg):
    # Check dimensional consistency of:
    # u^{n+1} = 2u^n - u^{n-1} + (c dt)^2 u_xx
    U = 1.0 * ureg.field
    L = 1.0 * ureg.meter
    c = 1.0 * (ureg.meter / ureg.second)

    Nx = 100
    C = 0.5
    dx = L / Nx
    dt = C * dx / c

    u_xx = U / (ureg.meter**2)
    term = (c * dt) ** 2 * u_xx
    assert term.dimensionality == U.dimensionality


def test_advection_cfl_numbers_dimensionless(ureg):
    L = 1.0 * ureg.meter
    c = 1.0 * (ureg.meter / ureg.second)

    for Nx, C in [(80, 0.8), (100, 0.8)]:
        dx = L / Nx
        dt = C * dx / c
        _assert_dimensionless(c * dt / dx)
        assert (c * dt / dx).to_base_units().magnitude == pytest.approx(C)


def test_burgers_equation_units_consistent(ureg):
    # Snippet `src/book_snippets/burgers_equations_bc.py` corresponds to:
    # u_t + u u_x + v u_y = nu laplace(u)
    # Interpret u, v as velocities [L/T]; then all terms are [L/T^2].
    L = 1.0 * ureg.meter
    T = 1.0 * ureg.second
    u = 1.0 * (L / T)

    u_t = u / T
    u_x = u / L
    adv = u * u_x

    nu = 1.0 * (L**2 / T)
    lap_u = u / (L**2)
    visc = nu * lap_u

    assert u_t.dimensionality == adv.dimensionality
    assert u_t.dimensionality == visc.dimensionality


def test_logistic_ode_units_consistent(ureg):
    # Logistic ODE: u_t = r u (1 - u/K)
    # r is 1/T, u and K share units.
    U = 1.0 * ureg.field
    T = 1.0 * ureg.second
    r = 1.0 / T
    K = 1.0 * ureg.field

    rhs = r * U * (1.0 - U / K)
    assert rhs.dimensionality == (U / T).dimensionality


def test_time_dependent_bc_units_consistent(ureg):
    # Snippet `src/book_snippets/time_dependent_bc_sine.py` uses:
    # u(0,t) = A sin(omega t)
    U = 1.0 * ureg.field
    T = 1.0 * ureg.second

    A = 1.0 * ureg.field
    omega = 1.0 / T
    t = 0.3 * T
    _assert_dimensionless(omega * t)

    bc = A * 0.0  # sin(...) is dimensionless; use placeholder for units.
    assert bc.dimensionality == U.dimensionality


def test_maxwell_fdtd_units_consistent(ureg):
    # 1D Maxwell (Yee) updates:
    #   E^{n+1} = E^n + (dt/eps) * dH/dx
    #   H^{n+1/2} = H^{n-1/2} + (dt/mu) * dE/dx
    E = 1.0 * (ureg.volt / ureg.meter)
    H = 1.0 * (ureg.ampere / ureg.meter)
    eps = 1.0 * (ureg.farad / ureg.meter)
    mu = 1.0 * (ureg.henry / ureg.meter)

    dx = 0.01 * ureg.meter
    dt = 1e-10 * ureg.second

    dH_dx = H / ureg.meter
    dE_dx = E / ureg.meter

    e_update_term = (dt / eps) * dH_dx
    h_update_term = (dt / mu) * dE_dx

    assert e_update_term.dimensionality == E.dimensionality
    assert h_update_term.dimensionality == H.dimensionality


def test_maxwell_cfl_number_dimensionless(ureg):
    # CFL: C = c dt / dx
    c = 3e8 * (ureg.meter / ureg.second)
    dx = 0.01 * ureg.meter
    dt = 0.9 * dx / c

    _assert_dimensionless(c * dt / dx)
    assert (c * dt / dx).to_base_units().magnitude == pytest.approx(0.9)
