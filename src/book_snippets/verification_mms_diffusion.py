import numpy as np
from devito import Constant, Eq, Grid, Operator, TimeFunction, solve


def solve_diffusion_exact(Nx, alpha=1.0, T=0.1, F=0.4):
    """Solve diffusion equation and compare with exact eigenfunction solution.

    Uses exact solution: u(x,t) = sin(pi*x) * exp(-alpha*pi^2*t)
    which satisfies u_t = alpha * u_xx with u(0,t) = u(L,t) = 0.
    """
    L = 1.0
    dx = L / Nx
    dt = F * dx**2 / alpha
    Nt = int(T / dt)

    grid = Grid(shape=(Nx + 1,), extent=(L,))
    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)
    t_dim = grid.stepping_dim

    x_vals = np.linspace(0, L, Nx + 1)

    # Exact solution: eigenfunction of diffusion operator
    def u_exact(x, t):
        return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)

    # Initial condition from exact solution
    u.data[0, :] = u_exact(x_vals, 0)

    # Diffusion equation: u_t = alpha * u_xx
    a = Constant(name="a")
    pde = u.dt - a * u.dx2
    update = Eq(u.forward, solve(pde, u.forward), subdomain=grid.interior)

    bc_left = Eq(u[t_dim + 1, 0], 0.0)
    bc_right = Eq(u[t_dim + 1, Nx], 0.0)

    op = Operator([update, bc_left, bc_right])

    # Run all time steps at once
    op(time=Nt, dt=dt, a=alpha)

    # Compare to exact solution
    t_final = Nt * dt
    u_exact_final = u_exact(x_vals, t_final)

    # Determine which buffer has the final solution
    final_idx = Nt % 2
    error = float(np.max(np.abs(u.data[final_idx, :] - u_exact_final)))

    return error, dx


def convergence_test_mms(grid_sizes):
    """Run MMS convergence test for diffusion equation."""
    errors = []
    dx_vals = []

    for Nx in grid_sizes:
        error, dx = solve_diffusion_exact(Nx)
        errors.append(error)
        dx_vals.append(dx)

    # Compute rates
    rates = []
    for i in range(len(errors) - 1):
        rate = np.log(errors[i] / errors[i + 1]) / np.log(2)
        rates.append(float(rate))
    return rates


RESULT = convergence_test_mms([20, 40, 80, 160])
