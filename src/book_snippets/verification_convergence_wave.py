import numpy as np
from devito import Eq, Grid, Operator, TimeFunction


def solve_wave_equation(Nx, L=1.0, T=0.5, c=1.0, C=0.5):
    dx = L / Nx
    dt = C * dx / c
    Nt = int(T / dt)

    grid = Grid(shape=(Nx + 1,), extent=(L,))
    t_dim = grid.stepping_dim
    u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

    x_vals = np.linspace(0, L, Nx + 1)
    u.data[0, :] = np.sin(np.pi * x_vals)
    u.data[1, :] = np.sin(np.pi * x_vals) * np.cos(np.pi * c * dt)

    update = Eq(u.forward, 2 * u - u.backward + (c * dt) ** 2 * u.dx2, subdomain=grid.interior)
    bc_left = Eq(u[t_dim + 1, 0], 0.0)
    bc_right = Eq(u[t_dim + 1, Nx], 0.0)

    op = Operator([update, bc_left, bc_right])
    op(time=Nt, dt=dt)

    t_final = Nt * dt
    u_exact = np.sin(np.pi * x_vals) * np.cos(np.pi * c * t_final)
    # For time_order=2, buffer has 3 slots; final solution is at Nt % 3
    final_idx = Nt % 3
    error = float(np.max(np.abs(u.data[final_idx, :] - u_exact)))
    return error, dx


def convergence_test(grid_sizes):
    errors = []
    dx_values = []

    for Nx in grid_sizes:
        error, dx = solve_wave_equation(Nx)
        errors.append(error)
        dx_values.append(dx)

    rates = []
    for i in range(len(errors) - 1):
        rate = np.log(errors[i] / errors[i + 1]) / np.log(dx_values[i] / dx_values[i + 1])
        rates.append(float(rate))
    return rates


# Use grid sizes that avoid numerical resonance issues
RESULT = convergence_test([25, 50, 100, 200])
