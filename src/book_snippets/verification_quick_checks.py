import numpy as np
from devito import Eq, Grid, Operator, TimeFunction


def check_mass_conservation(Nx=50, alpha=1.0, T=0.1, F=0.4):
    """Check mass conservation for diffusion with Neumann BCs (approximated)."""
    L = 1.0
    dx = L / Nx
    dt = F * dx**2 / alpha
    Nt = int(T / dt)

    grid = Grid(shape=(Nx + 1,), extent=(L,))
    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)

    # Symmetric initial condition
    x_vals = np.linspace(0, L, Nx + 1)
    u.data[0, :] = np.exp(-((x_vals - 0.5) ** 2) / 0.01)

    # Diffusion with zero-flux BCs (approximate via copying)
    t_dim = grid.stepping_dim
    update = Eq(u.forward, u + alpha * dt * u.dx2, subdomain=grid.interior)
    bc_left = Eq(u[t_dim + 1, 0], u[t_dim + 1, 1])
    bc_right = Eq(u[t_dim + 1, Nx], u[t_dim + 1, Nx - 1])

    op = Operator([update, bc_left, bc_right])

    mass_initial = float(np.sum(u.data[0, :]) * dx)
    op(time=Nt, dt=dt)
    mass_final = float(np.sum(u.data[0, :]) * dx)

    return abs(mass_final - mass_initial)


def check_symmetry(Nx=50, alpha=1.0, T=0.1, F=0.4):
    """Check symmetry preservation for symmetric initial conditions."""
    L = 1.0
    dx = L / Nx
    dt = F * dx**2 / alpha
    Nt = int(T / dt)

    grid = Grid(shape=(Nx + 1,), extent=(L,))
    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)

    # Symmetric initial condition centered at L/2
    x_vals = np.linspace(0, L, Nx + 1)
    u.data[0, :] = np.exp(-((x_vals - 0.5) ** 2) / 0.01)

    t_dim = grid.stepping_dim
    update = Eq(u.forward, u + alpha * dt * u.dx2, subdomain=grid.interior)
    bc_left = Eq(u[t_dim + 1, 0], 0.0)
    bc_right = Eq(u[t_dim + 1, Nx], 0.0)

    op = Operator([update, bc_left, bc_right])
    op(time=Nt, dt=dt)

    # Check symmetry: left half vs reversed right half
    u_left = u.data[0, : Nx // 2]
    u_right = u.data[0, Nx // 2 + 1 :][::-1]
    symmetry_error = float(np.max(np.abs(u_left - u_right)))

    return symmetry_error


RESULT = {
    "mass_change": check_mass_conservation(),
    "symmetry_error": check_symmetry(),
}
