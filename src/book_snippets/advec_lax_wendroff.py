import numpy as np
from devito import Constant, Eq, Grid, Operator, TimeFunction


def solve_advection_lax_wendroff(L, c, Nx, T, C, I):
    """Lax-Wendroff scheme for 1D advection."""
    dx = L / Nx
    dt = C * dx / c
    Nt = int(T / dt)

    grid = Grid(shape=(Nx + 1,), extent=(L,))
    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)

    x_coords = np.linspace(0, L, Nx + 1)
    u.data[0, :] = I(x_coords)

    courant = Constant(name="C", value=C)

    # Lax-Wendroff: u - (C/2)*dx*u.dx + (C²/2)*dx²*u.dx2
    # u.dx  = centered first derivative
    # u.dx2 = centered second derivative
    stencil = u - 0.5 * courant * dx * u.dx + 0.5 * courant**2 * dx**2 * u.dx2
    update = Eq(u.forward, stencil)

    # Periodic boundary conditions
    t_dim = grid.stepping_dim
    bc_left = Eq(u[t_dim + 1, 0], u[t_dim, Nx])
    bc_right = Eq(u[t_dim + 1, Nx], u[t_dim + 1, 0])

    op = Operator([update, bc_left, bc_right])
    op(time=Nt, dt=dt)

    return u.data[0, :].copy(), x_coords


def I_gaussian(x):
    """Gaussian pulse initial condition."""
    return np.exp(-0.5 * ((x - 0.25) / 0.05) ** 2)


# Test the Lax-Wendroff scheme
u_final, x = solve_advection_lax_wendroff(
    L=1.0, c=1.0, Nx=100, T=0.5, C=0.8, I=I_gaussian
)
RESULT = {"max_u": float(np.max(u_final)), "u_shape": u_final.shape}
