import numpy as np
from devito import Constant, Eq, Grid, Operator, TimeFunction


def solve_advection_upwind(L, c, Nx, T, C, I):
    """Upwind scheme for 1D advection."""
    # Grid setup
    dx = L / Nx
    dt = C * dx / c
    Nt = int(T / dt)

    grid = Grid(shape=(Nx + 1,), extent=(L,))
    (x_dim,) = grid.dimensions

    u = TimeFunction(name="u", grid=grid, time_order=1, space_order=1)

    # Set initial condition
    x_coords = np.linspace(0, L, Nx + 1)
    u.data[0, :] = I(x_coords)

    # Courant number as constant
    courant = Constant(name="C", value=C)

    # Upwind stencil: u^{n+1} = u - C*(u - u[x-dx])
    u_minus = u.subs(x_dim, x_dim - x_dim.spacing)
    stencil = u - courant * (u - u_minus)
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


# Test the upwind scheme
u_final, x = solve_advection_upwind(L=1.0, c=1.0, Nx=100, T=0.5, C=0.8, I=I_gaussian)
RESULT = {"max_u": float(np.max(u_final)), "u_shape": u_final.shape}
