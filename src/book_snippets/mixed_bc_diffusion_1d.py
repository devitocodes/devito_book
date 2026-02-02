import numpy as np
from devito import Eq, Grid, Operator, TimeFunction

# Mixed boundary conditions: Dirichlet on left, Neumann (copy) on right.
L = 1.0
Nx = 80
alpha = 1.0
F = 0.4

dx = L / Nx
dt = F * dx**2 / alpha

grid = Grid(shape=(Nx + 1,), extent=(L,))
t = grid.stepping_dim
u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)

x = np.linspace(0.0, L, Nx + 1)
u.data[0, :] = np.exp(-((x - 0.25) ** 2) / (2 * 0.05**2))

update = Eq(u.forward, u + alpha * dt * u.dx2, subdomain=grid.interior)

bc_left = Eq(u[t + 1, 0], 0.0)
bc_right = Eq(u[t + 1, Nx], u[t + 1, Nx - 1])  # du/dx = 0 (copy trick)

op = Operator([update, bc_left, bc_right])
op.apply(time_m=0, time_M=0)

RESULT = {
    "left_boundary": float(u.data[1, 0]),
    "right_copy_error": float(abs(u.data[1, -1] - u.data[1, -2])),
}
