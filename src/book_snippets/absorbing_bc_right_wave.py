import numpy as np
from devito import Eq, Grid, Operator, TimeFunction

# First-order absorbing boundary condition at the right boundary (1D wave).
L = 1.0
Nx = 200
c = 1.0
C = 0.9

dx = L / Nx
dt = C * dx / c
Nt = 10

grid = Grid(shape=(Nx + 1,), extent=(L,))
t = grid.stepping_dim
u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

x = np.linspace(0.0, L, Nx + 1)
u.data[0, :] = np.exp(-((x - 0.8) ** 2) / (2 * 0.03**2))
u.data[1, :] = u.data[0, :]

update = Eq(u.forward, 2 * u - u.backward + (c * dt) ** 2 * u.dx2, subdomain=grid.interior)
bc_left = Eq(u[t + 1, 0], 0.0)

dx_sym = grid.spacing[0]
bc_right_absorbing = Eq(u[t + 1, Nx], u[t, Nx] - c * dt / dx_sym * (u[t, Nx] - u[t, Nx - 1]))

op = Operator([update, bc_left, bc_right_absorbing])
op(time=Nt, dt=dt)

RESULT = float(np.max(np.abs(u.data[0, :])))
